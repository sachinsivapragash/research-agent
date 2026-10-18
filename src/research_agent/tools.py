"""Tools the agent can call.

Each tool has a name, a description and a JSON schema (what the model sees), plus a
plain Python function. Tools return strings and raise exceptions on failure; the agent
turns exceptions into error results so one bad call never crashes the run.
"""

from __future__ import annotations

import ast
import ipaddress
import json
import operator
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

MAX_PAGE_CHARS = 6000
MAX_DOWNLOAD_BYTES = 1_000_000


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., str]
    cacheable: bool = True  # same input -> same output, safe to reuse within a run
    untrusted: bool = False  # output comes from the internet; wrap it before the model sees it

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Holds tools, validates calls and caches results. Safe to use from several threads."""

    def __init__(self, cache: bool = True) -> None:
        self._tools: dict[str, Tool] = {}
        self._cache_enabled = cache
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self.cache_hits = 0

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def _get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return tool

    def is_untrusted(self, name: str) -> bool:
        return name in self._tools and self._tools[name].untrusted

    @staticmethod
    def _key(name: str, args: dict[str, Any]) -> str:
        return name + ":" + json.dumps(args, sort_keys=True, default=str)

    def is_cached(self, name: str, args: dict[str, Any]) -> bool:
        with self._lock:
            return self._key(name, args) in self._cache

    def _validate(self, tool: Tool, args: dict[str, Any]) -> None:
        if not isinstance(args, dict):
            raise TypeError("tool input must be an object")
        schema = tool.input_schema
        missing = [k for k in schema.get("required", []) if k not in args]
        if missing:
            raise ValueError(f"missing required argument(s): {', '.join(missing)}")
        unknown = [k for k in args if k not in schema.get("properties", {})]
        if unknown:
            raise ValueError(f"unknown argument(s): {', '.join(unknown)}")

    def run(self, name: str, args: dict[str, Any]) -> str:
        tool = self._get(name)
        self._validate(tool, args)
        use_cache = self._cache_enabled and tool.cacheable
        key = self._key(name, args)
        if use_cache:
            with self._lock:
                if key in self._cache:
                    self.cache_hits += 1
                    return self._cache[key]
        output = tool.func(**args)
        if use_cache:
            with self._lock:
                self._cache[key] = output
        return output


# ---------------------------------------------------------------- calculator

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("only numbers and + - * / // % ** ( ) are allowed")


def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression (no eval, no names, no calls)."""
    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval_node(tree)
    return str(round(result, 10) if isinstance(result, float) else result)


# ---------------------------------------------------------------------- notes


@dataclass
class Notes:
    items: list[str] = field(default_factory=list)

    def save(self, text: str) -> str:
        self.items.append(text.strip())
        return f"Saved note #{len(self.items)}."

    def read(self) -> str:
        if not self.items:
            return "No notes yet."
        return "\n".join(f"{i}. {note}" for i, note in enumerate(self.items, start=1))


# ----------------------------------------------------------------- web search


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo via the `ddgs` package) and return titles, URLs, snippets."""
    from ddgs import DDGS  # lazy import: tests never need the network package

    max_results = max(1, min(int(max_results), 10))
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return "No results found."
    lines = []
    for i, item in enumerate(results, start=1):
        title, href, body = item.get("title", ""), item.get("href", ""), item.get("body", "")
        lines.append(f"{i}. {title}\n   {href}\n   {body}")
    return "\n".join(lines)


# ------------------------------------------------------------------ fetch page


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return " ".join(parser.parts)


def check_url_is_public(url: str) -> None:
    """Block non-http(s) URLs and anything that resolves to a private address (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("only http and https URLs are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve host: {parsed.hostname}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            raise ValueError("refusing to fetch a private or local address")


def fetch_page(url: str) -> str:
    """Download a web page and return its readable text (truncated)."""
    from research_agent.http import get_with_retry

    check_url_is_public(url)
    response = get_with_retry(url, stream=True, allow_redirects=False)
    if response.is_redirect:
        location = response.headers.get("Location", "")
        raise ValueError(f"page redirects to {location}; call fetch_page again with that URL")
    raw = response.raw.read(MAX_DOWNLOAD_BYTES, decode_content=True)
    html = raw.decode(response.encoding or "utf-8", errors="replace")
    text = html_to_text(html)
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + " ...[truncated]"
    return text or "Page had no readable text."


# -------------------------------------------------------------------- registry


def _register_integrations(registry: ToolRegistry) -> None:
    from research_agent import integrations

    def one_text_arg(
        name: str, description: str, arg: str, func: Callable[..., str], untrusted: bool = True
    ) -> Tool:
        return Tool(
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {arg: {"type": "string"}},
                "required": [arg],
            },
            func=func,
            untrusted=untrusted,
        )

    registry.register(
        one_text_arg(
            "wikipedia_summary",
            "Get a short encyclopedia summary of a topic from Wikipedia.",
            "title",
            integrations.wikipedia_summary,
        )
    )
    registry.register(
        one_text_arg(
            "github_repo_info",
            "Get public facts (stars, language, last push) about a GitHub repo as 'owner/name'.",
            "repo",
            integrations.github_repo_info,
        )
    )
    registry.register(
        Tool(
            name="arxiv_search",
            description="Search arXiv for research papers on a topic.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "1 to 8, default 3"},
                },
                "required": ["query"],
            },
            func=integrations.arxiv_search,
            untrusted=True,
        )
    )
    registry.register(
        one_text_arg(
            "weather",
            "Get the current weather for a city.",
            "city",
            integrations.weather,
            untrusted=False,
        )
    )


def default_registry(notes: Notes | None = None, cache: bool = True) -> ToolRegistry:
    notes = notes or Notes()
    registry = ToolRegistry(cache=cache)
    _register_integrations(registry)
    registry.register(
        Tool(
            name="web_search",
            description="Search the web. Returns titles, URLs and short snippets.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "1 to 10, default 5"},
                },
                "required": ["query"],
            },
            func=web_search,
            untrusted=True,
        )
    )
    registry.register(
        Tool(
            name="fetch_page",
            description="Download a public web page and return its readable text.",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full http(s) URL"}},
                "required": ["url"],
            },
            func=fetch_page,
            untrusted=True,
        )
    )
    registry.register(
        Tool(
            name="calculator",
            description="Evaluate an arithmetic expression such as (1.07 ** 10) * 1000.",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            func=calculator,
        )
    )
    registry.register(
        Tool(
            name="save_note",
            description="Save a short finding (with its source URL) to use in the final report.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            func=notes.save,
            cacheable=False,
        )
    )
    registry.register(
        Tool(
            name="read_notes",
            description="Read back all notes saved so far.",
            input_schema={"type": "object", "properties": {}},
            func=notes.read,
            cacheable=False,
        )
    )
    return registry
