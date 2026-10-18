"""LLM layer.

The agent only depends on the small `LLM` protocol below, so any provider can be
plugged in. `AnthropicLLM` talks to the Claude API. `ScriptedLLM` replays a fixed
list of responses, which is used for unit tests and for the offline `--demo` mode.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass
class ToolCall:
    """A request from the model to run one tool."""

    id: str
    name: str
    input: dict[str, Any]
    # Provider extras that must be sent back unchanged (e.g. Gemini thought signatures)
    extra: dict[str, Any] | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )

    def cost(self, input_per_mtok: float | None, output_per_mtok: float | None) -> float | None:
        """Cost in dollars, or None when prices are not configured."""
        if input_per_mtok is None or output_per_mtok is None:
            return None
        return (
            self.input_tokens * input_per_mtok + self.output_tokens * output_per_mtok
        ) / 1_000_000


@dataclass
class LLMResponse:
    """One model turn: optional text plus zero or more tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def to_message(self) -> dict[str, Any]:
        """Convert to an assistant message in content-block format."""
        blocks: list[dict[str, Any]] = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for call in self.tool_calls:
            block: dict[str, Any] = {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.input,
            }
            if call.extra:
                block["extra"] = call.extra
            blocks.append(block)
        return {"role": "assistant", "content": blocks}


class LLM(Protocol):
    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...


class AnthropicLLM:
    """Claude API client. Reads ANTHROPIC_API_KEY from the environment.

    The SDK already retries rate limits and server errors with backoff; `max_retries`
    controls how many times.
    """

    def __init__(
        self, model: str | None = None, max_tokens: int = 2048, max_retries: int = 4
    ) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it first, or run with --demo "
                "to try the agent offline."
            )
        import anthropic  # imported lazily so tests and --demo work without it

        self._client = anthropic.Anthropic(max_retries=max_retries)
        self.model = model or os.environ.get("RESEARCH_AGENT_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(getattr(block, "text", ""))
            elif kind == "tool_use":
                calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        input=dict(getattr(block, "input", {}) or {}),
                    )
                )
        usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
        return LLMResponse(text="\n".join(text_parts).strip(), tool_calls=calls, usage=usage)


class ScriptedLLM:
    """Replays a fixed list of responses. No network, fully deterministic."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        if self.calls >= len(self._responses):
            raise RuntimeError("ScriptedLLM ran out of scripted responses")
        response = self._responses[self.calls]
        self.calls += 1
        return response


class OpenAICompatLLM:
    """Any server that speaks the OpenAI chat-completions format with tool calling.

    Works with Ollama (free, runs on your PC), and hosted services that offer an
    OpenAI-compatible endpoint. Configure with a base URL, model name and optional key:

        RESEARCH_AGENT_BASE_URL=http://localhost:11434/v1   (Ollama)
        RESEARCH_AGENT_MODEL=llama3.1
        RESEARCH_AGENT_API_KEY=...                          (not needed for Ollama)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 2048,
        attempts: int = 3,
        session: Any = None,
        sleep: Any = time.sleep,
    ) -> None:
        if not base_url:
            raise RuntimeError("RESEARCH_AGENT_BASE_URL is not set.")
        if not model:
            raise RuntimeError(
                "No model chosen. Set RESEARCH_AGENT_MODEL or pass --model, for example "
                "--model llama3.1 when using Ollama."
            )
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.attempts = attempts
        self._session = session
        self._sleep = sleep

    # ----------------------------------------------------------- conversion

    @staticmethod
    def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                out.append({"role": message["role"], "content": content})
            elif message["role"] == "assistant":
                text = "\n".join(b["text"] for b in content if b["type"] == "text")
                calls = []
                for b in content:
                    if b["type"] != "tool_use":
                        continue
                    call: dict[str, Any] = {
                        "id": b["id"],
                        "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                    }
                    if b.get("extra"):
                        call["extra_content"] = b["extra"]
                    calls.append(call)
                entry: dict[str, Any] = {"role": "assistant", "content": text or None}
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
            else:  # user turn made of tool results
                for b in content:
                    if b["type"] == "tool_result":
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": b["tool_use_id"],
                                "content": b["content"],
                            }
                        )
        return out

    @staticmethod
    def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    # ------------------------------------------------------------- request

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        import requests

        client = self._session or requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                response = client.post(self.url, json=payload, headers=headers, timeout=120)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
            else:
                if response.status_code in (429, 500, 502, 503, 504):
                    last = requests.HTTPError(f"HTTP {response.status_code} from {self.url}")
                else:
                    if response.status_code >= 400:
                        raise RuntimeError(
                            f"Model server returned HTTP {response.status_code}: "
                            f"{response.text[:300]}"
                        )
                    data: dict[str, Any] = response.json()
                    return data
            if attempt < self.attempts - 1:
                self._sleep(min(8.0, 0.5 * 2**attempt))
        assert last is not None
        raise RuntimeError(f"Could not reach the model server at {self.url}: {last}")

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_openai_messages(system, messages),
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = self.to_openai_tools(tools)
        data = self._post(payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Model server returned no choices.")
        message = choices[0].get("message", {})
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function", {})
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    name=function.get("name", ""),
                    input=args,
                    extra=raw.get("extra_content") or None,
                )
            )
        usage = data.get("usage") or {}
        return LLMResponse(
            text=(message.get("content") or "").strip(),
            tool_calls=calls,
            usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
        )


def live_configured() -> bool:
    """True when some real model is configured (Claude key or an OpenAI-compatible server)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("RESEARCH_AGENT_BASE_URL"))


def make_llm(provider: str | None = None, model: str | None = None) -> LLM:
    """Build the real model client from flags and environment variables."""
    base_url = os.environ.get("RESEARCH_AGENT_BASE_URL", "")
    chosen = (
        provider
        or os.environ.get("RESEARCH_AGENT_PROVIDER")
        or ("openai" if base_url else "anthropic")
    )
    if chosen == "openai":
        return OpenAICompatLLM(
            base_url=base_url,
            model=model or os.environ.get("RESEARCH_AGENT_MODEL", ""),
            api_key=os.environ.get("RESEARCH_AGENT_API_KEY"),
        )
    if chosen == "anthropic":
        return AnthropicLLM(model=model)
    raise RuntimeError(f"Unknown provider '{chosen}'. Use 'anthropic' or 'openai'.")
