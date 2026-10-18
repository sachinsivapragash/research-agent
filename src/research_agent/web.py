"""Local web UI: `research-agent-ui` then open http://127.0.0.1:8765

Standard library only. The browser opens a Server-Sent Events stream at `/api/run`;
every event the agent emits (model turns, tool starts, tool results, the final answer
and its citation check) is pushed to the page as it happens.

Binds to 127.0.0.1 by default, so it is only reachable from your own computer.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlparse

from research_agent.agent import Agent
from research_agent.demo import DEMO_QUESTION, demo_llm, demo_registry
from research_agent.llm import LLM
from research_agent.tools import Notes, ToolRegistry, default_registry
from research_agent.tracing import Tracer

MAX_QUESTION_CHARS = 500
MAX_CONCURRENT_RUNS = 2
_DONE = object()


def _index_html() -> bytes:
    return resources.files("research_agent").joinpath("static/index.html").read_bytes()


class AgentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], demo_delay: float = 1.0) -> None:
        super().__init__(address, Handler)
        self.demo_delay = demo_delay
        self.slots = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)

    @property
    def live_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))


class Handler(BaseHTTPRequestHandler):
    server: AgentServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass  # keep the terminal clean

    # ---------------------------------------------------------------- helpers

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict[str, Any]) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    # ----------------------------------------------------------------- routes

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, _index_html(), "text/html; charset=utf-8")
        elif url.path == "/api/config":
            self._json(
                HTTPStatus.OK,
                {
                    "live": self.server.live_available,
                    "model": os.environ.get("RESEARCH_AGENT_MODEL", "claude-sonnet-5-5"),
                    "demo_question": DEMO_QUESTION,
                },
            )
        elif url.path == "/api/run":
            self._run(parse_qs(url.query))
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _run(self, query: dict[str, list[str]]) -> None:
        mode = (query.get("mode") or ["demo"])[0]
        question = (query.get("q") or [""])[0].strip()

        llm: LLM
        tools: ToolRegistry
        if mode == "demo":
            question = question or DEMO_QUESTION
            llm = demo_llm(self.server.demo_delay)
            tools = demo_registry(self.server.demo_delay * 0.8)
        elif mode == "live":
            if not question:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "question is empty"})
            if not self.server.live_available:
                return self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "ANTHROPIC_API_KEY is not set on the server. Use Demo mode."},
                )
            from research_agent.llm import AnthropicLLM

            llm = AnthropicLLM()
            tools = default_registry(Notes())
        else:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "mode must be demo or live"})

        if len(question) > MAX_QUESTION_CHARS:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "question is too long"})
        if not self.server.slots.acquire(blocking=False):
            return self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "busy, try again"})

        try:
            self._stream(question, llm, tools)
        finally:
            self.server.slots.release()

    def _stream(self, question: str, llm: LLM, tools: ToolRegistry) -> None:
        events: queue.Queue[Any] = queue.Queue()
        agent = Agent(llm, tools, tracer=Tracer(listener=events.put))

        def work() -> None:
            try:
                agent.run(question)
            except Exception as exc:  # noqa: BLE001 - surface any failure to the page
                events.put({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                events.put(_DONE)

        threading.Thread(target=work, daemon=True).start()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        while True:
            item = events.get()
            if item is _DONE:
                payload = {"event": "done"}
            else:
                payload = item
            try:
                self.wfile.write(f"data: {json.dumps(payload, default=str)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return  # the browser went away; the agent thread finishes on its own
            if item is _DONE:
                return


def serve(host: str = "127.0.0.1", port: int = 8765, demo_delay: float = 1.0) -> AgentServer:
    return AgentServer((host, port), demo_delay=demo_delay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-agent-ui", description="Open the 3D web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    args = parser.parse_args(argv)

    server = serve(args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}"
    mode = "live + demo" if server.live_available else "demo only (set ANTHROPIC_API_KEY for live)"
    print(f"research-agent UI running at {url}  [{mode}]  Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
