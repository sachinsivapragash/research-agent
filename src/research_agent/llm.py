"""LLM layer.

The agent only depends on the small `LLM` protocol below, so any provider can be
plugged in. `AnthropicLLM` talks to the Claude API. `ScriptedLLM` replays a fixed
list of responses, which is used for unit tests and for the offline `--demo` mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"


@dataclass
class ToolCall:
    """A request from the model to run one tool."""

    id: str
    name: str
    input: dict[str, Any]


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
            blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
            )
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
