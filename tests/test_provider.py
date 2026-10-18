"""Tests for the OpenAI-compatible provider (Ollama, Gemini, ...). No network."""

from __future__ import annotations

from typing import Any

import pytest

from research_agent.llm import OpenAICompatLLM, live_configured, make_llm


class FakeResponse:
    def __init__(self, status: int, data: dict[str, Any] | None = None) -> None:
        self.status_code = status
        self._data = data or {}
        self.text = str(self._data)

    def json(self) -> dict[str, Any]:
        return self._data


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def post(self, url: str, json: Any, headers: Any, timeout: Any) -> FakeResponse:
        self.payloads.append(json)
        self.headers.append(headers)
        return self.responses.pop(0)


def make(session: FakeSession, key: str | None = "k") -> OpenAICompatLLM:
    return OpenAICompatLLM("http://x/v1/", "m", api_key=key, session=session, sleep=lambda s: None)


def test_message_conversion() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking"},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "calculator",
                    "input": {"expression": "1+1"},
                },
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "2"}]},
    ]
    out = OpenAICompatLLM.to_openai_messages("sys", messages)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"expression": "1+1"}'
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "2"}


def test_tool_conversion() -> None:
    spec = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    out = OpenAICompatLLM.to_openai_tools(spec)
    assert out[0]["function"]["parameters"] == {"type": "object"}


def test_complete_parses_tool_calls_and_usage() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "2*3"}',
                            },
                        },
                        {"id": "b", "function": {"name": "weather", "arguments": "not json"}},
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }
    session = FakeSession([FakeResponse(200, data)])
    resp = make(session).complete("s", [{"role": "user", "content": "q"}], [])
    assert resp.tool_calls[0].input == {"expression": "2*3"}
    assert resp.tool_calls[0].id  # generated when the server sends none
    assert resp.tool_calls[1].input == {}
    assert resp.usage.input_tokens == 10 and resp.usage.output_tokens == 4
    assert session.headers[0]["Authorization"] == "Bearer k"
    assert "tools" not in session.payloads[0]


def test_text_answer() -> None:
    data = {"choices": [{"message": {"content": " done "}}]}
    resp = make(FakeSession([FakeResponse(200, data)]), key=None).complete("s", [], [])
    assert resp.text == "done" and not resp.tool_calls


def test_retries_then_succeeds() -> None:
    session = FakeSession([FakeResponse(429), FakeResponse(200, {"choices": [{"message": {}}]})])
    make(session).complete("s", [], [])
    assert len(session.payloads) == 2


def test_gives_up_after_attempts() -> None:
    session = FakeSession([FakeResponse(503)] * 3)
    with pytest.raises(RuntimeError):
        make(session).complete("s", [], [])


def test_client_error_is_not_retried() -> None:
    session = FakeSession([FakeResponse(401, {"error": "bad key"})])
    with pytest.raises(RuntimeError) as err:
        make(session).complete("s", [], [])
    assert "401" in str(err.value) and len(session.payloads) == 1


def test_requires_model_and_url() -> None:
    with pytest.raises(RuntimeError):
        OpenAICompatLLM("", "m")
    with pytest.raises(RuntimeError):
        OpenAICompatLLM("http://x", "")


def test_make_llm_and_live_configured(monkeypatch: Any) -> None:
    for name in ("ANTHROPIC_API_KEY", "RESEARCH_AGENT_BASE_URL", "RESEARCH_AGENT_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    assert not live_configured()
    monkeypatch.setenv("RESEARCH_AGENT_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("RESEARCH_AGENT_MODEL", "llama3.1")
    assert live_configured()
    llm = make_llm()
    assert isinstance(llm, OpenAICompatLLM) and llm.model == "llama3.1"
    with pytest.raises(RuntimeError):
        make_llm("nope")


def test_extra_content_round_trip() -> None:
    sig = {"google": {"thought_signature": "abc123"}}
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "weather", "arguments": '{"city": "Pune"}'},
                            "extra_content": sig,
                        }
                    ]
                }
            }
        ]
    }
    first = make(FakeSession([FakeResponse(200, data)])).complete("s", [], [])
    assert first.tool_calls[0].extra == sig
    message = first.to_message()
    assert message["content"][0]["extra"] == sig
    out = OpenAICompatLLM.to_openai_messages("s", [message])
    assert out[1]["tool_calls"][0]["extra_content"] == sig
