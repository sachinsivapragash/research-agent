"""Tests for the web UI server and the live event stream (no network, no API key)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from research_agent.agent import Agent
from research_agent.demo import FAKE, demo_llm, demo_registry
from research_agent.tracing import Tracer
from research_agent.web import serve


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    srv = serve(port=0, demo_delay=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read().decode()


def read_events(body):
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_index_page_is_served(server):
    status, ctype, body = get(server + "/")
    assert status == 200
    assert ctype.startswith("text/html")
    assert "<canvas" in body
    assert "evidence lab" in body


def test_config_reports_demo_only_without_key(server):
    _, _, body = get(server + "/api/config")
    data = json.loads(body)
    assert data["live"] is False
    assert data["demo_question"]


def test_demo_run_streams_full_event_sequence(server):
    status, ctype, body = get(server + "/api/run?mode=demo")
    assert status == 200
    assert ctype.startswith("text/event-stream")
    events = read_events(body)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start"
    assert kinds[-1] == "done"
    assert kinds.count("tool_start") == kinds.count("tool_result") == 6
    end = next(e for e in events if e["event"] == "run_end")
    assert end["finished"] is True
    assert end["ungrounded"] == [FAKE]
    assert len(end["grounded"]) == 3
    results = [e for e in events if e["event"] == "tool_result"]
    assert any(e["cached"] for e in results)
    assert any(e["urls"] for e in results)


def test_live_mode_without_key_is_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        get(server + "/api/run?mode=live&q=hello")
    assert err.value.code == 400
    assert "ANTHROPIC_API_KEY" in err.value.read().decode()


def test_bad_mode_and_long_question_are_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        get(server + "/api/run?mode=evil")
    assert err.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        get(server + "/api/run?mode=demo&q=" + "x" * 600)
    assert err.value.code == 400


def test_unknown_path_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        get(server + "/nope")
    assert err.value.code == 404


def test_tracer_listener_receives_every_event():
    seen = []
    tracer = Tracer(listener=seen.append)
    Agent(demo_llm(0), demo_registry(0), tracer=tracer).run("q")
    assert [e["event"] for e in seen] == [e["event"] for e in tracer.events]
    assert seen[0]["event"] == "run_start"
    assert seen[-1]["event"] == "run_end"
