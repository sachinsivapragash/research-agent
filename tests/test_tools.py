import pytest

from research_agent.tools import (
    Notes,
    calculator,
    check_url_is_public,
    default_registry,
    html_to_text,
)


def test_calculator_basic():
    assert calculator("2 + 3 * 4") == "14"
    assert calculator("(1.07 ** 10) * 1000") == "1967.1513572896"


def test_calculator_rejects_code():
    with pytest.raises(ValueError):
        calculator("__import__('os').system('echo hi')")
    with pytest.raises(ValueError):
        calculator("2 ** 1000")


def test_calculator_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        calculator("1 / 0")


def test_notes_roundtrip():
    notes = Notes()
    assert notes.read() == "No notes yet."
    assert notes.save("fact one") == "Saved note #1."
    assert notes.read() == "1. fact one"


def test_html_to_text_skips_scripts():
    html = (
        "<html><head><title>x</title></head>"
        "<body><script>bad()</script><p>Hello</p><p>World</p></body></html>"
    )
    assert html_to_text(html) == "Hello World"


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com", "http://127.0.0.1/admin", "http://localhost:8000"],
)
def test_url_guard_blocks_unsafe(url):
    with pytest.raises(ValueError):
        check_url_is_public(url)


def test_registry_specs_and_unknown_tool():
    registry = default_registry()
    assert set(registry.names()) == {
        "web_search",
        "fetch_page",
        "calculator",
        "save_note",
        "read_notes",
        "wikipedia_summary",
        "github_repo_info",
        "arxiv_search",
        "weather",
    }
    for spec in registry.specs():
        assert {"name", "description", "input_schema"} <= spec.keys()
    with pytest.raises(KeyError):
        registry.run("nope", {})
