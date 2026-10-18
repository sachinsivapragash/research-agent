import pytest

from research_agent import integrations


class FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def fake_get(routes):
    """Build a replacement for _http_get that answers by URL substring."""

    def _get(url, params=None):
        for key, response in routes.items():
            if key in url:
                return response
        raise AssertionError(f"unexpected URL: {url}")

    return _get


def test_wikipedia_summary(monkeypatch):
    payload = {
        "title": "MQTT",
        "extract": "MQTT is a lightweight messaging protocol.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/MQTT"}},
    }
    monkeypatch.setattr(
        integrations, "_http_get", fake_get({"wikipedia.org": FakeResponse(payload)})
    )
    out = integrations.wikipedia_summary("MQTT")
    assert "lightweight messaging protocol" in out
    assert "https://en.wikipedia.org/wiki/MQTT" in out


def test_github_repo_info(monkeypatch):
    payload = {
        "full_name": "octocat/hello",
        "description": "demo",
        "language": "Python",
        "stargazers_count": 5,
        "forks_count": 2,
        "open_issues_count": 1,
        "pushed_at": "2026-10-01T00:00:00Z",
        "html_url": "https://github.com/octocat/hello",
    }
    monkeypatch.setattr(
        integrations, "_http_get", fake_get({"api.github.com": FakeResponse(payload)})
    )
    out = integrations.github_repo_info("octocat/hello")
    assert "Stars: 5" in out
    assert "Python" in out


@pytest.mark.parametrize("bad", ["", "no-slash", "a/b/c", "../etc/passwd", "a b/c"])
def test_github_repo_rejects_bad_names(bad):
    with pytest.raises(ValueError):
        integrations.github_repo_info(bad)


ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>A Study of
      Agents</title>
    <summary>We study agents that use tools.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
  </entry>
</feed>"""


def test_arxiv_search(monkeypatch):
    monkeypatch.setattr(
        integrations, "_http_get", fake_get({"arxiv.org": FakeResponse(text=ARXIV_XML)})
    )
    out = integrations.arxiv_search("agents")
    assert "A Study of Agents" in out
    assert "Ada Lovelace, Alan Turing" in out
    assert "arxiv.org/abs/1234.5678v1" in out


def test_arxiv_no_results(monkeypatch):
    empty = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    monkeypatch.setattr(
        integrations, "_http_get", fake_get({"arxiv.org": FakeResponse(text=empty)})
    )
    assert integrations.arxiv_search("zzzz") == "No papers found."


def test_weather(monkeypatch):
    place = {"name": "Coimbatore", "country": "India", "latitude": 11.0, "longitude": 77.0}
    geo = {"results": [place]}
    current = {"temperature_2m": 28.5, "relative_humidity_2m": 70, "wind_speed_10m": 9.0}
    forecast = {"current": current}
    monkeypatch.setattr(
        integrations,
        "_http_get",
        fake_get(
            {
                "geocoding-api": FakeResponse(geo),
                "api.open-meteo.com": FakeResponse(forecast),
            }
        ),
    )
    out = integrations.weather("Coimbatore")
    assert "Coimbatore, India" in out
    assert "28.5 C" in out


def test_weather_unknown_city(monkeypatch):
    monkeypatch.setattr(integrations, "_http_get", fake_get({"geocoding-api": FakeResponse({})}))
    assert "Could not find" in integrations.weather("Nowhereville")
