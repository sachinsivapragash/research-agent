"""Integrations with public web APIs (no API keys needed).

Each function takes plain arguments and returns a short text summary the model can read.
All network access goes through `_http_get`, so tests replace that one function and never
touch the network. `_http_get` retries rate limits and server errors with backoff.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote

from research_agent.http import get_with_retry


def _http_get(url: str, params: dict[str, Any] | None = None) -> Any:
    return get_with_retry(url, params=params)


# ------------------------------------------------------------------ Wikipedia


def wikipedia_summary(title: str) -> str:
    """Short encyclopedia summary of a topic (English Wikipedia)."""
    if not title.strip():
        raise ValueError("title must not be empty")
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title.strip(), safe="")
    data = _http_get(url).json()
    extract = data.get("extract") or "No summary available."
    page = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    return f"{data.get('title', title)}: {extract}\nSource: {page}".strip()


# --------------------------------------------------------------------- GitHub

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def github_repo_info(repo: str) -> str:
    """Public facts about a GitHub repository, given as 'owner/name'."""
    repo = repo.strip()
    if not _REPO_RE.match(repo) or ".." in repo:
        raise ValueError("repo must look like 'owner/name'")
    data = _http_get(f"https://api.github.com/repos/{repo}").json()
    return (
        f"{data.get('full_name')}: {data.get('description') or 'no description'}\n"
        f"Language: {data.get('language')} | Stars: {data.get('stargazers_count')} | "
        f"Forks: {data.get('forks_count')} | Open issues: {data.get('open_issues_count')}\n"
        f"Last push: {data.get('pushed_at')}\n"
        f"Source: {data.get('html_url')}"
    )


# ---------------------------------------------------------------------- arXiv

_ATOM = {"a": "http://www.w3.org/2005/Atom"}


def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search arXiv for research papers. Returns title, authors, link and abstract start."""
    max_results = max(1, min(int(max_results), 8))
    text = _http_get(
        "https://export.arxiv.org/api/query",
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results},
    ).text
    root = ET.fromstring(text)
    entries = root.findall("a:entry", _ATOM)
    if not entries:
        return "No papers found."
    lines = []
    for i, entry in enumerate(entries, start=1):
        title = " ".join((entry.findtext("a:title", "", _ATOM) or "").split())
        summary = " ".join((entry.findtext("a:summary", "", _ATOM) or "").split())
        link = entry.findtext("a:id", "", _ATOM)
        authors = ", ".join(
            (a.findtext("a:name", "", _ATOM) or "") for a in entry.findall("a:author", _ATOM)
        )
        lines.append(f"{i}. {title}\n   Authors: {authors}\n   {link}\n   {summary[:300]}...")
    return "\n".join(lines)


# -------------------------------------------------------------------- Weather


def weather(city: str) -> str:
    """Current weather for a city (Open-Meteo)."""
    if not city.strip():
        raise ValueError("city must not be empty")
    geo = _http_get(
        "https://geocoding-api.open-meteo.com/v1/search", {"name": city.strip(), "count": 1}
    ).json()
    results = geo.get("results") or []
    if not results:
        return f"Could not find a place called '{city}'."
    place = results[0]
    forecast = _http_get(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        },
    ).json()
    now = forecast.get("current", {})
    return (
        f"{place.get('name')}, {place.get('country', '')}: "
        f"{now.get('temperature_2m')} C, humidity {now.get('relative_humidity_2m')}%, "
        f"wind {now.get('wind_speed_10m')} km/h"
    )
