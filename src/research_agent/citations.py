"""Citation grounding check.

Language models sometimes cite URLs they never opened, or invent them outright. This
module checks every URL in the final answer against the text the tools actually
returned during the run. A URL counts as grounded only if it appeared in a tool result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`|]+")
_TRAILING = ".,;:!?*_"


def normalize(url: str) -> str:
    url = url.rstrip(_TRAILING)
    url = url.split("#", 1)[0]
    url = re.sub(r"^https?://(www\.)?", "", url, flags=re.IGNORECASE)
    return url.rstrip("/").lower()


def extract_urls(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _URL_RE.findall(text):
        seen.setdefault(match.rstrip(_TRAILING), None)
    return list(seen)


@dataclass
class CitationReport:
    cited: list[str] = field(default_factory=list)
    grounded: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        """Share of cited URLs that were really seen. None when nothing was cited."""
        if not self.cited:
            return None
        return len(self.grounded) / len(self.cited)


def check_citations(answer: str, evidence: list[str]) -> CitationReport:
    """Compare URLs in `answer` with URLs and fetched pages found in `evidence`."""
    seen = {normalize(u) for text in evidence for u in extract_urls(text)}
    report = CitationReport()
    for url in extract_urls(answer):
        report.cited.append(url)
        (report.grounded if normalize(url) in seen else report.ungrounded).append(url)
    return report
