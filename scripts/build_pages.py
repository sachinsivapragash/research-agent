"""Build the static GitHub Pages demo into docs/.

Runs the offline demo agent once, records every event, and copies the UI next to it.
The page replays the recording in the browser, so no Python server is needed.

    python scripts/build_pages.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_agent.agent import Agent  # noqa: E402
from research_agent.demo import DEMO_QUESTION, demo_llm, demo_registry  # noqa: E402
from research_agent.tracing import Tracer  # noqa: E402


def main() -> None:
    events: list[dict] = []
    Agent(demo_llm(1.0), demo_registry(0.8), tracer=Tracer(listener=events.append)).run(
        DEMO_QUESTION
    )
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "demo-events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    shutil.copy(ROOT / "src/research_agent/static/index.html", docs / "index.html")
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Wrote {len(events)} events and index.html to {docs}")


if __name__ == "__main__":
    main()
