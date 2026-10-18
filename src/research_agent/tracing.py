"""Run tracing: one JSON object per line (JSONL) for every event in a run.

A trace lets you replay and debug a run after the fact: what the model asked for,
which tools ran, how long each took, and how many tokens each turn used.
An optional `listener` receives every event as it happens, which is how the web UI
streams a run live.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

Listener = Callable[[dict[str, Any]], None]


class Tracer:
    def __init__(self, path: str | Path | None = None, listener: Listener | None = None) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.path = Path(path) if path else None
        self.listener = listener
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **data: Any) -> None:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "event": event, **data}
        with self._lock:
            self.events.append(record)
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        if self.listener:
            self.listener(record)
