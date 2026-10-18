"""Usage limits for a publicly hosted Live mode.

Protects the owner's model quota: each visitor (by IP) gets a few live runs per hour,
and the whole site has a daily cap. Both are off unless set, so local use is unlimited.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class LiveLimiter:
    def __init__(self, per_ip_hour: int = 0, per_day: int = 0) -> None:
        self.per_ip_hour = per_ip_hour
        self.per_day = per_day
        self._ip: dict[str, deque[float]] = defaultdict(deque)
        self._day: deque[float] = deque()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.per_ip_hour or self.per_day)

    def check(self, ip: str, now: float | None = None) -> str | None:
        """Record a run and return None, or return the reason it is not allowed."""
        t = time.time() if now is None else now
        with self._lock:
            while self._day and t - self._day[0] > 86400:
                self._day.popleft()
            hits = self._ip[ip]
            while hits and t - hits[0] > 3600:
                hits.popleft()
            if not hits:
                self._ip.pop(ip, None)
                hits = self._ip[ip]
            if self.per_day and len(self._day) >= self.per_day:
                return "Daily live limit for this site reached. Try tomorrow, or use Demo mode."
            if self.per_ip_hour and len(hits) >= self.per_ip_hour:
                return f"Limit: {self.per_ip_hour} live runs per hour per visitor. Try Demo mode."
            hits.append(t)
            self._day.append(t)
            return None
