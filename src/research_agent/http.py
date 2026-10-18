"""HTTP GET with retries and exponential backoff.

Retries on connection errors, timeouts, HTTP 429 and 5xx. Honors a numeric
`Retry-After` header when the server sends one. Other 4xx errors fail at once,
because retrying a bad request never helps.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

log = logging.getLogger("research_agent")

USER_AGENT = "research-agent/0.2 (+https://github.com/sachinsivapragash/research-agent)"
RETRY_STATUSES = {429, 500, 502, 503, 504}


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:
    """Full-jitter exponential backoff: random value in [0, min(cap, base * 2**attempt)]."""
    return random.uniform(0, min(cap, base * (2**attempt)))


def get_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 15,
    attempts: int = 4,
    sleep: Any = time.sleep,
    session: Any = None,
    **kwargs: Any,
) -> Any:
    import requests

    client = session or requests
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = client.get(url, params=params, timeout=timeout, headers=headers, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            log.debug("GET %s failed (%s), attempt %d", url, exc, attempt + 1)
        else:
            if response.status_code not in RETRY_STATUSES:
                response.raise_for_status()
                return response
            last_error = requests.HTTPError(f"HTTP {response.status_code} from {url}")
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit() and attempt < attempts - 1:
                sleep(min(float(retry_after), 30.0))
                continue
        if attempt < attempts - 1:
            sleep(backoff_delay(attempt))

    assert last_error is not None
    raise last_error
