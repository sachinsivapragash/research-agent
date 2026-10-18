"""Conversation memory with a size limit.

Messages after the first user message come in (assistant, user) pairs: the model asks
for tools, then we reply with the tool results. When the history gets too long we drop
the oldest whole pairs, so a tool call is never separated from its result.
"""

from __future__ import annotations

from typing import Any


class Memory:
    def __init__(self, max_messages: int = 24) -> None:
        if max_messages < 3:
            raise ValueError("max_messages must be at least 3")
        self.max_messages = max_messages
        self.messages: list[dict[str, Any]] = []

    def add(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self._trim()

    def _trim(self) -> None:
        # Keep message 0 (the original question). Drop oldest pairs after it.
        while len(self.messages) > self.max_messages and len(self.messages) >= 3:
            del self.messages[1:3]

    def __len__(self) -> int:
        return len(self.messages)
