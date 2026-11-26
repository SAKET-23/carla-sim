"""Exact-stamp synchroniser: a set is complete only when every key has a message with the same stamp."""

from collections.abc import Sequence
from typing import Any


class StampSync:
    def __init__(self, keys: Sequence[str], max_pending: int = 20) -> None:
        self.keys, self.max_pending, self.dropped = tuple(keys), max_pending, 0
        self._pending: dict[float, dict[str, Any]] = {}  # insertion order = arrival order of first message

    def add(self, key: str, stamp: float, msg: Any) -> dict[str, Any] | None:
        slot = self._pending.setdefault(stamp, {})
        slot[key] = msg
        if len(slot) == len(self.keys):
            # an emitted set also discards older incomplete ones: they can never be used
            for s in [s for s in self._pending if s <= stamp]:
                del self._pending[s]
            return slot
        while len(self._pending) > self.max_pending:
            del self._pending[next(iter(self._pending))]
            self.dropped += 1
        return None
