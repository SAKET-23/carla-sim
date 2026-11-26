"""Tick ownership and the command deadline (architecture.md section 5). Pure core; reused by the CARLA bridge."""

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from hybrid_av.types import ActuatorCommand

_STAMP_TOL = 1e-6
_BRAKE = ActuatorCommand(0.0, 0.0, 1.0)
T = TypeVar("T")


class StampMailbox(Generic[T]):
    """Thread-safe store of recent payloads (commands, policy actions) keyed by the stamp of the tick they answer."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._items: deque[tuple[float, T]] = deque(maxlen=16)

    def put(self, stamp: float, cmd: T) -> None:
        with self._cond:
            self._items.append((stamp, cmd))
            self._cond.notify_all()

    def wait_for(self, stamp: float, timeout: float) -> T | None:
        def find():
            return next((c for s, c in reversed(self._items) if abs(s - stamp) <= _STAMP_TOL), None)

        with self._cond:
            return self._cond.wait_for(find, timeout)


CommandMailbox = StampMailbox


@dataclass
class TickStats:
    ticks: int = 0
    late_commands: int = 0
    warmup_ticks: int = 0  # late commands during the first `warmup_ticks` ticks (ECC-21)


class TickLoop:
    def __init__(
        self,
        step: Callable[[ActuatorCommand], tuple[float, bool]],
        publish: Callable[[float], None],
        mailbox: CommandMailbox,
        command_timeout: float = 1.0,
        warmup_ticks: int = 20,
        initial_command: ActuatorCommand = _BRAKE,
    ) -> None:
        self.step, self.publish, self.mailbox = step, publish, mailbox
        self.command_timeout, self.warmup, self.initial = command_timeout, warmup_ticks, initial_command

    def run(self, max_ticks: int | None = None) -> TickStats:
        stats, last = TickStats(), self.initial
        while True:
            t, done = self.step(last)
            stats.ticks += 1
            self.publish(t)
            cmd = self.mailbox.wait_for(
                t, self.command_timeout
            )  # also after the final step: one command per tick
            if cmd is None:
                if stats.ticks > self.warmup:
                    stats.late_commands += 1
                else:
                    stats.warmup_ticks += 1
            else:
                last = cmd
            if done or stats.ticks == max_ticks:
                return stats
