"""Sample time budget tracking, quota pauses, and checkpoint steering."""
from __future__ import annotations

import time
from collections.abc import Callable

from inspect_ai.event import SampleLimitEvent


class BudgetTracker:
    """Track elapsed and remaining time budget, accounting for holds and checkpoints."""

    def __init__(self, budget_sec: float, clock: Callable[[], float] | None = None, *, checkpoints: tuple[float, ...] = ()):
        self.checkpoints = checkpoints
        self.budget_sec = float(budget_sec)
        self._clock = clock if clock is not None else time.monotonic
        self.start = self._clock()
        self.held_sec = 0.0
        self._delivered: set[float] = set()

    def elapsed(self) -> float:
        return self._clock() - self.start

    def remaining(self) -> float:
        return self.budget_sec - self.elapsed()

    def credit(self, seconds: float) -> None:
        """Exclude time the agent was not working (e.g. awaiting an operator) from the budget."""
        self.start += seconds

    def hold(self, seconds: float) -> None:
        """Exclude a quota hold from the budget, counting it against the hold allowance."""
        self.credit(seconds)
        self.held_sec += seconds

    def due_checkpoints(self) -> list[float]:
        """Return and mark checkpoint milestones that have elapsed."""
        due = [
            f
            for f in self.checkpoints
            if f not in self._delivered and self.elapsed() >= f * self.budget_sec
        ]
        self._delivered.update(due)
        return due

    def until_next_checkpoint(self) -> float:
        """Seconds until the next due checkpoint or remaining budget."""
        pending = [f * self.budget_sec - self.elapsed() for f in self.checkpoints if f not in self._delivered]
        return max(0, min(pending)) if pending else max(0, self.remaining())


def time_budget_depleted_event(budget_sec: float, span_id: str | None = None) -> SampleLimitEvent:
    """Create a SampleLimitEvent when a solver exhausts its time budget."""
    return SampleLimitEvent(
        type="time",
        message=f"Time budget depleted ({budget_sec:g}s limit reached)",
        limit=float(budget_sec),
        span_id=span_id,
    )
