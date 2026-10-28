"""Lifecycle, input formatting, and completion policies for Codex runs."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from inspect_ai.model import ChatMessage
from inspect_ai.solver import TaskState
from inspect_ai.util import SandboxEnvironment

from .runtime.budget import BudgetTracker

ReceiptReader = Callable[[dict[str, Any], str | None, str | None], str | None]
CODEX_MAX_GOAL_CHARS = 4000


def truncate_goal_objective(objective: str, max_chars: int = CODEX_MAX_GOAL_CHARS) -> str:
    if len(objective) <= max_chars:
        return objective
    notice = f"\n\n[Objective truncated to {max_chars} chars; full instructions in first turn]"
    allowed = max_chars - len(notice)
    if allowed <= 0:
        return objective[:max_chars]
    truncated = objective[:allowed]
    last_break = max(truncated.rfind("\n\n"), truncated.rfind("\n"), truncated.rfind(". "))
    if last_break > allowed // 2:
        truncated = truncated[:last_break].rstrip()
    return f"{truncated}{notice}"


def split_history(messages: list[ChatMessage]) -> tuple[list[ChatMessage], ChatMessage | None]:
    """Earlier messages and the current user message (None when the last message is not a user's)."""
    if messages and messages[-1].role == "user":
        return messages[:-1], messages[-1]
    return list(messages), None


def task_input(state: TaskState) -> tuple[str, str]:
    """The current user message; earlier messages are replayed as native thread history.

    The goal objective is the latest user message, so a conversation that ends
    with the assistant keeps working toward it.
    """
    _, current = split_history(state.messages)
    objective = next((m.text for m in reversed(state.messages) if m.role == "user"), "")
    return truncate_goal_objective(objective), current.text if current is not None else ""


def accepted_answer(client: Any, answers: list[str]) -> str:
    return client.submitted_answer or next((a for a in reversed(answers) if a), "")


@dataclass(frozen=True)
class RunPolicy:
    """Execution policy controlling task input, completion condition, and quota handling.

    Completion modes:
    - ``"turn"``: Completes when the root turn finishes successfully.
    - ``"goal"``: Waits for the native thread goal to complete.
    - ``"submission"``: Accepts only the answer returned by ``receipt_reader``.

    Quota management:
    - ``quota_hold_sec``: Max seconds to pause and wait for account rate limits to reset.
      Paused time is credited to Inspect working limits and excluded from the agent timeout.
    """

    completion: Literal["turn", "goal", "submission"] = "turn"
    input_builder: Callable[[TaskState], tuple[str, str]] = task_input
    receipt_reader: ReceiptReader | None = None
    prepare: Callable[[SandboxEnvironment, TaskState, BudgetTracker], Awaitable[None]] | None = None
    thread_started: Callable[[SandboxEnvironment, str], Awaitable[None]] | None = None
    config_overrides: tuple[str, ...] = ()
    excluded_tools: frozenset[str] = frozenset()
    mcp_tool_aliases: tuple[tuple[str, str, str], ...] = ()
    checkpoint_fractions: tuple[float, ...] = ()
    checkpoint_message: Callable[[float, float, float], str] | None = None
    terminal_nudge: str | None = None
    continuation: str = "Continue working toward the objective."
    max_turn_failures: int = 3
    quota_hold_sec: float = 0.0
    quota_poll_sec: float = 60.0
    budget_resumed: Callable[[SandboxEnvironment, BudgetTracker], Awaitable[None]] | None = None

    def __post_init__(self) -> None:
        if self.completion not in ("turn", "goal", "submission"):
            raise ValueError("completion must be turn, goal, or submission")
        if self.completion == "submission" and self.receipt_reader is None:
            raise ValueError("submission completion requires an authoritative receipt_reader")
        if self.checkpoint_fractions and self.checkpoint_message is None:
            raise ValueError("checkpoint fractions require a checkpoint_message formatter")
        if any(not 0 < f < 1 for f in self.checkpoint_fractions):
            raise ValueError("checkpoint fractions must be between zero and one")
        if self.max_turn_failures < 0:
            raise ValueError("max_turn_failures must be nonnegative")
        if not (math.isfinite(self.quota_hold_sec) and self.quota_hold_sec >= 0):
            raise ValueError("quota_hold_sec must be finite and nonnegative")
        if not (math.isfinite(self.quota_poll_sec) and self.quota_poll_sec > 0):
            raise ValueError("quota_poll_sec must be finite and positive")
