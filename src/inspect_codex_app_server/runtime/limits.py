"""Inspect token limit enforcement, usage aggregation, and transcript events."""

from __future__ import annotations

import logging

from inspect_ai._util.working import report_sample_waiting_time
from inspect_ai.event import Event
from inspect_ai.log._samples import set_active_sample_total_tokens
from inspect_ai.log._transcript import transcript
from inspect_ai.model import ModelUsage
from inspect_ai.model._model import (
    model_usage_context_var,
    sample_model_usage_context_var,
    sample_total_tokens,
    set_model_usage,
)
from inspect_ai.util._limit import (
    check_cost_limit,
    check_token_limit,
    check_turn_limit,
    cost_limit_tree,
    record_model_usage,
    turn_limit_tree,
)

logger = logging.getLogger(__name__)


def emit_event(event: Event, *, best_effort: bool = False) -> None:
    """Append an event to the active sample transcript."""
    if best_effort:
        try:
            transcript()._event(event)
        except Exception as e:  # pragma: no cover - transcript optional in unit tests
            logger.debug("Failed to emit %s event: %s", getattr(event, "event", "?"), e)
    else:
        transcript()._event(event)


def update_event(event: Event, *, best_effort: bool = False) -> None:
    """Update an already-emitted (pending) event in the transcript."""
    if best_effort:
        try:
            transcript()._event_updated(event)
        except Exception as e:  # pragma: no cover - transcript optional in unit tests
            logger.debug("Failed to update %s event: %s", getattr(event, "event", "?"), e)
    else:
        transcript()._event_updated(event)


def record_usage(model: str, usage: ModelUsage) -> None:
    """Accumulate token usage into Inspect sample and task registries."""
    set_model_usage(model, usage, sample_model_usage_context_var.get(None))
    set_model_usage(model, usage, model_usage_context_var.get(None))
    # Update live token counter used for real-time throughput calculations.
    set_active_sample_total_tokens(sample_total_tokens())


def record_limit_usage(usage: ModelUsage) -> None:
    """Meter a usage delta against every active token limit scope."""
    record_model_usage(usage)


def record_waiting_time(seconds: float) -> None:
    """Credit time the sample spent unable to work against working limits."""
    report_sample_waiting_time(seconds)


def require_supported_budgets(cost_limit: float | None = None, harness: str = "agent server") -> None:
    """Ensure configured limits are supported by the app-server integration."""
    unsupported = set()
    if cost_limit is not None:
        unsupported.add("cost")
    for name, tree in (("cost", cost_limit_tree), ("turn", turn_limit_tree)):
        if tree.is_suspended():
            continue
        node = tree.get()
        while node is not None:
            if node.limit is not None:
                unsupported.add(name)
            node = node.parent
    if unsupported:
        raise ValueError(
            f"{harness} native inference does not support Inspect {'/'.join(sorted(unsupported))} "
            "limits: App Server does not report authoritative model-call counts or costs. "
            "Use token, message, or time limits instead."
        )


def check_limits(harness: str = "agent server") -> None:
    """Raise ``LimitExceededError`` when any scoped limit has been exceeded.

    Checks token, cost, and turn limits active in the current context. Message
    limits raise naturally when ``TaskState.messages`` grows past the cap.
    """
    require_supported_budgets(harness=harness)
    check_token_limit()
    check_cost_limit()
    check_turn_limit()
