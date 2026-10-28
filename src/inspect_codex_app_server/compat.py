"""Compatibility layer for Inspect AI event, accounting, and process utilities."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Sequence

from inspect_ai._util.exception import TerminateSampleError
from inspect_ai.agent._types import (
    DEFAULT_CONTINUE_PROMPT,
    DEFAULT_CONTINUE_PROMPT_NO_SUBMIT,
    DEFAULT_SUBMIT_PROMPT,
)
from inspect_ai.event import Event
from inspect_ai.model import GenerateConfig, ModelUsage
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.solver import TaskState
from inspect_ai.tool import Skill, ToolCall
from inspect_ai.tool._tool_def import tool_def_fields
from inspect_ai.util import (
    Checkpointer,
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    checkpointer,
)
from inspect_ai.util._limit import LimitExceededError
from inspect_ai.util._sandbox.exec_remote import (
    ExecCompleted,
    ExecOutput,
    ExecStderr,
    ExecStdout,
)

from .runtime.limits import (
    check_limits as _check_limits,
)
from .runtime.limits import (
    emit_event as _emit_event,
)
from .runtime.limits import (
    record_limit_usage,
    record_usage,
    record_waiting_time,
)
from .runtime.limits import (
    require_supported_budgets as _require_supported_budgets,
)
from .runtime.limits import (
    update_event as _update_event,
)
from .runtime.process import kill_process, process_events

__all__ = [
    "DEFAULT_CONTINUE_PROMPT",
    "DEFAULT_CONTINUE_PROMPT_NO_SUBMIT",
    "DEFAULT_SUBMIT_PROMPT",
    "ExecCompleted",
    "ExecOutput",
    "ExecRemoteProcess",
    "ExecRemoteStreamingOptions",
    "ExecStderr",
    "ExecStdout",
    "LimitExceededError",
    "TerminateSampleError",
    "USAGE_FIELDS",
    "check_limits",
    "emit_event",
    "kill_process",
    "process_events",
    "record_limit_usage",
    "record_response_usage",
    "record_usage",
    "record_waiting_time",
    "update_event",
]

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "input_tokens_cache_read",
    "input_tokens_cache_write",
)



def emit_event(event: Event) -> None:
    """Append an event to the active sample transcript (best effort)."""
    _emit_event(event, best_effort=True)


def update_event(event: Event) -> None:
    """Update an already-emitted (pending) event in the transcript."""
    _update_event(event, best_effort=True)


def require_supported_budgets(cost_limit: float | None = None) -> None:
    """Ensure the configured limits are supported by the Codex App Server integration."""
    _require_supported_budgets(cost_limit, harness="Codex")


def check_limits() -> None:
    """Raise ``LimitExceededError`` when any scoped limit has been exceeded.

    Checks token, cost, and turn limits active in the current context. Message
    limits raise naturally when ``TaskState.messages`` grows past the cap.
    """
    _check_limits(harness="Codex")


def record_response_usage(model: str, usage: ModelUsage) -> None:
    """Record response token usage into Inspect's live throughput registry."""
    from inspect_ai.model._throughput import record_generate

    qualified_model = model if model.startswith("codex/") else f"codex/{model}"
    record_generate(qualified_model, usage)


def reassign_usage(previous_model: str, model: str, usage: ModelUsage) -> None:
    """Move already-metered tokens between model registries, without remetering."""
    from inspect_ai.model._model import (
        model_usage_context_var,
        sample_model_usage_context_var,
        set_model_usage,
    )

    negative = ModelUsage(**{field: -(getattr(usage, field) or 0) for field in USAGE_FIELDS})
    for registry in (sample_model_usage_context_var.get(None), model_usage_context_var.get(None)):
        set_model_usage(previous_model, negative, registry)
        set_model_usage(model, usage, registry)
        if registry is not None and all(not getattr(registry[previous_model], field) for field in USAGE_FIELDS):
            del registry[previous_model]


def tool_schema(tool: Any) -> tuple[str, str | None, dict[str, Any]]:
    """Extract (name, description, JSON-schema parameters) from an Inspect Tool."""
    fields = tool_def_fields(tool)
    return (
        fields.name,
        fields.description,
        fields.parameters.model_dump(mode="json", exclude_none=True),
    )


@contextmanager
def native_thread_span(span_id: str | None):
    """Attribute Inspect-owned executions to an existing native agent span."""
    from inspect_ai.util._span import _current_span_id

    token = _current_span_id.set(span_id)
    try:
        yield
    finally:
        _current_span_id.reset(token)


def current_span_id() -> str | None:
    """The span Inspect opened around this execution (e.g. by ``as_solver``)."""
    from inspect_ai.util._span import current_span_id as current

    return current()


def active_generate_config() -> GenerateConfig:
    """The evaluation's GenerateConfig (``inspect eval --reasoning-effort`` etc.)."""
    from inspect_ai.model._generate_config import active_generate_config as active

    return active()


def have_tool_approval() -> bool:
    """True when an Inspect approval policy is active in this context."""
    from inspect_ai.approval._apply import have_tool_approval as have

    return have()


async def apply_tool_approval(call: ToolCall, history: list[ChatMessage]) -> tuple[bool, Any]:
    """Ask Inspect's active approver about a native action; it records the ApprovalEvent."""
    from inspect_ai.approval._apply import apply_tool_approval as apply

    return await apply("", call, None, history)


def sample_state() -> TaskState | None:
    """The running sample's TaskState (``None`` outside a sample)."""
    from inspect_ai.solver._task_state import sample_state as current

    return current()


def remove_submit_tool(messages: list[ChatMessage], name: str) -> list[ChatMessage]:
    """Drop submit calls from a conversation, as react() does with ``keep_in_messages=False``."""
    from inspect_ai.agent._react import _remove_submit_tool

    return _remove_submit_tool(messages, name)


def read_skills(skills: Sequence[str | Path | Skill]) -> list[Skill]:
    """Resolve and validate skills (unique names), as Inspect's skill tool does."""
    from inspect_ai.tool import read_skills as read
    from inspect_ai.tool._tools._skill.validate import check_unique_skill_names

    resolved = read(skills)
    check_unique_skill_names(resolved)
    return resolved


@asynccontextmanager
async def sample_checkpointer() -> AsyncGenerator[Checkpointer | None]:
    """The sample's checkpoint session, or ``None`` when nothing would be checkpointed.

    That is outside a sample, with checkpointing off, and when nested in an
    agent that owns the session (Inspect hands nested agents an inert one).
    """
    from inspect_ai.log._samples import sample_active
    from inspect_ai.util._checkpoint.checkpointer_noop import _NoopCheckpointer

    if sample_active() is None:
        yield None
        return
    async with checkpointer() as session:
        yield None if isinstance(session, _NoopCheckpointer) else session
