"""Map Codex App Server events onto Inspect AI transcript structures.

Converts JSON-RPC notifications and turn items into Inspect events:
- Maps agent messages, reasoning blocks, and tool executions to Inspect ChatMessages.
- Emits ToolEvents, SpanBegin/EndEvents for subagents, and ModelEvents for activity.
- Deduplicates redelivered items using deterministic identifiers.
- Reconciles token usage and updates Inspect registries and live throughput metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import shortuuid
from inspect_ai.event import (
    ApprovalEvent,
    CompactionEvent,
    InfoEvent,
    InputEvent,
    InterruptEvent,
    ModelEvent,
    ReviewEvent,
    SpanBeginEvent,
    SpanEndEvent,
    ToolEvent,
)
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    StopReason,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall, ToolCallContent, ToolCallError
from pydantic import BaseModel, Field

from . import compat
from .policy import ReceiptReader

logger = logging.getLogger(__name__)

_RAW_EVENT_LIMIT = 5000
_RAW_STRING_LIMIT = 4096

# Collaboration tool names from the pinned schema.
_COLLAB_TOOLS_HIDDEN = {"wait", "poll"}
# Native plan step statuses in Inspect's update_plan vocabulary.
PLAN_STATUSES = {"pending": "pending", "inProgress": "in_progress", "completed": "completed"}


def _truncate_deep(value: Any) -> Any:
    """Bound raw evidence size without dropping structure."""
    if isinstance(value, str):
        return value if len(value) <= _RAW_STRING_LIMIT else value[:_RAW_STRING_LIMIT] + "…[truncated]"
    if isinstance(value, dict):
        return {k: _truncate_deep(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_deep(v) for v in value]
    return value


def _optional_delta(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return max(0, (a or 0) - (b or 0))


def _usage_delta(current: ModelUsage, baseline: ModelUsage) -> ModelUsage:
    """Non-negative componentwise delta between current usage and a baseline."""
    return ModelUsage(
        input_tokens=max(0, current.input_tokens - baseline.input_tokens),
        output_tokens=max(0, current.output_tokens - baseline.output_tokens),
        total_tokens=max(0, current.total_tokens - baseline.total_tokens),
        reasoning_tokens=max(0, (current.reasoning_tokens or 0) - (baseline.reasoning_tokens or 0)) or None,
        input_tokens_cache_read=_optional_delta(
            current.input_tokens_cache_read, baseline.input_tokens_cache_read
        ),
        input_tokens_cache_write=_optional_delta(
            current.input_tokens_cache_write, baseline.input_tokens_cache_write
        ),
    )


def _usage_envelope(a: ModelUsage, b: ModelUsage) -> ModelUsage:
    """Componentwise envelope (maximum) of two usage records."""
    return ModelUsage(**{
        field: max(getattr(a, field) or 0, getattr(b, field) or 0)
        for field in compat.USAGE_FIELDS
    })


def _copy_usage(u: ModelUsage) -> ModelUsage:
    return u.model_copy()


def _usage_add(base: ModelUsage, delta: ModelUsage) -> ModelUsage:
    base.input_tokens += delta.input_tokens
    base.output_tokens += delta.output_tokens
    base.total_tokens += delta.total_tokens
    if delta.reasoning_tokens:
        base.reasoning_tokens = (base.reasoning_tokens or 0) + delta.reasoning_tokens
    if delta.input_tokens_cache_read:
        base.input_tokens_cache_read = (
            (base.input_tokens_cache_read or 0) + delta.input_tokens_cache_read
        )
    if delta.input_tokens_cache_write:
        base.input_tokens_cache_write = (
            (base.input_tokens_cache_write or 0) + delta.input_tokens_cache_write
        )
    return base


class _ThreadUsage(BaseModel):
    """Monotonic per-thread usage accounting."""

    last_reported_total: ModelUsage | None = None
    running_total: ModelUsage = Field(default_factory=ModelUsage)
    response_total: ModelUsage = Field(default_factory=ModelUsage)

    def record_total(self, new_total: ModelUsage) -> ModelUsage:
        """Meter lifetime counters using a high-water mark per field.

        Duplicate/stale reports never lower the baseline. A reset must use a
        new thread id; otherwise it cannot be distinguished from stale data.
        """
        prev = self.last_reported_total
        if prev is None:
            self.last_reported_total = _copy_usage(new_total)
            return _copy_usage(new_total)
        delta = _usage_delta(new_total, prev)
        _usage_add(prev, delta)
        return delta


@dataclass
class SubagentState:
    """A subagent thread, independent of how many turns it runs."""

    thread_id: str
    name: str
    tool_span_id: str
    agent_span_id: str
    call_id: str
    parent_thread_id: str | None = None
    model_name: str | None = None
    prompt: str = ""
    reports: list[str] = field(default_factory=list)
    messages: list[ChatMessage] = field(default_factory=list)
    agent_path: str | None = None
    pending: list[ContentReasoning | ContentText] = field(default_factory=list)
    tool_event: ToolEvent | None = None
    tool_span: SpanBeginEvent | None = None
    commands_count: int = 0
    completed: bool = False
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def drain_pending(self) -> list[ContentReasoning | ContentText]:
        drained, self.pending = self.pending, []
        return drained


class EventStreamMapper:
    """Collects and maps streamed Codex thread items into Inspect AI structures."""

    def __init__(self, model_name: str | None = None, state: TaskState | None = None, *,
                 receipt_reader: ReceiptReader | None = None,
                 mcp_tool_aliases: tuple[tuple[str, str, str], ...] = ()):
        self.receipt_reader = receipt_reader
        self.mcp_tool_aliases = {(server, tool): name for server, tool, name in mcp_tool_aliases}
        self.model_name = model_name or "codex"
        self.state = state
        self.messages: list[ChatMessage] = []

        self.root_thread_id: str | None = None
        self.root_agent_span_id: str | None = None
        self.subagents: dict[str, SubagentState] = {}

        # Root-thread state
        self._root_pending: list[ContentReasoning | ContentText] = []
        self.root_usage = _ThreadUsage()
        self._child_usage: dict[str, _ThreadUsage] = {}
        self._thread_models: dict[str, str] = {}

        # Ledger: deduplicate redelivered items and maintain deterministic message IDs.
        self._seen_items: set[tuple[str | None, str]] = set()
        self._item_starts: dict[tuple[str | None, str], datetime] = {}
        self._item_times: dict[tuple[str | None, str], tuple[datetime, datetime | None]] = {}
        self._delegation_calls: dict[tuple[str | None, str], str] = {}
        self.thread_metadata: dict[str, dict[str, Any]] = {}
        self.inspect_tool_calls: set[tuple[str, str]] = set()
        self.inspect_exchanges: dict[tuple[str, str], tuple[ChatMessageAssistant, list[ChatMessage]]] = {}
        self.responses: dict[tuple[str, str], dict[str, Any]] = {}
        self.tearing_down = False
        self.raw_events: list[dict[str, Any]] = []
        self.dropped_event_count = 0

        self.submitted_answers: list[str] = []
        self._active_asst_msgs: dict[str | None, ChatMessageAssistant] = {}
        self._active_model_events: dict[str | None, ModelEvent] = {}
        self._last_model_events: dict[str | None, ModelEvent] = {}
        self._pending_response_usage: dict[str | None, ModelUsage] = {}
        self._last_diff_text: str | None = None
        self._last_goal_key: tuple[Any, Any] | str | None = None
        self._last_plan_text: str | None = None
        self._plan_updates = 0
        # Native safety reroutes, and the serving model of each thread's rerouted turn.
        self.reroutes: list[dict[str, Any]] = []
        self._rerouted: dict[str | None, tuple[str | None, str]] = {}
        self._last_event_time: datetime | None = None

    # ------------------------------------------------------------------
    # Raw evidence retention
    # ------------------------------------------------------------------

    def retain_raw(self, method: str, params: dict[str, Any]) -> None:
        raw_ts = (
            params.get("emittedAtMs")
            or params.get("completedAtMs")
            or params.get("startedAtMs")
        )
        parsed_ts = _native_datetime(raw_ts)
        if parsed_ts is None:
            # Turn timestamps use Unix seconds; item lifecycle fields use milliseconds.
            turn = params.get("turn") or {}
            parsed_ts = _native_datetime(turn.get("completedAt") or turn.get("startedAt"), milliseconds=False)
        if parsed_ts:
            self._last_event_time = parsed_ts
        if len(self.raw_events) < _RAW_EVENT_LIMIT:
            self.raw_events.append({"method": method, "params": _truncate_deep(params)})
        else:
            self.dropped_event_count += 1

    def _emit_raw_item(self, item: dict[str, Any], reason: str) -> None:
        logger.debug("Retaining unmapped item (%s): type=%s id=%s", reason, item.get("type"), item.get("id"))
        compat.emit_event(
            InfoEvent(source="codex-raw-item", data={"reason": reason, "item": _truncate_deep(item)})
        )

    # ------------------------------------------------------------------
    # Thread/scope helpers
    # ------------------------------------------------------------------

    def _thread_scope(self, thread_id: str | None) -> tuple[str, SubagentState | None]:
        """Return (usage-state key, subagent or None for root)."""
        if thread_id and self.root_thread_id and thread_id != self.root_thread_id:
            return thread_id, self.subagents.get(thread_id)
        return self.root_thread_id or "root", None

    def _thread_usage(self, thread_id: str | None) -> _ThreadUsage:
        if thread_id and self.root_thread_id and thread_id != self.root_thread_id:
            return self._child_usage.setdefault(thread_id, _ThreadUsage())
        return self.root_usage

    def thread_tokens(self, thread_id: str | None) -> int:
        """Total tokens the thread has used so far."""
        return self._thread_usage(thread_id).running_total.total_tokens

    @property
    def thread_ids(self) -> set[str]:
        """Threads observed in this run, including children restored from a checkpoint."""
        threads = set(self.subagents) | self.thread_metadata.keys() | self._child_usage.keys()
        if self.root_thread_id:
            threads.add(self.root_thread_id)
        return threads

    def checkpoint_accounting(self) -> dict[str, Any]:
        """Capture deduplication and lifetime counters already charged to Inspect.

        Response usage awaiting its thread's next ModelEvent is saved, not
        attached, so a checkpoint never changes the live transcript.
        """
        return {
            "root_thread_id": self.root_thread_id,
            "root_usage": self.root_usage.model_dump(),
            "child_usage": {tid: usage.model_dump() for tid, usage in self._child_usage.items()},
            "responses": list(self.responses.values()),
            "thread_metadata": self.thread_metadata,
            "thread_models": self._thread_models,
            "reroutes": self.reroutes,
            "rerouted": self._rerouted,
            "pending_usage": [[tid, usage.model_dump()] for tid, usage in self._pending_response_usage.items()],
        }

    def restore_accounting(self, data: dict[str, Any]) -> None:
        """Restore without recording usage: Inspect restores its registries and limit scopes."""
        self.root_thread_id = data["root_thread_id"]
        self.root_usage = _ThreadUsage.model_validate(data["root_usage"])
        self._child_usage = {tid: _ThreadUsage.model_validate(usage) for tid, usage in data["child_usage"].items()}
        self.responses = {(record["threadId"], record["responseId"]): record for record in data["responses"]}
        self.thread_metadata = data["thread_metadata"]
        self._thread_models = data["thread_models"]
        self.reroutes = data["reroutes"]
        self._rerouted = {tid: (turn, model) for tid, (turn, model) in data["rerouted"].items()}
        self._pending_response_usage = {tid: ModelUsage.model_validate(usage) for tid, usage in data["pending_usage"]}

    def _thread_model(self, thread_id: str | None) -> str:
        if thread_id in self._rerouted:
            return self._rerouted[thread_id][1]
        return self._configured_model(thread_id)

    def _configured_model(self, thread_id: str | None) -> str:
        if thread_id in self._thread_models:
            return self._thread_models[thread_id]
        _, sub = self._thread_scope(thread_id)
        if thread_id and self.root_thread_id and thread_id != self.root_thread_id:
            return (sub.model_name if sub is not None else None) or "codex/unknown"
        return self.model_name

    def thread_messages(self, thread_id: str | None) -> list[ChatMessage]:
        if thread_id == self.root_thread_id:
            return self._snapshot_input()
        sub = self.subagents.get(thread_id)
        return list(sub.messages) if sub else []

    def thread_span(self, thread_id: str) -> str | None:
        return self._thread_span(thread_id)

    def _thread_span(self, thread_id: str | None) -> str | None:
        _, sub = self._thread_scope(thread_id)
        return sub.agent_span_id if sub is not None else self.root_agent_span_id

    def _thread_pending(self, thread_id: str | None) -> list[ContentReasoning | ContentText]:
        _, sub = self._thread_scope(thread_id)
        return sub.pending if sub is not None else self._root_pending

    # ------------------------------------------------------------------
    # Event emission helpers
    # ------------------------------------------------------------------

    def _snapshot_input(self) -> list[ChatMessage]:
        """Snapshot conversation input BEFORE appending the message being emitted."""
        if self.state is not None:
            return list(self.state.messages)
        return list(self.messages)

    def _commit_messages(self, *msgs: ChatMessage) -> None:
        # Preserve message exchanges for final state even if a message limit is exceeded.
        self.messages.extend(msgs)
        for msg in msgs:
            if self.state is not None and not any(
                getattr(m, "id", None) == msg.id for m in self.state.messages
            ):
                self.state.messages.append(msg)

    def _commit_thread_messages(self, thread_id: str | None, *messages: ChatMessage) -> None:
        _, sub = self._thread_scope(thread_id)
        if sub is None:
            self._commit_messages(*messages)
        else:
            sub.messages.extend(messages)

    def _emit_model_event(
        self,
        thread_id: str | None,
        asst_msg: ChatMessageAssistant,
        input_snapshot: list[ChatMessage] | None = None,
        usage: ModelUsage | None = None,
        timestamp: datetime | None = None,
    ) -> ModelEvent:
        """Emit a reconstructed ModelEvent for a completed item.

        ``input`` is the conversation snapshot taken before this message was
        appended; ``output.time``/``working_time`` stay unknown because Codex
        owns generation and item events carry no model timing.
        """
        model = self._thread_model(thread_id)
        if usage is None:
            usage = self._pending_response_usage.pop(thread_id, None)
        output = ModelOutput(model=model, choices=[ChatCompletionChoice(message=asst_msg)], usage=usage)
        event_ts = timestamp or datetime.now(timezone.utc)
        event = ModelEvent(
            model=model,
            input=input_snapshot if input_snapshot is not None else self._snapshot_input(),
            tools=[],
            tool_choice="auto",
            config=GenerateConfig(),
            output=output,
            span_id=self._thread_span(thread_id),
            timestamp=event_ts,
            completed=event_ts,
        )
        compat.emit_event(event)
        self._last_model_events[thread_id] = event
        return event

    def _drain_thread_pending(self, thread_id: str | None) -> list[ContentReasoning | ContentText]:
        pending = self._thread_pending(thread_id)
        drained = pending[:]
        pending.clear()
        return drained

    def _record_tool_call_and_message(
        self,
        thread_id: str | None,
        item_id: str,
        tc: ToolCall,
        tool_msg: ChatMessageTool,
        err: ToolCallError | None = None,
    ) -> None:
        pending = self._thread_pending(thread_id)
        active_asst = self._active_asst_msgs.get(thread_id)
        active_event = self._active_model_events.get(thread_id)

        if not pending and active_asst is not None and active_event is not None:
            calls = active_asst.tool_calls or []
            active_asst.tool_calls = [*calls, tc]
            compat.update_event(active_event)
            self._commit_thread_messages(thread_id, tool_msg)
            self._emit_tool_event(thread_id, tc, tool_msg.content, err, message_id=tool_msg.id)
            return

        started, _ = self._item_times.get((thread_id, tc.id)) or self._item_times.get(
            (thread_id, item_id), (None, None)
        )
        model_ts = (started - timedelta(milliseconds=1)) if started else None

        asst_content = self._drain_thread_pending(thread_id)
        asst_msg = ChatMessageAssistant(
            id=_msg_id(thread_id, item_id, "asst"),
            content=asst_content,
            tool_calls=[tc],
            model=self._thread_model(thread_id),
        )
        input_snapshot = self.thread_messages(thread_id)
        self._commit_thread_messages(thread_id, asst_msg, tool_msg)
        event = self._emit_model_event(thread_id, asst_msg, input_snapshot=input_snapshot, timestamp=model_ts)
        self._active_asst_msgs[thread_id] = asst_msg
        self._active_model_events[thread_id] = event
        self._emit_tool_event(thread_id, tc, tool_msg.content, err, message_id=tool_msg.id)

    def _emit_tool_event(
        self,
        thread_id: str | None,
        tool_call: ToolCall,
        result: Any,
        error: ToolCallError | None = None,
        message_id: str | None = None,
    ) -> None:
        native_id = tool_call.id.removeprefix("call_")
        if (thread_id, native_id) in self.inspect_tool_calls or (thread_id, tool_call.id) in self.inspect_tool_calls:
            return  # Inspect's execute_tools already owns the execution event.
        started, completed = self._item_times.get((thread_id, tool_call.id)) or self._item_times.get(
            (thread_id, native_id), (datetime.now(timezone.utc), None)
        )
        span_id = shortuuid.uuid()
        parent = self._thread_span(thread_id)
        if parent:
            compat.emit_event(
                SpanBeginEvent(id=span_id, parent_id=parent, type="tool", name=tool_call.function, timestamp=started)
            )
        compat.emit_event(
            ToolEvent(
                id=tool_call.id,
                function=tool_call.function,
                arguments=tool_call.arguments,
                view=tool_call.view,
                result=result,
                error=error,
                span_id=span_id if parent else None,
                message_id=message_id,
                timestamp=started,
                completed=completed,
            )
        )
        if parent:
            compat.emit_event(SpanEndEvent(id=span_id, timestamp=completed or started))

    # ------------------------------------------------------------------
    # Notification entry points
    # ------------------------------------------------------------------

    def on_item_started(self, method: str, params: dict[str, Any]) -> None:
        item = params.get("item") or {}
        self._item_starts[(params.get("threadId"), item["id"])] = (
            _native_datetime(params.get("startedAtMs")) or datetime.now(timezone.utc)
        )

    def on_item_completed(self, method: str, params: dict[str, Any]) -> None:
        """Handle ``item/completed`` notifications from Codex app-server."""
        self.retain_raw(method, params)
        item = params.get("item", {})
        item_type = item.get("type")
        item_id = item.get("id") or f"anon-{len(self.raw_events)}"
        thread_id = params.get("threadId")

        key = (thread_id, item_id)
        if key in self._seen_items:
            logger.debug("Dropping duplicate item delivery: type=%s id=%s", item_type, item_id)
            return
        self._seen_items.add(key)
        completed = _native_datetime(params.get("completedAtMs"))
        started = self._item_starts.pop(key, None)
        duration = item.get("durationMs")
        if started is None and isinstance(duration, (int, float)) and duration >= 0:
            completed = completed or datetime.now(timezone.utc)
            started = completed - timedelta(milliseconds=duration)
        elif started is not None:
            completed = completed or datetime.now(timezone.utc)
        self._item_times[key] = (started or datetime.now(timezone.utc), completed)

        logger.debug("Mapping item/completed: type=%s id=%s thread=%s", item_type, item_id, thread_id)

        # Route known item types or retain unknown items as raw evidence.
        self._convert_item(thread_id, item, item_type, item_id)

    def _convert_item(
        self, thread_id: str | None, item: dict[str, Any], item_type: str, item_id: str
    ) -> None:
        exchange = self.inspect_exchanges.pop((thread_id, item_id), None)
        if exchange is not None:
            assistant, messages = exchange
            assistant.content = list(self._thread_pending(thread_id))
            self._thread_pending(thread_id).clear()
            assistant.model = self._thread_model(thread_id)
            snapshot = self.thread_messages(thread_id)
            if thread_id == self.root_thread_id:
                self._commit_messages(assistant, *messages)
            elif thread_id is not None:
                sub = self.subagents.get(thread_id)
                if sub is None:
                    sub = self._start_subagent(thread_id, f"subagent_{thread_id[:8]}", item_id, invented=True)
                sub.messages.extend([assistant, *messages])
            self._emit_model_event(thread_id, assistant, snapshot)
            # Later native calls follow the bridged result, so they cannot join an earlier assistant.
            self._active_asst_msgs.pop(thread_id, None)
            self._active_model_events.pop(thread_id, None)
            return
        # Subagent lifecycle
        if item_type == "subAgentActivity":
            self._on_subagent_activity(thread_id, item)
            return

        # Collaboration orchestration
        if item_type == "collabAgentToolCall":
            self._on_collab_tool_call(thread_id, item, item_id)
            return

        # Initialize subagent state if a child thread is observed without a prior spawn event.
        if thread_id and self.root_thread_id and thread_id != self.root_thread_id and thread_id not in self.subagents:
            self._start_subagent(thread_id, f"subagent_{thread_id[:8]}", item_id, invented=True)

        # Shared item converters keep root and child tools consistent.
        if item_type == "reasoning":
            self._map_reasoning(thread_id, item)

        elif item_type == "agentMessage":
            self._map_agent_message(thread_id, item)

        elif item_type == "commandExecution":
            self._map_command_execution(thread_id, item, item_id)

        elif item_type == "fileChange":
            self._map_file_change(thread_id, item, item_id)

        elif item_type == "dynamicToolCall":
            self._map_dynamic_tool(thread_id, item, item_id)

        elif item_type == "mcpToolCall":
            self._map_mcp_tool(thread_id, item, item_id)

        elif item_type in ("webSearch", "imageView", "imageGeneration", "sleep"):
            self._map_generic_tool(thread_id, item, item_id)

        elif item_type == "plan":
            compat.emit_event(
                InfoEvent(source="plan", data=item.get("text") or item, span_id=self._thread_span(thread_id))
            )

        elif item_type in ("enteredReviewMode", "exitedReviewMode"):
            compat.emit_event(
                InfoEvent(source=item_type, data=_truncate_deep(item), span_id=self._thread_span(thread_id))
            )

        elif item_type == "contextCompaction":
            self._map_context_compaction(thread_id, item_id)

        elif item_type == "userMessage":
            self._map_user_message(thread_id, item, item_id)

        elif item_type in ("hookPrompt", "functionCallOutput"):
            self._emit_raw_item(item, f"unmapped {item_type}")

        else:
            self._emit_raw_item(item, "unknown item type")

    # ------------------------------------------------------------------
    # Thread item converters
    # ------------------------------------------------------------------

    def _map_user_message(self, thread_id: str | None, item: dict[str, Any], item_id: str) -> None:
        content = item.get("content")
        text = ""
        if isinstance(content, list):
            text = _content_items_text(content)
        elif isinstance(content, str):
            text = content.strip()
        if not text:
            text = (item.get("text") or "").strip()
        if not text:
            return

        _, sub = self._thread_scope(thread_id)
        if sub is not None:
            # As on the root: earlier output precedes the message, and later calls follow it.
            self._flush_subagent_message(sub)
            self._active_asst_msgs.pop(thread_id, None)
            self._active_model_events.pop(thread_id, None)
            sub.messages.append(ChatMessageUser(id=_msg_id(thread_id, item_id, "user"), content=text))
            compat.emit_event(InputEvent(input=text, input_ansi=text, span_id=sub.agent_span_id))
            return

        # Deduplicate if the exact same message was already the last user message.
        last_user_text: str | None = None
        if self.messages:
            if self.messages[-1].role == "user":
                last_user_text = (self.messages[-1].text or "").strip()
        elif self.state and self.state.messages:
            for m in reversed(self.state.messages):
                if m.role == "user":
                    last_user_text = (m.text or "").strip()
                    break

        if last_user_text is not None:
            if last_user_text == text:
                return
            if last_user_text in text or text in last_user_text:
                if self.state and self.state.messages:
                    for m in reversed(self.state.messages):
                        if m.role == "user" and (m.text or "").strip() == last_user_text:
                            m.content = text
                            break
                if self.messages:
                    for m in reversed(self.messages):
                        if m.role == "user" and (m.text or "").strip() == last_user_text:
                            m.content = text
                            break
                return

        self._flush_thread_content(thread_id)
        self._active_asst_msgs.pop(thread_id, None)
        self._active_model_events.pop(thread_id, None)

        user_msg = ChatMessageUser(
            id=_msg_id(thread_id, item_id, "user"),
            content=text,
        )
        self._commit_messages(user_msg)
        compat.emit_event(InputEvent(input=text, input_ansi=text))

    def _map_reasoning(self, thread_id: str | None, item: dict[str, Any]) -> None:
        summary_text = _join_strs(item.get("summary"))
        reasoning_text = _join_strs(item.get("content"))
        if not reasoning_text and summary_text:
            reasoning_text, summary_text = summary_text, ""
        self._active_asst_msgs.pop(thread_id, None)
        self._active_model_events.pop(thread_id, None)
        if reasoning_text:
            self._thread_pending(thread_id).append(
                ContentReasoning(
                    reasoning=reasoning_text,
                    summary=summary_text if summary_text and summary_text != reasoning_text else None,
                )
            )

    def _map_agent_message(self, thread_id: str | None, item: dict[str, Any]) -> None:
        text = (item.get("text") or "").strip()
        if text:
            self._active_asst_msgs.pop(thread_id, None)
            self._active_model_events.pop(thread_id, None)
            self._thread_pending(thread_id).append(ContentText(text=text))
            _, sub = self._thread_scope(thread_id)
            if sub is not None:
                sub.reports.append(text)
                started, completed = self._item_times.get((thread_id, item.get("id")), (None, None))
                self._flush_subagent_message(sub, timestamp=completed or started)

    def _map_command_execution(self, thread_id: str | None, item: dict[str, Any], item_id: str) -> None:
        _, sub = self._thread_scope(thread_id)
        if sub is not None:
            sub.commands_count += 1
        command = item.get("command", "")
        tc_id = _call_id(item_id)
        tc = ToolCall(id=tc_id, function="bash", arguments={"command": command}, type="function")

        exit_code = item.get("exitCode", 0)
        out = item.get("aggregatedOutput") or ""
        content = f"Exit code: {exit_code}\n{out}" if exit_code not in (0, None) and out else out
        err = (
            ToolCallError(message=f"Command failed with exit code {exit_code}", type="unknown")
            if exit_code not in (0, None)
            else None
        )
        tool_msg = ChatMessageTool(
            id=_msg_id(thread_id, item_id, "tool"),
            tool_call_id=tc_id,
            function="bash",
            content=content or "(empty output)",
            error=err,
        )
        self._record_tool_call_and_message(thread_id, item_id, tc, tool_msg, err)

    def _map_file_change(self, thread_id: str | None, item: dict[str, Any], item_id: str) -> None:
        changes = item.get("changes") or []
        first_path = changes[0].get("path", "") if changes else "/app"
        tc_id = _call_id(item_id)

        diff_chunks = []
        view_chunks = []
        for c in changes:
            p = c.get("path", "")
            d = (c.get("diff") or "").strip()
            if d:
                diff_chunks.append(f"--- {p}\n{d}")
                view_chunks.append(f"**{p}**\n```diff\n{d}\n```")
            elif p:
                diff_chunks.append(f"--- {p}\n(modified)")
                view_chunks.append(f"**{p}** (modified)")

        view_content = "\n\n".join(view_chunks) if view_chunks else "*(no diff content)*"
        call_view = ToolCallContent(
            title=f"edit: {first_path}",
            format="markdown",
            content=view_content,
        )
        tc = ToolCall(
            id=tc_id,
            function="text_editor",
            arguments={"command": "edit", "path": first_path, "changes": changes},
            view=call_view,
            type="function",
        )
        status = item.get("status", "applied")
        content = f"Status: {status}\n" + "\n\n".join(diff_chunks) if diff_chunks else f"Status: {status}"
        err = (
            ToolCallError(message=f"File change status: {status}", type="unknown")
            if status in ("failed", "declined")
            else None
        )
        tool_msg = ChatMessageTool(
            id=_msg_id(thread_id, item_id, "tool"),
            tool_call_id=tc_id,
            function="text_editor",
            content=content.strip(),
            error=err,
        )
        self._record_tool_call_and_message(thread_id, item_id, tc, tool_msg, err)

    def _map_dynamic_tool(self, thread_id: str | None, item: dict[str, Any], item_id: str) -> None:
        tool_name = item.get("tool", "tool")
        args = item.get("arguments", {})
        tc_id = _call_id(item_id)

        tc = ToolCall(
            id=tc_id,
            function=tool_name,
            arguments=args if isinstance(args, dict) else {"arg": args},
            type="function",
        )
        resp_text = _content_items_text(item.get("contentItems"))
        err = (
            ToolCallError(message="Tool reported failure", type="unknown")
            if item.get("success") is False
            else None
        )
        tool_msg = ChatMessageTool(
            id=_msg_id(thread_id, item_id, "tool"),
            tool_call_id=tc_id,
            function=tool_name,
            content=resp_text or "(completed)",
            error=err,
        )
        self._record_tool_call_and_message(thread_id, item_id, tc, tool_msg, err)

    def _map_mcp_tool(self, thread_id: str | None, item: dict[str, Any], item_id: str) -> None:
        server = item.get("server", "mcp")
        tool = item.get("tool", "tool")
        args = item.get("arguments", {})
        tc_id = _call_id(item_id)
        fn_name = self.mcp_tool_aliases.get((server, tool), f"{server}_{tool}" if server else tool)

        tc = ToolCall(
            id=tc_id,
            function=fn_name,
            arguments=args if isinstance(args, dict) else {"arguments": args},
            type="function",
        )

        # Schema: mcpToolCall.result.content / result.structuredContent
        result = item.get("result") or {}
        answer = self.receipt_reader(item, self.root_thread_id, thread_id) if self.receipt_reader else None
        if answer:
            self.submitted_answers.append(answer)
        content_parts: list[str] = []
        if isinstance(result, dict):
            for part in result.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    content_parts.append(str(part["text"]))
            structured = result.get("structuredContent")
            if structured:
                content_parts.append(f"[structured] {structured!r}")
        resp_text = "\n".join(p for p in content_parts if p)
        error_info = item.get("error") or (resp_text if item.get("status") == "failed" else None)
        err = ToolCallError(message=str(error_info), type="unknown") if error_info else None
        tool_msg = ChatMessageTool(
            id=_msg_id(thread_id, item_id, "tool"),
            tool_call_id=tc_id,
            function=fn_name,
            content=resp_text or "(no output)",
            error=err,
        )
        self._record_tool_call_and_message(thread_id, item_id, tc, tool_msg, err)

    def _map_generic_tool(self, thread_id: str | None, item: dict[str, Any], item_id: str) -> None:
        item_type = item.get("type", "tool")
        tc_id = _call_id(item_id)
        if item_type == "webSearch":
            # Inspect's web_search shape: the viewer renders ``query``.
            function = "web_search"
            action = item.get("action") or {}
            arguments = {"query": item.get("query") or action.get("query") or ""}
            arguments.update({k: action[k] for k in ("url", "pattern") if action.get(k)})
        else:
            function = item_type
            arguments = {k: v for k, v in item.items() if k not in ("id", "type", "status")}
        tc = ToolCall(id=tc_id, function=function, arguments=arguments, type="function")
        status = item.get("status")
        if item_type == "webSearch" and item.get("results"):
            result_text = "\n".join(
                f"{r.get('title', '')}: {r.get('url', '')}" for r in item["results"] if isinstance(r, dict)
            )
        elif status:
            result_text = str(status)
        else:
            result_text = "(completed)"
        err = (
            ToolCallError(message=f"{item_type} failed", type="unknown")
            if item.get("failure") or status == "failed"
            else None
        )
        tool_msg = ChatMessageTool(
            id=_msg_id(thread_id, item_id, "tool"),
            tool_call_id=tc_id,
            function=function,
            content=result_text,
            error=err,
        )
        self._record_tool_call_and_message(thread_id, item_id, tc, tool_msg, err)

    def _map_context_compaction(self, thread_id: str | None, item_id: str) -> None:
        # Flush any pending content across the compaction boundary first.
        self._flush_thread_content(thread_id)
        # Context size before compaction is not reported in the event payload.
        times = self._item_times.get((thread_id, item_id))
        duration = (times[1] - times[0]).total_seconds() if (times and times[0] and times[1]) else 0.0
        c_ts = times[0] if (times and times[0]) else datetime.now(timezone.utc)
        # Supply compaction duration for Inspect's CompactionEventView header formatting.
        compat.emit_event(
            CompactionEvent(
                type="summary",
                role=self._thread_model(thread_id),
                source="codex",
                tokens_before=None,
                tokens_after=None,
                span_id=self._thread_span(thread_id),
                working_start=max(0.0, duration),
                metadata={"compaction_id": item_id},
                timestamp=c_ts,
            )
        )

    # ------------------------------------------------------------------
    # Subagents
    # ------------------------------------------------------------------

    def _start_subagent(
        self,
        thread_id: str,
        name: str,
        item_id: str,
        prompt: str = "",
        parent_thread_id: str | None = None,
        model_name: str | None = None,
        invented: bool = False,
    ) -> SubagentState:
        if thread_id in self.subagents:
            sub = self.subagents[thread_id]
            if prompt:
                if sub.prompt != prompt:
                    sub.messages.insert(0, ChatMessageUser(content=prompt))
                    compat.emit_event(InputEvent(input=prompt, input_ansi=prompt, span_id=sub.agent_span_id))
                sub.prompt = prompt
            if model_name:
                sub.model_name = model_name
            if parent_thread_id:
                sub.parent_thread_id = parent_thread_id
                if sub.tool_span:
                    sub.tool_span.parent_id = self._thread_span(parent_thread_id)
                    compat.update_event(sub.tool_span)
            sub.name = name
            if sub.tool_event:
                sub.tool_event.arguments.update(subagent_type=name, prompt=sub.prompt)
                compat.update_event(sub.tool_event)
            return sub

        prompt_text = prompt or "[Delegated prompt unavailable in native evidence]"
        tool_span_id = shortuuid.uuid()
        agent_span_id = shortuuid.uuid()
        tc_id = _call_id(item_id)

        parent_span = (
            self._thread_span(parent_thread_id) if parent_thread_id else self.root_agent_span_id
        )

        tc = ToolCall(
            id=tc_id,
            function="agent",
            arguments={"subagent_type": name, "prompt": prompt_text},
            type="function",
        )
        # Snapshot input before appending the new assistant message to avoid self-reference.
        input_snapshot = self._snapshot_input()
        parent_id = parent_thread_id or self.root_thread_id
        parent_pending = self._thread_pending(parent_id)
        content = list(parent_pending)
        parent_pending.clear()
        asst_msg = ChatMessageAssistant(
            id=_msg_id(parent_thread_id or self.root_thread_id, item_id, "asst"),
            content=content,
            tool_calls=[tc],
            model=self._thread_model(parent_id),
        )
        if parent_id == self.root_thread_id:
            self._commit_messages(asst_msg)
        else:
            input_snapshot = self.thread_messages(parent_id)
            parent_sub = self.subagents.get(parent_id)
            if parent_sub:
                parent_sub.messages.append(asst_msg)
        self._emit_model_event(parent_id, asst_msg, input_snapshot)

        tool_event = ToolEvent(
            id=tc_id,
            function="agent",
            arguments={"subagent_type": name, "prompt": prompt_text},
            view=ToolCallContent(title=f"agent: {name}", format="markdown", content="{{prompt}}"),
            agent=name,
            agent_span_id=agent_span_id,
            span_id=tool_span_id,
            pending=True,
        )

        sub = SubagentState(
            thread_id=thread_id,
            name=name,
            tool_span_id=tool_span_id,
            agent_span_id=agent_span_id,
            call_id=tc_id,
            parent_thread_id=parent_thread_id or self.root_thread_id,
            model_name=self._thread_models.get(thread_id) or model_name,
            prompt=prompt_text,
            tool_event=tool_event,
        )
        self.subagents[thread_id] = sub

        sub.tool_span = SpanBeginEvent(id=tool_span_id, parent_id=parent_span, type="tool", name="agent")
        compat.emit_event(sub.tool_span)
        compat.emit_event(tool_event)
        compat.emit_event(
            SpanBeginEvent(id=agent_span_id, parent_id=tool_span_id, type="agent", name=name)
        )
        if prompt:
            sub.messages.append(ChatMessageUser(content=prompt))
            compat.emit_event(InputEvent(input=prompt, input_ansi=prompt, span_id=agent_span_id))
        if invented:
            compat.emit_event(
                InfoEvent(
                    source="codex-subagent",
                    data={
                        "note": "subagent thread observed without a spawn item; parent call reconstructed",
                        "threadId": thread_id,
                    },
                )
            )
        return sub

    def _on_subagent_activity(self, parent_thread_id: str | None, item: dict[str, Any]) -> None:
        kind = item.get("kind")
        agent_thread_id = item.get("agentThreadId")
        agent_path = item.get("agentPath", "")
        if not agent_thread_id:
            return
        if kind == "started":
            name = agent_path.split("/")[-1] if "/" in agent_path else (agent_path or "subagent")
            sub = self._start_subagent(
                agent_thread_id, name, item.get("id") or agent_thread_id,
                parent_thread_id=parent_thread_id,
                prompt=self._delegation_calls.pop((parent_thread_id, item["id"]), ""),
            )
            sub.agent_path = agent_path
        elif kind in ("completed", "interrupted"):
            self._complete_subagent(agent_thread_id, interrupted=(kind == "interrupted"))
        elif kind == "interacted":
            message = self._delegation_calls.pop((parent_thread_id, item["id"]), None)
            sub = self.subagents.get(agent_thread_id)
            if message is not None:
                if agent_thread_id == self.root_thread_id:
                    self.append_user_message(message)
                elif sub:
                    sub.messages.append(ChatMessageUser(content=message))
                    compat.emit_event(InputEvent(input=message, input_ansi=message, span_id=sub.agent_span_id))
            if sub:
                compat.emit_event(InfoEvent(source="codex-child-task", data=item, span_id=sub.agent_span_id))

    def _on_collab_tool_call(self, thread_id: str | None, item: dict[str, Any], item_id: str) -> None:
        collab_tool = item.get("tool")
        if collab_tool in _COLLAB_TOOLS_HIDDEN:
            return
        if collab_tool == "spawnAgent":
            # Schema: senderThreadId, receiverThreadIds, prompt, model.
            prompt = item.get("prompt")
            model = item.get("model")
            sender = item.get("senderThreadId") or thread_id
            for receiver in item.get("receiverThreadIds") or []:
                if receiver:
                    self._start_subagent(
                        receiver,
                        f"subagent_{str(receiver)[:8]}",
                        item_id,
                        prompt=str(prompt) if prompt else "",
                        parent_thread_id=sender,
                        model_name=model,
                    )
            return
        if collab_tool == "closeAgent":
            for receiver in item.get("receiverThreadIds") or []:
                if receiver in self.subagents:
                    self._complete_subagent(receiver, final=True)
            return
        # Filter out polling/waiting items from transcripts; retain state transitions in raw evidence.
        self._emit_raw_item(item, f"collab tool {collab_tool}")

    def _flush_subagent_message(self, sub: SubagentState, timestamp: datetime | None = None) -> None:
        content = sub.drain_pending()
        if not content:
            return
        model = self._thread_model(sub.thread_id)
        asst_msg = ChatMessageAssistant(
            id=_msg_id(sub.thread_id, f"msg-{len(sub.messages)}", "asst"),
            content=content,
            model=model,
        )
        input_snapshot = list(sub.messages)
        sub.messages.append(asst_msg)
        event_ts = timestamp or self._last_event_time or datetime.now(timezone.utc)
        event = self._emit_model_event(
            sub.thread_id,
            asst_msg,
            input_snapshot=input_snapshot,
            timestamp=event_ts,
        )
        self._active_asst_msgs[sub.thread_id] = asst_msg
        self._active_model_events[sub.thread_id] = event

    def _complete_subagent(self, thread_id: str, interrupted: bool = False, *, final: bool = False) -> None:
        sub = self.subagents.get(thread_id)
        if not sub or sub.completed:
            return
        if final:
            for child in list(self.subagents.values()):
                if child.parent_thread_id == thread_id:
                    self._complete_subagent(child.thread_id, interrupted=interrupted, final=True)
        self._flush_subagent_message(sub)
        self._flush_pending_usage(thread_id)

        report = "\n\n".join(sub.reports).strip()
        if not report:
            if interrupted:
                report = f"Subagent '{sub.name}' interrupted."
            elif sub.commands_count > 0:
                report = f"Subagent '{sub.name}' completed (executed {sub.commands_count} commands)."
            else:
                report = f"Subagent '{sub.name}' completed."

        now_utc = datetime.now(timezone.utc)
        tool_msg = ChatMessageTool(
            id=_msg_id(sub.thread_id, sub.call_id, "tool"),
            tool_call_id=sub.call_id,
            function="agent",
            content=report,
        )
        parent = self.subagents.get(sub.parent_thread_id)
        parent_messages = parent.messages if parent else self.messages
        if parent or sub.parent_thread_id == self.root_thread_id:
            existing = next((m for m in parent_messages if m.id == tool_msg.id), None)
            if existing is None:
                if parent:
                    parent.messages.append(tool_msg)
                else:
                    self._commit_messages(tool_msg)
            else:
                existing.content = report

        if interrupted:
            compat.emit_event(
                InterruptEvent(
                    source="system",
                    interrupted="tool_call",
                    interrupted_tool_call_id=sub.call_id,
                    span_id=sub.agent_span_id,
                )
            )
        if sub.tool_event:
            sub.tool_event._set_result(
                result=report,
                truncated=None,
                error=None,
                working_time=(now_utc - sub.tool_event.timestamp).total_seconds(),
                agent=sub.name,
                failed=None,
                message_id=tool_msg.id,
                agent_span_id=sub.agent_span_id,
            )
            sub.tool_event.completed = now_utc
            compat.update_event(sub.tool_event)
        if final:
            compat.emit_event(SpanEndEvent(id=sub.agent_span_id))
            compat.emit_event(SpanEndEvent(id=sub.tool_span_id))
            sub.completed = True

    # ------------------------------------------------------------------
    # Turn / usage / misc handlers
    # ------------------------------------------------------------------

    def on_turn_completed(self, method: str, params: dict[str, Any]) -> None:
        """Flush pending content for the completed turn.

        A child's identity remains reusable after turn/task completion.
        Actual thread closure or finalization closes its span.
        """
        self.retain_raw(method, params)
        thread_id = params.get("threadId")
        self._active_asst_msgs.pop(thread_id, None)
        self._active_model_events.pop(thread_id, None)
        _, sub = self._thread_scope(thread_id)
        if sub is not None:
            self._flush_subagent_message(sub)
            return
        self._flush_thread_content(thread_id)

    def _flush_thread_content(self, thread_id: str | None) -> None:
        _, sub = self._thread_scope(thread_id)
        if sub is not None:
            self._flush_subagent_message(sub)
            return
        pending = self._thread_pending(thread_id)
        if not pending:
            return
        drained = pending[:]
        pending.clear()
        self._active_asst_msgs.pop(thread_id, None)
        self._active_model_events.pop(thread_id, None)
        input_snapshot = self._snapshot_input()
        asst_msg = ChatMessageAssistant(
            id=_msg_id(thread_id, f"turn-flush-{len(self.messages)}", "asst"),
            content=drained,
            model=self._thread_model(thread_id),
        )
        self._commit_messages(asst_msg)
        self._emit_model_event(thread_id, asst_msg, input_snapshot=input_snapshot)

    def _drain_root_pending(self) -> list[ContentReasoning | ContentText]:
        return self._drain_thread_pending(self.root_thread_id)

    def _meter_thread(self, thread_id: str | None) -> None:
        usage = self._thread_usage(thread_id)
        native = usage.last_reported_total or ModelUsage()
        calls = usage.response_total
        # Use componentwise maximum between lifetime and per-response totals to avoid double counting.
        candidate = _usage_envelope(native, calls)
        delta = _usage_delta(candidate, usage.running_total)
        _usage_add(usage.running_total, delta)
        compat.record_usage(self._thread_model(thread_id), delta)
        if not self.tearing_down:
            compat.record_limit_usage(delta)
            compat.check_limits()

    def on_response_completed(self, method: str, params: dict[str, Any]) -> None:
        key = (params["threadId"], params["responseId"])
        prior = self.responses.get(key)
        if prior is not None and (prior.get("usage") is not None or params.get("usage") is None):
            return
        thread_id = params.get("threadId")
        model = self._thread_model(thread_id)
        rerouted = thread_id in self._rerouted
        attribution = ("model-rerouted" if rerouted else
                       "unobserved" if model == "codex/unknown" else "thread-configuration")
        record = dict(params, configuredModel=self._configured_model(thread_id), modelAttribution=attribution,
                      inputRepresentation="unobserved", servingModel=model if rerouted else "unobserved")
        self.responses[key] = record
        usage = params.get("usage")
        if usage is not None:
            model_usage = _to_usage(usage)
            if method == "rawResponse/completed":
                # Record usage before limit checks can trigger.
                compat.record_response_usage(model, model_usage)
            _usage_add(self._thread_usage(thread_id).response_total, model_usage)
            self._meter_thread(thread_id)
            active_event = self._active_model_events.get(thread_id)
            if active_event is not None and active_event.output is not None:
                if active_event.output.usage is None:
                    active_event.output.usage = model_usage
                else:
                    _usage_add(active_event.output.usage, model_usage)
                compat.update_event(active_event)
            else:
                if thread_id in self._pending_response_usage:
                    _usage_add(self._pending_response_usage[thread_id], model_usage)
                else:
                    self._pending_response_usage[thread_id] = model_usage

    def on_token_usage(self, method: str, params: dict[str, Any]) -> None:
        """Handle ``thread/tokenUsage/updated`` with monotonic delta accounting.

        Totals are per-thread lifetime counters. Each report contributes the
        delta since the previous report exactly once: to Inspect's token-limit
        scopes, to the sample/task usage registries (keyed by the thread's
        configured model). Response records are reconciled independently by native
        response ID; reconstructed activity events never receive usage.
        """
        self.retain_raw(method, params)
        usage_data = params.get("tokenUsage", {})
        total = usage_data.get("total") or {}
        thread_id = params.get("threadId")

        usage_state = self._thread_usage(thread_id)

        if total:
            usage_state.record_total(_to_usage(total))
            self._meter_thread(thread_id)

    def on_turn_started(self, method: str, params: dict[str, Any]) -> None:
        self.retain_raw(method, params)
        thread_id = params.get("threadId")
        turn_id = (params.get("turn") or {}).get("id")
        if thread_id in self._rerouted and self._rerouted[thread_id][0] != turn_id:
            del self._rerouted[thread_id]
        self._active_asst_msgs.pop(thread_id, None)
        self._active_model_events.pop(thread_id, None)
        ts = self._last_event_time or datetime.now(timezone.utc)
        compat.emit_event(InfoEvent(source="codex-turn", data=params, span_id=self._thread_span(thread_id), timestamp=ts))

    def on_thread_metadata(self, method: str, params: dict[str, Any]) -> None:
        """Native parent IDs and configured settings supersede provisional data."""
        thread = params.get("thread") or params.get("threadSettings") or {}
        thread_id = thread.get("id") or params.get("threadId")
        if not thread_id:
            return
        metadata = self.thread_metadata.setdefault(thread_id, {})
        previous_model = self._thread_model(thread_id)
        metadata.update(thread)
        if metadata.get("model"):
            self._thread_models[thread_id] = metadata["model"]
            if previous_model == "codex/unknown":
                # Attribute tokens to the thread model once observed.
                compat.reassign_usage(previous_model, metadata["model"], self._thread_usage(thread_id).running_total)
                for (response_thread, _), record in self.responses.items():
                    if response_thread == thread_id and record["configuredModel"] == previous_model:
                        record.update(configuredModel=metadata["model"],
                                      modelAttribution="first-observed-thread-configuration")
                compat.emit_event(InfoEvent(source="codex-model-attribution", data={
                    "threadId": thread_id, "configuredModel": metadata["model"],
                    "modelAttribution": "first-observed-thread-configuration", "servingModel": "unobserved",
                }, span_id=self._thread_span(thread_id), timestamp=self._last_event_time or datetime.now(timezone.utc)))
            if thread_id == self.root_thread_id:
                self.model_name = metadata["model"]
        parent = metadata.get("parentThreadId")
        if parent:
            sub = self.subagents.get(thread_id)
            name = sub.name if sub and sub.agent_path else metadata.get("agentNickname") or f"subagent_{thread_id[:8]}"
            self._start_subagent(thread_id, name, f"thread-{thread_id}", parent_thread_id=parent, model_name=metadata.get("model"), invented=sub is None)
        elif thread_id in self.subagents and metadata.get("model"):
            self.subagents[thread_id].model_name = metadata["model"]

    def on_thread_closed(self, method: str, params: dict[str, Any]) -> None:
        self._complete_subagent(params["threadId"], final=True)

    def on_raw_item(self, method: str, params: dict[str, Any]) -> None:
        """Retain delegation attempts until native activity confirms acceptance.

        The pinned core records raw calls before dispatch (stream_events_utils.rs),
        and emits started/interacted with that call ID only after spawn/send
        succeeds. The activity's thread ID owns attribution, including relative
        targets; attempted calls alone never establish delivered input.
        """
        item = params.get("item") or {}
        parent = params.get("threadId")
        call_id = item.get("call_id")
        if item.get("type") == "function_call_output":
            self._delegation_calls.pop((parent, call_id), None)
            return
        if item.get("type") != "function_call":
            return
        try:
            args = json.loads(item.get("arguments") or "{}")
        except json.JSONDecodeError:
            return
        if not isinstance(args, dict):
            return
        name = item.get("name", "").rsplit(".", 1)[-1]
        if (name in ("spawn_agent", "followup_task", "send_message")
                and call_id and isinstance(args.get("message"), str)):
            self._delegation_calls[(parent, call_id)] = args["message"]
            compat.emit_event(InfoEvent(source="codex-delegation", data={
                "parentThreadId": parent, "delivery": "attempted", "call": item,
            }, timestamp=self._last_event_time or datetime.now(timezone.utc)))

    def on_model_rerouted(self, method: str, params: dict[str, Any]) -> None:
        """Record a safety reroute and credit the rest of the turn to the serving model.

        The pinned core reports a serving-model mismatch at most once per turn
        and does not identify the serving model of later turns, so usage and
        events are attributed to ``toModel`` until the thread's next turn.
        """
        self.retain_raw(method, params)
        thread_id = params.get("threadId")
        self.reroutes.append({key: params.get(key) for key in ("threadId", "turnId", "fromModel", "toModel", "reason")})
        if params.get("toModel"):
            self._rerouted[thread_id] = (params.get("turnId"), params["toModel"])
        compat.emit_event(
            InfoEvent(
                source="codex-model-rerouted",
                data=params,
                span_id=self._thread_span(thread_id),
                timestamp=self._last_event_time or datetime.now(timezone.utc),
            )
        )

    def on_plan_updated(self, method: str, params: dict[str, Any]) -> None:
        """Present a native plan update as the ``update_plan`` call Codex made.

        Arguments use Inspect's ``update_plan`` shape so the viewer renders
        the checklist.
        """
        self.retain_raw(method, params)
        thread_id = params.get("threadId")
        steps = [
            {"step": str(s.get("step", "")), "status": PLAN_STATUSES.get(s.get("status"), "pending")}
            for s in params.get("plan") or [] if isinstance(s, dict)
        ]
        arguments: dict[str, Any] = {"plan": steps}
        if params.get("explanation"):
            arguments["explanation"] = params["explanation"]
        plan_str = json.dumps([thread_id, arguments], sort_keys=True)
        if plan_str == self._last_plan_text:
            return
        self._last_plan_text = plan_str
        self._plan_updates += 1
        item_id = f"plan-{params.get('turnId')}-{self._plan_updates}"
        tc = ToolCall(id=_call_id(item_id), function="update_plan", arguments=arguments, type="function")
        tool_msg = ChatMessageTool(
            id=_msg_id(thread_id, item_id, "tool"), tool_call_id=tc.id, function="update_plan", content="Plan updated",
        )
        self._record_tool_call_and_message(thread_id, item_id, tc, tool_msg)

    def on_goal_updated(self, method: str, params: dict[str, Any]) -> None:
        self.retain_raw(method, params)
        goal_data = params.get("goal") or params
        goal_key = (
            (goal_data.get("objective"), goal_data.get("status"))
            if isinstance(goal_data, dict)
            else json.dumps(goal_data, sort_keys=True)
            if isinstance(goal_data, list)
            else str(goal_data)
        )
        if goal_key == self._last_goal_key:
            return
        self._last_goal_key = goal_key
        compat.emit_event(InfoEvent(
            source="goal", data=_truncate_deep(goal_data),
            span_id=self.root_agent_span_id,
            timestamp=self._last_event_time or datetime.now(timezone.utc),
        ))

    def on_diff_updated(self, method: str, params: dict[str, Any]) -> None:
        # Retain git diffs in evidence while skipping transcript InfoEvents to avoid timeline bloat.
        self.retain_raw(method, params)

    def on_review_event(self, method: str, params: dict[str, Any]) -> None:
        """Handle auto-approval review notifications (started and completed)."""
        self.retain_raw(method, params)
        review = params.get("review") or {}
        status = review.get("status")
        # Retain unmapped approval states in native evidence.
        ts = self._last_event_time or datetime.now(timezone.utc)
        compat.emit_event(InfoEvent(
            source="codex-auto-review", data=params,
            span_id=self._thread_span(params.get("threadId")),
            timestamp=ts,
        ))
        if status not in ("approved", "denied", "timedOut", "aborted"):
            return
        decision = {"approved": "continue", "denied": "terminate", "timedOut": "escalate", "aborted": "escalate"}[status]
        tc = ToolCall(id=params["targetItemId"], function="review", arguments=params.get("action") or {})
        compat.emit_event(
            ReviewEvent(
                message=f"Automated tool execution review ({method.rsplit('/', 1)[-1]})",
                call=tc,
                reviewer="codex-auto-review",
                decision=decision,
                explanation=review.get("rationale"),
                span_id=self._thread_span(params.get("threadId")),
                timestamp=ts,
            )
        )

    def on_approval_requested(self, method: str, params: dict[str, Any]) -> None:
        """Emit ApprovalEvent evidence for server approval requests."""
        call_id = params.get("itemId") or params.get("id") or str(len(self.raw_events))
        cmd = params.get("command") or params.get("cmd") or params.get("file") or _truncate_deep(params)
        fn_name = "bash" if "command" in method.lower() or "exec" in method.lower() else "edit_file"
        tc = ToolCall(
            id=call_id,
            function=fn_name,
            arguments={"command": cmd} if fn_name == "bash" else {"file": cmd},
        )
        compat.emit_event(
            ApprovalEvent(
                message=f"Auto-accepted {fn_name} (explicit unattended policy): {cmd}",
                call=tc,
                approver="auto",
                decision="approve",
                span_id=self._thread_span(params.get("threadId")),
            )
        )

    def approval_call(self, method: str, params: dict[str, Any]) -> ToolCall:
        """The tool call an Inspect approver reviews for a native approval request.

        Uses the identity and function of the item it gates, as later mapped.
        """
        call_id = _call_id(params.get("itemId") or str(len(self.raw_events)))
        if method in ("item/fileChange/requestApproval", "applyPatchApproval"):
            arguments = {"command": "edit", "path": params.get("grantRoot") or params.get("cwd") or "",
                         "reason": params.get("reason")}
            return ToolCall(id=call_id, function="text_editor", arguments=arguments)
        command = params.get("command")
        if isinstance(command, list):
            command = " ".join(str(part) for part in command)
        return ToolCall(id=call_id, function="bash", arguments={"command": command or ""})

    # ------------------------------------------------------------------
    # Solver-facing helpers
    # ------------------------------------------------------------------

    def append_user_message(self, text: str) -> None:
        user_msg = ChatMessageUser(id=_msg_id(self.root_thread_id, f"user-{len(self.messages)}", "user"), content=text)
        self._commit_messages(user_msg)
        compat.emit_event(InputEvent(input=text, input_ansi=text))

    def append_assistant_message(self, text: str) -> None:
        asst_msg = ChatMessageAssistant(
            id=_msg_id(self.root_thread_id, f"note-{len(self.messages)}", "asst"),
            content=[ContentText(text=text)],
            model=self.model_name,
        )
        input_snapshot = self._snapshot_input()
        self._commit_messages(asst_msg)
        self._emit_model_event(self.root_thread_id, asst_msg, input_snapshot=input_snapshot)

    def emit_reconstruction_label(self, timestamp: datetime | None = None) -> None:
        """Record, once per run, what the emitted model events actually are."""
        compat.emit_event(
            InfoEvent(
                source="codex-reconstruction",
                data={
                    "note": (
                        "ModelEvents in this transcript are reconstructed from Codex "
                        "item/completed notifications. Codex owns model generation; "
                        "input is a pre-commit conversation snapshot, and model call "
                        "timing/tools/config are not observable and left unknown."
                    )
                },
                span_id=self.root_agent_span_id,
                timestamp=timestamp or datetime.now(timezone.utc),
            )
        )

    @property
    def root_running_usage(self) -> ModelUsage:
        return self.root_usage.running_total

    def _flush_pending_usage(self, thread_id: str | None = None) -> None:
        """Attach uncommitted response tokens to the thread's active or last ModelEvent."""
        tids = [thread_id] if thread_id is not None else list(self._pending_response_usage.keys())
        for tid in tids:
            pending_usage = self._pending_response_usage.pop(tid, None)
            if pending_usage and (pending_usage.total_tokens or 0) > 0:
                target_event = self._active_model_events.get(tid) or self._last_model_events.get(tid)
                if target_event is not None and target_event.output is not None:
                    if target_event.output.usage is None:
                        target_event.output.usage = pending_usage
                    else:
                        _usage_add(target_event.output.usage, pending_usage)
                    compat.update_event(target_event)

    def finalize_state(
        self,
        state: TaskState,
        final_submission: str,
        error: dict[str, str] | None = None,
        *, interrupted: bool = False,
        stop_reason: StopReason = "stop",
    ) -> TaskState:
        """Commit mapped messages and output structure to TaskState."""
        # Finalize open subagents and flush remaining pending content.
        for sub in list(self.subagents.values()):
            if not sub.completed:
                self._complete_subagent(sub.thread_id, interrupted=interrupted, final=True)
        self._flush_thread_content(self.root_thread_id)
        self._flush_pending_usage()

        existing_ids = {getattr(m, "id", None) for m in state.messages if getattr(m, "id", None)}
        for m in self.messages:
            m_id = getattr(m, "id", None)
            if not m_id or m_id not in existing_ids:
                state.messages.append(m)
                if m_id:
                    existing_ids.add(m_id)

        state.metadata["codex_raw_event_count"] = len(self.raw_events)
        state.metadata["codex_raw_events_dropped"] = self.dropped_event_count
        state.metadata["codex_threads"] = self.thread_metadata
        state.metadata["codex_accounting"] = {
            "observed_responses": len(self.responses),
            "responses_missing_usage": sum(p.get("usage") is None for p in self.responses.values()),
            "unattributed_tokens_by_thread": {
                tid: max(0, usage.running_total.total_tokens - usage.response_total.total_tokens)
                for tid, usage in [(self.root_thread_id or "root", self.root_usage), *self._child_usage.items()]
            },
            "provider_inputs": "unobserved",
            "failed_inference_attempts": "not fully observable",
        }
        if error:
            state.metadata["codex_error"] = error
        if self.reroutes:
            state.metadata["codex_safety"] = {"reroutes": self.reroutes}

        # Set the final output choice for scorers while preserving transcript history.
        answer_message = ChatMessageAssistant(
            id=_msg_id(self.root_thread_id, "final", "asst"),
            content=final_submission,
            model=self.model_name,
        )

        # Aggregate token usage across all threads into the final output.
        total_usage = _copy_usage(self.root_usage.running_total)
        for child_usage in self._child_usage.values():
            _usage_add(total_usage, child_usage.running_total)

        state.output = ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(message=answer_message, stop_reason=stop_reason)],
            completion=final_submission,
            usage=total_usage,
        )
        state.completed = True
        return state


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _native_datetime(value: Any, *, milliseconds: bool = True) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000 if milliseconds else value, tz=timezone.utc)
    return None


def _call_id(item_id: str) -> str:
    return item_id if item_id.startswith("call_") else f"call_{item_id}"


def _msg_id(thread_id: str | None, item_id: str, role: str) -> str:
    tid = (thread_id or "root")[:12]
    return f"msg_{tid}_{role}_{item_id}"


def _join_strs(value: Any) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(s) for s in value if s).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _content_items_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    parts: list[str] = []
    for c in items:
        if isinstance(c, dict) and c.get("text"):
            parts.append(str(c["text"]))
    return "\n".join(parts)


def _to_usage(d: dict[str, Any]) -> ModelUsage:
    # Calculate non-cached input tokens.
    cached = d.get("cachedInputTokens") or 0
    written = d.get("cacheWriteInputTokens") or 0
    return ModelUsage(
        input_tokens=max(0, d.get("inputTokens", 0) - cached - written),
        output_tokens=d.get("outputTokens", 0),
        total_tokens=d.get("totalTokens", 0),
        reasoning_tokens=d.get("reasoningOutputTokens", 0),
        input_tokens_cache_read=d.get("cachedInputTokens"),
        input_tokens_cache_write=d.get("cacheWriteInputTokens"),
    )
