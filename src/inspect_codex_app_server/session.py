"""Own native Codex sessions, execution control, accounting and recovery."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, Sequence, TypedDict

import anyio
import shortuuid
from inspect_ai.agent import AgentAttempts, AgentInterrupted, AgentState, agent_channel
from inspect_ai.event import (
    InfoEvent,
    InterruptEvent,
    SampleLimitEvent,
    SpanBeginEvent,
    SpanEndEvent,
)
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentAudio,
    ContentImage,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
    StopReason,
)
from inspect_ai.review import ReviewPolicy
from inspect_ai.scorer import score
from inspect_ai.solver import TaskState
from inspect_ai.tool import Skill, Tool, install_skills
from inspect_ai.util import Checkpointer, message_limit, sandbox

from . import _generated_protocol as p
from . import compat
from .auth import account_summary, require_authentication
from .client import (
    ApprovalDecision,
    CodexClient,
    ExecRemoteJsonRpcTransport,
    JsonRpcError,
)
from .container import (
    CODEX_HOME,
    ensure_container_codex,
    get_container_ip,
    resolve_service_container,
)
from .evidence import NativeEvidence
from .mapping import EventStreamMapper
from .policy import RunPolicy, accepted_answer, split_history
from .runtime.budget import (
    BudgetTracker,
    time_budget_depleted_event,
)
from .runtime.modelapi import register_placeholder_modelapi
from .subagent import CodexSubagent, role_config, role_overrides
from .tools import _media_url, inspect_tool_handler

logger = logging.getLogger(__name__)


# Default unattended execution policy: automatic approvals and full container access.
class _UnattendedPolicy(TypedDict):
    approval_policy: Literal["never"]
    sandbox_mode: Literal["danger-full-access"]


UNATTENDED_POLICY: _UnattendedPolicy = {
    "approval_policy": "never",
    "sandbox_mode": "danger-full-access",
}

# With an Inspect approver active, ``untrusted`` is the pinned native policy
# that requests approval before commands (``on-request`` leaves it to the model).
INSPECT_APPROVAL_POLICY: Literal["untrusted"] = "untrusted"

# GenerateConfig fields applied to Codex; any other field set is reported.
APPLIED_GENERATE_CONFIG = frozenset({"reasoning_effort", "reasoning_summary", "response_schema"})

CONTINUATION_GRACE_SEC = 20.0

# Inactivity threshold between turns before reporting a stalled session.
STALL_TIMEOUT_SEC = 900.0

# Backoff before retrying a transiently failed turn/start: 1s, 2s, 4s... capped.
TURN_START_BACKOFF_CAP_SEC = 10.0

# Cadence at which waiting time is credited during quota holds.
QUOTA_CREDIT_SEC = 1.0

# Native warnings recorded as InfoEvents and in codex_warnings (configWarning fails fast).
NATIVE_WARNINGS = ("configWarning", "warning", "deprecationNotice", "guardianWarning")

# Native web search modes (config ``web_search``).
WebSearchMode = Literal["disabled", "cached", "indexed", "live"]
WEB_SEARCH_MODES: tuple[WebSearchMode, ...] = ("disabled", "cached", "indexed", "live")

# Native turn errors that end a run normally with react's stop reasons.
POLICY_BLOCK_ERRORS = frozenset({"cyberPolicy", "misalignmentPolicyViolation", "tooManyDenials"})
CONTEXT_LIMIT_ERRORS = frozenset({"contextWindowExceeded", "sessionBudgetExceeded"})
NORMAL_STOPS: dict[str, StopReason] = {"policy_block": "content_filter", "context_limit": "model_length"}

# Terminal goal statuses indicating completion.
GOAL_SUCCESS_STATUSES = frozenset({"complete", "completed", "done"})


class CodexExecutionError(Exception):
    """Terminal execution failure; propagates so Inspect records a sample error."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"[{kind}] {detail}")
        self.kind = kind
        self.detail = detail


# Register placeholder ModelAPI so Inspect CLI and eval_set can resolve codex/* models.
register_placeholder_modelapi("codex")


def _format_turn_error(turn_obj: dict[str, Any] | None) -> str:
    """Format turn failure details into a human-readable string."""
    if not isinstance(turn_obj, dict):
        return "Turn failed"
    err = turn_obj.get("error")
    status = turn_obj.get("status", "failed")
    if not err and (turn_obj.get("message") or turn_obj.get("codexErrorInfo")):
        err = turn_obj
    if isinstance(err, dict):
        info = err.get("codexErrorInfo") or err.get("code")
        if isinstance(info, dict):
            info = next(iter(info.keys()), "error")
        msg = err.get("message") or err.get("details") or err.get("additionalDetails")
        if msg and info:
            return f"[{info}] {msg}"
        if info:
            return f"[{info}] Turn failed"
        if msg:
            return str(msg)
        return f"Turn error ({status}): {err}"
    if isinstance(err, str) and err.strip():
        return err.strip()
    return f"Turn failed with status '{status}'"


def _classify_rpc_error(exc: JsonRpcError) -> tuple[str, bool]:
    if exc.code in (-32600, -32601, -32602):
        return "config", False
    return _classify_turn_error({"error": exc.data or {}})


def _classify_turn_error(turn: dict[str, Any]) -> tuple[str, bool]:
    """Classify Codex errors into category and whether they are transient (retryable)."""
    error = turn.get("error") or {}
    if not isinstance(error, dict):
        return "turn_failure", False
    info = error.get("codexErrorInfo")
    tag = next(iter(info), "other") if isinstance(info, dict) else info
    details = info.get(tag) if isinstance(info, dict) else None
    status = details.get("httpStatusCode") if isinstance(details, dict) else None
    if tag == "unauthorized" or status in (401, 403):
        return "auth", False
    if tag in ("badRequest", "modelNotFound") or status in (400, 404, 422):
        return "config", False
    if tag == "usageLimitExceeded":
        return "quota", False
    if tag in POLICY_BLOCK_ERRORS:
        return "policy_block", False
    if tag in CONTEXT_LIMIT_ERRORS:
        return "context_limit", False
    transient = tag in {"rateLimitExceeded", "flexUnavailable", "serverOverloaded", "internalServerError"}
    transient = transient or status == 429 or (isinstance(status, int) and 500 <= status < 600)
    return "turn_failure", transient


def _usage_recovery(limits: dict[str, Any], baseline: dict[str, Any], now: float) -> str | None:
    """Determine if rate limit indicators suggest usage can resume."""

    def snapshots(response: dict[str, Any]) -> list[dict[str, Any]]:
        return [response.get("rateLimits") or {}, *(response.get("rateLimitsByLimitId") or {}).values()]

    def has_credits(response: dict[str, Any]) -> bool:
        return any(
            (s.get("credits") or {}).get("hasCredits") or (s.get("credits") or {}).get("unlimited")
            for s in snapshots(response)
        )

    allowed = limits.get("ordinaryUsageAllowed")
    if allowed is True:
        return "usage_allowed"
    if has_credits(limits) and not has_credits(baseline):
        return "credits_added"
    if allowed is None:
        # If ordinaryUsageAllowed is not explicit, check whether exhausted rate limit windows have reset.
        resets = [
            w.get("resetsAt")
            for s in snapshots(limits)
            for w in (s.get("primary"), s.get("secondary"))
            if isinstance(w, dict) and w.get("usedPercent", 0) >= 100
        ]
        if resets and all(isinstance(r, int) and r <= now for r in resets):
            return "windows_reset"
    return None


def _catalog_entry(
    data: list[Any], model: str | None, effort: str | None, owner: str = "", *, requested: bool = True
) -> dict[str, Any] | None:
    """The catalog entry for ``model`` (the default when None); unsupported efforts fail fast.

    An unknown model fails only when the run ``requested`` it: Codex's own
    configuration may select a model the catalog does not describe.
    """
    entries = [m for m in data if isinstance(m, dict)]
    if model:
        entry = next((m for m in entries if m.get("id") == model), None)
    else:
        entry = next((m for m in entries if m.get("isDefault")), None)
    if model and entry is None and requested:
        raise CodexExecutionError("config", f"Model {model!r}{owner} is not in the native model catalog")
    supported = entry.get("supportedReasoningEfforts") if entry else None
    if effort and supported:
        efforts = [e.get("reasoningEffort") for e in supported if isinstance(e, dict)]
        if effort not in efforts:
            raise CodexExecutionError(
                "config",
                f"Reasoning effort {effort!r}{owner} is not supported by {entry.get('id') if entry else model}; "
                f"supported: {', '.join(map(str, efforts))}",
            )
    return entry


def _enforce_role_limits(
    limits: dict[str, int],
    client: CodexClient,
    mapper: EventStreamMapper,
    state: TaskState,
    span_id: str,
    failures: list[str],
    interrupted: set[tuple[str, str]],
    method: str,
    params: dict[str, Any],
) -> None:
    """Interrupt a child's turn once its role's token limit is used up.

    Runs in the client's reader, so the interrupt is sent from a task. The
    limit is recorded once the interrupt is acknowledged; a failed interrupt
    is reported and retried on the thread's next update.
    """
    thread = params.get("thread") or params.get("threadSettings") or {}
    thread_id = params.get("threadId") or thread.get("id")
    role = mapper.thread_metadata.get(thread_id or "", {}).get("agentRole")
    limit = limits.get(role or "")
    turn_id = client.active_turn.get(thread_id or "")
    if not (thread_id and limit and turn_id) or (thread_id, turn_id) in interrupted:
        return
    tokens = mapper.thread_tokens(thread_id)
    if tokens < limit:
        return
    interrupted.add((thread_id, turn_id))
    record = {"thread_id": thread_id, "role": role, "token_limit": limit, "tokens": tokens, "turn_id": turn_id}

    async def interrupt() -> None:
        try:
            await client.interrupt_turn(thread_id, turn_id)
        except Exception as exc:
            failures.append(f"subagent limit interrupt {thread_id}: {type(exc).__name__}: {exc}")
            interrupted.discard((thread_id, turn_id))
            return
        state.metadata.setdefault("codex_subagent_limits", []).append(record)
        compat.emit_event(InfoEvent(source="codex-subagent-limit", data=record, span_id=span_id))

    task = asyncio.create_task(interrupt())
    task.add_done_callback(lambda t: t.cancelled() or t.exception())


# sh -c scripts (positional args: source, destination) moving Codex's session files.
MIRROR_SESSIONS = 'test -d "$1/sessions" || exit 0; rm -rf "$2.new" && mkdir -p "$2.new" && cp -a "$1/sessions" "$2.new/" && rm -rf "$2" && mv "$2.new" "$2"'
RESTORE_SESSIONS = 'test -d "$1/sessions" && mkdir -p "$2" && rm -rf "$2/sessions" && cp -a "$1/sessions" "$2/"'


def _steer_race(exc: JsonRpcError) -> bool:
    """The pinned turn processor rejects steering a turn that already finished."""
    return exc.code == -32600 and (
        exc.message == "no active turn to steer" or exc.message.startswith("expected active turn id `")
    )


def _set_generate_config_fields(config: GenerateConfig) -> list[str]:
    return sorted(config.model_dump(exclude_none=True))


def _turn_info(completed_data: dict[str, Any]) -> dict[str, Any]:
    turn = completed_data.get("turn")
    if isinstance(turn, dict):
        return turn
    return completed_data if isinstance(completed_data, dict) else {}


def _input_items(text: str, current: ChatMessage | None) -> tuple[list[p.UserInput], list[str]]:
    """Build Codex ``turn/start`` input items from text plus the current message's images and audio.

    Text, images and audio are bridged; other content types are reported as
    unsupported rather than silently dropped.
    """
    items: list[p.UserInput] = []
    unsupported: list[str] = []
    if text:
        items.append({"type": "text", "text": text, "text_elements": []})

    blocks = current.content if current is not None and isinstance(current.content, list) else []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "image" or hasattr(block, "image"):
            image = getattr(block, "image", "")
            if isinstance(image, str) and image.startswith(("http://", "https://", "data:image/")):
                items.append({"type": "image", "url": image})
            elif isinstance(image, str) and image:
                items.append({"type": "localImage", "path": image})
            else:
                unsupported.append("image(content)")
        elif isinstance(block, ContentAudio):
            audio = block.audio
            if audio.startswith("data:"):
                items.append({"type": "audio", "url": audio})
            elif audio.startswith(("http://", "https://")):
                # The pinned core omits audio it would have to fetch.
                raise CodexExecutionError("config", "Codex does not fetch audio URLs; pass a file path or data URL")
            else:
                items.append({"type": "localAudio", "path": audio})
        elif block_type not in ("text", None):
            unsupported.append(str(block_type or "unknown"))
    return items, unsupported


async def _history_items(messages: list[ChatMessage]) -> tuple[list[dict[str, Any]], int]:
    """Earlier conversation as raw Responses items for ``thread/inject_items``.

    System messages become developer messages. Reasoning is omitted and
    counted; other content Codex history cannot carry fails fast.
    """
    items: list[dict[str, Any]] = []
    omitted = 0
    for message in messages:
        content = [ContentText(text=message.content)] if isinstance(message.content, str) else message.content
        if isinstance(message, ChatMessageTool):
            unsupported = [block.type for block in content if not isinstance(block, ContentText)]
            output = message.error.message if message.error else message.text
            if not unsupported:
                items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": output})
                continue
        else:
            parts: list[dict[str, Any]] = []
            unsupported = []
            for block in content:
                if isinstance(block, ContentText):
                    if block.text:
                        kind = "output_text" if message.role == "assistant" else "input_text"
                        parts.append({"type": kind, "text": block.text})
                elif isinstance(block, ContentReasoning) and message.role == "assistant":
                    omitted += 1
                elif isinstance(block, ContentImage) and message.role == "user":
                    parts.append(
                        {
                            "type": "input_image",
                            "image_url": await _media_url(block.image, "image/png"),
                            "detail": block.detail,
                        }
                    )
                else:
                    unsupported.append(block.type)
            if parts and not unsupported:
                role = "developer" if message.role == "system" else message.role
                items.append({"type": "message", "role": role, "content": parts})
            if isinstance(message, ChatMessageAssistant):
                items.extend(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.function,
                        "arguments": json.dumps(call.arguments),
                    }
                    for call in message.tool_calls or []
                )
        if unsupported:
            raise CodexExecutionError(
                "config", f"Codex cannot replay {unsupported[0]} content from an earlier {message.role} message"
            )
    return items, omitted


async def _stage_local_media(items: list[p.UserInput], env: Any) -> None:
    """Copy host input images and audio into the sandbox before passing local paths."""
    for item in items:
        if item["type"] != "localImage" and item["type"] != "localAudio":
            continue
        path = Path(item["path"]).expanduser()
        data = await asyncio.to_thread(path.read_bytes)
        target = f"/tmp/codex-input-{shortuuid.uuid()}{path.suffix}"
        await env.write_file(target, data)
        item["path"] = target


# Folder in the sandbox user's home (which checkpoints back up) mirroring Codex's sessions.
CHECKPOINT_DIR = ".codex-inspect-checkpoint"


@dataclass
class _Checkpoint:
    """This run's checkpoint session and the Codex state it carries across a resume."""

    checkpointer: Checkpointer
    session: dict[str, Any]
    """Root thread, goal and counters at the last turn boundary."""
    resume: bool
    conversation: Callable[[], list[ChatMessage]] | None = None
    snapshot: Callable[[], dict[str, Any]] | None = None


@dataclass(frozen=True)
class SessionConfig:
    """Validated options shared by executions of one solver or agent."""

    model: str | None
    transport: Literal["stdio", "websocket"]
    reasoning_effort: str | None
    base_instructions: str | None
    developer_instructions: str | None
    codex_path: str | None
    auth_path: str | None
    ws_port: int
    auto_compact_token_limit: int | None
    cwd: str
    sandbox_name: str | None
    timeout_sec: float
    max_child_agents: int
    authentication: Literal["existing", "chatgpt", "api_key"]
    policy: RunPolicy
    reasoning_summary: Literal["none", "concise", "detailed", "auto"] | None
    retry_refusals: int | None
    attempts: AgentAttempts | None
    continue_with: Callable[[AgentState], Awaitable[str | None]] | None
    skills: Sequence[Skill]
    web_search: WebSearchMode | None
    roles: Sequence[CodexSubagent]
    service_tier: str | None


class CodexSession:
    """Own one native execution, its accounting, control flow and teardown."""

    def __init__(
        self,
        config: SessionConfig,
        state: TaskState,
        tools: list[Tool],
        root_span_id: str | None,
        review: list[ReviewPolicy] | None,
        checkpoint: _Checkpoint | None,
    ) -> None:
        self.config = config
        self.state = state
        self.tools = tools
        self.root_span_id = root_span_id
        self.review = review
        self.checkpoint = checkpoint
        self.resume_thread = (
            checkpoint.session.get("thread_id") if checkpoint is not None and checkpoint.resume else None
        )
        self.mapper = EventStreamMapper(
            model_name=config.model or "codex",
            state=state,
            receipt_reader=config.policy.receipt_reader,
            mcp_tool_aliases=config.policy.mcp_tool_aliases,
        )
        if self.resume_thread and checkpoint is not None:
            accounting = checkpoint.session.get("accounting")
            if accounting is None:
                raise CodexExecutionError("checkpoint", "Checkpoint lacks Codex accounting state; start a new sample")
            self.mapper.restore_accounting(accounting)
        self.role_limits = {role.name: role.token_limit for role in config.roles if role.token_limit}
        self.thread_id = ""
        self.turn_id: str | None = None
        self.stop_reason: StopReason | None = None
        self.unsupported_settings: list[str] = []
        self.normal_finish = False
        self.goal_ok = False
        self.pending_failure: CodexExecutionError | None = None

    async def run(self) -> TaskState:
        await self._provision()
        self._register_handlers()
        self.execution_timeout = asyncio.timeout(self.budget.remaining())
        try:
            async with self.execution_timeout, agent_channel() as self.channel:
                await self._initialize()
                await self._configure_thread()
                await self._execute_turns()
                self.normal_finish = True
                return self.state

        except TimeoutError:
            if not self.execution_timeout.expired():
                raise
            self.client.raise_if_failed()
            if self.pending_failure is not None and not self.client.submit_event.is_set():
                raise self.pending_failure
            compat.emit_event(time_budget_depleted_event(self.config.timeout_sec, span_id=self.root_agent_span_id))
            compat.emit_event(
                InterruptEvent(
                    source="limit",
                    interrupted="generate" if self.turn_id else "between_turns",
                    span_id=self.root_agent_span_id,
                )
            )
            self.normal_finish = True
            return self.state

        finally:
            cleanup_task = asyncio.create_task(self._cleanup())
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    await asyncio.shield(cleanup_task)
                    raise
            if self.normal_finish:
                self.client.raise_if_failed()
                last = (
                    self.client.turn_data(self.thread_id, self.client.submitted_turn_id)
                    if self.thread_id and self.client.submitted_turn_id
                    else self.client.latest_turn_data(self.thread_id)
                    if self.thread_id
                    else None
                )
                if last and _turn_info(last).get("status") == "failed" and self.stop_reason is None:
                    self.state.metadata["codex_partial_submission"] = self.state.output.completion
                    kind, _ = _classify_turn_error(_turn_info(last))
                    raise CodexExecutionError(kind, _format_turn_error(_turn_info(last)))

    async def _provision(self) -> None:
        compat.require_supported_budgets(self.state.cost_limit)
        compat.check_limits()
        # Resolve model name (strip provider prefix, keep None for server default)
        self.resolved_model = self.config.model
        if not self.resolved_model and self.state.model:
            model_str = str(self.state.model)
            self.resolved_model = (
                model_str.split("/", 1)[-1] if "/" in model_str and not model_str.startswith("http") else model_str
            )
        if self.resolved_model in ("default", "codex/default"):
            self.resolved_model = None
        self.mapper.model_name = self.resolved_model or "codex"

        # Inspect's task-level limits also apply outside this native deadline.
        self.budget = BudgetTracker(
            self.config.timeout_sec,
            clock=asyncio.get_running_loop().time,
            checkpoints=self.config.policy.checkpoint_fractions,
        )
        self.env = sandbox(self.config.sandbox_name) if self.config.sandbox_name else sandbox()

        setup_timeout = asyncio.timeout(self.budget.remaining())
        try:
            async with setup_timeout:
                self.container_id = await resolve_service_container(self.env, self.config.sandbox_name or "default")
                logger.info(
                    "Initializing Codex Solver for sample %s in container %s", self.state.sample_id, self.container_id
                )

                in_container_bin, self.codex_home, self.codex_version = await ensure_container_codex(
                    env=self.env,
                    container_id=self.container_id,
                    codex_binary_path=self.config.codex_path,
                    auth_path=self.config.auth_path,
                )

                if self.config.policy.prepare is not None:
                    await self.config.policy.prepare(self.env, self.state, self.budget)
                if self.config.skills:
                    # Codex discovers user skills in $CODEX_HOME/skills; this
                    # run's skills replace any installed by an earlier run.
                    skills_dir = f"{self.codex_home or CODEX_HOME}/skills"
                    await self.env.exec(["rm", "-rf", skills_dir], user="root")
                    await install_skills(self.config.skills, sandbox=self.env, dir=skills_dir)
                self.role_args: list[str] = []
                for role in self.config.roles:
                    config_file = f"{self.codex_home or CODEX_HOME}/agent-roles/{role.name}.toml"
                    await self.env.write_file(config_file, role_config(role))
                    self.role_args.extend(role_overrides(role, config_file))
                self.home_dir = self.codex_home or CODEX_HOME
                self.mirror = ""
                if self.checkpoint is not None:
                    user_home = (await self.env.exec(["sh", "-c", 'printf %s "$HOME"'])).stdout.strip()
                    if not user_home.startswith("/"):
                        raise CodexExecutionError("checkpoint", "The sandbox user's home is unknown")
                    self.mirror = f"{user_home.rstrip('/')}/{CHECKPOINT_DIR}"
                    if self.resume_thread:
                        restored_sessions = await self.env.exec(
                            ["sh", "-c", RESTORE_SESSIONS, "sh", self.mirror, self.home_dir], user="root"
                        )
                        if restored_sessions.returncode != 0:
                            raise CodexExecutionError(
                                "checkpoint", f"No mirrored Codex session to resume in {self.mirror}"
                            )

                # Explicit arguments win over the evaluation's GenerateConfig.
                self.generate_config = compat.active_generate_config()
                self.effective_effort = self.config.reasoning_effort or self.generate_config.reasoning_effort
                self.effective_summary = self.config.reasoning_summary or self.generate_config.reasoning_summary
                self.output_schema = (
                    self.generate_config.response_schema.json_schema.model_dump(mode="json", exclude_none=True)
                    if self.generate_config.response_schema
                    else None
                )
                self.approval_policy: Literal["never", "untrusted"] = (
                    INSPECT_APPROVAL_POLICY if compat.have_tool_approval() else UNATTENDED_POLICY["approval_policy"]
                )
                self.run_policy = {
                    "approval_policy": self.approval_policy,
                    "sandbox_mode": UNATTENDED_POLICY["sandbox_mode"],
                }

                # Transport setup
                ws_url = None
                auth_token = None
                self.daemon_transport: ExecRemoteJsonRpcTransport | None = None
                server_environment = {"CODEX_HOME": self.codex_home or CODEX_HOME}

                if self.config.transport == "websocket":
                    import secrets

                    auth_token = secrets.token_hex(32)
                    token_path = f"/tmp/codex_ws_{str(self.container_id)[:8]}.token"
                    await self.env.write_file(token_path, auth_token.encode("utf-8"))
                    container_ip = await get_container_ip(self.container_id)
                    ws_url = f"ws://{container_ip}:{self.config.ws_port}"
                    daemon_cmd = [
                        in_container_bin,
                        *self._policy_args(),
                        "-c",
                        "features.multi_agent_v2=true",
                        "-c",
                        f"features.multi_agent_v2.max_concurrent_threads_per_session={self.config.max_child_agents}",
                        *(["-c", f'model_reasoning_effort="{self.effective_effort}"'] if self.effective_effort else []),
                        "app-server",
                        "--listen",
                        f"ws://0.0.0.0:{self.config.ws_port}",
                        "--ws-auth",
                        "capability-token",
                        "--ws-token-file",
                        token_path,
                    ]
                    # Launch the WebSocket daemon inside the container via exec_remote.
                    self.daemon_transport = ExecRemoteJsonRpcTransport(
                        self.env, daemon_cmd, cwd=self.config.cwd, user="root", environment=server_environment
                    )

                stdio_cmd = [
                    in_container_bin,
                    *self._policy_args(),
                    "-c",
                    "features.multi_agent_v2=true",
                    "-c",
                    f"features.multi_agent_v2.max_concurrent_threads_per_session={self.config.max_child_agents}",
                    *(["-c", f'model_reasoning_effort="{self.effective_effort}"'] if self.effective_effort else []),
                    *(["-c", f'model="{self.resolved_model}"'] if self.resolved_model else []),
                    "app-server",
                    "--stdio",
                ]

        except TimeoutError:
            if not setup_timeout.expired():
                raise
            raise CodexExecutionError("setup_timeout", "Codex setup exceeded the agent deadline") from None

        self.client = CodexClient(
            transport=self.config.transport,
            env=self.env,
            command=stdio_cmd,
            cwd=self.config.cwd,
            user="root",
            environment=server_environment,
            ws_url=ws_url,
            auth_token=auth_token,
            receipt_reader=self.config.policy.receipt_reader,
        )

    def _register_handlers(self) -> None:
        self.evidence = NativeEvidence(str(self.state.sample_id))
        self.client.evidence = self.evidence
        self.state.metadata["codex_native_evidence"] = self.evidence.metadata
        # An agent run already has the agent span Inspect opened for it.
        self.owns_span = self.root_span_id is None
        self.root_agent_span_id = self.root_span_id or shortuuid.uuid()
        self.mapper.root_agent_span_id = self.root_agent_span_id
        if self.owns_span:
            compat.emit_event(SpanBeginEvent(id=self.root_agent_span_id, type="agent", name="codex"))
        self.mapper.emit_reconstruction_label()

        if self.approval_policy == INSPECT_APPROVAL_POLICY:
            self.client.approval_decider = self._decide_approval

        self.client.register_notification_handler("item/completed", self.mapper.on_item_completed)
        self.client.register_notification_handler("item/started", self.mapper.on_item_started)
        self.client.register_notification_handler("thread/started", self.mapper.on_thread_metadata)
        self.client.register_notification_handler("thread/metadata/read", self.mapper.on_thread_metadata)
        self.client.register_notification_handler("thread/settings/updated", self.mapper.on_thread_metadata)
        self.client.register_notification_handler("thread/closed", self.mapper.on_thread_closed)
        self.client.register_notification_handler("rawResponseItem/completed", self.mapper.on_raw_item)
        self.client.register_notification_handler("rawResponse/completed", self.mapper.on_response_completed)
        self.client.register_notification_handler("thread/tokenUsage/updated", self.mapper.on_token_usage)
        if self.role_limits:
            enforce_limits = partial(
                _enforce_role_limits,
                self.role_limits,
                self.client,
                self.mapper,
                self.state,
                self.root_agent_span_id,
                self.evidence.metadata["failures"],
                set(),
            )
            # Metadata is fetched asynchronously and can arrive after usage.
            # Recheck on role discovery and on turns reusing an exhausted child.
            for method in (
                "thread/tokenUsage/updated",
                "thread/started",
                "thread/metadata/read",
                "thread/settings/updated",
                "turn/started",
            ):
                self.client.register_notification_handler(method, enforce_limits)
        self.client.register_notification_handler("turn/completed", self.mapper.on_turn_completed)
        self.client.register_notification_handler("turn/started", self.mapper.on_turn_started)
        self.client.register_notification_handler("turn/plan/updated", self.mapper.on_plan_updated)
        self.client.register_notification_handler("thread/goal/updated", self.mapper.on_goal_updated)
        self.client.register_notification_handler("turn/diff/updated", self.mapper.on_diff_updated)
        self.client.register_notification_handler("item/autoApprovalReview/started", self.mapper.on_review_event)
        self.client.register_notification_handler("item/autoApprovalReview/completed", self.mapper.on_review_event)
        self.client.register_notification_handler("model/rerouted", self.mapper.on_model_rerouted)
        self.native_warnings: list[dict[str, Any]] = []

        for method in NATIVE_WARNINGS:
            self.client.register_notification_handler(method, self._record_warning)
        # Record reconnecting stream errors (willRetry=True) as warnings; terminal errors end the turn.
        self.client.register_notification_handler(
            "error", lambda method, params: self._record_warning(method, params) if params.get("willRetry") else None)
        self.client.register_notification_handler("mcpServer/startupStatus/updated", self._mcp_startup)
        self.client.register_approval_handler(self.mapper.on_approval_requested)

        self.auth_metadata: dict[str, Any] = {"observed": False, "expected": self.config.authentication}
        self.state.metadata["codex_authentication"] = self.auth_metadata

        self.client.register_notification_handler("account/updated", self._account_updated)
        self.client.register_notification_handler("account/metadata/read", self._account_observed)
        self.client.register_notification_handler("account/rateLimits/updated", self._rate_limits_updated)
        self._publish_account()

    async def _initialize(self) -> None:
        if self.daemon_transport is not None:
            await self.daemon_transport.start()
        await self.client.start()

        # Initialize handshake (experimental capability opt-in)
        await self.client.rpc.initialize(
            {
                "clientInfo": {
                    "name": "inspect-ai",
                    "title": "Inspect AI Codex Solver",
                    "version": "2.0.0",
                },
                "capabilities": {"experimentalApi": True},
            }
        )
        await self.client.send_notification("initialized")

        try:
            account = await self.client.rpc.account_read({"refreshToken": False})
        except JsonRpcError as exc:
            if exc.code != -32601:
                raise
            if self.config.authentication != "existing":
                raise CodexExecutionError("auth", "Native account verification is unavailable") from exc
        else:
            self._account_observed("account/read", account)

    async def _configure_thread(self) -> None:
        # Read the server's model catalog (checked once the provider is known)
        actual_model = self.resolved_model or "codex"
        catalog: list[Any] | None = None
        try:
            data: list[Any] = []
            cursor: str | None = None
            while True:
                # Hidden models can still be selected, so they are listed too.
                page = await self.client.rpc.model_list(
                    {"limit": 100, "includeHidden": True, **({"cursor": cursor} if cursor else {})},
                    timeout=15.0,
                )
                data.extend(page.get("data") or [])
                cursor = page.get("nextCursor")
                if not cursor:
                    break
            catalog = data
        except JsonRpcError as exc:
            if exc.code != -32601:
                raise
            self.unsupported_settings.append("model_catalog_unsupported")

        # Bridge configured Inspect tools as Codex dynamic tools
        dynamic_tools: list[p.DynamicToolSpec] = []
        for tool in self.tools:
            name, description, parameters = compat.tool_schema(tool)
            if name in self.config.policy.excluded_tools:
                continue
            dynamic_tools.append(
                {
                    "type": "function",
                    "name": name,
                    "description": description or "",
                    "inputSchema": parameters,
                }
            )
            self.client.register_tool_handler(name, inspect_tool_handler(tool, self.mapper, self.review))

        # Start thread with configured instructions and sandbox policy
        instructions = self.config.developer_instructions
        thread_params: p.ThreadStartParams = {
            "approvalPolicy": self.approval_policy,
            "sandbox": UNATTENDED_POLICY["sandbox_mode"],
            "cwd": self.config.cwd,
            "dynamicTools": dynamic_tools,
            "experimentalRawEvents": True,
        }
        if instructions:
            thread_params["developerInstructions"] = instructions
        if self.config.base_instructions:
            thread_params["baseInstructions"] = self.config.base_instructions
        if self.resolved_model:
            thread_params["model"] = self.resolved_model

        thread_res: dict[str, Any]
        if self.resume_thread:
            # The restored session keeps its instructions and dynamic tools.
            resume_params: p.ThreadResumeParams = {
                "threadId": self.resume_thread,
                "approvalPolicy": self.approval_policy,
                "sandbox": UNATTENDED_POLICY["sandbox_mode"],
                "cwd": self.config.cwd,
            }
            if self.resolved_model:
                resume_params["model"] = self.resolved_model
            thread_res = dict(await self.client.rpc.thread_resume(resume_params))
            # Only thread/start enables raw events: resumed responses carry no
            # per-response usage records or delegated child input.
            self.unsupported_settings.append("raw_events_unavailable_after_resume")
        else:
            thread_res = dict(await self.client.rpc.thread_start(thread_params))
        self.thread_id = thread_res.get("thread", {}).get("id")
        if not self.thread_id:
            raise CodexExecutionError("config", f"Thread start returned no thread id: {thread_res}")
        # Configuration warnings arrive with the initialize response, so all
        # have been seen by now. Codex would run with settings it ignored.
        ignored = [w["summary"] for w in self.native_warnings if w["type"] == "configWarning"]
        if ignored:
            raise CodexExecutionError("config", f"Codex configuration warning: {ignored[0]}")
        if self.config.skills:
            # Codex skips a skill it cannot load. Skills the workspace itself
            # defines (.codex/skills, .agents/skills) are not the run's.
            listed = await self.client.rpc.skills_list({"cwds": [self.config.cwd]})
            installed = tuple(f"{self.home_dir}/skills/{skill.name}/" for skill in self.config.skills)
            errors = [error for entry in listed.get("data") or [] for error in entry.get("errors") or []
                      if str(error.get("path", "")).startswith(installed)]
            if errors:
                raise CodexExecutionError("config", f"Codex could not load skill {errors[0]['path']}: "
                                                    f"{errors[0]['message']}")
        # The catalog describes Codex's built-in OpenAI provider; models
        # served by a configured provider are not listed in it.
        model_provider = thread_res.get("modelProvider")
        actual_model = thread_res.get("model") or self.resolved_model
        if catalog is not None and model_provider in (None, "openai"):
            requested = self.resolved_model is not None
            entry = _catalog_entry(catalog, actual_model, self.effective_effort, requested=requested)
            if not actual_model and entry:
                actual_model = entry.get("id") or actual_model
            for role in self.config.roles:
                _catalog_entry(
                    catalog, role.model or actual_model, role.reasoning_effort, f" for subagent {role.name!r}",
                    requested=requested or role.model is not None,
                )
            tiers = [t.get("id") for t in (entry or {}).get("serviceTiers") or [] if isinstance(t, dict)]
            if self.config.service_tier and entry and self.config.service_tier not in tiers:
                raise CodexExecutionError(
                    "config",
                    f"Service tier {self.config.service_tier!r} is not offered for {entry.get('id')}; "
                    f"offered: {', '.join(map(str, tiers)) or 'none'}",
                )
        elif catalog is not None:
            self.unsupported_settings.append(f"model_catalog_not_applicable:{model_provider}")
        self.mapper.root_thread_id = self.thread_id
        self.client.set_root_thread(self.thread_id)
        if self.config.policy.thread_started is not None:
            await self.config.policy.thread_started(self.env, self.thread_id)

        actual_model = actual_model or "codex"
        self.mapper.model_name = actual_model

        # Set goal and prepare first-turn input
        self.goal_objective, first_input_text = self.config.policy.input_builder(self.state)
        self.goal_ok = False

        # Earlier messages become native thread history; the current
        # user message (with its images and audio) is the first turn.
        history, current = split_history(self.state.messages)
        self.input_items: list[p.UserInput]
        self.input_items, input_unsupported = _input_items(first_input_text, current)
        if current is not None:
            original = current.content
            attachments = [block for block in original if block.type != "text"] if isinstance(original, list) else []
            current.content = [ContentText(text=first_input_text), *attachments] if attachments else first_input_text
        else:
            first_input_text = first_input_text or self.config.policy.continuation
            self.input_items = [{"type": "text", "text": first_input_text, "text_elements": []}]
            self.mapper.append_user_message(first_input_text)
        replayed: dict[str, int] = {}
        if history and not self.resume_thread:
            # A resumed session already holds its history.
            history_items, reasoning_omitted = await _history_items(history)
            await self.client.rpc.thread_inject_items({"threadId": self.thread_id, "items": history_items})
            replayed = {"history_items": len(history_items), "history_reasoning_omitted": reasoning_omitted}

        if any(item["type"] in ("audio", "localAudio") for item in self.input_items):
            # Codex replaces audio with a notice for models its catalog
            # does not list as accepting audio, whatever the provider.
            listed = next((m for m in catalog or [] if isinstance(m, dict) and m.get("id") == actual_model), {})
            if "audio" not in (listed.get("inputModalities") or []):
                raise CodexExecutionError(
                    "config", f"Model {actual_model!r} does not accept audio input in the native model catalog"
                )
        await _stage_local_media(self.input_items, self.env)
        self.unsupported_settings.extend(input_unsupported)
        self.unsupported_settings.extend(
            f"generate_config_not_applied:{field}"
            for field in _set_generate_config_fields(self.generate_config)
            if field not in APPLIED_GENERATE_CONFIG
        )

        self.state.metadata["codex_run"] = {
            "model": actual_model,
            "effort": self.effective_effort,
            "reasoning_summary": self.effective_summary,
            "output_schema": self.output_schema is not None,
            "skills": [skill.name for skill in self.config.skills],
            "subagents": [role.name for role in self.config.roles],
            "web_search": self.config.web_search,
            "service_tier": self.config.service_tier,
            "policy": self.run_policy,
            "codex_version": self.codex_version,
            "inspect_version": version("inspect-ai"),
            "adapter_version": version("inspect-codex-app-server"),
            "experimental_api": True,
            "experimental_features": ["dynamic_tools", "raw_events"],
            "completion": self.config.policy.completion,
            "max_child_agents": self.config.max_child_agents,
            "transport": self.config.transport,
            "thread_id": self.thread_id,
            "goal_initialized": self.goal_ok,
            "unsupported_budgets": ["cost", "model_generation_turns"],
            **replayed,
        }

    async def _execute_turns(self) -> None:
        # Autonomous execution loop
        self.consecutive_failures = 0
        self.first_turn = True
        self.turn_id = None
        self.last_turn_failed = False
        self.pending_checkpoint: list[p.UserInput] | None = None
        self.seen_turns: set[str] = set()
        self.attempt_count = 0
        self.refusal_retries = 0
        if self.resume_thread and self.checkpoint is not None:
            self.goal_ok = bool(self.checkpoint.session.get("goal_ok"))
            self.attempt_count = int(self.checkpoint.session.get("attempt_count", 0))
            self.refusal_retries = int(self.checkpoint.session.get("refusal_retries", 0))

        self.best_effort_nudged = False
        self.last_activity: tuple[int, int] = (-1, -1)
        self.last_activity_at = time.monotonic()

        if self.checkpoint is not None:
            self.checkpoint.conversation = lambda: self._conversation(self.state.output.completion).messages
            self.checkpoint.snapshot = self._checkpoint_state
        if self.resume_thread and self.checkpoint is not None:
            # The interrupted attempt's turn is gone: continue the restored session.
            self.first_turn = False
            report = self.checkpoint.checkpointer.restored
            notice = report.message if report and report.message and not report.transparent else None
            if self.goal_ok:
                await self.client.set_goal(self.thread_id, objective=self.goal_objective, status="active")
            text = "\n\n".join(filter(None, [notice, self.config.policy.continuation]))
            self.turn_id = await self._start_turn([{"type": "text", "text": text, "text_elements": []}])
            self.state.metadata["codex_run"]["resumed_thread"] = self.thread_id

        while True:
            self.client.raise_if_failed()
            compat.check_limits()
            compat.require_supported_budgets(self.state.cost_limit)
            remaining_time = self.budget.remaining()
            if remaining_time <= 0.0:
                logger.info("Time budget depleted (%.1fs remaining); stopping execution", remaining_time)
                compat.emit_event(time_budget_depleted_event(self.config.timeout_sec, span_id=self.root_agent_span_id))
                compat.emit_event(
                    InterruptEvent(source="limit", interrupted="between_turns", span_id=self.root_agent_span_id)
                )
                break

            if self.client.submit_event.is_set():
                if await self._accept_submission():
                    logger.info("Submit event set; stopping execution loop")
                    break
                continue

            self.client.raise_if_failed()

            if self.turn_id is None:
                if self.first_turn:
                    self.turn_id = await self._start_turn(self.input_items)
                    self.first_turn = False
                    try:
                        initial_turn = self.client.turn_data(self.thread_id, self.turn_id)
                        if (
                            self.config.policy.completion != "turn"
                            and self.config.continue_with is None
                            and not self.client.submit_event.is_set()
                            and (initial_turn is None or _turn_info(initial_turn).get("status") != "failed")
                        ):
                            await self.client.set_goal(self.thread_id, objective=self.goal_objective, status="active")
                            self.goal_ok = True
                    except JsonRpcError as exc:
                        if exc.code != -32601:
                            raise
                        self.unsupported_settings.append("thread_goal_unsupported")
                    self.state.metadata["codex_run"]["goal_initialized"] = self.goal_ok
                else:
                    if self.last_turn_failed and self.pending_failure is not None:
                        self.turn_id = await self._start_turn(
                            [{"type": "text", "text": self.config.policy.continuation, "text_elements": []}]
                        )
                        continue
                    status = await self._goal_status()
                    if status is not None and status != "active":
                        if await self._finish_or_nudge(status, remaining_time):
                            break
                        continue
                    due = self.budget.due_checkpoints()
                    if due and self.config.policy.checkpoint_message is not None:
                        self.pending_checkpoint = [
                            {
                                "type": "text",
                                "text": self.config.policy.checkpoint_message(
                                    due[-1], self.budget.elapsed(), self.config.timeout_sec
                                ),
                                "text_elements": [],
                            }
                        ]
                    if self.pending_checkpoint:
                        # Deliver scheduled checkpoint steering reminder as its own turn.
                        self.turn_id = await self._start_turn(self.pending_checkpoint)
                        self.pending_checkpoint = None
                    else:
                        # After a failed turn, retry immediately: a failed turn
                        # is not goal completion, so there is nothing to wait for.
                        try:
                            with self.channel.turn_scope():
                                server_turn = (
                                    None
                                    if self.last_turn_failed
                                    else await self.client.wait_for_next_turn(
                                        self.thread_id,
                                        self.seen_turns,
                                        min(CONTINUATION_GRACE_SEC, max(0.0, self.budget.remaining())),
                                    )
                                )
                        except AgentInterrupted:
                            await self._operator_interrupt()
                            continue
                        if server_turn is not None:
                            logger.info("Codex continued goal server-side (turn %s)", server_turn)
                            self.turn_id = server_turn
                            self.seen_turns.add(server_turn)
                        else:
                            # Re-read after the grace period; the goal may have
                            # stopped or submitted while we were waiting.
                            self.client.raise_if_failed()
                            compat.check_limits()
                            if self.client.submit_event.is_set():
                                continue
                            status = await self._goal_status()
                            if status is not None and status != "active":
                                if await self._finish_or_nudge(status, remaining_time):
                                    break
                                continue
                            # The status RPC itself can overlap a server start.
                            server_turn = await self.client.wait_for_next_turn(self.thread_id, self.seen_turns, 0)
                            if server_turn:
                                self.seen_turns.add(server_turn)
                                self.turn_id = server_turn
                                continue
                            if self.goal_ok and not self.last_turn_failed:
                                # Wait for Codex to continue the active goal; detect stalls if no events arrive.
                                if self._poll_stalled():
                                    raise CodexExecutionError(
                                        "stall",
                                        f"No Codex activity for {STALL_TIMEOUT_SEC:.0f}s between turns",
                                    )
                                continue
                            self.turn_id = await self._start_turn(
                                [{"type": "text", "text": self.config.policy.continuation, "text_elements": []}]
                            )

            logger.info("Codex turn %s in progress (remaining=%.1fs)", self.turn_id, remaining_time)
            if self.client.submit_event.is_set():
                continue
            due = self.budget.due_checkpoints()
            if due and self.config.policy.checkpoint_message is not None:
                self.pending_checkpoint = [
                    {
                        "type": "text",
                        "text": self.config.policy.checkpoint_message(
                            due[-1], self.budget.elapsed(), self.config.timeout_sec
                        ),
                        "text_elements": [],
                    }
                ]
            if self.pending_checkpoint:
                reminder = self.pending_checkpoint
                try:
                    await self.client.rpc.turn_steer(
                        {"threadId": self.thread_id, "expectedTurnId": self.turn_id, "input": reminder},
                        timeout=min(10.0, max(0.1, self.budget.remaining())),
                    )
                    self.pending_checkpoint = None
                except JsonRpcError as exc:
                    # If the turn finished before steering was applied, keep the message for the next turn.
                    if not _steer_race(exc):
                        raise
                    self.pending_checkpoint = reminder
            try:
                with self.channel.turn_scope():
                    completed_data = await self.client.wait_for_turn(
                        thread_id=self.thread_id,
                        turn_id=self.turn_id,
                        timeout=max(0.01, min(self.budget.remaining(), self.budget.until_next_checkpoint())),
                    )
            except AgentInterrupted:
                await self._operator_interrupt()
                continue
            except (TimeoutError, asyncio.TimeoutError):
                if self.budget.remaining() > 0.0:
                    continue  # Wake for steering; the native turn stays active.
                logger.warning("Codex turn timed out after %.1fs; interrupting", remaining_time)
                compat.emit_event(
                    SampleLimitEvent(
                        type="time",
                        message=f"Turn execution timed out after {remaining_time:.1f}s",
                        limit=float(self.config.timeout_sec),
                        span_id=self.root_agent_span_id,
                    )
                )
                compat.emit_event(
                    InterruptEvent(source="limit", interrupted="generate", span_id=self.root_agent_span_id)
                )
                try:
                    await self.client.interrupt_turn(self.thread_id, self.turn_id)
                except Exception as exc:
                    self.evidence.metadata["failures"].append(f"deadline interrupt: {type(exc).__name__}: {exc}")
                break

            self.turn_id = None
            self.client.raise_if_failed()
            compat.check_limits()
            await self._tick()

            # An accepted submission during the turn ends the sample immediately.
            if self.client.submit_event.is_set():
                if await self._accept_submission():
                    compat.emit_event(
                        InterruptEvent(source="system", interrupted="generate", span_id=self.root_agent_span_id)
                    )
                    break
                continue

            turn_info = _turn_info(completed_data)
            turn_status = turn_info.get("status")
            if turn_status == "interrupted":
                self.state.metadata["codex_turn_status"] = turn_status
                break
            turn_error = turn_info.get("error") or completed_data.get("error")
            is_failed = (turn_status == "failed") or (bool(turn_error) and turn_status != "completed")

            # A completed root submit remains available even during a failed turn.
            flag = accepted_answer(self.client, self.mapper.submitted_answers)
            if flag:
                if await self._accept_submission():
                    logger.info("Submission found after turn: %s", flag)
                    break
                continue

            if is_failed:
                self.consecutive_failures += 1
                self.last_turn_failed = True
                err_desc = _format_turn_error(turn_info)
                kind, retryable = _classify_turn_error(turn_info)
                self.pending_failure = CodexExecutionError(kind, err_desc)
                self.state.metadata.setdefault("codex_turn_failures", []).append(turn_info)
                logger.warning(
                    "Codex turn failed (%d/%d): %s",
                    self.consecutive_failures,
                    self.config.policy.max_turn_failures,
                    err_desc,
                )
                if kind in NORMAL_STOPS:
                    if kind == "policy_block" and self.refusal_retries < (self.config.retry_refusals or 0):
                        self.refusal_retries += 1
                        self.consecutive_failures = 0
                        self.last_turn_failed = False
                        self.pending_failure = None
                        self.turn_id = await self._start_turn(
                            [{"type": "text", "text": self.config.policy.continuation, "text_elements": []}]
                        )
                        continue
                    # A provider refusal or exhausted context is an observed
                    # outcome, not an execution error: finish like react().
                    self.stop_reason = NORMAL_STOPS[kind]
                    self.state.metadata["codex_stop"] = {
                        "stop_reason": self.stop_reason,
                        "error": err_desc,
                        "refusal_retries": self.refusal_retries,
                    }
                    if kind == "policy_block":
                        self.state.metadata["codex_policy_block"] = err_desc
                    self.pending_failure = None
                    break
                self.mapper.append_assistant_message(
                    f"[Turn Failed]: {err_desc} (retry {self.consecutive_failures}/{self.config.policy.max_turn_failures})"
                )
                if kind == "quota" and await self._quota_hold(err_desc):
                    self.consecutive_failures = 0
                    self.last_turn_failed = False
                    self.pending_failure = None
                    if self.config.policy.budget_resumed is not None:
                        await self.config.policy.budget_resumed(self.env, self.budget)
                    if self.goal_ok:
                        # Reactivate the goal now that quota has recovered.
                        await self.client.set_goal(self.thread_id, status="active")
                    else:
                        self.turn_id = await self._start_turn(
                            [{"type": "text", "text": self.config.policy.continuation, "text_elements": []}]
                        )
                    continue
                if kind == "quota":
                    held = self.budget.held_sec
                    raise CodexExecutionError(
                        kind,
                        f"Execution failed ({err_desc})."
                        + (f" Usage did not recover within the {held:.0f}s quota hold." if held else ""),
                    )
                if not retryable:
                    raise CodexExecutionError(kind, f"Execution failed ({err_desc}).")
                if self.consecutive_failures > self.config.policy.max_turn_failures:
                    raise CodexExecutionError(
                        kind,
                        f"Execution failed ({err_desc}). Aborted after {self.config.policy.max_turn_failures} recovery attempts.",
                    )
            else:
                self.consecutive_failures = 0
                self.last_turn_failed = False
                self.pending_failure = None
                if await self._deliver_posted() or await self._continue_after_turn():
                    continue
                if self.config.policy.completion == "turn" or self.config.continue_with is not None:
                    break
                logger.info("Codex turn completed; continuing toward goal")

        # The final snapshot belongs to teardown, after native execution
        # stops and all available tail evidence has been consumed.
        if self.pending_failure is not None and not self.client.submit_event.is_set():
            raise self.pending_failure
        self.client.raise_if_failed()
        compat.check_limits()

    def _policy_args(self) -> list[str]:
        args: list[str] = []
        for key, value in self.run_policy.items():
            args.extend(["-c", f'{key}="{value}"'])
        if self.effective_summary:
            args.extend(["-c", f'model_reasoning_summary="{self.effective_summary}"'])
        if self.config.web_search is not None:
            args.extend(["-c", f'web_search="{self.config.web_search}"'])
        if self.config.auto_compact_token_limit is not None:
            args.extend(
                [
                    "-c",
                    f"model_auto_compact_token_limit={self.config.auto_compact_token_limit}",
                ]
            )
        args.extend(self.role_args)
        for override in self.config.policy.config_overrides:
            args.extend(["-c", override])
        return args

    async def _decide_approval(self, method: str, params: dict[str, Any]) -> ApprovalDecision:
        call = self.mapper.approval_call(method, params)
        with compat.native_thread_span(self.mapper.thread_span(params.get("threadId") or "")):
            approved, decision = await compat.apply_tool_approval(
                call,
                self.mapper.thread_messages(params.get("threadId")),
            )
        if approved and not (decision and decision.modified):
            return "accept"
        # Codex cannot run a modified native action; the approver's
        # ApprovalEvent records the modification that was declined.
        return "terminate" if decision and decision.decision == "terminate" else "decline"

    def _record_warning(self, method: str, params: dict[str, Any]) -> None:
        entry = {"type": method, **{key: value for key, value in params.items() if value is not None}}
        self.native_warnings.append(entry)
        self.state.metadata.setdefault("codex_warnings", []).append(entry)
        compat.emit_event(InfoEvent(source="codex-warning", data=entry, span_id=self.root_agent_span_id))

    def _mcp_startup(self, method: str, params: dict[str, Any]) -> None:
        if params.get("status") != "failed":
            return
        thread_id = params.get("threadId")
        child = bool(thread_id and self.thread_id and thread_id != self.thread_id)
        if child or self.normal_finish or self.mapper.tearing_down:
            # A child thread or completed run retains existing tools; record as a warning.
            self._record_warning(method, params)
            return
        # An MCP server failure on the root thread during setup leaves required tools missing.
        raise CodexExecutionError("config", params.get("error") or f"MCP server {params.get('name')!r} failed to start")

    def _publish_account(self) -> None:
        # Snapshot metadata so transcript history reflects account state at observation time.
        summary = dict(self.auth_metadata)
        self.state.store.set("codex_authentication", summary)
        compat.emit_event(
            InfoEvent(
                source="codex-authentication",
                data=summary,
                span_id=self.root_agent_span_id,
            )
        )

    def _account_observed(self, method: str, response: Mapping[str, Any]) -> None:
        self.auth_metadata.update(account_summary(response))
        self._publish_account()
        try:
            require_authentication(self.config.authentication, self.auth_metadata)
        except ValueError as exc:
            raise CodexExecutionError("auth", str(exc)) from exc

    def _account_updated(self, method: str, params: dict[str, Any]) -> None:
        previously_observed = self.auth_metadata["observed"]
        mode = params.get("authMode")
        self.auth_metadata.update(
            type="apiKey" if mode == "apikey" else mode, plan_type=params.get("planType"), email=None, observed=False
        )
        self._publish_account()
        # Ensure updated account information matches the expected authentication mode.
        if previously_observed:
            try:
                require_authentication(self.config.authentication, self.auth_metadata)
            except ValueError as exc:
                raise CodexExecutionError("auth", str(exc)) from exc

    def _rate_limits_updated(self, method: str, params: dict[str, Any]) -> None:
        self.state.metadata["codex_rate_limits"] = params

    async def _start_turn(self, items: list[p.UserInput]) -> str:
        """Start a turn; transient failures retry with backoff, terminal ones abort."""
        params: p.TurnStartParams = {
            "threadId": self.thread_id,
            "input": items,
            "approvalPolicy": self.approval_policy,
            "cwd": self.config.cwd,
        }
        if self.effective_effort:
            params["effort"] = self.effective_effort
        if self.config.service_tier:
            params["serviceTier"] = self.config.service_tier
        if self.output_schema is not None:
            params["outputSchema"] = self.output_schema
        if self.resolved_model:
            params["model"] = self.resolved_model
        while True:
            try:
                res = await self.client.rpc.turn_start(params, timeout=30.0)
            except JsonRpcError as e:
                kind, retryable = _classify_rpc_error(e)
                if not retryable:
                    raise CodexExecutionError(kind, f"turn/start failed: {e.message}")
                self.consecutive_failures += 1
                if self.consecutive_failures > self.config.policy.max_turn_failures:
                    raise CodexExecutionError("turn_failure", f"turn/start failed after retries: {e.message}")
                backoff = min(2.0 ** (self.consecutive_failures - 1), TURN_START_BACKOFF_CAP_SEC)
                await asyncio.sleep(min(backoff, max(0.0, self.budget.remaining())))
                self.client.raise_if_failed()
                continue
            started_id = res.get("turn", {}).get("id") if isinstance(res, dict) else None
            if not started_id:
                raise CodexExecutionError("protocol", "turn/start returned no turn id")
            self.seen_turns.add(started_id)
            return started_id

    async def _goal_status(self) -> str | None:
        try:
            goal = await self.client.get_goal(self.thread_id)
            return (goal.get("goal") or goal).get("status")
        except JsonRpcError as exc:
            if exc.code == -32601:
                return None
            raise

    async def _quota_hold(self, detail: str) -> bool:
        """Wait within policy.quota_hold_sec for the account's usage to recover.

        Held time is credited to Inspect working limits and removed
        from the agent budget. Returns False when the hold is
        disabled, exhausted, or cannot observe recovery.
        """
        if self.budget.held_sec >= self.config.policy.quota_hold_sec or self.budget.remaining() <= 0.0:
            return False
        loop = asyncio.get_running_loop()
        hold: dict[str, Any] = {"error": detail, "held_sec": 0.0, "outcome": "exhausted", "probe_errors": []}
        self.state.metadata.setdefault("codex_quota_holds", []).append(hold)
        compat.emit_event(
            InfoEvent(source="codex-quota-hold", data={"phase": "started", **hold}, span_id=self.root_agent_span_id)
        )
        last = loop.time()

        def credit() -> None:
            nonlocal last
            now = loop.time()
            compat.record_waiting_time(now - last)
            self.budget.hold(now - last)
            hold["held_sec"] += now - last
            last = now
            self._reschedule_deadline()

        async def read_limits() -> dict[str, Any] | None:
            try:
                return dict(await self.client.rpc.account_rateLimits_read({}, timeout=15.0))
            except JsonRpcError as exc:
                if exc.code == -32601:
                    raise
                hold["probe_errors"].append(f"{exc.code}: {exc.message}")
                return None

        try:
            try:
                baseline = await read_limits() or {}
            except JsonRpcError:
                hold["outcome"] = "unobservable"
                return False
            hold["rate_limits"] = baseline
            next_probe = loop.time() + self.config.policy.quota_poll_sec
            while self.budget.held_sec < self.config.policy.quota_hold_sec:
                await asyncio.sleep(
                    max(
                        0.0,
                        min(
                            QUOTA_CREDIT_SEC,
                            next_probe - loop.time(),
                            self.config.policy.quota_hold_sec - self.budget.held_sec,
                        ),
                    )
                )
                credit()
                self.client.raise_if_failed()
                if loop.time() < next_probe:
                    continue
                limits = await read_limits()
                next_probe = loop.time() + self.config.policy.quota_poll_sec
                if limits is not None:
                    hold["rate_limits"] = limits
                    outcome = _usage_recovery(limits, baseline, time.time())
                    if outcome:
                        hold["outcome"] = outcome
                        return True
            return False
        finally:
            credit()
            compat.emit_event(
                InfoEvent(source="codex-quota-hold", data={"phase": "ended", **hold}, span_id=self.root_agent_span_id)
            )

    def _poll_stalled(self) -> bool:
        """True when between-turns polling saw no new activity for STALL_TIMEOUT_SEC."""
        current = (len(self.mapper.raw_events), self.client.turn_started_seq(self.thread_id))
        if current != self.last_activity:
            self.last_activity = current
            self.last_activity_at = time.monotonic()
            return False
        return (time.monotonic() - self.last_activity_at) >= STALL_TIMEOUT_SEC

    async def _finish_or_nudge(self, status: str, remaining: float) -> bool:
        """Decide how to end a sample whose goal is no longer active.

        Returns True when the sample should finish. Otherwise one
        best-effort submission turn is started (latched) so the sample
        never ends without an answer while budget remains.
        """
        if self.pending_failure is not None:
            # Propagate fatal turn failures even if the goal completed.
            raise self.pending_failure
        if (
            status in GOAL_SUCCESS_STATUSES
            or self.best_effort_nudged
            or remaining <= 0.0
            or self.config.policy.terminal_nudge is None
        ):
            self.state.metadata["codex_goal_status"] = status
            logger.info("Codex goal reports status %r; finishing", status)
            return True
        self.best_effort_nudged = True
        logger.info("Codex goal reports %r; requesting a best-effort submission first", status)
        self.turn_id = await self._start_turn(
            [{"type": "text", "text": self.config.policy.terminal_nudge, "text_elements": []}]
        )
        return False

    def _conversation(self, answer: str) -> AgentState:
        """The conversation mapped so far, with ``answer`` as its output."""
        known = {m.id for m in self.state.messages if m.id}
        snapshot = AgentState(
            messages=[*self.state.messages, *(m for m in self.mapper.messages if not m.id or m.id not in known)]
        )
        snapshot.output = ModelOutput.from_content(self.mapper.model_name, answer)
        return snapshot

    async def _accept_submission(self) -> bool:
        """True when the submission ends the run.

        Otherwise the scored answer was incorrect with attempts left:
        the submission is withdrawn and ``incorrect_message`` steers the
        active turn (or starts one), as react() replies to the model.
        """
        if self.config.attempts is None:
            return True
        self.attempt_count += 1
        if self.attempt_count >= self.config.attempts.attempts:
            return True
        # Score the conversation so far with the answer as output, as react() does.
        attempt = self._conversation(accepted_answer(self.client, self.mapper.submitted_answers))
        scores = await score(attempt)
        if self.config.attempts.score_value(scores[0].value) == 1.0:
            return True
        message = self.config.attempts.incorrect_message
        if not isinstance(message, str):
            message = await message(attempt, scores)
        self.client.clear_submission()
        self.mapper.submitted_answers.clear()
        if self.goal_ok:
            # Submitting may have completed the native goal; the run goes on.
            await self.client.set_goal(self.thread_id, objective=self.goal_objective, status="active")
        items: list[p.UserInput] = [{"type": "text", "text": message, "text_elements": []}]
        active = self.client.active_turn.get(self.thread_id)
        if active:
            try:
                await self.client.rpc.turn_steer(
                    {"threadId": self.thread_id, "expectedTurnId": active, "input": items},
                    timeout=min(10.0, max(0.1, self.budget.remaining())),
                )
                self.turn_id = active
                return False
            except JsonRpcError as exc:
                if not _steer_race(exc):
                    raise
        self.turn_id = await self._start_turn(items)
        return False

    def _reschedule_deadline(self) -> None:
        """Move the agent deadline to match the budget after credited or held time."""
        if not self.execution_timeout.expired():
            self.execution_timeout.reschedule(asyncio.get_running_loop().time() + self.budget.remaining())

    def _credit_wait(self, seconds: float) -> None:
        """Credit time spent waiting on an operator to Inspect limits and the agent budget."""
        compat.record_waiting_time(seconds)
        self.budget.credit(seconds)

    def _record_intervention(self, kind: str, messages: list[ChatMessageUser], waited: float = 0.0) -> str:
        text = "\n\n".join(m.text for m in messages)
        self.state.metadata.setdefault("codex_operator_interventions", []).append(
            {"type": kind, "turn_id": self.turn_id, "message": text, "waited_sec": round(waited, 3)}
        )
        return text

    async def _deliver_posted(self) -> bool:
        """Start a turn with operator messages posted while Codex worked; True if one started."""
        # Codex already has its task input, so never block for an initial message.
        posted = await self.channel.before_turn([ChatMessageUser(content=self.goal_objective)])
        if not posted:
            return False
        text = self._record_intervention("message", posted)
        self.turn_id = await self._start_turn([{"type": "text", "text": text, "text_elements": []}])
        return True

    async def _operator_interrupt(self) -> None:
        """Stop native work for an operator interrupt, then resume with their follow-up.

        Root and child turns are interrupted and the goal paused, so
        Codex does nothing while the operator decides. The wait for
        the follow-up is credited, as react() blocks for it.
        """
        compat.emit_event(
            InterruptEvent(
                source="user_cancel",
                interrupted="generate" if self.turn_id else "between_turns",
                span_id=self.root_agent_span_id,
            )
        )
        if self.goal_ok:
            await self.client.set_goal(self.thread_id, objective=self.goal_objective, status="paused")
        for active_thread, active_turn in list(self.client.active_turn.items()):
            if active_turn:
                await self.client.interrupt_turn(active_thread, active_turn)
        # Inspect tools and approvals must stop before the follow-up redirects Codex.
        await self.client.cancel_tools()
        if self.turn_id:
            with suppress(TimeoutError):
                await self.client.wait_for_turn(thread_id=self.thread_id, turn_id=self.turn_id, timeout=10.0)
        loop = asyncio.get_running_loop()
        started = loop.time()
        self.execution_timeout.reschedule(None)
        follow_up_task = asyncio.create_task(self.channel.after_cancel([]))
        last_credit = started
        try:
            # Keep Inspect's working limits credited while its monitor runs.
            while not follow_up_task.done():
                await asyncio.wait([follow_up_task], timeout=0.1)
                now = loop.time()
                self._credit_wait(now - last_credit)
                last_credit = now
            follow_up = [m for m in await follow_up_task if isinstance(m, ChatMessageUser)]
        finally:
            self._credit_wait(loop.time() - last_credit)
            follow_up_task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(follow_up_task, return_exceptions=True)
            self._reschedule_deadline()
        waited = loop.time() - started
        text = self._record_intervention("interrupt", follow_up, waited)
        if self.goal_ok:
            await self.client.set_goal(self.thread_id, objective=self.goal_objective, status="active")
        self.turn_id = await self._start_turn([{"type": "text", "text": text, "text_elements": []}])

    async def _continue_after_turn(self) -> bool:
        """Ask ``continue_with`` after a completed turn; True starts another turn."""
        if self.config.continue_with is None:
            return False
        message = await self.config.continue_with(self._conversation(self.client.completed_root_answer))
        if message is None:
            return False
        self.turn_id = await self._start_turn([{"type": "text", "text": message, "text_elements": []}])
        return True

    def _checkpoint_state(self) -> dict[str, Any]:
        # Inspect captures tracked state and its usage registries synchronously.
        # Snapshot here so tokens arriving during session mirroring are included.
        return {
            "thread_id": self.thread_id,
            "goal_ok": self.goal_ok,
            "attempt_count": self.attempt_count,
            "refusal_retries": self.refusal_retries,
            "accounting": self.mapper.checkpoint_accounting(),
        }

    async def _tick(self) -> None:
        """Turn boundary: mirror Codex's sessions where checkpoints see them, then tick."""
        if self.checkpoint is None:
            return
        mirrored = await self.env.exec(["sh", "-c", MIRROR_SESSIONS, "sh", self.home_dir, self.mirror], user="root")
        if mirrored.returncode != 0:
            raise CodexExecutionError("checkpoint", f"Could not mirror Codex sessions: {mirrored.stderr}")
        await self.checkpoint.checkpointer.tick()

    async def _cleanup(self) -> None:
        self.mapper.tearing_down = True
        loop = asyncio.get_running_loop()
        # Allow sufficient time for the server to drain queued RPCs and exit.
        deadline = loop.time() + 90.0
        failures = self.evidence.metadata["failures"]
        with anyio.CancelScope(shield=True), message_limit(None):

            async def bounded(name: str, operation: Any, limit: float) -> None:
                try:
                    async with asyncio.timeout(max(0, min(limit, deadline - loop.time()))):
                        await operation()
                except Exception as exc:
                    failures.append(f"{name}: {type(exc).__name__}: {exc}")

            async def pause_goal(target_thread_id: str) -> None:
                await self.client.rpc.thread_goal_set(
                    {"threadId": target_thread_id, "status": "paused"}, timeout=10.0, _allow_failed=True
                )

            if self.thread_id and self.goal_ok:
                await bounded("pause goal", partial(pause_goal, self.thread_id), 10.0)
            for tid, active in list(self.client.active_turn.items()):
                if active:
                    await bounded(
                        f"interrupt {tid}", lambda tid=tid, active=active: self.client.interrupt_turn(tid, active), 10.0
                    )
            try:
                await self.client.close(timeout=max(0, min(50.0, deadline - loop.time() - 15.0)))
            finally:
                if self.daemon_transport is not None:
                    # Terminate the WebSocket daemon process in the container.
                    self.client.shutdown["forced"] = True
                    self.client.shutdown["daemon_termination"] = "unconfirmed"
                    daemon = self.daemon_transport

                    async def stop_daemon() -> None:
                        await daemon.stop()
                        self.client.shutdown["daemon_termination"] = "owned-process-kill"

                    await bounded("stop websocket daemon", stop_daemon, 3.0)
                self.evidence.metadata["native_snapshot"] = (
                    "stopped"
                    if self.client.shutdown.get("transport_stopped")
                    and (
                        self.daemon_transport is None
                        or self.client.shutdown.get("daemon_termination") == "owned-process-kill"
                    )
                    else "process-stop-unconfirmed"
                )
                await bounded(
                    "native export", lambda: self.evidence.export(self.env, self.container_id, self.codex_home), 12.0
                )
                await bounded("rollout reconciliation", lambda: self.evidence.reconcile(self.mapper), 12.0)
                flag = accepted_answer(self.client, self.mapper.submitted_answers)
                if self.config.policy.completion != "submission":
                    flag = self.client.completed_root_answer
                if flag and not self.normal_finish:
                    self.state.metadata["codex_partial_submission"] = flag
                self.mapper.finalize_state(
                    self.state, flag, interrupted=not self.normal_finish, stop_reason=self.stop_reason or "stop"
                )
                if self.unsupported_settings:
                    self.state.metadata["codex_unsupported_settings"] = sorted(set(self.unsupported_settings))
                self.evidence.close(shutdown=self.client.shutdown, accounting=self.state.metadata["codex_accounting"])
                if self.owns_span:
                    compat.emit_event(SpanEndEvent(id=self.root_agent_span_id))
