"""Inspect solver entry points for native Codex app-server execution."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from inspect_ai.agent import AgentAttempts, AgentState
from inspect_ai.approval import ApprovalPolicy
from inspect_ai.approval import approval as approval_scope
from inspect_ai.model import ChatMessage
from inspect_ai.review import ReviewPolicy
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Skill, Tool

from . import compat
from .policy import RunPolicy
from .session import (
    WEB_SEARCH_MODES,
    CodexSession,
    SessionConfig,
    WebSearchMode,
    _Checkpoint,
)
from .session import CodexExecutionError as CodexExecutionError
from .subagent import CodexSubagent


class NativeSolver(Protocol):
    """A Solver that also accepts the agent-execution options of ``codex_agent``."""

    async def __call__(
        self,
        state: TaskState,
        generate: Generate | None = None,
        *,
        tools: list[Tool] | None = None,
        root_span_id: str | None = None,
        approval: list[ApprovalPolicy] | None = None,
        review: list[ReviewPolicy] | None = None,
    ) -> TaskState: ...


def create_solver(
    model: str | None = None,
    transport: Literal["stdio", "websocket"] = "stdio",
    reasoning_effort: str | None = None,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    codex_path: str | None = None,
    auth_path: str | None = None,
    ws_port: int = 4500,
    auto_compact_token_limit: int | None = None,
    cwd: str = "/app",
    sandbox_name: str | None = None,
    timeout_sec: float = 600.0,
    max_child_agents: int = 4,
    authentication: Literal["existing", "chatgpt", "api_key"] = "existing",
    policy: RunPolicy | None = None,
    reasoning_summary: Literal["none", "concise", "detailed", "auto"] | None = None,
    retry_refusals: int | None = None,
    attempts: AgentAttempts | None = None,
    continue_with: Callable[[AgentState], Awaitable[str | None]] | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    web_search: WebSearchMode | None = None,
    subagents: Sequence[CodexSubagent] | None = None,
    service_tier: str | None = None,
) -> NativeSolver:
    """Create a solver instance configured with the specified parameters and policy.

    The returned solver also accepts keyword-only ``tools``, ``root_span_id``,
    ``approval`` and ``review`` for agent execution (see ``codex_agent``).

    ``retry_refusals`` continues after that many policy-blocked turns before
    finishing with ``content_filter``. ``attempts`` scores each accepted
    submission and steers ``incorrect_message`` while attempts remain.
    ``continue_with`` decides after each completed turn: a message continues,
    ``None`` finishes; it replaces native goal continuation. Both hooks see
    the conversation mapped so far, with the candidate answer as output.

    ``skills`` are installed as Codex skills for the run; ``web_search`` sets
    the native web search mode. ``subagents`` declares native agent roles.
    ``service_tier`` requests a service tier for every turn.
    """
    policy = policy or RunPolicy()
    if authentication not in ("existing", "chatgpt", "api_key"):
        raise ValueError("authentication must be existing, chatgpt, or api_key")
    if timeout_sec <= 0 or not math.isfinite(timeout_sec):
        raise ValueError("timeout_sec must be finite and positive")
    if max_child_agents < 1:
        raise ValueError("max_child_agents must be positive")
    if not cwd.startswith("/"):
        raise ValueError("cwd must be an absolute sandbox path")
    if retry_refusals is not None and retry_refusals < 0:
        raise ValueError("retry_refusals must be nonnegative")
    if attempts is not None and policy.completion != "submission":
        raise ValueError("attempts require submission completion")
    if web_search is not None and web_search not in WEB_SEARCH_MODES:
        raise ValueError(f"web_search must be one of {', '.join(WEB_SEARCH_MODES)}")
    # Read (and validate) skills before any sample runs.
    resolved_skills = compat.read_skills(skills) if skills else []
    roles = list(subagents or [])
    if len({role.name for role in roles}) != len(roles):
        raise ValueError("subagent names must be unique")

    config = SessionConfig(
        model=model,
        transport=transport,
        reasoning_effort=reasoning_effort,
        base_instructions=base_instructions,
        developer_instructions=developer_instructions,
        codex_path=codex_path,
        auth_path=auth_path,
        ws_port=ws_port,
        auto_compact_token_limit=auto_compact_token_limit,
        cwd=cwd,
        sandbox_name=sandbox_name,
        timeout_sec=timeout_sec,
        max_child_agents=max_child_agents,
        authentication=authentication,
        policy=policy,
        reasoning_summary=reasoning_summary,
        retry_refusals=retry_refusals,
        attempts=attempts,
        continue_with=continue_with,
        skills=tuple(resolved_skills),
        web_search=web_search,
        roles=tuple(roles),
        service_tier=service_tier,
    )

    async def solve(
        state: TaskState,
        generate: Generate | None = None,
        *,
        tools: list[Tool] | None = None,
        root_span_id: str | None = None,
        approval: list[ApprovalPolicy] | None = None,
        review: list[ReviewPolicy] | None = None,
    ) -> TaskState:
        # Approval policies replace the active ones for the whole run, as in
        # react(); the client's reader tasks inherit this context at start.
        with approval_scope(approval) if approval is not None else nullcontext():
            tools = list(state.tools) if tools is None else tools
            async with compat.sample_checkpointer() as checkpointer:
                if checkpointer is None:
                    return await CodexSession(config, state, tools, root_span_id, review, None).run()
                # As react() does, the conversation and output are restored on resume;
                # Codex's own session comes back through its mirrored session files.
                session: dict[str, Any] = {}
                checkpoint = _Checkpoint(checkpointer, session, resume=checkpointer.attempt == "resume")
                restored = checkpointer.track(
                    "codex_session",
                    lambda: checkpoint.snapshot() if checkpoint.snapshot else dict(session),
                    {},
                    value_type=dict[str, Any],
                )
                state.messages = checkpointer.track(
                    "messages",
                    lambda: checkpoint.conversation() if checkpoint.conversation else state.messages,
                    state.messages,
                    value_type=list[ChatMessage],
                )
                state.output = checkpointer.track("output", lambda: state.output, state.output)
                if checkpointer.attempt == "resume_for_scoring":
                    return state
                if checkpoint.resume:
                    session.update(restored)
                return await CodexSession(config, state, tools, root_span_id, review, checkpoint).run()

    return solve


@solver(name="codex_app_server")
def codex_app_server(
    model: str | None = None,
    transport: Literal["stdio", "websocket"] = "stdio",
    reasoning_effort: str | None = None,
    base_instructions: str | None = None,
    developer_instructions: str | None = None,
    codex_path: str | None = None,
    auth_path: str | None = None,
    ws_port: int = 4500,
    auto_compact_token_limit: int | None = None,
    cwd: str = "/app",
    sandbox_name: str | None = None,
    timeout_sec: float = 600.0,
    max_child_agents: int = 4,
    authentication: Literal["existing", "chatgpt", "api_key"] = "existing",
    policy: RunPolicy | None = None,
    reasoning_summary: Literal["none", "concise", "detailed", "auto"] | None = None,
    retry_refusals: int | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    web_search: WebSearchMode | None = None,
    subagents: Sequence[CodexSubagent] | None = None,
    service_tier: str | None = None,
) -> Solver:
    """Inspect AI solver that runs Codex App Server sessions inside a Docker sandbox.

    Args:
        model: Optional model identifier (defaults to server default).
        transport: Transport mechanism ("stdio" or "websocket").
        reasoning_effort: Reasoning effort (defaults to the evaluation's GenerateConfig).
        base_instructions: Optional base system prompt instructions.
        developer_instructions: Optional developer instructions for the thread.
        codex_path: Explicit path to the host Codex binary.
        auth_path: Explicit path to the host auth.json.
        ws_port: Port used for WebSocket transport.
        auto_compact_token_limit: Token threshold for auto-compaction.
        cwd: Container working directory (default "/app").
        sandbox_name: Named sandbox environment if using multi-sandbox compose.
        timeout_sec: Maximum execution time budget in seconds (default 600.0).
        max_child_agents: Maximum concurrent subagents per sample (default 4).
        authentication: Required auth mode ("existing", "chatgpt", or "api_key").
        policy: Optional RunPolicy for custom completion, receipts, and quota holds.
        reasoning_summary: Reasoning summary detail (defaults to the evaluation's GenerateConfig).
        retry_refusals: Continue after this many policy-blocked turns before
            finishing with ``stop_reason="content_filter"``.
        skills: Agent skills (directories or ``Skill``) installed as Codex skills.
        web_search: Native web search mode (Codex's default when omitted).
        subagents: Native agent roles Codex can spawn children with.
        service_tier: Service tier requested for every turn (for example
            ``"priority"`` or ``"flex"``). With Codex's OpenAI provider the
            model must offer it.

    Returns:
        Solver function compatible with Inspect AI tasks.
    """
    return create_solver(
        model=model,
        transport=transport,
        reasoning_effort=reasoning_effort,
        base_instructions=base_instructions,
        developer_instructions=developer_instructions,
        codex_path=codex_path,
        auth_path=auth_path,
        ws_port=ws_port,
        auto_compact_token_limit=auto_compact_token_limit,
        cwd=cwd,
        sandbox_name=sandbox_name,
        timeout_sec=timeout_sec,
        max_child_agents=max_child_agents,
        authentication=authentication,
        policy=policy,
        reasoning_summary=reasoning_summary,
        retry_refusals=retry_refusals,
        skills=skills,
        web_search=web_search,
        subagents=subagents,
        service_tier=service_tier,
    )
