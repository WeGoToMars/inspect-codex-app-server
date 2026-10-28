"""Inspect agent running a native Codex session.

``codex_agent`` composes with ``handoff()``, ``as_tool()``, ``as_solver()`` and
``run()`` like Inspect's built-in agents. Codex owns inference, context and
native tools; Inspect owns bridged tools, approvals, reviews, limits and scoring.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Sequence

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentContinue,
    AgentState,
    AgentSubmit,
    agent,
    agent_with,
)
from inspect_ai.approval import ApprovalPolicy
from inspect_ai.model import ChatMessageUser
from inspect_ai.review import ReviewPolicy
from inspect_ai.tool import (
    Skill,
    Tool,
    ToolDef,
    ToolResult,
    ToolSource,
    mcp_connection,
    tool,
)

from . import compat
from .mapping import _content_items_text
from .policy import ReceiptReader, RunPolicy
from .solver import WebSearchMode, create_solver
from .subagent import CodexSubagent


@agent
def codex_agent(
    *,
    name: str | None = None,
    description: str | None = None,
    prompt: str | None = None,
    tools: Sequence[Tool | ToolDef | ToolSource] | None = None,
    approval: list[ApprovalPolicy] | None = None,
    review: list[ReviewPolicy] | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: Literal["none", "concise", "detailed", "auto"] | None = None,
    base_instructions: str | None = None,
    cwd: str = "/app",
    sandbox_name: str | None = None,
    timeout_sec: float = 600.0,
    max_child_agents: int = 4,
    authentication: Literal["existing", "chatgpt", "api_key"] = "existing",
    transport: Literal["stdio", "websocket"] = "stdio",
    codex_path: str | None = None,
    auth_path: str | None = None,
    ws_port: int = 4500,
    auto_compact_token_limit: int | None = None,
    policy: RunPolicy | None = None,
    submit: AgentSubmit | bool = False,
    attempts: int | AgentAttempts = 1,
    on_continue: str | AgentContinue | None = None,
    retry_refusals: int | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    web_search: WebSearchMode | None = None,
    subagents: Sequence[CodexSubagent] | None = None,
    service_tier: str | None = None,
) -> Agent:
    """Native Codex agent.

    Without ``submit`` the run ends when Codex finishes its turn, and its final
    answer is the output. With ``submit`` a bridged submit tool (react's, or
    ``AgentSubmit.tool``) is the only way to finish: Codex keeps working toward
    the task through its native goal until it submits, and the submitted
    answer is the output. Codex manages its own context, so react's
    ``compaction`` and ``truncation`` have no counterpart (see
    ``auto_compact_token_limit``).

    Codex plans with its native ``update_plan`` (shown as Inspect's plan
    checklist), so deepagent's ``todo_write`` is not needed. For deepagent's
    ``memory``, pass Inspect's ``memory()`` tool in ``tools``. Child agents
    are Codex's own (``subagents`` declares roles for them); Codex's
    multi-agent v2 has no depth setting, so deepagent's ``max_depth`` has no
    counterpart.

    Args:
       name: Agent name (required when using with ``handoff()`` or ``as_tool()``).
       description: Agent description (required when using with ``handoff()`` or ``as_tool()``).
       prompt: Agent instructions, given to Codex as developer instructions.
       tools: Inspect tools bridged to Codex as dynamic tools. Tool sources
          (e.g. MCP servers) stay connected for the whole run.
       approval: Approval policies replacing the active ones for this agent.
          While any Inspect approval policy is active, native commands and file
          changes also require its approval (a rejection is declined to Codex).
       review: Review policies for results of bridged tool calls.
       model: Codex model (defaults to the evaluated ``codex/<model>``).
       reasoning_effort: Reasoning effort (defaults to the evaluation's GenerateConfig).
       reasoning_summary: Reasoning summary detail (defaults to the GenerateConfig).
       base_instructions: Replacement for Codex's base instructions.
       cwd: Absolute working directory inside the sandbox.
       sandbox_name: Inspect sandbox running Codex (defaults to the default sandbox).
       timeout_sec: Wall-clock budget for this agent run.
       max_child_agents: Concurrent native child agents.
       authentication: Required native authentication; never a fallback.
       transport: App-server transport.
       codex_path: Host Codex executable to install (defaults to the pinned bundle).
       auth_path: Host ``auth.json`` to install.
       ws_port: WebSocket port for ``transport="websocket"``.
       auto_compact_token_limit: Native auto-compaction threshold.
       policy: Task-owned completion, input and lifecycle policy.
       submit: Finish only through a submit tool (``True`` for react's default).
          The output is the submitted answer, so ``answer_only`` and
          ``answer_delimiter`` do not apply.
       attempts: Scored submissions allowed (requires ``submit``). An incorrect
          answer is withdrawn and ``incorrect_message`` steers Codex onward.
       on_continue: After a turn ends without finishing. With ``submit`` a
          string replaces the native goal's continuation message; a function
          (as for react) decides after each completed turn instead of the native
          goal. Returning an ``AgentState`` may only append user messages, which
          are sent as the next turn, since Codex owns its conversation history.
       retry_refusals: Continue after this many policy-blocked turns before
          finishing with ``stop_reason="content_filter"``.
       skills: Agent skills (directories or ``Skill``) installed as Codex skills,
          which Codex discovers and loads natively.
       web_search: Native web search mode (Codex's default when omitted).
       subagents: Native agent roles Codex can spawn children with.
       service_tier: Service tier requested for every turn (for example
          ``"priority"`` or ``"flex"``). With Codex's OpenAI provider the
          model must offer it.

    Returns:
        Codex agent.
    """
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts
    submit_config = AgentSubmit() if submit is True else (submit or None)
    agent_tools: list[Tool | ToolDef | ToolSource] = list(tools or [])
    submit_name: str | None = None
    continue_with = None
    if submit_config is None:
        if attempts.attempts > 1:
            raise ValueError("attempts require a submit tool (submit=True)")
        if isinstance(on_continue, str):
            raise ValueError("Passing a string to on_continue with no submit tool is not permitted")
    else:
        if policy is not None and (policy.completion != "turn" or policy.receipt_reader is not None):
            raise ValueError("submit sets the completion policy; pass a policy without completion or receipt_reader")
        submit_tool = (copy(submit_config.tool) if isinstance(submit_config.tool, ToolDef) else
                       ToolDef(submit_config.tool or _default_submit(), name=submit_config.name,
                               description=submit_config.description))
        if submit_tool.max_output is None:
            submit_tool.max_output = 0  # The answer is the completion; never truncate it.
        submit_name = submit_tool.name
        agent_tools.append(submit_tool)
        prompt = "\n\n".join(filter(None, [prompt, compat.DEFAULT_SUBMIT_PROMPT.strip().replace("{submit}", submit_name)]))
        policy = replace(
            policy or RunPolicy(), completion="submission", receipt_reader=_submission_reader(submit_name),
            continuation=(on_continue if isinstance(on_continue, str) else compat.DEFAULT_CONTINUE_PROMPT).strip()
            .replace("{submit}", submit_name),
        )
    if on_continue is not None and not isinstance(on_continue, str):
        continue_with = _continue_with(on_continue, (
            compat.DEFAULT_CONTINUE_PROMPT.replace("{submit}", submit_name) if submit_name else compat.DEFAULT_CONTINUE_PROMPT_NO_SUBMIT
        ).strip())

    native = create_solver(
        model=model, transport=transport, reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary, base_instructions=base_instructions,
        developer_instructions=prompt, codex_path=codex_path, auth_path=auth_path,
        ws_port=ws_port, auto_compact_token_limit=auto_compact_token_limit, cwd=cwd,
        sandbox_name=sandbox_name, timeout_sec=timeout_sec, max_child_agents=max_child_agents,
        authentication=authentication, policy=policy, retry_refusals=retry_refusals,
        attempts=attempts if submit_name else None, continue_with=continue_with,
        skills=skills, web_search=web_search, subagents=subagents, service_tier=service_tier,
    )

    async def execute(state: AgentState) -> AgentState:
        sample = compat.sample_state()
        if sample is None:
            raise RuntimeError("codex_agent must run inside an Inspect sample (it needs the sample's sandbox)")
        # Sample-level metadata, store and limits are shared; this
        # conversation and its output are the agent's own (as in score()).
        conversation = copy(sample)
        conversation.messages = state.messages
        async with mcp_connection(agent_tools):
            try:
                await native(
                    conversation, tools=await _resolve_tools(agent_tools),
                    root_span_id=compat.current_span_id(), approval=approval, review=review,
                )
            finally:
                state.messages = conversation.messages
                state.output = conversation.output
        if submit_name and not (submit_config and submit_config.keep_in_messages):
            state.messages = compat.remove_submit_tool(state.messages, submit_name)
        return state

    if name is not None or description is not None:
        return agent_with(execute, name=name, description=description)
    return execute


@tool(name="submit")
def _default_submit() -> Tool:
    async def execute(answer: str) -> ToolResult:
        """Submit an answer for evaluation.

        Args:
          answer (str): Submitted answer
        """
        return answer

    return execute


def _submission_reader(name: str) -> ReceiptReader:
    """A successful root-thread call of the submit tool is the answer."""

    def read(item: dict[str, Any], root_thread_id: str | None, thread_id: str | None) -> str | None:
        if (item.get("type") == "dynamicToolCall" and item.get("tool") == name and item.get("success") is True
                and thread_id == root_thread_id):
            return _content_items_text(item.get("contentItems")) or None
        return None

    return read


def _continue_with(on_continue: AgentContinue, default: str) -> Callable[[AgentState], Awaitable[str | None]]:
    """Adapt react's ``on_continue`` to a message for the next Codex turn (``None`` stops)."""

    async def decide(state: AgentState) -> str | None:
        before = len(state.messages)
        result = await on_continue(state)
        if result is True:
            return default
        if result is False:
            return None
        if isinstance(result, str):
            return result
        added = result.messages[before:]
        if result.messages[:before] != state.messages[:before] or not added or not all(
                isinstance(m, ChatMessageUser) for m in added):
            raise ValueError("on_continue may only append user messages to a Codex conversation")
        return "\n\n".join(m.text for m in added)

    return decide


async def _resolve_tools(tools: Sequence[Tool | ToolDef | ToolSource]) -> list[Tool]:
    resolved: list[Tool] = []
    for item in tools:
        if isinstance(item, ToolSource):
            resolved.extend(await item.tools())
        elif isinstance(item, ToolDef):
            resolved.append(item.as_tool())
        else:
            resolved.append(item)
    return resolved
