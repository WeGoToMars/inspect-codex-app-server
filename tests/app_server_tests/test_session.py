"""Session failures, retries, stalls, and scoped resource limits.

Scripted app-server events exercise the solver boundary and teardown;
these cases do not start the Codex executable.
"""

from __future__ import annotations

import asyncio

import pytest
from inspect_ai.util import cost_limit, message_limit, turn_limit

from inspect_codex_app_server import RunPolicy, compat
from inspect_codex_app_server.client import CodexTransportFailure
from inspect_codex_app_server.solver import CodexExecutionError, create_solver

from .support import (
    ScriptedAppServer,
    command_item,
    configure_common,
    make_state,
)


@pytest.mark.asyncio
async def test_terminal_failure_raises_sample_error(solver_env, scripted):
    """Terminal infrastructure failures propagate with no completed answer."""
    configure_common(scripted)
    # The first turn fails without submitting an answer.
    failure = {"threadId": "root-thread", "turn": {"id": "turn-1", "status": "failed", "error": {"message": "transport failed", "codexErrorInfo": "rpcError"}}}

    def on_turn_start(server: ScriptedAppServer, params: dict) -> None:
        server.notify("turn/completed", failure)

    scripted.hooks["turn/start"] = on_turn_start

    state = make_state()
    solve = create_solver()
    with pytest.raises(CodexExecutionError) as excinfo:
        await asyncio.wait_for(solve(state, generate=None), timeout=30.0)
    assert "transport failed" in str(excinfo.value)
    assert state.output.completion == ""
    assert scripted.stopped
    assert state.metadata["codex_native_evidence"]["closed"]


@pytest.mark.asyncio
async def test_transport_death_raises(solver_env, scripted):
    configure_common(scripted)

    death = object()

    async def lines_until_death():
        while True:
            item = await scripted.queue.get()
            if item is death:
                raise RuntimeError("connection reset")
            yield item

    scripted.lines = lines_until_death  # type: ignore[method-assign]

    def on_turn_start(server: ScriptedAppServer, params: dict) -> None:
        server.notify("turn/completed", {"threadId": "root-thread", "turn": {"id": "turn-1", "status": "completed"}})

        # Wake the blocked reader as a dying stream would; a separate flag
        # cannot interrupt queue.get() and previously waited for a later RPC.
        scripted.queue.put_nowait(death)

    scripted.hooks["turn/start"] = on_turn_start

    state = make_state()
    solve = create_solver()
    with pytest.raises((CodexExecutionError, CodexTransportFailure), match="connection reset"):
        await asyncio.wait_for(solve(state, generate=None), timeout=2.0)


@pytest.mark.asyncio
async def test_limit_exceeded_stops_codex_and_propagates(solver_env, scripted, usage_contexts):
    """A scoped token limit stops the run and preserves all reported usage."""
    from inspect_ai.util import token_limit

    configure_common(scripted)

    def on_turn_start(server: ScriptedAppServer, params: dict) -> None:
        server.notify("thread/tokenUsage/updated", {
            "threadId": "root-thread",
            "tokenUsage": {"total": {"inputTokens": 10_000, "outputTokens": 0, "totalTokens": 10_000}},
        })
        server.notify("turn/completed", {"threadId": "root-thread", "turn": {"id": "turn-1", "status": "completed"}})

    scripted.hooks["turn/start"] = on_turn_start

    state = make_state()
    solve = create_solver()
    from inspect_codex_app_server import compat

    with token_limit(10), pytest.raises(compat.LimitExceededError):
        await asyncio.wait_for(solve(state, generate=None), timeout=30.0)
    assert scripted.stopped, "A completed turn still requires owned process closure"
    assert state.output.usage.total_tokens == 10_000


@pytest.mark.asyncio
async def test_stall_watchdog_aborts_silent_run(solver_env, scripted, monkeypatch):
    """Silence between turns beyond the threshold surfaces a sample error."""
    import inspect_codex_app_server.session as solver_mod

    configure_common(scripted)
    monkeypatch.setattr(solver_mod, "CONTINUATION_GRACE_SEC", 0.05)
    monkeypatch.setattr(solver_mod, "STALL_TIMEOUT_SEC", 0.25)

    def on_turn_start(server: ScriptedAppServer, params: dict) -> None:
        server.notify("turn/completed", {"threadId": "root-thread", "turn": {"id": "turn-1", "status": "completed"}})

    scripted.hooks["turn/start"] = on_turn_start

    state = make_state()
    solve = create_solver(policy=RunPolicy(completion="goal"))
    with pytest.raises(CodexExecutionError) as excinfo:
        await asyncio.wait_for(solve(state, generate=None), timeout=20.0)
    assert "stall" in str(excinfo.value)
    assert scripted.stopped


@pytest.mark.asyncio
async def test_turn_start_persistent_failure_aborts(solver_env, scripted, monkeypatch):
    import inspect_codex_app_server.session as solver_mod

    configure_common(scripted)
    monkeypatch.setattr(solver_mod, "TURN_START_BACKOFF_CAP_SEC", 0.01)
    attempts = {"n": 0}

    def failing_turn_start(params: dict) -> dict:
        attempts["n"] += 1
        return {"error": {"code": -32603, "message": "provider overloaded", "data": {"codexErrorInfo": "serverOverloaded"}}}

    scripted.responses["turn/start"] = failing_turn_start

    state = make_state()
    solve = create_solver(policy=RunPolicy(completion="goal"))
    with pytest.raises(CodexExecutionError) as excinfo:
        await asyncio.wait_for(solve(state, generate=None), timeout=20.0)
    assert "turn/start failed after retries" in str(excinfo.value)
    assert attempts["n"] == 4  # MAX_TURN_FAILURES (3) retries plus the aborting attempt


@pytest.mark.asyncio
async def test_message_limit_cleans_up_and_preserves_evidence(solver_env, scripted):
    configure_common(scripted)
    scripted.hooks["turn/start"] = lambda server, params: server.notify(
        "item/completed", {"threadId": "root-thread", "item": command_item("cmd")}
    )
    state = make_state()
    with message_limit(1), pytest.raises(compat.LimitExceededError):
        await asyncio.wait_for(create_solver()(state, None), 2)
    assert state.completed
    assert any(m.role == "assistant" for m in state.messages)
    assert any(m.role == "tool" for m in state.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [lambda: cost_limit(1), lambda: turn_limit(1)])
async def test_unsupported_budget_fails_before_start(solver_env, scripted, budget):
    with budget(), pytest.raises(ValueError, match="does not support Inspect"):
        await create_solver()(make_state(), None)
    assert scripted.requests == []


def test_unlimited_inner_scope_does_not_hide_unsupported_parent():
    with cost_limit(1), cost_limit(None), pytest.raises(ValueError, match="cost"):
        compat.require_supported_budgets()
