"""Token accounting, model attribution, throughput, and retained evidence.

Fixed protocol events and real rollout files establish independent totals,
deduplication, checkpoint counters, and recovery after partial writes.
No Codex process or live provider is used.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from inspect_ai.model import _throughput
from inspect_ai.util import token_limit

from inspect_codex_app_server import compat
from inspect_codex_app_server.evidence import NativeEvidence
from inspect_codex_app_server.mapping import EventStreamMapper

from .support import (
    command_item,
    completed_item,
    make_mapper,
    make_state,
    model_events,
    usage_params,
)


def test_usage_accumulates_across_samples(usage_contexts):
    """Independent sample totals of 100 and 200 contribute 300 to the task."""
    sample_ctx, task_ctx = usage_contexts
    from inspect_ai.model._model import sample_model_usage_context_var

    # Sample A runs in its own sample context
    t_a = sample_model_usage_context_var.set({})
    make_mapper().on_token_usage("", usage_params("root", 100))
    sample_a = dict(sample_ctx.get())
    sample_model_usage_context_var.reset(t_a)

    # Sample B runs in a fresh sample context, same task context
    t_b = sample_model_usage_context_var.set({})
    make_mapper().on_token_usage("", usage_params("root", 200))
    sample_b = dict(sample_ctx.get())
    sample_model_usage_context_var.reset(t_b)

    assert sample_a["codex/test"].total_tokens == 100
    assert sample_b["codex/test"].total_tokens == 200
    assert task_ctx.get()["codex/test"].total_tokens == 300


def test_late_child_configuration_moves_only_its_unknown_usage(usage_contexts, captured_events):
    """An asynchronous thread/read must preserve both usage and attribution uncertainty."""
    mapper = make_mapper()
    mapper.on_token_usage("", usage_params("child", 120))
    mapper.on_token_usage("", usage_params("other-child", 90))
    mapper.on_response_completed("", {"threadId": "child", "responseId": "first"})
    mapper.on_thread_metadata("thread/metadata/read", {"thread": {
        "id": "child", "parentThreadId": "root", "model": "child-model",
    }})
    mapper.on_token_usage("", usage_params("child", 150))
    mapper.on_thread_metadata("thread/metadata/read", {"thread": {
        "id": "child", "parentThreadId": "root", "model": "child-model",
    }})
    for context in usage_contexts:
        usage = context.get()
        assert usage["child-model"].total_tokens == 150
        assert usage["codex/unknown"].total_tokens == 90
        assert sum(value.total_tokens for value in usage.values()) == 240
    record = mapper.responses[("child", "first")]
    assert record["configuredModel"] == "child-model"
    assert record["modelAttribution"] == "first-observed-thread-configuration"
    assert record["servingModel"] == "unobserved"


def test_lifetime_usage_is_not_paired_with_reconstructed_items(captured_events, usage_contexts):
    """Item boundaries cannot identify a native inference; aggregate usage remains metered."""
    mapper = make_mapper(make_state())
    mapper.on_item_completed("", completed_item("root", command_item("a")))
    mapper.on_token_usage("", usage_params("root", 100, last=100))
    mapper.on_item_completed("", completed_item("root", {"type": "agentMessage", "id": "final", "text": "Done"}))
    mapper.on_turn_completed("", {"threadId": "root", "turn": {"id": "t1"}})
    assert all(e.output.usage is None for e in model_events(captured_events))
    assert mapper.finalize_state(mapper.state, "").output.usage.total_tokens == 100


@pytest.mark.parametrize("thread,item_type,order", [
    ("root", "commandExecution", "before_item"),
    ("root", "commandExecution", "after_item"),
    ("root", "commandExecution", "after_turn"),
    ("child", "commandExecution", "before_item"),
    ("child", "commandExecution", "after_item"),
    ("child", "agentMessage", "before_item"),
])
def test_response_usage_survives_item_and_turn_ordering(
    captured_events, usage_contexts, thread, item_type, order,
):
    """Pending, active, and finalized events retain summed response usage in their own thread."""
    state = make_state()
    mapper = make_mapper(state)
    mapper.root_agent_span_id = "root-span"
    if thread == "child":
        mapper.on_thread_metadata("thread/metadata/read", {"thread": {
            "id": "child", "parentThreadId": "root", "model": "codex/test"}})
    item = (command_item("item", command="pwd") if item_type == "commandExecution" else
            {"type": "agentMessage", "id": "item", "text": "Child answer"})
    responses = [{"threadId": thread, "responseId": f"r{number}", "usage": {
        "inputTokens": inputs, "outputTokens": outputs, "totalTokens": inputs + outputs,
    }} for number, (inputs, outputs) in enumerate(((100, 10), (200, 20)))]

    if order == "before_item":
        for response in responses:
            mapper.on_response_completed("rawResponse/completed", response)
    mapper.on_item_completed("item/completed", completed_item(thread, item))
    if order == "after_item":
        for response in responses:
            mapper.on_response_completed("rawResponse/completed", response)
    elif order == "after_turn":
        mapper.on_response_completed("rawResponse/completed", responses[0])
        mapper.on_turn_completed("turn/completed", {"threadId": thread, "turn": {"id": "turn"}})
        mapper.on_response_completed("rawResponse/completed", responses[1])
    mapper.finalize_state(state, "answer")

    span = mapper.thread_span(thread)
    events = {e.uuid: e for e in model_events(captured_events) if e.span_id == span}
    event, = events.values()
    assert event.output.usage.input_tokens == 300
    assert event.output.usage.output_tokens == 30
    assert event.output.usage.total_tokens == 330
    assert state.output.usage.total_tokens == 330
    for context in usage_contexts:
        assert context.get()["codex/test"].total_tokens == 330


def test_checkpoint_keeps_usage_queued_for_the_next_model_event(captured_events, usage_contexts):
    """Firing a checkpoint changes no live event; a resumed mapper attaches the queued usage."""
    mapper = make_mapper(make_state())
    mapper.on_item_completed("", completed_item("root", command_item("ls")))
    mapper.on_item_completed("", completed_item("root", {"type": "reasoning", "id": "r", "content": ["next"]}))
    mapper.on_response_completed("rawResponse/completed", {"threadId": "root", "responseId": "r1", "usage": {
        "inputTokens": 7, "outputTokens": 3, "totalTokens": 10}})
    accounting = mapper.checkpoint_accounting()

    for current in (mapper, make_mapper(make_state())):
        if current is not mapper:
            current.restore_accounting(accounting)
        current.on_item_completed("", completed_item("root", command_item("cat")))
    usage = {e.output.choices[0].message.tool_calls[0].id: e.output.usage for e in model_events(captured_events)}
    assert usage["call_ls"] is None
    assert usage["call_cat"].total_tokens == 10
    assert [e.output.usage.total_tokens for e in model_events(captured_events)[-2:]] == [10, 10]


def test_cache_usage_duplicates_and_stale_reports(usage_contexts):
    mapper = make_mapper(model_name="old-model")
    for total, cached in [(100, 60), (200, 120), (200, 120), (100, 60), (200, 120)]:
        params = usage_params("root", total)
        params["tokenUsage"]["total"].update(
            cachedInputTokens=cached, cacheWriteInputTokens=cached // 3
        )
        mapper.on_token_usage("", params)
    assert mapper.root_running_usage.total_tokens == 200
    assert mapper.root_running_usage.input_tokens == 40
    assert mapper.root_running_usage.input_tokens_cache_read == 120
    assert mapper.root_running_usage.input_tokens_cache_write == 40
    _, task = usage_contexts
    assert task.get()["old-model"].total_tokens == 200
    assert task.get()["old-model"].input_tokens_cache_read == 120


def test_child_usage_before_spawn_is_isolated(captured_events, usage_contexts):
    mapper = make_mapper(model_name="old-model")
    mapper.on_token_usage("", usage_params("child", 100))
    mapper.on_token_usage("", usage_params("root", 200))
    mapper.on_item_completed(
        "item/completed",
        {
            "threadId": "root",
            "item": {
                "id": "spawn",
                "type": "collabAgentToolCall",
                "tool": "spawnAgent",
                "senderThreadId": "root",
                "receiverThreadIds": ["child"],
            },
        },
    )
    mapper.on_token_usage("", usage_params("child", 150))
    assert mapper.root_running_usage.total_tokens == 200
    assert mapper.thread_tokens("child") == 150


def test_reroute_credits_the_rest_of_the_turn_to_the_serving_model(usage_contexts, captured_events):
    state = make_state()
    mapper = make_mapper(state, model_name="old-model")
    mapper.on_turn_started("turn/started", {"threadId": "root", "turn": {"id": "t1"}})
    mapper.on_model_rerouted(
        "model/rerouted",
        {
            "threadId": "root",
            "turnId": "t1",
            "fromModel": "old-model",
            "toModel": "new-model",
            "reason": "highRiskCyberActivity",
        },
    )
    mapper.on_token_usage("", usage_params("root", 100))
    assert mapper.model_name == "old-model"
    assert usage_contexts[0].get()["new-model"].total_tokens == 100
    warning = next(event for event in captured_events if getattr(event, "source", None) == "codex-model-rerouted")
    assert warning.data["toModel"] == "new-model" and warning.data["turnId"] == "t1"
    mapper.on_response_completed("", {"threadId": "root", "responseId": "rerouted-response"})
    assert {key: mapper.responses[("root", "rerouted-response")][key]
            for key in ("configuredModel", "servingModel", "modelAttribution")} == {
        "configuredModel": "old-model", "servingModel": "new-model", "modelAttribution": "model-rerouted"}

    # Codex reports the serving model per turn, so the next turn is credited as configured.
    mapper.on_turn_started("turn/started", {"threadId": "root", "turn": {"id": "t2"}})
    mapper.on_token_usage("", usage_params("root", 130))
    assert usage_contexts[0].get()["old-model"].total_tokens == 30
    mapper.finalize_state(state, "")
    assert state.metadata["codex_safety"] == {"reroutes": [{
        "threadId": "root", "turnId": "t1", "fromModel": "old-model", "toModel": "new-model",
        "reason": "highRiskCyberActivity"}]}


@pytest.fixture
def throughput(monkeypatch):
    _throughput.init_model_throughput()
    monkeypatch.setattr(_throughput, "time", SimpleNamespace(monotonic=lambda: 100.0))
    try:
        yield lambda: _throughput.throughput_snapshot(now=110.0)
    finally:
        _throughput.init_model_throughput()


def test_live_response_rates_exclude_duplicate_lifetime_and_recovered_usage(
    throughput, usage_contexts, captured_events
):
    mapper = EventStreamMapper("root-model")
    mapper.root_thread_id = "root"
    mapper.on_thread_metadata("thread/metadata/read", {"thread": {
        "id": "child", "parentThreadId": "root", "model": "worker-model",
    }})
    # The lifetime total already covers both the live and later recovered call.
    mapper.on_token_usage("thread/tokenUsage/updated", {
        "threadId": "root", "tokenUsage": {"total": {
            "inputTokens": 620, "outputTokens": 180, "totalTokens": 800,
        }},
    })
    assert throughput() == {}

    response = {
        "threadId": "root", "responseId": "same-id",
        "usage": {"inputTokens": 380, "outputTokens": 120, "totalTokens": 500},
    }
    mapper.on_response_completed("rawResponse/completed", {
        "threadId": "root", "responseId": "same-id",
    })
    assert throughput() == {}  # Missing usage cannot establish a token rate.
    mapper.on_response_completed("rawResponse/completed", response)
    mapper.on_response_completed("rawResponse/completed", response)
    mapper.on_response_completed("rollout/token_usage_record", response)
    mapper.on_response_completed("rawResponse/completed", {
        "threadId": "child", "responseId": "same-id",
        "usage": {"inputTokens": 140, "outputTokens": 60, "totalTokens": 200},
    })
    mapper.on_response_completed("rollout/token_usage_record", {
        "threadId": "root", "responseId": "recovered-only",
        "usage": {"inputTokens": 240, "outputTokens": 60, "totalTokens": 300},
    })

    views = throughput()
    assert set(views) == {"codex/root-model", "codex/worker-model"}
    root, child = views["codex/root-model"], views["codex/worker-model"]
    assert (root.requests, root.output_tokens, root.total_tokens) == (1, 120, 500)
    assert (root.output_tokens_per_second, root.requests_per_minute) == (12, 6)
    assert (child.requests, child.output_tokens, child.total_tokens) == (1, 60, 200)
    assert (child.output_tokens_per_second, child.requests_per_minute) == (6, 6)
    # Recovery still contributes to final token accounting exactly once.
    for context in usage_contexts:
        assert context.get()["root-model"].total_tokens == 800
        assert context.get()["worker-model"].total_tokens == 200


def test_response_reaching_token_limit_still_updates_throughput(
    throughput, usage_contexts
):
    mapper = EventStreamMapper("codex/root-model")
    mapper.root_thread_id = "root"
    with token_limit(100), pytest.raises(compat.LimitExceededError):
        mapper.on_response_completed("rawResponse/completed", {
            "threadId": "root", "responseId": "last-response",
            "usage": {"inputTokens": 80, "outputTokens": 40, "totalTokens": 120},
        })
    view = throughput()["codex/root-model"]
    assert (view.requests, view.output_tokens_per_second, view.total_tokens) == (1, 4, 120)


def usage_record(response, tokens, *, thread="root", session="root", total=None):
    usage = {"input_tokens": tokens, "cached_input_tokens": 0,
             "output_tokens": 0, "reasoning_output_tokens": 0, "total_tokens": tokens}
    return json.dumps({"timestamp": "2026-10-02T00:00:00Z", "type": "token_usage_record", "payload": {
        "thread_id": thread, "turn_id": "turn", "session_id": session, "root_turn_id": "turn",
        "response_id": response, "usage": usage, "turn_token_usage": usage,
        "thread_token_usage": {**usage, "input_tokens": total or tokens, "total_tokens": total or tokens},
    }}) + "\n"


async def test_partial_rollouts_preserve_late_root_and_child_usage(
    captured_events, usage_contexts
):
    evidence = NativeEvidence("partial-rollout")
    native = evidence.directory / "native" / "sessions"
    native.mkdir(parents=True)
    (native / "root.jsonl").write_text(
        usage_record("wire-response", 10)
        + '{"type":"token_usage_record","payload":\n'
        + usage_record("late-response", 15, total=25)
    )
    (native / "child.jsonl").write_bytes(
        usage_record("child-response", 20, thread="child").encode()
        + b'{"type":"response_item","payload":{"text":"\xe9\x9b'
    )
    # A shared native home can contain earlier agent runs; these are not this run's children.
    (native / "unrelated.jsonl").write_text(
        usage_record("unrelated-response", 1000, thread="previous-root", session="previous-root"))
    state = make_state(model="codex/gpt-5.4")
    mapper = EventStreamMapper("gpt-5.4", state)
    mapper.root_thread_id = "root"
    mapper.tearing_down = True
    # No notification from the child arrived; its records name the root's session.
    mapper.on_response_completed("rawResponse/completed", {
        "threadId": "root", "responseId": "wire-response", "usage": {
            "inputTokens": 10, "outputTokens": 0, "totalTokens": 10,
        },
    })
    try:
        await evidence.reconcile(mapper)
        mapper.finalize_state(state, "", interrupted=True)
        assert state.output.usage.total_tokens == 45
        assert state.metadata["codex_accounting"]["observed_responses"] == 3
        assert evidence.metadata["reconciled_usage_records"] == 3
        assert "previous-root" not in state.metadata["codex_accounting"]["unattributed_tokens_by_thread"]
        assert evidence.metadata["rollout_reconciliation"] == "partial"
        assert len(evidence.metadata["failures"]) == 2
        evidence.close(shutdown={"stream_drained": False, "forced": True},
                       accounting=state.metadata["codex_accounting"])
        manifest = json.loads((evidence.directory / "manifest.json").read_text())
        assert manifest["closed"] and manifest["truncated"]
        assert manifest["accounting"]["observed_responses"] == 3
    finally:
        if not evidence.metadata["closed"]:
            evidence.close(shutdown={"stream_drained": False}, accounting={})


async def test_bounded_reconciliation_retains_already_charged_records(
    captured_events, usage_contexts, monkeypatch
):
    evidence = NativeEvidence("bounded-rollout")
    native = evidence.directory / "native"
    native.mkdir()
    (native / "rollout.jsonl").write_text(usage_record("first", 10) + usage_record("second", 20, total=30))
    state = make_state(model="codex/gpt-5.4")
    mapper = EventStreamMapper("gpt-5.4", state)
    mapper.root_thread_id = "root"
    mapper.tearing_down = True
    record = mapper.on_response_completed

    def on_response(method, params):
        record(method, params)
        # Impose cleanup cancellation at observed progress; the next JSONL
        # checkpoint must preserve this record and its published count.
        asyncio.current_task().cancel()

    monkeypatch.setattr(mapper, "on_response_completed", on_response)
    task = asyncio.create_task(evidence.reconcile(mapper))
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        mapper.finalize_state(state, "", interrupted=True)
        assert state.output.usage.total_tokens == 10
        assert evidence.metadata["reconciled_usage_records"] == 1
        assert evidence.metadata["rollout_reconciliation"] == "interrupted"
        evidence.close(shutdown={"stream_drained": True}, accounting=state.metadata["codex_accounting"])
        manifest = json.loads((evidence.directory / "manifest.json").read_text())
        assert manifest["reconciled_usage_records"] == 1
        assert manifest["rollout_reconciliation"] == "interrupted"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if not evidence.metadata["closed"]:
            evidence.close(shutdown={"stream_drained": False}, accounting={})


def test_native_evidence_redacts_account_rpc_fields_and_retains_full_content(tmp_path, monkeypatch):
    """Pinned account RPCs carry credentials in both directions, with independent IDs."""
    from inspect_codex_app_server.evidence import NativeEvidence

    monkeypatch.setenv("CODEX_NATIVE_LOG_DIR", str(tmp_path))
    evidence = NativeEvidence("account-evidence")
    evidence.record("out", {"id": 1, "method": "account/login/start", "params": {
        "type": "apiKey", "apiKey": "CLIENT_CREDENTIAL_SENTINEL",
    }})
    evidence.record("write", {"requestId": 1, "status": "completed"})
    evidence.record("in", {"id": 1, "method": "account/chatgptAuthTokens/refresh", "params": {
        "reason": "unauthorized", "previousAccountId": "account-id",
    }})
    evidence.record("out", {"id": 1, "result": {
        "accessToken": "SERVER_CREDENTIAL_SENTINEL", "chatgptAccountId": "account-id",
    }})
    evidence.record("in", {"id": 1, "result": {"type": "apiKey"}})
    prompt = "FULL_CONTENT_SENTINEL" + "x" * 20000
    evidence.record("out", {"id": 2, "method": "turn/start", "params": {"input": [
        {"type": "text", "text": prompt, "text_elements": []},
    ]}})
    evidence.close(shutdown={"stream_drained": True}, accounting={})
    raw = (evidence.directory / "protocol.jsonl").read_text()
    assert "CLIENT_CREDENTIAL_SENTINEL" not in raw and "SERVER_CREDENTIAL_SENTINEL" not in raw
    rows = [json.loads(line) for line in raw.splitlines()]
    assert rows[0]["message"]["params"]["apiKey"] == "[redacted]"
    assert rows[3]["message"]["result"]["accessToken"] == "[redacted]"
    assert rows[4]["auth_fields_redacted"]
    assert rows[5]["message"]["params"]["input"][0]["text"] == prompt
    manifest = json.loads((evidence.directory / "manifest.json").read_text())
    assert manifest["closed"] and not manifest["truncated"] and manifest["events"] == 6
