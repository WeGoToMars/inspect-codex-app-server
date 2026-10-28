"""Saved Inspect results, authentication transitions, and quota recovery.

Real Inspect evaluations against scripted app-server peers verify accepted
answers, visible execution errors, live usage metrics, and credited quota waits.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import match

from inspect_codex_app_server import RunPolicy, codex_app_server

from .support import patch_session


class Peer:
    def __init__(self, account, *, failure=None, switch_account=False, continuation=False):
        self.account = account
        self.failure = failure
        self.switch_account = switch_account
        self.continuation = continuation
        self.requests = []
        self.queue = asyncio.Queue()
        self.stopped = False

    async def start(self):
        pass

    def notify(self, method, params):
        self.queue.put_nowait(json.dumps({"method": method, "params": params}))

    async def write_line(self, line):
        msg = json.loads(line)
        if "id" not in msg or "method" not in msg:
            return
        method, params = msg["method"], msg.get("params", {})
        self.requests.append((method, params))
        responses = {
            "initialize": {},
            "account/read": {"account": self.account, "requiresOpenaiAuth": True},
            "model/list": {"data": []},
            "thread/start": {"thread": {"id": "root"}},
            "thread/inject_items": {},
            "turn/start": {"turn": {"id": "turn-1"}},
            "thread/goal/set": {"goal": {"status": params.get("status")}},
            "thread/goal/get": {"goal": {"status": "completed"}},
        }
        assert method in responses, f"Unexpected RPC: {method}"
        if method == "thread/goal/get" and self.continuation:
            # The native continuation can start and finish before the pending
            # goal-status reply reaches the client. No client turn was started.
            self.notify("turn/started", {"threadId": "root", "turn": {"id": "turn-2"}})
            self.notify("item/completed", {
                "threadId": "root", "turnId": "turn-2",
                "item": {"type": "agentMessage", "id": "final-2", "phase": "final_answer", "text": "42"},
            })
            self.notify("turn/completed", {
                "threadId": "root", "turn": {"id": "turn-2", "status": "completed"},
            })
        self.queue.put_nowait(json.dumps({"id": msg["id"], "result": responses[method]}))
        if method == "turn/start":
            if self.switch_account:
                self.notify("account/updated", {"authMode": "apikey"})
            self.notify("account/rateLimits/updated", {"rateLimits": {"primary": {"usedPercent": 75}}})
            for thread, phase, text in [
                ("child", "final_answer", "CHILD_ANSWER_MUST_NOT_SCORE"),
                ("root", "commentary", "COMMENTARY_MUST_NOT_SCORE"),
                ("root", "final_answer", "21" if self.continuation else "42"),
            ]:
                self.notify("item/completed", {
                    "threadId": thread, "turnId": "turn-1",
                    "item": {"type": "agentMessage", "id": f"{thread}-{phase}", "phase": phase, "text": text},
                })
            self.notify("turn/completed", {
                "threadId": "root", "turn": {"id": "turn-1",
                    "status": "failed" if self.failure else "completed",
                    "error": self.failure},
            })

    async def lines(self):
        while (line := await self.queue.get()) is not None:
            yield line

    async def stop(self):
        self.stopped = True
        self.queue.put_nowait(None)


class Sandbox:
    async def exec(self, *args, **kwargs):
        # No native files exist in this protocol-only scenario.
        return SimpleNamespace(returncode=1, stdout="", stderr="")


def run_evaluation(tmp_path, monkeypatch, peer, **options):
    monkeypatch.setenv("CODEX_NATIVE_LOG_DIR", str(tmp_path / "native"))
    patch_session(monkeypatch, peer, Sandbox())
    task = Task(
        dataset=[Sample(input=[ChatMessageSystem(content="Use decimal notation."),
                               ChatMessageUser(content="What is 19 + 23?")], target="42")],
        solver=codex_app_server(**{"authentication": "chatgpt", **options}),
        scorer=match(),
    )
    log, = eval(task, model="codex/default", log_dir=str(tmp_path / "logs"), display="none")
    saved = read_eval_log(log.location)
    assert peer.stopped
    sample, = saved.samples
    evidence = sample.metadata["codex_native_evidence"]
    manifest = json.loads((Path(evidence["directory"]) / "manifest.json").read_text())
    assert manifest["closed"] and manifest["shutdown"]["stream_drained"]
    return sample


def test_subscription_run_persists_only_completed_root_answer(tmp_path, monkeypatch):
    peer = Peer({"type": "chatgpt", "planType": "plus", "email": "solver@example.invalid"})
    sample = run_evaluation(tmp_path, monkeypatch, peer, max_child_agents=1, cwd="/workspace")
    assert sample.error is None
    assert sample.output.completion == "42"
    assert next(iter(sample.scores.values())).value == "C"
    assert sample.metadata["codex_authentication"] == {
        "expected": "chatgpt", "type": "chatgpt", "plan_type": "plus",
        "email": "solver@example.invalid",
        "requires_openai_auth": True, "observed": True,
    }
    assert sample.metadata["codex_rate_limits"]["rateLimits"]["primary"]["usedPercent"] == 75
    assert sample.metadata["codex_run"]["max_child_agents"] == 1
    assert sample.metadata["codex_run"]["experimental_api"] is True
    start = next(params for method, params in peer.requests if method == "thread/start")
    assert start["cwd"] == "/workspace"
    injected, = [params for method, params in peer.requests if method == "thread/inject_items"]
    assert injected == {"threadId": "root", "items": [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Use decimal notation."}]}]}
    turn = next(params for method, params in peer.requests if method == "turn/start")
    assert turn["input"] == [{"type": "text", "text": "What is 19 + 23?", "text_elements": []}]


@pytest.mark.parametrize("account,authentication,email", [
    ({"type": "chatgpt", "planType": "pro", "email": "solver@example.invalid",
      "accessToken": "CREDENTIAL_MUST_NOT_APPEAR"}, "chatgpt", "solver@example.invalid"),
    ({"type": "apiKey", "apiKey": "CREDENTIAL_MUST_NOT_APPEAR"}, "api_key", None),
])
def test_account_identity_is_live_before_inference_and_persisted(
    tmp_path, monkeypatch, account, authentication, email
):
    from inspect_ai._control.store import sample_store
    from inspect_ai.log._samples import sample_active

    class IdentityPeer(Peer):
        live_identity = None

        async def write_line(self, line):
            if json.loads(line).get("method") == "turn/start":
                active = sample_active()
                self.live_identity = await sample_store(
                    active.eval_id, str(active.sample.id), active.epoch,
                    keys=["codex_authentication"], full=True,
                )
            await super().write_line(line)

    peer = IdentityPeer(account)
    sample = run_evaluation(tmp_path, monkeypatch, peer, authentication=authentication)
    assert sample.error is None
    live = peer.live_identity
    assert live["status"] == "running"
    identity = live["store"]["codex_authentication"]
    assert identity["observed"] is True
    assert identity["type"] == account["type"] and identity["email"] == email
    assert identity["plan_type"] == account.get("planType")
    assert sample.metadata["codex_authentication"] == identity
    assert sample.store["codex_authentication"] == identity
    observations = [event.data for event in sample.events
                    if event.event == "info" and event.source == "codex-authentication"]
    assert observations[-1] == identity
    assert "CREDENTIAL_MUST_NOT_APPEAR" not in json.dumps(observations)
    assert "CREDENTIAL_MUST_NOT_APPEAR" not in json.dumps(identity)


def test_task_token_rate_updates_while_native_solver_is_running(tmp_path, monkeypatch):
    from inspect_ai._control.state import current_eval_summaries

    class UsagePeer(Peer):
        live_summaries = None

        def notify(self, method, params):
            if method == "turn/completed":
                response = {
                    "threadId": "root", "responseId": "root-response",
                    "usage": {"inputTokens": 380, "outputTokens": 120, "totalTokens": 500},
                }
                self.notify("rawResponse/completed", response)
                self.notify("thread/tokenUsage/updated", {
                    "threadId": "root", "tokenUsage": {"total": response["usage"]},
                })
                self.notify("thread/tokenUsage/updated", {
                    "threadId": "child", "tokenUsage": {"total": {
                        "inputTokens": 140, "outputTokens": 60, "totalTokens": 200,
                    }},
                })
                self.notify("rawResponse/completed", response)
            super().notify(method, params)

        async def lines(self):
            async for line in super().lines():
                # Resuming after yield means the reader processed this event.
                yield line
                if json.loads(line).get("method") == "thread/tokenUsage/updated":
                    self.live_summaries = await current_eval_summaries(started_at=0)

    peer = UsagePeer({"type": "chatgpt"})
    sample = run_evaluation(tmp_path, monkeypatch, peer)
    assert sample.error is None
    summary, = peer.live_summaries
    assert summary["status"] == "running"
    assert summary["samples"]["in_flight"] == 1
    assert summary["total_tokens"] == 700
    assert summary["tokens_per_second"] > 0
    assert sum(usage.total_tokens for usage in sample.model_usage.values()) == 700


def test_subscription_constraint_rejects_api_billing_before_inference(tmp_path, monkeypatch):
    peer = Peer({"type": "apiKey"})
    sample = run_evaluation(tmp_path, monkeypatch, peer)
    assert sample.error and "Required chatgpt authentication" in sample.error.message
    assert not sample.scores
    assert not any(method == "turn/start" for method, _ in peer.requests)


def test_account_switch_is_an_error_without_a_scored_answer(tmp_path, monkeypatch):
    peer = Peer({"type": "chatgpt", "email": "previous@example.invalid"}, switch_account=True)
    sample = run_evaluation(tmp_path, monkeypatch, peer)
    assert sample.error and "Required chatgpt authentication" in sample.error.message
    assert not sample.scores
    assert sample.metadata["codex_authentication"]["email"] is None
    assert sample.store["codex_authentication"]["email"] is None
    observations = [event.data for event in sample.events
                    if event.event == "info" and event.source == "codex-authentication"]
    assert observations[-2]["email"] == "previous@example.invalid"
    assert observations[-1]["type"] == "apiKey" and observations[-1]["email"] is None


def test_exhausted_subscription_is_an_error_without_retry_or_score(tmp_path, monkeypatch):
    peer = Peer({"type": "chatgpt"}, failure={
        "message": "Plan allowance exhausted", "codexErrorInfo": "usageLimitExceeded",
    })
    sample = run_evaluation(tmp_path, monkeypatch, peer)
    assert sample.error and "[quota]" in sample.error.message
    assert not sample.scores
    assert sample.output.completion == ""
    assert sum(method == "turn/start" for method, _ in peer.requests) == 1


def test_submission_policy_requires_a_receipt_reader():
    with pytest.raises(ValueError, match="authoritative receipt_reader"):
        RunPolicy(completion="submission")


def no_receipt(item, root_thread_id, thread_id):
    return None


@pytest.mark.parametrize("completion,expected", [("goal", "42"), ("submission", "")])
def test_task_policy_controls_completion_in_saved_log(tmp_path, monkeypatch, completion, expected):
    peer = Peer({"type": "chatgpt"})
    sample = run_evaluation(tmp_path, monkeypatch, peer, policy=RunPolicy(
        completion=completion, receipt_reader=no_receipt,
    ))
    assert sample.error is None
    assert sample.output.completion == expected
    # Submission mode must not promote the agent's conversation answer when
    # no authoritative receipt exists, even though the native goal completed.
    assert next(iter(sample.scores.values())).value == ("C" if expected else "I")
    assert sample.metadata["codex_goal_status"] == "completed"


def test_fast_goal_continuation_preserves_its_completed_answer(tmp_path, monkeypatch):
    peer = Peer({"type": "chatgpt"}, continuation=True)
    sample = run_evaluation(tmp_path, monkeypatch, peer, policy=RunPolicy(completion="goal"))
    assert sample.error is None
    assert sample.output.completion == "42"
    assert next(iter(sample.scores.values())).value == "C"
    assert sum(method == "turn/start" for method, _ in peer.requests) == 1


USAGE_LIMIT = {"message": "You've hit your usage limit.", "codexErrorInfo": "usageLimitExceeded"}


class QuotaPeer:
    """Fails turn-1 on the usage limit, serves scripted rate-limit reads, then
    completes turn-2: client-started in turn mode, native continuation after
    the goal is reactivated in goal mode."""

    def __init__(self, reads, *, goal=False):
        self.reads = list(reads)
        self.goal = goal
        self.requests = []
        self.queue = asyncio.Queue()
        self.stopped = False
        self.finished = False

    async def start(self):
        pass

    def send(self, msg):
        self.queue.put_nowait(json.dumps(msg))

    def finish_turn(self, turn, *, failed):
        if not failed:
            self.send({"method": "turn/started", "params": {"threadId": "root", "turn": {"id": turn}}})
            self.send({"method": "item/completed", "params": {"threadId": "root", "turnId": turn, "item": {
                "type": "agentMessage", "id": f"{turn}-final", "phase": "final_answer", "text": "42"}}})
            self.finished = True
        self.send({"method": "turn/completed", "params": {"threadId": "root", "turn": {
            "id": turn, "status": "failed" if failed else "completed", "error": USAGE_LIMIT if failed else None}}})

    async def write_line(self, line):
        msg = json.loads(line)
        if "id" not in msg or "method" not in msg:
            return
        method, params = msg["method"], msg.get("params", {})
        self.requests.append((method, params))
        starts = sum(m == "turn/start" for m, _ in self.requests)
        responses = {
            "initialize": {},
            "account/read": {"account": {"type": "chatgpt"}, "requiresOpenaiAuth": True},
            "model/list": {"data": []},
            "thread/start": {"thread": {"id": "root"}},
            "thread/inject_items": {},
            "turn/start": {"turn": {"id": f"turn-{starts}"}},
            "thread/goal/set": {"goal": {"status": params.get("status")}},
            "thread/goal/get": {"goal": {"status": "complete" if self.finished else "active"}},
        }
        if method == "account/rateLimits/read":
            result = self.reads.pop(0)
        else:
            assert method in responses, f"Unexpected RPC: {method}"
            result = responses[method]
        self.send({"id": msg["id"], "result": result})
        if method == "turn/start":
            self.finish_turn(f"turn-{starts}", failed=starts == 1) if not self.goal else None
        elif method == "thread/goal/set" and self.goal:
            # Objective setup follows turn-1's start; reactivation follows the hold.
            self.finish_turn("turn-1", failed=True) if "objective" in params else self.finish_turn("turn-2", failed=False)

    async def lines(self):
        while (line := await self.queue.get()) is not None:
            yield line

    async def stop(self):
        self.stopped = True
        self.queue.put_nowait(None)


@pytest.mark.parametrize("completion", ["turn", "goal"])
def test_quota_hold_resumes_when_usage_recovers(tmp_path, monkeypatch, completion):
    peer = QuotaPeer([{"ordinaryUsageAllowed": False}] * 2 + [{"ordinaryUsageAllowed": True}],
                     goal=completion == "goal")
    resumed = []

    async def budget_resumed(env, budget):
        resumed.append(budget.held_sec)

    sample = run_evaluation(tmp_path, monkeypatch, peer, policy=RunPolicy(
        completion=completion, quota_hold_sec=30.0, quota_poll_sec=0.2, continuation="Resume.",
        budget_resumed=budget_resumed,
    ))
    assert sample.error is None
    assert sample.output.completion == "42"
    assert next(iter(sample.scores.values())).value == "C"
    hold, = sample.metadata["codex_quota_holds"]
    assert hold["outcome"] == "usage_allowed" and hold["probe_errors"] == []
    # Two failed polls at 0.2s precede recovery; Inspect excludes the hold from working time.
    assert hold["held_sec"] >= 0.4
    assert sample.total_time - sample.working_time >= hold["held_sec"] - 0.01
    # Task-owned budget views are refreshed once, after the whole hold is excluded.
    assert resumed == [pytest.approx(hold["held_sec"])]
    phases = [event.data["phase"] for event in sample.events
              if event.event == "info" and event.source == "codex-quota-hold"]
    assert phases == ["started", "ended"]
    turn_texts = [params["input"][0]["text"] for method, params in peer.requests if method == "turn/start"]
    if completion == "goal":
        # Native continuation owns the resumed turn; the client only reactivates the goal.
        assert len(turn_texts) == 1
        assert ("thread/goal/set", {"threadId": "root", "status": "active"}) in peer.requests
    else:
        assert turn_texts[1:] == ["Resume."]


def test_exhausted_quota_hold_is_an_error_without_retry(tmp_path, monkeypatch):
    peer = QuotaPeer([{"ordinaryUsageAllowed": False}] * 20)
    sample = run_evaluation(tmp_path, monkeypatch, peer, policy=RunPolicy(
        quota_hold_sec=0.5, quota_poll_sec=0.2,
    ))
    assert sample.error and "[quota]" in sample.error.message
    assert "did not recover" in sample.error.message
    assert not sample.scores
    hold, = sample.metadata["codex_quota_holds"]
    assert hold["outcome"] == "exhausted" and hold["held_sec"] >= 0.5
    assert sum(method == "turn/start" for method, _ in peer.requests) == 1
