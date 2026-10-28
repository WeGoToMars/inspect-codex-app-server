"""Inspect agent composition, tool governance, turn control, and checkpoints.

Scripted app-server replies drive real Inspect evaluations to verify agent
integration, operator interventions, submission attempts, and restored state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import asynccontextmanager

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import AgentAttempts, AgentState, as_solver, as_tool
from inspect_ai.agent._channel import Cancel, UserMessage
from inspect_ai.approval import ApprovalPolicy, auto_approver
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentAudio,
    ContentReasoning,
    ContentText,
)
from inspect_ai.scorer import match
from inspect_ai.solver import Generate, TaskState, chain, solver
from inspect_ai.tool import Tool, ToolCall, tool
from inspect_ai.util import SandboxEnvironment

from inspect_codex_app_server import CodexSubagent, codex_agent, compat
from inspect_codex_app_server.policy import RunPolicy

from .support import ScriptedAppServer, configure_common, patch_session, usage_params


class ApprovingServer(ScriptedAppServer):
    """Asks for one native command approval, then completes the turn with an answer."""

    def __init__(self) -> None:
        super().__init__()
        self.approval_replies: list[dict] = []
        configure_common(self)
        self.responses["model/list"] = {"data": []}
        self.hooks["turn/start"] = self.on_turn_start

    def on_turn_start(self, server: ScriptedAppServer, params: dict) -> None:
        if self.ask_approval:
            self.server_request(900, "item/commandExecution/requestApproval", {
                "threadId": "root-thread", "turnId": "turn-1", "itemId": "cmd-1",
                "startedAtMs": 0, "command": "rm -rf build",
            })
        else:
            self.finish()

    ask_approval = False

    def finish(self) -> None:
        if self.approval_replies and self.approval_replies[-1] == {"decision": "cancel"}:
            # Native cancel interrupts the turn; there is no answer.
            self.notify("turn/completed", {"threadId": "root-thread", "turn": {"id": "turn-1", "status": "interrupted"}})
            return
        self.notify("item/completed", {"threadId": "root-thread", "turnId": "turn-1", "item": {
            "type": "agentMessage", "id": "final", "phase": "final_answer", "text": "42"}})
        self.notify("turn/completed", {"threadId": "root-thread", "turn": {"id": "turn-1", "status": "completed"}})

    async def write_line(self, line: str) -> None:
        msg = json.loads(line)
        if "method" not in msg and msg.get("id") == 900:
            self.approval_replies.append(msg.get("result") or msg)
            self.finish()
            return
        await super().write_line(line)


def run_task(tmp_path, monkeypatch, server, task_solver, **eval_options):
    server.sandbox_calls = []

    class Sandbox:
        async def exec(self, *args, **kwargs):
            from types import SimpleNamespace
            server.sandbox_calls.append(("exec", args[0]))
            # Shell steps (home lookup, session mirroring) succeed; probes find nothing.
            if args[0][0] == "sh":
                return SimpleNamespace(returncode=0, stdout="/home/agent", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        async def write_file(self, file, contents):
            server.sandbox_calls.append(("write_file", file))

    SandboxEnvironment.register(Sandbox)  # Accepted where Inspect requires an environment.

    monkeypatch.setenv("CODEX_NATIVE_LOG_DIR", str(tmp_path / "native"))
    patch_session(monkeypatch, server, Sandbox(), lambda kwargs: setattr(server, "command", kwargs["command"]))
    task = Task(dataset=[Sample(input="What is 19 + 23?", target="42")], solver=task_solver, scorer=match())
    log, = eval(task, model="codex/default", log_dir=str(tmp_path / "logs"), display="none", **eval_options)
    sample, = read_eval_log(log.location).samples
    return sample


def requests(server, method):
    return [params for name, params in server.requests if name == method]


def test_agent_as_solver_uses_inspect_agent_span(tmp_path, monkeypatch):
    server = ApprovingServer()
    sample = run_task(tmp_path, monkeypatch, server,
                      as_solver(codex_agent(name="coder", prompt="Answer tersely.")))
    assert sample.error is None
    assert sample.output.completion == "42"
    assert next(iter(sample.scores.values())).value == "C"
    assert requests(server, "thread/start")[0]["developerInstructions"] == "Answer tersely."
    agent_spans = [e for e in sample.events if e.event == "span_begin" and e.type == "agent"]
    # The Codex run lives in the span Inspect opened for the agent; no nested "codex" span.
    assert [e.name for e in agent_spans] == ["coder"]
    model_events = [e for e in sample.events if e.event == "model"]
    assert model_events and all(e.span_id == agent_spans[0].id for e in model_events)


def test_agent_as_tool_returns_codex_answer(tmp_path, monkeypatch):
    server = ApprovingServer()
    delegate = as_tool(codex_agent(name="coder", description="Solves arithmetic.", tools=[LookupSource()]))
    original_write = server.write_line
    returned = []

    def request_lookup(srv, params):
        srv.server_request(901, "item/tool/call", {
            "threadId": "root-thread", "turnId": "turn-1", "callId": "lookup-call",
            "tool": "lookup", "arguments": {"key": "LOOKUP_EXECUTED"},
        })

    async def finish_lookup(line):
        message = json.loads(line)
        if "method" not in message and message.get("id") == 901:
            returned.append(message["result"])
            server.finish()
        else:
            await original_write(line)

    server.hooks["turn/start"] = request_lookup
    server.write_line = finish_lookup

    @solver
    def delegating():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.output.completion = str(await delegate(input="What is 19 + 23?"))
            return state
        return solve

    sample = run_task(tmp_path, monkeypatch, server, delegating())
    assert sample.error is None
    assert sample.output.completion == "42"
    assert requests(server, "turn/start")[0]["input"][0]["text"] == "What is 19 + 23?"
    assert returned == [{"success": True, "contentItems": [{"type": "inputText", "text": "LOOKUP_EXECUTED"}]}]
    executed, = [e for e in sample.events if e.event == "tool" and e.function == "lookup"]
    assert executed.result == "LOOKUP_EXECUTED"


@tool
def lookup() -> Tool:
    async def execute(key: str) -> str:
        """Look up a value.

        Args:
          key: Key to look up.
        """
        return key

    return execute


class LookupSource:
    """A ToolSource (as MCP servers are) resolved when the agent runs."""

    async def tools(self) -> list[Tool]:
        return [lookup()]


@pytest.mark.parametrize("decision, native", [("approve", "accept"), ("reject", "decline")])
def test_inspect_approval_decides_native_commands(tmp_path, monkeypatch, decision, native):
    server = ApprovingServer()
    server.ask_approval = True
    policy = [ApprovalPolicy(auto_approver(decision), "*")]
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(approval=policy)))
    assert sample.error is None
    assert requests(server, "thread/start")[0]["approvalPolicy"] == "untrusted"
    assert server.approval_replies == [{"decision": native}]
    approval, = [e for e in sample.events if e.event == "approval"]
    assert (approval.decision, approval.call.function, approval.call.arguments) == (
        decision, "bash", {"command": "rm -rf build"})


def test_inspect_approval_terminate_ends_sample(tmp_path, monkeypatch):
    server = ApprovingServer()
    server.ask_approval = True
    policy = [ApprovalPolicy(auto_approver("terminate"), "*")]
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(approval=policy)))
    assert server.approval_replies == [{"decision": "cancel"}]
    assert sample.error is None, "Termination ends the sample for scoring, as for bridged tools"
    assert [e.type for e in sample.events if e.event == "sample_limit"] == ["operator"]
    assert sample.output.completion != "42"


def test_without_approval_policy_native_actions_stay_unattended(tmp_path, monkeypatch):
    server = ApprovingServer()
    server.ask_approval = True
    run_task(tmp_path, monkeypatch, server, as_solver(codex_agent()))
    assert requests(server, "thread/start")[0]["approvalPolicy"] == "never"
    assert server.approval_replies == [{"decision": "accept"}]


def test_generate_config_maps_to_native_turns(tmp_path, monkeypatch):
    from inspect_ai.model import ResponseSchema
    from inspect_ai.util import JSONSchema

    server = ApprovingServer()
    server.responses["model/list"] = {"data": [{"id": "test", "isDefault": True, "supportedReasoningEfforts": [
        {"reasoningEffort": "high"}], "serviceTiers": [{"id": "priority", "name": "Fast", "description": ""}]}]}
    schema = ResponseSchema(name="answer", json_schema=JSONSchema(type="object"))
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(service_tier="priority")),
                      reasoning_effort="high", reasoning_summary="concise", response_schema=schema, temperature=0.2)
    assert sample.error is None
    turn, = requests(server, "turn/start")
    assert (turn["effort"], turn["outputSchema"], turn["serviceTier"]) == ("high", {"type": "object"}, "priority")
    assert sample.metadata["codex_run"]["reasoning_summary"] == "concise"
    assert "generate_config_not_applied:temperature" in sample.metadata["codex_unsupported_settings"]


# Parametrize reasoning effort: "high" (valid), "low" (unsupported), None (model missing from catalog).
@pytest.mark.parametrize("listed_effort", ["high", "low", None])
def test_catalog_validation_uses_the_model_selected_by_the_server(tmp_path, monkeypatch, listed_effort):
    server = ApprovingServer()
    server.responses["thread/start"] = {
        "thread": {"id": "root-thread"}, "model": "configured-model", "modelProvider": "openai"}
    server.responses["model/list"] = {"data": [
        {"id": "catalog-default", "isDefault": True,
         "supportedReasoningEfforts": [{"reasoningEffort": "low"}]},
        *([{"id": "configured-model", "isDefault": False,
            "supportedReasoningEfforts": [{"reasoningEffort": listed_effort}]}] if listed_effort else []),
    ]}
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(
        reasoning_effort="high", subagents=[CodexSubagent(
            name="reviewer", description="Reviews answers.", reasoning_effort="high")],
    )))
    assert requests(server, "model/list")[0]["includeHidden"] is True
    if listed_effort != "low":
        assert sample.error is None
        assert sample.output.model == "configured-model"
        assert sample.output.completion == "42"
    else:
        assert sample.error is not None and "not supported by configured-model" in sample.error.message
        assert not requests(server, "turn/start")


@pytest.mark.parametrize("catalog, message", [
    ({"data": [{"id": "test", "isDefault": True, "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]},
     "Reasoning effort 'high' is not supported"),
    ({"data": [{"id": "other", "isDefault": False}]}, "Model 'gpt-x' is not in the native model catalog"),
    ({"data": [{"id": "test", "isDefault": True, "serviceTiers": [{"id": "priority", "name": "Fast", "description": ""}]}]},
     "Service tier 'flex' is not offered for test; offered: priority"),
])
def test_unsupported_model_settings_fail_before_inference(tmp_path, monkeypatch, catalog, message):
    server = ApprovingServer()
    server.responses["model/list"] = catalog
    agent = (codex_agent(model="gpt-x") if "gpt-x" in message else codex_agent(service_tier="flex")
             if "flex" in message else codex_agent(reasoning_effort="high"))
    sample = run_task(tmp_path, monkeypatch, server, as_solver(agent))
    assert sample.error is not None and message in sample.error.message
    assert not requests(server, "turn/start")


AUDIO_DATA = "data:audio/wav;base64,UklGRg=="


@pytest.mark.parametrize("modalities, audio, error", [
    (["text", "audio"], None, None),
    (["text"], None, "Model 'test' does not accept audio input in the native model catalog"),
    (["text", "audio"], "https://example.com/clip.wav", "Codex does not fetch audio URLs"),
])
def test_audio_input_reaches_codex_when_the_model_accepts_it(tmp_path, monkeypatch, modalities, audio, error):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    server = ApprovingServer()
    server.responses["model/list"] = {"data": [{"id": "test", "isDefault": True, "inputModalities": modalities}]}
    sources = [audio] if audio else [AUDIO_DATA, str(clip)]

    @solver
    def attach_audio():
        # Sample media are already data URLs; a solver can add paths and URLs.
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.messages[0].content = [ContentText(text=state.messages[0].text),
                                         *[ContentAudio(audio=source, format="wav") for source in sources]]
            return state
        return solve

    sample = run_task(tmp_path, monkeypatch, server, chain(attach_audio(), as_solver(codex_agent())))
    if error:
        assert sample.error is not None and error in sample.error.message
        assert not requests(server, "turn/start")
        return
    assert sample.error is None
    _, data_audio, local_audio = requests(server, "turn/start")[0]["input"]
    assert data_audio == {"type": "audio", "url": AUDIO_DATA}
    assert local_audio["type"] == "localAudio" and ("write_file", local_audio["path"]) in server.sandbox_calls


CALL = ToolCall(id="call-1", function="lookup", arguments={"key": 1})
PRIOR = [ChatMessageSystem(content="Be brief."), ChatMessageUser(content="First question."),
         ChatMessageAssistant(content=[ContentReasoning(reasoning="thinking"), ContentText(text="Looking.")],
                              tool_calls=[CALL]),
         ChatMessageTool(content="looked up", tool_call_id="call-1", function="lookup")]
REPLAYED = [
    {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Be brief."}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "First question."}]},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Looking."}]},
    {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": '{"key": 1}'},
    {"type": "function_call_output", "call_id": "call-1", "output": "looked up"},
]


@pytest.mark.parametrize("conversation, replayed, turn_text, error", [
    ([*PRIOR, ChatMessageUser(content="Second question.")], REPLAYED, "Second question.", None),
    ([*PRIOR, ChatMessageAssistant(content="Found it.")], [*REPLAYED, {
        "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Found it."}]}],
     RunPolicy().continuation, None),
    ([ChatMessageUser(content=[ContentAudio(audio=AUDIO_DATA, format="wav")]), ChatMessageUser(content="Now?")],
     None, None, "Codex cannot replay audio content from an earlier user message"),
])
def test_earlier_messages_are_replayed_as_native_history(tmp_path, monkeypatch, conversation, replayed, turn_text,
                                                         error):
    @solver
    def converse():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.messages = list(conversation)
            return state
        return solve

    server = ApprovingServer()
    sample = run_task(tmp_path, monkeypatch, server, chain(converse(), as_solver(codex_agent())))
    if error:
        assert sample.error is not None and error in sample.error.message
        assert not requests(server, "thread/inject_items") and not requests(server, "turn/start")
        return
    assert sample.error is None
    assert requests(server, "thread/inject_items") == [{"threadId": "root-thread", "items": replayed}]
    assert requests(server, "turn/start")[0]["input"] == [{"type": "text", "text": turn_text, "text_elements": []}]
    assert sample.metadata["codex_run"]["history_reasoning_omitted"] == 1
    # The transcript keeps the replayed conversation ahead of Codex's turn.
    assert [m.role for m in sample.messages[:len(conversation)]] == [m.role for m in conversation]


class ScriptedModelServer(ScriptedAppServer):
    """Plays one scripted model action per root turn.

    ``("submit", answer)`` calls the bridged submit tool, ``("answer", text)``
    ends the turn with a final answer and ``("fail", tag)`` fails the turn with
    that native ``codexErrorInfo``.
    """

    def __init__(self, *script: tuple[str, str]) -> None:
        super().__init__()
        configure_common(self)
        self.responses["model/list"] = {"data": []}
        self.script = list(script)
        self.turns = 0
        self.interrupted: set[tuple[str, str]] = set()
        self.responses["turn/start"] = self.start_turn
        self.responses["turn/steer"] = {"error": {"code": -32600, "message": "no active turn to steer"}}
        self.hooks["turn/start"] = self.play
        self.hooks["turn/interrupt"] = self.interrupt
        self.responses["thread/read"] = self.read_thread
        self.responses["thread/resume"] = {"thread": {"id": "root-thread"}}

    child_role: str | None = None

    def read_thread(self, params: dict) -> dict:
        if params["threadId"] != "child-thread":
            return {"thread": {"id": params["threadId"]}}
        return {"thread": {"id": "child-thread", "agentRole": self.child_role, "parentThreadId": "root-thread"}}

    def start_turn(self, params: dict) -> dict:
        self.turns += 1
        return {"turn": {"id": f"turn-{self.turns}"}}

    def play(self, server: ScriptedAppServer, params: dict) -> None:
        action, value = self.script.pop(0)
        turn = f"turn-{self.turns}"
        if action == "plan_and_search":
            self.notify("turn/plan/updated", {"threadId": "root-thread", "turnId": turn, "explanation": "Two steps.",
                                              "plan": [{"step": "Search", "status": "inProgress"},
                                                       {"step": "Answer", "status": "pending"}]})
            self.notify("item/completed", {"threadId": "root-thread", "turnId": turn, "item": {
                "type": "webSearch", "id": "search-1", "query": "19 + 23",
                "action": {"type": "search", "query": "19 + 23"}}})
            self.play(server, params)
        elif action == "hang":
            # Root and child turns keep working until interrupted.
            self.notify("turn/started", {"threadId": "root-thread", "turn": {"id": turn, "status": "inProgress"}})
            self.notify("turn/started", {"threadId": "child-thread", "turn": {"id": "child-turn",
                                                                              "status": "inProgress"}})
        elif action in ("spawn", "spawn_complete"):
            # A child with the given role starts, then reports its token usage.
            self.child_role = value
            self.notify("item/completed", {"threadId": "root-thread", "turnId": turn, "item": {
                "type": "subAgentActivity", "id": "spawn-1", "kind": "started", "agentThreadId": "child-thread",
                "agentPath": "/root/review"}})
            self.notify("turn/started", {"threadId": "child-thread", "turn": {
                "id": "child-turn", "status": "inProgress"}})
            # These reports can arrive before the asynchronous thread/read
            # response supplies the role. No delay should be needed for enforcement.
            for total in (60, 150):
                self.notify("thread/tokenUsage/updated", usage_params("child-thread", total))
            if action == "spawn_complete":
                self.notify("turn/completed", {"threadId": "child-thread", "turn": {
                    "id": "child-turn", "status": "completed"}})
                self.play_answer(turn, "42")
        elif action == "post":
            self.operator(UserMessage(ChatMessageUser(content=value, source="operator")))
            self.play_answer(turn, "41")
        elif action == "submit":
            self.server_request(800 + self.turns, "item/tool/call", {
                "threadId": "root-thread", "turnId": turn, "callId": f"submit-{self.turns}",
                "tool": "submit", "arguments": {"answer": value}})
        elif action == "answer":
            self.play_answer(turn, value)
        else:
            self.complete(turn, error={"message": f"{value} stopped the turn", "codexErrorInfo": value})

    def play_answer(self, turn: str, text: str) -> None:
        self.notify("item/completed", {"threadId": "root-thread", "turnId": turn, "item": {
            "type": "agentMessage", "id": f"answer-{self.turns}", "phase": "final_answer", "text": text}})
        self.complete(turn)

    def interrupt(self, server: ScriptedAppServer, params: dict) -> None:
        self.interrupted.add((params["threadId"], params["turnId"]))
        self.notify("turn/completed", {"threadId": params["threadId"], "turn": {
            "id": params["turnId"], "status": "interrupted"}})
        if self.child_role and params["threadId"] == "child-thread":
            self.play_answer(f"turn-{self.turns}", "42")

    operator = None  # Set by tests acting as the operator on the run's agent channel.

    def complete(self, turn: str, error: dict | None = None) -> None:
        self.notify("turn/completed", {"threadId": "root-thread", "turn": {
            "id": turn, "status": "failed" if error else "completed", "error": error}})

    async def write_line(self, line: str) -> None:
        msg = json.loads(line)
        if "method" not in msg and str(msg.get("id", "")).startswith("8"):
            turn = f"turn-{msg['id'] - 800}"
            result = msg["result"]
            self.notify("item/completed", {"threadId": "root-thread", "turnId": turn, "item": {
                "type": "dynamicToolCall", "id": f"submit-{msg['id'] - 800}", "tool": "submit",
                "arguments": {}, "success": result["success"], "contentItems": result["contentItems"]}})
            self.complete(turn)
            return
        await super().write_line(line)


def turn_inputs(server):
    return [params["input"][0]["text"] for params in requests(server, "turn/start")]


def test_incorrect_submission_is_scored_and_steered_onward(tmp_path, monkeypatch):
    server = ScriptedModelServer(("submit", "41"), ("submit", "42"))
    notify, play, receipts = server.notify, server.play, []

    def record(method, params):
        if params.get("item", {}).get("type") == "dynamicToolCall":
            receipts.append(params)
        notify(method, params)

    def redeliver_then_play(srv, params):  # The withdrawn attempt's receipt arrives again.
        if receipts:
            notify("item/completed", receipts[0])
        play(srv, params)

    server.notify = record
    server.hooks["turn/start"] = redeliver_then_play
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(submit=True, attempts=2)))
    assert sample.error is None
    assert sample.output.completion == "42"
    assert next(iter(sample.scores.values())).value == "C"
    thread, = requests(server, "thread/start")
    assert "submit" in [t["name"] for t in thread["dynamicTools"]]
    assert "call the submit() tool" in thread["developerInstructions"]
    assert turn_inputs(server)[1] == AgentAttempts().incorrect_message
    attempt_scores = [e for e in sample.events if e.event == "score" and e.intermediate]
    assert [e.score.value for e in attempt_scores] == ["I"], "The incorrect attempt is scored in the transcript"
    # As with react(), submit calls do not remain in the conversation.
    assert not [m for m in sample.messages if m.role == "tool" and m.function == "submit"]


def test_exhausted_attempts_keep_the_last_submission(tmp_path, monkeypatch):
    server = ScriptedModelServer(("submit", "40"), ("submit", "41"))
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(submit=True, attempts=2)))
    assert sample.error is None
    assert sample.output.completion == "41"
    assert next(iter(sample.scores.values())).value == "I"
    assert len(requests(server, "turn/start")) == 2


def test_on_continue_decides_after_each_turn(tmp_path, monkeypatch):
    seen: list[str] = []

    async def on_continue(state: AgentState) -> bool | str:
        seen.append(state.output.completion)
        return "Check the arithmetic." if len(seen) == 1 else False

    server = ScriptedModelServer(("answer", "41"), ("answer", "42"))
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(on_continue=on_continue)))
    assert sample.error is None
    assert seen == ["41", "42"]
    assert turn_inputs(server) == ["What is 19 + 23?", "Check the arithmetic."]
    assert sample.output.completion == "42"
    assert not requests(server, "thread/goal/set"), "on_continue replaces native goal continuation"


@pytest.mark.parametrize("script, retries, completion, stop_reason", [
    ([("fail", "cyberPolicy")], None, "", "content_filter"),
    ([("fail", "cyberPolicy"), ("answer", "42")], 1, "42", "stop"),
    ([("fail", "contextWindowExceeded")], 3, "", "model_length"),
])
def test_refusals_and_context_exhaustion_finish_normally(tmp_path, monkeypatch, script, retries, completion,
                                                         stop_reason):
    server = ScriptedModelServer(*script)
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(retry_refusals=retries)))
    assert sample.error is None
    assert sample.output.completion == completion
    assert sample.output.choices[0].stop_reason == stop_reason
    assert len(requests(server, "turn/start")) == len(script)
    assert ("codex_policy_block" in sample.metadata) == (stop_reason == "content_filter")


@pytest.mark.parametrize("options, message", [
    ({"attempts": 2}, "attempts require a submit tool"),
    ({"on_continue": "Keep going."}, "no submit tool is not permitted"),
    ({"submit": True, "policy": RunPolicy(completion="goal")}, "submit sets the completion policy"),
])
def test_conflicting_agent_options_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        codex_agent(**options)


def test_skills_web_search_and_native_tools_in_inspect_shapes(tmp_path, monkeypatch):
    from inspect_ai.tool import Skill

    server = ScriptedModelServer(("plan_and_search", ""), ("answer", "42"))
    skill = Skill(name="unit-convert", description="Converts units.", instructions="Convert.")
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(skills=[skill], web_search="live")))
    assert sample.error is None
    # Installed fresh into the Codex home Codex runs with, where it discovers user skills.
    assert ("exec", ["rm", "-rf", "/isolated-codex-home/skills"]) in server.sandbox_calls
    assert ("write_file", "/isolated-codex-home/skills/unit-convert/SKILL.md") in server.sandbox_calls
    assert 'web_search="live"' in server.command
    assert (sample.metadata["codex_run"]["skills"], sample.metadata["codex_run"]["web_search"]) == (
        ["unit-convert"], "live")
    calls = {e.function: e.arguments for e in sample.events if e.event == "tool"}
    # The shapes Inspect's viewer renders for its own update_plan and web_search tools.
    assert calls["update_plan"] == {"explanation": "Two steps.", "plan": [
        {"step": "Search", "status": "in_progress"}, {"step": "Answer", "status": "pending"}]}
    assert calls["web_search"] == {"query": "19 + 23"}


def test_invalid_skills_fail_when_the_agent_is_created(tmp_path):
    with pytest.raises(Exception, match="SKILL.md"):
        codex_agent(skills=[tmp_path])


def attach_operator(monkeypatch, server, on_turn_started=None):
    """Act as the operator (as ACP does) on the agent channel each run opens."""
    import inspect_codex_app_server.session as implementation

    opened = implementation.agent_channel

    @asynccontextmanager
    async def channel():
        async with opened() as ch:
            server.operator = ch._post
            if on_turn_started is not None:
                ch.subscribe_turn_state(lambda state: state == "started" and on_turn_started(ch))
            yield ch

    monkeypatch.setattr(implementation, "agent_channel", channel)


def test_operator_interrupt_pauses_native_work_until_follow_up(tmp_path, monkeypatch):
    server = ScriptedModelServer(("hang", ""), ("submit", "42"))
    interrupted = []

    def interrupt_once(ch):
        if not interrupted:
            interrupted.append(True)
            ch._interrupt(Cancel())
            follow_up = ChatMessageUser(content="Use base 10.", source="operator")
            # The redirect arrives after the native deadline would expire.
            asyncio.get_running_loop().call_later(1.25, ch._post, UserMessage(follow_up))

    attach_operator(monkeypatch, server, interrupt_once)
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(submit=True, timeout_sec=1.0)),
                      working_limit=1.0)
    assert sample.error is None
    assert sample.output.completion == "42"
    assert [(r["threadId"], r["turnId"]) for r in requests(server, "turn/interrupt")] == [
        ("root-thread", "turn-1"), ("child-thread", "child-turn")]
    # Paused for the operator, resumed with the follow-up (and paused by cleanup at the end).
    assert [r["status"] for r in requests(server, "thread/goal/set")] == ["active", "paused", "active", "paused"]
    assert turn_inputs(server) == ["What is 19 + 23?", "Use base 10."]
    assert [e.source for e in sample.events if e.event == "interrupt"][:1] == ["user_cancel"]
    intervention, = sample.metadata["codex_operator_interventions"]
    assert (intervention["type"], intervention["message"]) == ("interrupt", "Use base 10.")
    assert intervention["waited_sec"] >= 1.0
    assert sample.total_time - sample.working_time >= 1.0, "Waiting for the operator is not working time"


def test_operator_interrupt_cancels_bridged_tools_before_waiting(tmp_path, monkeypatch):
    server = ScriptedModelServer(("hang", ""), ("answer", "42"))
    channels, cancelled_before_redirect, mutations = [], [], []
    attach_operator(monkeypatch, server, channels.append)

    @tool
    def pending_write() -> Tool:
        async def execute() -> str:
            """Wait before writing a value."""
            cancelled = asyncio.Event()
            release = asyncio.Event()
            channel = channels[0]
            channel._interrupt(Cancel())

            def redirect():
                cancelled_before_redirect.append(cancelled.is_set())
                release.set()
                channel._post(UserMessage(ChatMessageUser(content="Continue.", source="operator")))

            asyncio.get_running_loop().call_later(0.1, redirect)
            try:
                await release.wait()
                mutations.append("unexpected write")
                return "written"
            finally:
                cancelled.set()

        return execute

    play = server.play

    def start(srv, params):
        play(srv, params)
        if server.turns == 1:
            server.server_request(700, "item/tool/call", {
                "threadId": "root-thread", "turnId": "turn-1", "callId": "pending-write",
                "tool": "pending_write", "arguments": {},
            })

    server.hooks["turn/start"] = start
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(tools=[pending_write()])))
    assert sample.error is None
    assert sample.output.completion == "42"
    assert cancelled_before_redirect == [True]
    assert mutations == []


def test_deadline_interrupt_is_not_held_by_a_tool_ignoring_cancellation(tmp_path, monkeypatch):
    server = ScriptedModelServer(("hang", ""))
    release, released_at_interrupt = asyncio.Event(), []

    @tool
    def stuck() -> Tool:
        async def execute() -> str:
            """Keep running when cancelled."""
            asyncio.get_running_loop().call_later(3.0, release.set)
            while not release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await release.wait()
            return "late"

        return execute

    play, interrupt = server.play, server.interrupt

    def start(srv, params):
        play(srv, params)
        server.server_request(700, "item/tool/call", {
            "threadId": "root-thread", "turnId": "turn-1", "callId": "stuck", "tool": "stuck", "arguments": {}})

    def record_interrupt(srv, params):
        released_at_interrupt.append(release.is_set())
        interrupt(srv, params)

    server.hooks["turn/start"] = start
    server.hooks["turn/interrupt"] = record_interrupt
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(tools=[stuck()], timeout_sec=1.0)))
    assert sample.error is None, "The deadline stays a limit, not a transport failure"
    assert [e.type for e in sample.events if e.event == "sample_limit"] == ["time"]
    assert released_at_interrupt and not any(released_at_interrupt)


def test_operator_message_posted_during_a_turn_starts_the_next(tmp_path, monkeypatch):
    server = ScriptedModelServer(("post", "Also show the working."), ("answer", "42"))
    attach_operator(monkeypatch, server)
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent()))
    assert sample.error is None
    assert turn_inputs(server) == ["What is 19 + 23?", "Also show the working."]
    assert sample.output.completion == "42"
    assert [i["type"] for i in sample.metadata["codex_operator_interventions"]] == ["message"]


def test_sample_interrupt_tears_down_the_native_session(tmp_path, monkeypatch):
    from inspect_ai.log._samples import sample_active

    server = ScriptedModelServer(("hang", ""))
    attach_operator(monkeypatch, server, lambda ch: sample_active().interrupt("score"))
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent()))
    assert sample.error is None, "An operator-interrupted sample is still scored"
    assert server.stopped, "The app-server transport is closed"
    assert ("root-thread", "turn-1") in [(r["threadId"], r["turnId"]) for r in requests(server, "turn/interrupt")]


# "retried": the first interrupt fails; the child's next usage report retries it.
@pytest.mark.parametrize("role, interrupted", [("reviewer", True), ("reviewer", "retried"), ("default", False)])
def test_subagent_roles_are_declared_and_their_token_limits_enforced(tmp_path, monkeypatch, role, interrupted):
    server = ScriptedModelServer(("spawn" if interrupted else "spawn_complete", role))
    if interrupted == "retried":
        acknowledge, failed = server.hooks["turn/interrupt"], []

        def fail_first(params):
            if params["threadId"] == "child-thread" and not failed:
                return {"error": {"code": -32603, "message": "interrupt failed"}}
            return {}

        def retry_on_next_report(srv, params):
            if params["threadId"] == "child-thread" and not failed:
                failed.append(params)
                asyncio.get_running_loop().call_later(0.2, srv.notify, "thread/tokenUsage/updated",
                                                      usage_params("child-thread", 200))
                return
            acknowledge(srv, params)

        server.responses["turn/interrupt"] = fail_first
        server.hooks["turn/interrupt"] = retry_on_next_report
    reviewer = CodexSubagent(name="reviewer", description="Reviews changes.", prompt="Be strict.", token_limit=100)
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(subagents=[reviewer], timeout_sec=1.0)))
    assert sample.error is None
    assert not [e for e in sample.events if e.event == "sample_limit"]
    assert sample.output.completion == "42"
    config_file = "/isolated-codex-home/agent-roles/reviewer.toml"
    assert ("write_file", config_file) in server.sandbox_calls
    assert 'agents.reviewer.description="Reviews changes."' in server.command
    assert f'agents.reviewer.config_file="{config_file}"' in server.command
    child_interrupts = [r["turnId"] for r in requests(server, "turn/interrupt") if r["threadId"] == "child-thread"]
    limits = sample.metadata.get("codex_subagent_limits", [])
    failures = [f for f in sample.metadata["codex_native_evidence"]["failures"] if "subagent limit" in f]
    if interrupted == "retried":
        # Recorded only once acknowledged; the failed attempt is reported.
        assert child_interrupts == ["child-turn", "child-turn"]
        assert [(r["role"], r["tokens"]) for r in limits] == [("reviewer", 200)]
        assert len(failures) == 1 and "interrupt failed" in failures[0]
        return
    # Only the role's child is limited, once it reaches 100 tokens (at the 150 report).
    assert child_interrupts == (["child-turn"] if interrupted else [])
    assert [(r["role"], r["tokens"]) for r in limits] == ([("reviewer", 150)] if interrupted else [])
    assert failures == []


def test_subagent_models_are_validated_before_inference(tmp_path, monkeypatch):
    server = ApprovingServer()
    server.responses["model/list"] = {"data": [{"id": "test", "isDefault": True}]}
    role = CodexSubagent(name="reviewer", description="Reviews changes.", model="gpt-x")
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(subagents=[role])))
    assert sample.error is not None
    assert "Model 'gpt-x' for subagent 'reviewer' is not in the native model catalog" in sample.error.message
    assert not requests(server, "turn/start")


class FakeCheckpointer:
    """The sample's checkpoint session, recording ticks and serving restored values."""

    def __init__(self, attempt: str = "initial", restored: dict | None = None) -> None:
        self.attempt = attempt
        self.restored = None
        self.values = restored or {}
        self.callbacks: dict = {}
        self.events: list = []

    def track(self, key, callback, initial_value, *, value_type=None):
        self.callbacks[key] = callback
        return self.values.get(key, initial_value)

    async def tick(self) -> None:
        self.events.append(("tick", json.loads(json.dumps(self.callbacks["codex_session"]()))))


def use_checkpointer(monkeypatch, fake, server):
    import inspect_codex_app_server.compat as compat

    @asynccontextmanager
    async def session():
        yield fake

    monkeypatch.setattr(compat, "sample_checkpointer", session)
    return fake


MIRROR = "/home/agent/.codex-inspect-checkpoint"


@pytest.mark.parametrize("accounting_saved", [True, False])
def test_resumed_sample_continues_the_restored_codex_thread(tmp_path, monkeypatch, accounting_saved):
    from inspect_codex_app_server.session import MIRROR_SESSIONS, RESTORE_SESSIONS

    first = ScriptedModelServer(("answer", "Working on it."))
    play = first.play

    def first_turn(srv, params):
        srv.notify("thread/tokenUsage/updated", usage_params("root-thread", 25))
        play(srv, params)

    first.hooks["turn/start"] = first_turn
    saved = use_checkpointer(monkeypatch, FakeCheckpointer(), first)
    initial = run_task(tmp_path / "initial", monkeypatch, first, as_solver(codex_agent()))
    assert initial.error is None
    assert ("exec", ["sh", "-c", MIRROR_SESSIONS, "sh", "/isolated-codex-home", MIRROR]) in first.sandbox_calls
    session = saved.events[-1][1]
    # Inspect saves the tracked conversation with the session state.
    tracked = saved.callbacks["messages"]()
    assert "Working on it." in [m.text for m in tracked]
    session["goal_ok"] = True
    if not accounting_saved:
        del session["accounting"]

    server = ScriptedModelServer(("submit", "42"))
    play_resumed = server.play

    def resumed_turn(srv, params):
        # Codex reports lifetime usage, including the 25 tokens already checkpointed.
        srv.notify("thread/tokenUsage/updated", usage_params("root-thread", 30))
        play_resumed(srv, params)

    server.hooks["turn/start"] = resumed_turn
    use_checkpointer(monkeypatch, FakeCheckpointer("resume", {
        "codex_session": session, "messages": tracked,
    }), server)

    @solver
    def restore_inspect_usage():
        async def solve(state, generate):
            # Inspect restores these separately from the adapter's tracked state.
            for model, usage in initial.model_usage.items():
                compat.record_usage(model, usage)
                compat.record_limit_usage(usage)
            return state
        return solve

    sample = run_task(tmp_path / "resumed", monkeypatch, server,
                      chain(restore_inspect_usage(), as_solver(codex_agent(submit=True))), token_limit=35)
    if not accounting_saved:
        assert sample.error is not None and "lacks Codex accounting state" in sample.error.message
        assert not server.requests
        return
    assert sample.error is None
    assert ("exec", ["sh", "-c", RESTORE_SESSIONS, "sh", MIRROR, "/isolated-codex-home"]) in server.sandbox_calls
    assert not requests(server, "thread/start")
    assert [r["threadId"] for r in requests(server, "thread/resume")] == ["root-thread"]
    assert requests(server, "thread/goal/set")[0]["status"] == "active"
    assert turn_inputs(server) == [compat.DEFAULT_CONTINUE_PROMPT.strip().replace("{submit}", "submit")]
    assert sample.output.completion == "42"
    assert sample.output.usage.total_tokens == 30
    assert sum(usage.total_tokens for usage in sample.model_usage.values()) == 30
    assert sample.token_limit_usage == 30
    assert "Working on it." in [m.text for m in sample.messages], "The restored conversation is kept"
    assert sample.metadata["codex_run"]["resumed_thread"] == "root-thread"
    assert "raw_events_unavailable_after_resume" in sample.metadata["codex_unsupported_settings"]


def test_resume_for_scoring_returns_the_restored_result_without_codex(tmp_path, monkeypatch):
    from inspect_ai.model import ModelOutput

    server = ScriptedModelServer()
    use_checkpointer(monkeypatch, FakeCheckpointer("resume_for_scoring", {
        "messages": [ChatMessageUser(content="What is 19 + 23?")],
        "output": ModelOutput.from_content("codex", "42"),
    }), server)
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent()))
    assert sample.error is None
    assert not server.requests, "Scoring re-runs without starting Codex"
    assert next(iter(sample.scores.values())).value == "C"


def test_models_of_a_configured_provider_are_not_checked_against_the_openai_catalog(tmp_path, monkeypatch):
    server = ApprovingServer()
    server.responses["model/list"] = {"data": [{"id": "other", "isDefault": True}]}
    server.responses["thread/start"] = {"thread": {"id": "root-thread"}, "modelProvider": "local"}
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(model="gpt-x")))
    assert sample.error is None
    assert sample.output.completion == "42"
    assert "model_catalog_not_applicable:local" in sample.metadata["codex_unsupported_settings"]


def test_configuration_warnings_fail_before_inference(tmp_path, monkeypatch):
    server = ApprovingServer()
    server.hooks["initialize"] = lambda srv, params: srv.notify("configWarning", {
        "summary": "Codex is ignoring 1 unrecognized configuration setting: `sandbox` is ignored."})
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent()))
    assert sample.error is not None and "Codex configuration warning" in sample.error.message
    assert "`sandbox` is ignored" in sample.error.message
    assert not requests(server, "turn/start")


def test_runtime_warnings_are_recorded(tmp_path, monkeypatch):
    server = ApprovingServer()
    original = server.on_turn_start

    def warn_then_answer(srv, params):
        srv.notify("warning", {"threadId": "root-thread", "message": "Model metadata for `x` not found."})
        for retry in (True, False):
            srv.notify("error", {"threadId": "root-thread", "willRetry": retry, "error": {"message": "Reconnecting..."}})
        original(srv, params)

    server.hooks["turn/start"] = warn_then_answer
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent()))
    assert sample.error is None
    # Only errors Codex retries are warnings; a terminal one ends the turn.
    assert sample.metadata["codex_warnings"] == [
        {"type": "warning", "threadId": "root-thread", "message": "Model metadata for `x` not found."},
        {"type": "error", "threadId": "root-thread", "willRetry": True, "error": {"message": "Reconnecting..."}}]
    assert len([e for e in sample.events if e.event == "info" and e.source == "codex-warning"]) == 2


# A broken skill the workspace defines is not the run's and must not fail it.
@pytest.mark.parametrize("path", ["/isolated-codex-home/skills/unit-convert/SKILL.md",
                                  "/app/.codex/skills/unit-convert/SKILL.md"])
def test_skills_codex_cannot_load_fail_before_inference(tmp_path, monkeypatch, path):
    from inspect_ai.tool import Skill

    server = ApprovingServer()
    server.responses["skills/list"] = {"data": [{"cwd": "/app", "skills": [], "errors": [
        {"path": path, "message": "invalid YAML"}]}]}
    skill = Skill(name="unit-convert", description="Converts units.", instructions="Convert.")
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent(skills=[skill])))
    if path.startswith("/app/"):
        assert sample.error is None and requests(server, "turn/start")
    else:
        assert sample.error is not None and "could not load skill" in sample.error.message
        assert "invalid YAML" in sample.error.message and not requests(server, "turn/start")


# Only the root's live session loses tools; a child's or a late failure is a warning.
@pytest.mark.parametrize("failed_in", ["root", "child", "cleanup"])
def test_mcp_server_that_fails_to_start_fails_the_run(tmp_path, monkeypatch, failed_in):
    server = ApprovingServer()

    def fail_mcp(srv, thread):
        srv.notify("mcpServer/startupStatus/updated", {"threadId": thread, "name": "docs", "status": "failed",
                                                       "error": "MCP client for `docs` failed to start"})

    if failed_in == "root":  # The turn stays open until the failure ends the run.
        server.hooks["turn/start"] = lambda srv, params: fail_mcp(srv, "root-thread")
    elif failed_in == "child":
        def child_fails(srv, params):
            fail_mcp(srv, "child-thread")
            server.finish()

        server.hooks["turn/start"] = child_fails
    else:
        stop = server.stop

        async def fail_then_stop():  # Delivered while the session closes.
            fail_mcp(server, "root-thread")
            await stop()

        server.stop = fail_then_stop
    sample = run_task(tmp_path, monkeypatch, server, as_solver(codex_agent()))
    if failed_in == "root":
        assert sample.error is not None and "MCP client for `docs` failed to start" in sample.error.message
        return
    assert sample.error is None and sample.output.completion == "42"
    warning, = [w for w in sample.metadata["codex_warnings"] if w["type"] == "mcpServer/startupStatus/updated"]
    assert warning["threadId"] == f"{failed_in if failed_in == 'child' else 'root'}-thread"
