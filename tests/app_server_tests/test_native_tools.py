"""Pinned Codex tool approvals, interruption, and the Inspect tool bridge.

Local scripted model responses drive real commands and bridged tools.
Verify declined commands stay unexecuted, interruption permits redirect,
and Inspect approval modifications, termination, and audio are preserved.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from dataclasses import replace

import pytest
from aiohttp import web
from inspect_ai import Task, eval_async
from inspect_ai.approval import Approval, ApprovalPolicy, approver
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ContentAudio, ContentText
from inspect_ai.solver import solver
from inspect_ai.tool import tool

from inspect_codex_app_server import compat
from inspect_codex_app_server.client import CodexClient, ExecRemoteJsonRpcTransport
from inspect_codex_app_server.mapping import EventStreamMapper
from inspect_codex_app_server.session import INSPECT_APPROVAL_POLICY
from inspect_codex_app_server.tools import inspect_tool_handler

from .support import LocalSandbox, responses_peer


@pytest.mark.native
@pytest.mark.parametrize("policy, asked", [(INSPECT_APPROVAL_POLICY, True), ("never", False)])
async def test_native_approval_policy_gates_commands(evaluation_workspace, native_binary, policy, asked):
    count = 0

    async def respond(request):
        nonlocal count
        await request.json()
        count += 1
        item = (
            {"type": "function_call", "call_id": "c1", "name": "exec_command",
             "arguments": json.dumps({"cmd": "touch made-by-command"})}
            if count == 1 else
            {"type": "message", "role": "assistant", "id": f"m{count}", "phase": "final_answer",
             "content": [{"type": "output_text", "text": "done"}]}
        )
        events = [
            {"type": "response.created", "response": {"id": f"r{count}"}},
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": {"id": f"r{count}", "usage": {
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                "input_tokens_details": None, "output_tokens_details": None}}},
        ]
        return web.Response(text="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
                            content_type="text/event-stream")

    async with responses_peer(respond) as base_url:
        home = evaluation_workspace / "native-home"
        home.mkdir()
        (home / "config.toml").write_text(
            f'model="gpt-5.4"\nmodel_provider="scripted"\n[model_providers.scripted]\nname="scripted"\n'
            f'base_url="{base_url}"\nwire_api="responses"\nsupports_websockets=false\n'
            'requires_openai_auth=false\nrequest_max_retries=0\nstream_max_retries=0\n'
        )
        native = ExecRemoteJsonRpcTransport(LocalSandbox(), [native_binary, "app-server", "--stdio"],
                                            cwd=str(home), environment={"CODEX_HOME": str(home)})
        client = CodexClient(transport_factory=lambda: native)
        requested = []

        async def decline(method, params):
            requested.append((method, params.get("command")))
            return "decline"

        client.approval_decider = decline if policy == INSPECT_APPROVAL_POLICY else None
        try:
            await client.start()
            await client.rpc.initialize({"clientInfo": {"name": "test", "version": "0"},
                                         "capabilities": {"experimentalApi": True}})
            await client.send_notification("initialized")
            thread = await client.rpc.thread_start({"cwd": str(home), "approvalPolicy": policy,
                                                    "sandbox": "danger-full-access"})
            thread_id = thread["thread"]["id"]
            turn = await client.rpc.turn_start({"threadId": thread_id, "input": [
                {"type": "text", "text": "go", "text_elements": []}]})
            await client.wait_for_turn(thread_id=thread_id, turn_id=turn["turn"]["id"], timeout=60)
        finally:
            await client.close(timeout=20)

    if asked:
        assert [method for method, _ in requested] == ["item/commandExecution/requestApproval"]
        assert "touch made-by-command" in (requested[0][1] or "")
        assert not (home / "made-by-command").exists(), "A declined command must not run"
    else:
        assert (home / "made-by-command").exists()


CALLS = {
    "tool": {"type": "function_call", "call_id": "c1", "name": "pending", "arguments": "{}"},
    "approval": {"type": "function_call", "call_id": "c1", "name": "exec_command",
                 "arguments": json.dumps({"cmd": "touch made-by-command"})},
}


@pytest.mark.native
@pytest.mark.parametrize("waiting_on", ["tool", "approval"])
async def test_native_turn_interrupts_while_the_adapter_owes_a_reply(evaluation_workspace, native_binary,
                                                                     waiting_on):
    bodies = []

    async def respond(request):
        bodies.append(await request.json())
        number = len(bodies)
        item = CALLS[waiting_on] if number == 1 else {
            "type": "message", "role": "assistant", "id": f"m{number}", "phase": "final_answer",
            "content": [{"type": "output_text", "text": "done"}]}
        events = [
            {"type": "response.created", "response": {"id": f"r{number}"}},
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": {"id": f"r{number}", "usage": {
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                "input_tokens_details": None, "output_tokens_details": None}}},
        ]
        return web.Response(text="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
                            content_type="text/event-stream")

    async with responses_peer(respond) as base_url:
        home = evaluation_workspace / "native-home"
        home.mkdir()
        (home / "config.toml").write_text(
            f'model="gpt-5.4"\nmodel_provider="scripted"\n[model_providers.scripted]\nname="scripted"\n'
            f'base_url="{base_url}"\nwire_api="responses"\nsupports_websockets=false\n'
            'requires_openai_auth=false\nrequest_max_retries=0\nstream_max_retries=0\n'
        )
        native = ExecRemoteJsonRpcTransport(LocalSandbox(), [native_binary, "app-server", "--stdio"],
                                            cwd=str(home), environment={"CODEX_HOME": str(home)})
        client = CodexClient(transport_factory=lambda: native)
        waiting = asyncio.Event()

        async def never_answers(*args):
            waiting.set()
            await asyncio.Event().wait()

        client.register_tool_handler("pending", never_answers)
        client.approval_decider = never_answers
        try:
            await client.start()
            await client.rpc.initialize({"clientInfo": {"name": "test", "version": "0"},
                                         "capabilities": {"experimentalApi": True}})
            await client.send_notification("initialized")
            thread = await client.rpc.thread_start({
                "cwd": str(home), "approvalPolicy": INSPECT_APPROVAL_POLICY, "sandbox": "danger-full-access",
                "dynamicTools": [{"type": "function", "name": "pending", "description": "Never returns.",
                                  "inputSchema": {"type": "object", "properties": {}}}]})
            thread_id = thread["thread"]["id"]
            turn = await client.rpc.turn_start({"threadId": thread_id, "input": [
                {"type": "text", "text": "go", "text_elements": []}]})
            await asyncio.wait_for(waiting.wait(), 30)
            await client.interrupt_turn(thread_id, turn["turn"]["id"])
            interrupted = await client.wait_for_turn(thread_id, turn["turn"]["id"], timeout=20)
            await client.cancel_turn_tools(thread_id, turn["turn"]["id"])
            redirect = await client.rpc.turn_start({"threadId": thread_id, "input": [
                {"type": "text", "text": "redirect", "text_elements": []}]})
            redirected = await client.wait_for_turn(thread_id, redirect["turn"]["id"], timeout=30)
        finally:
            await client.close(timeout=20)

    assert interrupted["turn"]["status"] == "interrupted"
    assert redirected["turn"]["status"] == "completed"
    assert not (home / "made-by-command").exists(), "A command awaiting approval must not run"


@pytest.mark.native
@pytest.mark.parametrize("decision,audio_supported", [
    ("modify", True), ("modify", False), ("terminate", False),
])
async def test_native_inspect_approval_and_audio(
    evaluation_workspace, decision, audio_supported, native_binary
):
    binary = native_binary
    requests = []
    executed = []
    native_tool_completed = asyncio.Event()
    audio_bytes = io.BytesIO()
    with wave.open(audio_bytes, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\0\0" * 80)
    audio_path = evaluation_workspace / "probe.wav"
    audio_path.write_bytes(audio_bytes.getvalue())
    expected_audio = "data:audio/wav;base64," + base64.b64encode(
        audio_bytes.getvalue()
    ).decode()

    async def respond(request):
        requests.append(await request.json())
        n = len(requests)
        assert n <= 2, "Unexpected additional native inference"
        item = (
            {
                "type": "function_call",
                "call_id": "approval-audio-call",
                "name": "inspect_probe",
                "arguments": json.dumps({"value": 3}),
            }
            if n == 1
            else {
                "type": "message", "id": "finished", "role": "assistant",
                "content": [{"type": "output_text", "text": "FINISHED"}],
            }
        )
        events = [
            {"type": "response.created", "response": {"id": f"resp-{n}"}},
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": {"id": f"resp-{n}"}},
        ]
        return web.Response(
            text="".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            ),
            content_type="text/event-stream",
        )

    async with responses_peer(respond) as base_url:
        home = evaluation_workspace / "native-home"
        home.mkdir()
        # The documented native catalog override advertises the scripted peer's
        # capabilities; this does not claim gpt-5.4 accepts audio.
        catalog = home / "models.json"
        catalog.write_text(json.dumps({"models": [{
            "slug": "scripted-contract", "display_name": "Scripted contract",
            "supported_reasoning_levels": [], "shell_type": "shell_command",
            "visibility": "list", "supported_in_api": True, "priority": 0,
            "support_verbosity": False, "base_instructions": "Run the probe.",
            "truncation_policy": {"mode": "tokens", "limit": 10000},
            "experimental_supported_tools": [],
            "input_modalities": ["text", "audio"] if audio_supported else ["text"],
        }]}))
        (home / "config.toml").write_text(f'''model="scripted-contract"
    model_provider="scripted_contract"
    model_catalog_json="{catalog}"
    approval_policy="never"
    sandbox_mode="read-only"
    [model_providers.scripted_contract]
    name="Scripted contract"
    base_url="{base_url}"
    wire_api="responses"
    supports_websockets=false
    requires_openai_auth=false
    request_max_retries=0
    stream_max_retries=0
    ''')

        @tool(name="inspect_probe")
        def probe_tool():
            async def probe(value: int):
                """Return an approved value and audio.

                Args:
                    value: Integer probe value.
                """
                executed.append(value)
                return [ContentText(text=f"APPROVED_VALUE_{value}"),
                        ContentAudio(audio=str(audio_path), format="wav")]

            return probe

        @approver
        def probe_approver():
            async def approve(message, call, view, history):
                return Approval(
                    decision=decision,
                    modified=replace(call, arguments={"value": 9})
                    if decision == "modify" else None,
                    explanation="SCRIPTED_APPROVAL_SENTINEL",
                )

            return approve

        @solver
        def native_probe():
            async def solve(state, generate):
                native = ExecRemoteJsonRpcTransport(
                    LocalSandbox(), [binary, "app-server", "--stdio"], cwd=str(home),
                    environment={"CODEX_HOME": str(home)},
                )
                client = CodexClient(transport_factory=lambda: native)
                mapper = EventStreamMapper("scripted-contract", state)
                def on_item(method, params):
                    mapper.on_item_completed(method, params)
                    if params["item"]["type"] == "dynamicToolCall":
                        native_tool_completed.set()

                for method, handler in (
                    ("item/started", mapper.on_item_started),
                    ("item/completed", on_item),
                    ("turn/completed", mapper.on_turn_completed),
                ):
                    client.register_notification_handler(method, handler)
                probe = probe_tool()
                name, description, schema = compat.tool_schema(probe)
                client.register_tool_handler(name, inspect_tool_handler(probe, mapper))
                try:
                    await client.start()
                    await client.send_request("initialize", {
                        "clientInfo": {"name": "inspect-ai", "version": "2.0.0"},
                        "capabilities": {"experimentalApi": True},
                    })
                    await client.send_notification("initialized")
                    root = (await client.send_request("thread/start", {
                        "cwd": str(home), "dynamicTools": [{"type": "function",
                            "name": name, "description": description, "inputSchema": schema}],
                    }))["thread"]["id"]
                    mapper.root_thread_id = root
                    turn = await client.send_request("turn/start", {
                        "threadId": root,
                        "input": [{"type": "text", "text": "PROBE", "text_elements": []}],
                    })
                    await client.wait_for_turn(root, turn["turn"]["id"], 15)
                    return state
                finally:
                    try:
                        if decision == "terminate" and client.failure_event.is_set():
                            # Observe the failed native item before stopping inference,
                            # so this actually tests deduplication at that boundary.
                            await asyncio.wait_for(native_tool_completed.wait(), 5)
                    finally:
                        try:
                            for tid, active in list(client.active_turn.items()):
                                if active:
                                    await client.interrupt_turn(tid, active)
                        finally:
                            await asyncio.wait_for(client.close(), 15)
                            mapper.finalize_state(state, "")

            return solve

        logs = await eval_async(
            Task(dataset=[Sample(input="PROBE")], solver=native_probe(),
                 approval=[ApprovalPolicy(probe_approver(), "inspect_probe")]),
            model="codex/scripted-contract", log_dir=str(evaluation_workspace / "logs"),
        )
        log = read_eval_log(logs[0].location)
        sample = log.samples[0]
        approvals = [e for e in sample.events if e.event == "approval"]
        assert len(approvals) == 1 and approvals[0].decision == decision
        tools = [e for e in sample.events if e.event == "tool"
                 and e.function == "inspect_probe"]
        assert len(tools) == 1, tools
        if decision == "modify":
            assert log.status == "success" and sample.error is None, sample.error
            assert approvals[0].call.arguments == {"value": 3}
            assert approvals[0].modified.arguments == {"value": 9}
            assert executed == [9]
            assert len(requests) == 2
            output = next(item["output"] for item in requests[1]["input"]
                          if item.get("type") == "function_call_output"
                          and item.get("call_id") == "approval-audio-call")
            assert output == [
                {"type": "input_text", "text": "APPROVED_VALUE_9"},
                {"type": "input_audio", "audio_url": expected_audio} if audio_supported
                else {"type": "input_text", "text":
                      "audio content omitted because you do not support audio input"},
            ]
            tool_message = next(m for m in sample.messages if m.role == "tool")
            assert tool_message.content[1].audio == str(audio_path)
        else:
            assert executed == []
            assert sample.error is None, sample.error
            assert sample.limit.type == "operator"
            assert sample.limit.reason == "Tool call approver requested termination."
            assert tools[0].failed
