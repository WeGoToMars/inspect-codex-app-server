"""Bounded process shutdown preserves received output and stops owned work.

Real subprocesses cover forced termination and tail draining. The native
case runs pinned Codex against a local scripted provider with an active
child and bridged Inspect tool.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from aiohttp import web
from inspect_ai import Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.solver import solver
from inspect_ai.tool import tool
from inspect_ai.util import ExecRemoteProcess, ExecRemoteStreamingOptions
from inspect_ai.util._sandbox.exec_remote import (
    _KillResult,
    _PollResult,
    _WriteStdinResult,
)

from inspect_codex_app_server.client import CodexClient, ExecRemoteJsonRpcTransport
from inspect_codex_app_server.compat import ExecCompleted
from inspect_codex_app_server.evidence import NativeEvidence
from inspect_codex_app_server.mapping import EventStreamMapper
from inspect_codex_app_server.tools import inspect_tool_handler

from .support import LocalExecProcess, LocalSandbox, make_state, responses_peer


class KilledStreamProcess(LocalExecProcess):
    """Inspect stops killed iteration after draining, without ExecCompleted."""

    async def __anext__(self):
        event = await super().__anext__()
        if isinstance(event, ExecCompleted) and self._proc.returncode < 0:
            raise StopAsyncIteration
        return event

    async def kill(self):
        if self._proc.returncode is None:
            self._proc.kill()
        await self._proc.wait()


class KilledStreamSandbox(LocalSandbox):
    async def exec_remote(self, command, options=None):
        started = await super().exec_remote(command, options)
        return KilledStreamProcess(started._proc, self.chunk_size)


@pytest.mark.parametrize("poll_output", ['{"poll":"POLL_SENTINEL"}\n', ""])
async def test_inspect_poll_stdin_and_acknowledged_kill_keep_all_output(monkeypatch, poll_output):
    """Exercise pinned poll/stdin RPCs across returning and silent polls.

    Independent sequences 1/2/3 and distinct output records must survive the
    older poll's queue commit and the empty-poll sleep before termination.
    """
    process = ExecRemoteProcess(LocalSandbox(), ["codex"], ExecRemoteStreamingOptions(stdin_open=True), 5)
    process._pid = 7
    process._transport = SimpleNamespace(sandbox=SimpleNamespace(no_events=nullcontext))
    poll_entered, poll_release, write_entered, stop_entered, kill_entered = (asyncio.Event() for _ in range(5))
    calls = []

    async def rpc(method, params, result_type):
        calls.append((method, params))
        if method == "exec_remote_poll":
            assert len(calls) == 1, "Acknowledged kill must drain without another poll"
            poll_entered.set()
            await poll_release.wait()
            return _PollResult(state="running", seq=1, stdout=poll_output, stderr="")
        if method == "exec_remote_write_stdin":
            assert params == {"pid": 7, "data": "request\n", "ack_seq": 1}
            return _WriteStdinResult(seq=2, stdout='{"stdin":"STDIN_SENTINEL"}\n', stderr="")
        assert method == "exec_remote_kill"
        assert params == {"pid": 7, "ack_seq": 2}
        kill_entered.set()
        return _KillResult(seq=3, stdout='{"kill":"KILL_TAIL_SENTINEL"}', stderr="")

    monkeypatch.setattr(process, "_rpc", rpc)

    async def start(command, options):
        return process

    transport = ExecRemoteJsonRpcTransport(SimpleNamespace(exec_remote=start), ["codex"])
    await transport.start()

    async def collect():
        return [line async for line in transport.lines()]

    async def stop():
        stop_entered.set()
        await transport.stop()

    async def write():
        write_entered.set()
        await transport.write_line("request")

    reader = asyncio.create_task(collect())
    writing = stopping = None
    try:
        await asyncio.wait_for(poll_entered.wait(), 1)
        writing = asyncio.create_task(write())
        await asyncio.wait_for(write_entered.wait(), 1)
        stopping = asyncio.create_task(stop())
        await asyncio.wait_for(stop_entered.wait(), 1)
        assert not kill_entered.is_set()
        poll_release.set()
        await asyncio.wait_for(writing, 1)
        await asyncio.wait_for(stopping, 1)
        expected = ([poll_output.strip()] if poll_output else []) + [
            '{"stdin":"STDIN_SENTINEL"}', '{"kill":"KILL_TAIL_SENTINEL"}',
        ]
        assert await asyncio.wait_for(reader, 1) == expected
        assert process._last_seq == 3 and process._killed
        assert transport._proc is None
        assert len(calls) == 3
    finally:
        poll_release.set()
        for task in (reader, writing, stopping):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (reader, writing, stopping) if task is not None), return_exceptions=True)


@pytest.mark.native
async def test_native_shutdown_with_active_child_and_inspect_tool(evaluation_workspace, native_binary):
    """The installed binary is killed while root tool and child inference run.

    Two completed Responses each report 100 input and 10 output. The child's
    open stream has no final usage and must never receive invented token usage.
    """
    child_active, tool_active, tool_cancelled = (asyncio.Event() for _ in range(3))
    release_provider = asyncio.Event()
    completed_responses = []
    requests = []

    async def respond(request):
        body = await request.json()
        requests.append(body)
        number = len(requests)
        response_id = f"response-{number}"
        is_child = any(
            item.get("type") == "agent_message" and item.get("recipient") == "/root/worker"
            for item in body["input"]
        )
        if is_child:
            stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await stream.prepare(request)
            await stream.write(("event: response.created\ndata: " + json.dumps({
                "type": "response.created", "response": {"id": response_id},
            }) + "\n\n").encode())
            child_active.set()
            await release_provider.wait()
            return stream
        item = {
            "type": "function_call", "call_id": f"call-{number}",
            "name": "spawn_agent" if number == 1 else "inspect_wait",
            "arguments": json.dumps({"task_name": "worker", "message": "CHILD_WAIT", "fork_turns": "all"})
            if number == 1 else "{}",
        }
        if number == 1:
            item["namespace"] = "collaboration"
        completed_responses.append(response_id)
        events = [
            {"type": "response.created", "response": {"id": response_id}},
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": {"id": response_id, "usage": {
                "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                "input_tokens_details": None, "output_tokens_details": None,
            }}},
        ]
        return web.Response(text="".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        ), content_type="text/event-stream")

    async with responses_peer(respond) as base_url:
        home = evaluation_workspace / "native-home"
        home.mkdir()
        (home / "config.toml").write_text(f'''model="gpt-5.4"
    model_provider="scripted_contract"
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
        binary = native_binary

        @tool
        def inspect_wait():
            async def wait():
                """Wait until interrupted for the native shutdown contract."""
                tool_active.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    tool_cancelled.set()

            return wait

        @solver
        def native_shutdown():
            async def solve(state, generate):
                transport = ExecRemoteJsonRpcTransport(
                    KilledStreamSandbox(),
                    [binary, "-c", "features.multi_agent_v2=true", "app-server", "--stdio"],
                    cwd=str(home), environment={"CODEX_HOME": str(home)},
                )
                client = CodexClient(transport_factory=lambda: transport)
                evidence = NativeEvidence("native-active-shutdown")
                client.evidence = evidence
                mapper = EventStreamMapper("gpt-5.4", state)
                for method, handler in (
                    ("thread/started", mapper.on_thread_metadata),
                    ("thread/metadata/read", mapper.on_thread_metadata),
                    ("rawResponseItem/completed", mapper.on_raw_item),
                    ("rawResponse/completed", mapper.on_response_completed),
                    ("thread/tokenUsage/updated", mapper.on_token_usage),
                    ("item/started", mapper.on_item_started),
                    ("item/completed", mapper.on_item_completed),
                    ("turn/completed", mapper.on_turn_completed),
                ):
                    client.register_notification_handler(method, handler)
                wait_tool = inspect_wait()
                client.register_tool_handler("inspect_wait", inspect_tool_handler(wait_tool, mapper))
                try:
                    await client.start()
                    process = transport._proc._proc
                    await client.send_request("initialize", {
                        "clientInfo": {"name": "inspect-ai", "version": "2.0.0"},
                        "capabilities": {"experimentalApi": True},
                    })
                    await client.send_notification("initialized")
                    root = (await client.send_request("thread/start", {
                        "cwd": str(home), "experimentalRawEvents": True,
                        "dynamicTools": [{"name": "inspect_wait", "description": "Wait for interruption", "inputSchema": {
                            "type": "object", "properties": {}, "additionalProperties": False,
                        }}],
                    }))["thread"]["id"]
                    mapper.root_thread_id = root
                    await client.send_request("turn/start", {"threadId": root, "input": [
                        {"type": "text", "text": "ROOT_START", "text_elements": []},
                    ]})
                    async with asyncio.timeout(15):
                        await asyncio.gather(child_active.wait(), tool_active.wait())
                    assert len(client.active_turn) == 2
                    assert all(client.active_turn.values())
                    mapper.tearing_down = True
                    # Freeze the actual native process at observed tool/child
                    # progress, so EOF cannot complete and close must force a kill.
                    os.kill(process.pid, signal.SIGSTOP)
                    await asyncio.wait_for(client.close(timeout=1), 3)
                    assert process.returncode < 0
                    assert tool_cancelled.is_set()
                    assert client.shutdown["forced"]
                    assert client.shutdown["transport_stopped"]
                    assert client.shutdown["stream_drained"]
                    assert not client._request_tasks
                    # A local boundary test copies the stopped native filesystem;
                    # Docker export is covered by container acceptance separately.
                    shutil.copytree(home / "sessions", evidence.directory / "native" / "sessions")
                    await evidence.reconcile(mapper)
                    mapper.finalize_state(state, "", interrupted=True)
                    state.metadata["codex_native_evidence"] = evidence.metadata
                    evidence.close(shutdown=client.shutdown, accounting=state.metadata["codex_accounting"])
                    return state
                finally:
                    await client.close(timeout=2)
                    if not evidence.metadata["closed"]:
                        evidence.close(shutdown=client.shutdown, accounting={})

            return solve

        try:
            logs = await eval_async(Task(dataset=[Sample(input="ROOT_START")], solver=native_shutdown()),
                                    model="codex/gpt-5.4", log_dir=str(evaluation_workspace / "logs"))
            log = read_eval_log(logs[0].location)
            assert log.samples[0].error is None, log.samples[0].error
            sample = log.samples[0]
            assert len(completed_responses) == 2
            assert sample.output.usage.total_tokens == 220
            assert not sample.output.completion and not sample.scores
            assert any(event.event == "tool" and event.function == "inspect_wait" for event in sample.events)
            assert sample.metadata["codex_accounting"]["observed_responses"] == 2
            assert sample.metadata["codex_native_evidence"]["closed"]
            assert sample.metadata["codex_native_evidence"]["reconciled_usage_records"] >= 2
        finally:
            release_provider.set()


async def test_forced_process_shutdown_preserves_tail_and_cancels_tool(
    tmp_path, captured_events, usage_contexts
):
    peer = tmp_path / "peer.py"
    peer.write_text('''
import json, sys, time
print(json.dumps({"id": 71, "method": "item/tool/call", "params": {
    "threadId": "root", "turnId": "turn", "callId": "tool-call", "tool": "wait_tool", "arguments": {}}}), flush=True)
for line in sys.stdin:
    raise RuntimeError("No client request was expected")
# EOF cleanup emits a last response before the process stalls. A killed
# Inspect iterator has no ExecCompleted event to trigger framing flush.
sys.stdout.write(json.dumps({"method": "rawResponse/completed", "params": {
    "threadId": "root", "turnId": "turn", "responseId": "late-response",
    "usage": {"inputTokens": 30, "outputTokens": 5, "totalTokens": 35}}}))
sys.stdout.flush()
time.sleep(60)
''')
    transport = ExecRemoteJsonRpcTransport(
        KilledStreamSandbox(chunk_size=7), [sys.executable, str(peer)]
    )
    client = CodexClient(transport_factory=lambda: transport)
    evidence = NativeEvidence("forced-tail")
    client.evidence = evidence
    state = make_state(model="codex/gpt-5.4")
    mapper = EventStreamMapper("gpt-5.4", state)
    mapper.root_thread_id = "root"
    mapper.tearing_down = True
    client.register_notification_handler("rawResponse/completed", mapper.on_response_completed)
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def tool(params):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client.register_tool_handler("wait_tool", tool)
    await client.start()
    process = transport._proc._proc
    try:
        await asyncio.wait_for(started.wait(), 3)
        await asyncio.wait_for(client.close(timeout=0.5), 2)
        assert process.returncode is not None
        assert cancelled.is_set()
        assert client.shutdown["forced"] and client.shutdown["transport_stopped"]
        assert client.shutdown["stream_drained"]
        mapper.finalize_state(state, "", interrupted=True)
        assert state.output.usage.total_tokens == 35
        assert state.output.completion == ""
        evidence.close(shutdown=client.shutdown, accounting=state.metadata["codex_accounting"])
        manifest = json.loads((evidence.directory / "manifest.json").read_text())
        assert manifest["closed"] and not manifest["truncated"]
        protocol = (evidence.directory / "protocol.jsonl").read_text()
        assert "late-response" in protocol
    finally:
        await client.close(timeout=1)
        if not evidence.metadata["closed"]:
            evidence.close(shutdown=client.shutdown, accounting={})
