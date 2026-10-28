"""Checkpoint resume with pinned Codex and real session files.

Restore mirrored sessions into a fresh home and continue the same thread against
a local scripted provider. Verify history, instructions, tools, lifetime usage,
and the absence of raw response events after resume. Missing sessions provide
the rejection control.
"""

import json
import shutil
import subprocess

import pytest
from aiohttp import web

from inspect_codex_app_server.client import (
    CodexClient,
    ExecRemoteJsonRpcTransport,
    JsonRpcError,
)
from inspect_codex_app_server.evidence import NativeEvidence
from inspect_codex_app_server.mapping import EventStreamMapper
from inspect_codex_app_server.session import MIRROR_SESSIONS, RESTORE_SESSIONS

from .support import LocalSandbox, make_state, responses_peer


def codex_home(path, base_url):
    path.mkdir()
    (path / "config.toml").write_text(
        f'model="gpt-5.4"\nmodel_provider="scripted"\n[model_providers.scripted]\nname="scripted"\n'
        f'base_url="{base_url}"\nwire_api="responses"\nsupports_websockets=false\n'
        'requires_openai_auth=false\nrequest_max_retries=0\nstream_max_retries=0\n'
    )
    return path


async def in_session(binary, home, body):
    native = ExecRemoteJsonRpcTransport(LocalSandbox(), [binary, "app-server", "--stdio"], cwd=str(home),
                                        environment={"CODEX_HOME": str(home)})
    client = CodexClient(transport_factory=lambda: native)
    try:
        await client.start()
        await client.rpc.initialize({"clientInfo": {"name": "test", "version": "0"},
                                     "capabilities": {"experimentalApi": True}})
        await client.send_notification("initialized")
        return await body(client)
    finally:
        await client.close(timeout=20)


async def run_turn(client, thread_id, text):
    turn = await client.rpc.turn_start({"threadId": thread_id, "input": [
        {"type": "text", "text": text, "text_elements": []}]})
    await client.wait_for_turn(thread_id=thread_id, turn_id=turn["turn"]["id"], timeout=60)


@pytest.mark.native
@pytest.mark.parametrize("restored", [True, False])
async def test_native_thread_resumes_from_mirrored_sessions(
    evaluation_workspace, native_binary, restored, usage_contexts, captured_events,
):
    bodies = []

    async def respond(request):
        bodies.append(await request.json())
        number = len(bodies)
        events = [
            {"type": "response.created", "response": {"id": f"r{number}"}},
            {"type": "response.output_item.done", "item": {
                "type": "message", "role": "assistant", "id": f"m{number}", "phase": "final_answer",
                "content": [{"type": "output_text", "text": f"ANSWER_{number}"}]}},
            {"type": "response.completed", "response": {"id": f"r{number}", "usage": {
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                "input_tokens_details": None, "output_tokens_details": None}}},
        ]
        return web.Response(text="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
                            content_type="text/event-stream")

    async with responses_peer(respond) as base_url:
        first_home = codex_home(evaluation_workspace / "first-home", base_url)
        first_mapper = EventStreamMapper("gpt-5.4")

        raw = {"first": 0, "resumed": 0}

        def observe_usage(client, mapper):
            attempt = "first" if mapper is first_mapper else "resumed"
            client.register_notification_handler("rawResponse/completed", lambda method, params: (
                raw.__setitem__(attempt, raw[attempt] + 1), mapper.on_response_completed(method, params)))
            client.register_notification_handler("thread/tokenUsage/updated", mapper.on_token_usage)

        async def first_attempt(client):
            thread = await client.rpc.thread_start({
                "cwd": str(first_home), "approvalPolicy": "never", "sandbox": "danger-full-access",
                "developerInstructions": "DEVELOPER_SENTINEL",
                "experimentalRawEvents": True,
                "dynamicTools": [{"type": "function", "name": "lookup", "description": "Look up a value.",
                                  "inputSchema": {"type": "object", "properties": {}}}],
            })
            first_mapper.root_thread_id = thread["thread"]["id"]
            observe_usage(client, first_mapper)
            await run_turn(client, thread["thread"]["id"], "HISTORY_SENTINEL")
            return thread["thread"]["id"]

        thread_id = await in_session(native_binary, first_home, first_attempt)
        assert first_mapper.root_running_usage.total_tokens == 2
        # Round-trip the adapter's state as Inspect's checkpoint storage does.
        accounting = json.loads(json.dumps(first_mapper.checkpoint_accounting()))
        resumed_mapper = EventStreamMapper("gpt-5.4")
        resumed_mapper.restore_accounting(accounting)
        mirror = evaluation_workspace / "user-home" / ".codex-inspect-checkpoint"
        subprocess.run(["sh", "-c", MIRROR_SESSIONS, "sh", str(first_home), str(mirror)], check=True, timeout=10)

        second_home = codex_home(evaluation_workspace / "second-home", base_url)
        if restored:
            subprocess.run(["sh", "-c", RESTORE_SESSIONS, "sh", str(mirror), str(second_home)], check=True, timeout=10)

        async def resumed_attempt(client):
            observe_usage(client, resumed_mapper)
            await client.rpc.thread_resume({"threadId": thread_id, "cwd": str(second_home),
                                            "approvalPolicy": "never", "sandbox": "danger-full-access"})
            await run_turn(client, thread_id, "Continue.")

        if not restored:
            with pytest.raises(JsonRpcError, match="no rollout found"):
                await in_session(native_binary, second_home, resumed_attempt)
            return
        await in_session(native_binary, second_home, resumed_attempt)
        assert raw == {"first": 1, "resumed": 0}

        # Each fixed response reports two tokens. Lifetime counters and old
        # rollout records must not charge the checkpoint's first two again.
        evidence = NativeEvidence("resumed-accounting")
        try:
            shutil.copytree(second_home / "sessions", evidence.directory / "native" / "sessions")
            resumed_mapper.tearing_down = True
            await evidence.reconcile(resumed_mapper)
            state = resumed_mapper.finalize_state(make_state(), "ANSWER_2")
            assert state.output.usage.total_tokens == 4
            for context in usage_contexts:
                assert sum(usage.total_tokens for usage in context.get().values()) == 4
            assert evidence.metadata["reconciled_usage_records"] == 2
        finally:
            evidence.close(shutdown={"stream_drained": True}, accounting={})

    resumed = json.dumps(bodies[-1])
    assert "HISTORY_SENTINEL" in resumed and "ANSWER_1" in resumed, "The earlier history continues"
    assert "DEVELOPER_SENTINEL" in resumed
    assert "lookup" in [tool.get("name") for tool in bodies[-1]["tools"]], "Bridged tools survive the resume"
