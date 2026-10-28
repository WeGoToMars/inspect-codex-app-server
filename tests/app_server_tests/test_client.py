"""JSON-RPC dispatch, turn correlation, request deadlines, and cancellation.

Scripted app-server peers isolate client behavior. Static caller programs
also verify the generated request and response types.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time

import pytest

from inspect_codex_app_server.client import CodexClient, JsonRpcError

from .support import ScriptedAppServer, make_client


@pytest.mark.asyncio
async def test_account_update_refreshes_identity_without_blocking_reader(scripted):
    client = make_client(scripted)
    observations = asyncio.Queue()
    client.register_notification_handler(
        "account/metadata/read", lambda method, params: observations.put_nowait(params)
    )
    await client.start()
    try:
        current = {"type": "chatgpt", "email": "current@example.invalid", "planType": "pro"}

        def switch_during_read(server, params):
            # Deliver another change before the previous read's coroutine
            # resumes: its old identity must not overwrite the newer one.
            del server.hooks["account/read"]
            server.responses["account/read"] = {"account": current, "requiresOpenaiAuth": True}
            server.notify("account/updated", {"authMode": "chatgpt"})

        scripted.responses["account/read"] = {
            "account": {"type": "chatgpt", "email": "previous@example.invalid"},
            "requiresOpenaiAuth": True,
        }
        scripted.hooks["account/read"] = switch_during_read
        scripted.notify("account/updated", {"authMode": "chatgpt"})
        assert (await asyncio.wait_for(observations.get(), 1))["account"] == current
        assert observations.empty()

        scripted.responses["account/read"] = {"account": None, "requiresOpenaiAuth": True}
        scripted.notify("account/updated", {"authMode": None})
        observed = await asyncio.wait_for(observations.get(), 1)
        assert observed["account"] is None
        assert scripted.requests[-1] == ("account/read", {"refreshToken": False})
        assert client.failure is None
    finally:
        await client.close()





@pytest.mark.asyncio
async def test_write_failure_allows_subsequent_request(scripted):
    """A write failure must leave a live client usable for the next request."""
    write_line = scripted.write_line

    async def fail_one_write(line):
        if json.loads(line).get("method") == "write-failure":
            raise RuntimeError("WRITE_FAILURE_SENTINEL")
        await write_line(line)

    scripted.write_line = fail_one_write
    client = make_client(scripted)
    await client.start()
    try:
        with pytest.raises(RuntimeError, match="WRITE_FAILURE_SENTINEL"):
            await client.send_request("write-failure")
        assert await client.send_request("initialize", timeout=1) == {}
    finally:
        await client.close()



@pytest.mark.asyncio
async def test_unhandled_server_requests_get_typed_responses(scripted):
    """Server requests receive responses matching their individual wire schemas."""
    client = make_client(scripted)
    await client.start()
    try:
        scripted.server_request(1, "item/tool/requestUserInput", {"questions": []})
        reply = await asyncio.wait_for(scripted.client_replies.get(), 1.0)

        msg = json.loads(reply)
        assert msg["id"] == 1
        assert msg["result"] == {"answers": {}}

        scripted.server_request(2, "mcpServer/elicitation/request", {})
        msg = json.loads(await asyncio.wait_for(scripted.client_replies.get(), 1.0))
        assert msg["id"] == 2
        assert msg["result"] == {"action": "decline"}

        scripted.server_request(3, "currentTime/read", {})
        msg = json.loads(await asyncio.wait_for(scripted.client_replies.get(), 1.0))
        assert msg["id"] == 3
        assert isinstance(msg["result"]["currentTimeAt"], int)
        assert abs(msg["result"]["currentTimeAt"] - time.time()) < 5

        scripted.server_request(4, "account/chatgptAuthTokens/refresh", {})
        msg = json.loads(await asyncio.wait_for(scripted.client_replies.get(), 1.0))
        assert msg["id"] == 4
        assert "error" in msg

        scripted.server_request(5, "totally/unknown/method", {})
        msg = json.loads(await asyncio.wait_for(scripted.client_replies.get(), 1.0))
        assert msg["id"] == 5
        assert msg["error"]["code"] == -32601
    finally:
        await client.close()



@pytest.mark.asyncio
async def test_unknown_dynamic_tool_reports_failure(scripted):
    client = make_client(scripted)
    await client.start()
    try:
        scripted.server_request(1, "item/tool/call", {"tool": "missing", "arguments": {}, "threadId": "root",
                                                      "turnId": "turn", "callId": "missing-call"})

        msg = json.loads(await asyncio.wait_for(scripted.client_replies.get(), 1.0))
        assert msg["result"]["success"] is False
    finally:
        await client.close()



@pytest.mark.asyncio
async def test_interrupt_requires_native_acknowledgement(scripted):
    client = make_client(scripted)
    await client.start()
    scripted.responses["turn/interrupt"] = {
        "error": {"code": -32600, "message": "no active turn to interrupt"}
    }
    try:
        with pytest.raises(JsonRpcError, match="no active turn to interrupt"):
            await client.interrupt_turn("root", "finished")
        scripted.notify("turn/completed", {"threadId": "root", "turn": {"id": "finished", "status": "completed"}})
        await client.wait_for_turn("root", "finished", timeout=1)
        await client.interrupt_turn("root", "finished")
        scripted.responses["turn/interrupt"] = {}
        await client.interrupt_turn("root")
        assert scripted.requests[-1] == ("turn/interrupt", {"threadId": "root", "turnId": ""})
    finally:
        await client.close()


async def test_child_metadata_failing_during_close_is_not_a_run_failure(scripted):
    client = make_client(scripted)
    client.register_notification_handler("thread/metadata/read", lambda method, params: None)
    asked = asyncio.Event()

    def child_gone(params):
        asked.set()
        return {"error": {"code": -32600, "message": "thread not found"}}

    scripted.responses["thread/read"] = child_gone
    stop = scripted.stop

    async def spawn_then_stop():  # A child starts as the session closes.
        scripted.notify("item/completed", {"threadId": "root", "item": {
            "type": "subAgentActivity", "id": "spawn", "kind": "started", "agentThreadId": "late-child"}})
        await asyncio.wait_for(asked.wait(), 1)
        while not (client.failure or client.shutdown["failures"]):
            await asyncio.sleep(0)
        await stop()

    scripted.stop = spawn_then_stop
    await client.start()
    await client.close()
    assert client.failure is None
    assert any("late-child" in failure for failure in client.shutdown["failures"])


@pytest.mark.parametrize("cooperative", [True, False])
async def test_turn_tool_cancellation_is_scoped_and_failure_blocks_resume(scripted, cooperative):
    from inspect_codex_app_server.client import CodexTransportFailure

    client = make_client(scripted)
    started = asyncio.Queue()
    release = asyncio.Event()
    cancelled = set()
    tasks = []

    async def pending_tool(params):
        tasks.append(asyncio.current_task())
        started.put_nowait(params["callId"])
        try:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    if cooperative:
                        raise
            return {"success": True, "contentItems": []}
        finally:
            cancelled.add(params["callId"])

    client.register_tool_handler("pending", pending_tool)
    await client.start()
    try:
        for number, thread in enumerate(("root", "child")):
            scripted.server_request(number, "item/tool/call", {
                "threadId": thread, "turnId": "turn", "callId": thread,
                "tool": "pending", "arguments": {},
            })
        assert {await asyncio.wait_for(started.get(), 1), await asyncio.wait_for(started.get(), 1)} == {"root", "child"}
        # A tool that ignores cancellation cannot delay the interrupt itself.
        await client.interrupt_turn("root", "turn")
        assert scripted.requests == [("turn/interrupt", {"threadId": "root", "turnId": "turn"})]
        assert client.failure is None
        if cooperative:
            await client.cancel_turn_tools("root", "turn")
            assert cancelled == {"root"}, "Interrupting one turn leaves the other thread's tool running"
            scripted.server_request(10, "item/tool/call", {
                "threadId": "root", "turnId": "turn", "callId": "late",
                "tool": "pending", "arguments": {},
            })
            reply = json.loads(await asyncio.wait_for(scripted.client_replies.get(), 1))
            assert reply["result"]["success"] is False
            assert started.empty(), "Late tool calls cannot restart an interrupted turn"
        else:
            with pytest.raises(CodexTransportFailure, match="did not stop"):
                await client.cancel_turn_tools("root", "turn", timeout=0.01)
            with pytest.raises(CodexTransportFailure, match="did not stop"):
                await client.send_request("turn/start", {})
            assert [method for method, _ in scripted.requests] == ["turn/interrupt"]
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()



@pytest.mark.asyncio
async def test_stale_turn_completion_cannot_satisfy_new_wait(scripted):
    """A completion for an earlier turn cannot release a later turn's wait."""
    client = make_client(scripted)
    await client.start()
    try:
        # An old root completion arrives while no turn id is bound for waiting.
        scripted.notify(
            "turn/completed", {"threadId": "root", "turn": {"id": "old", "status": "completed"}}
        )
        with pytest.raises(TimeoutError):
            await client.wait_for_turn("root", turn_id="new", timeout=0.1)

        # Once the new turn completes, its data (and only its data) is returned.
        scripted.notify(
            "turn/completed", {"threadId": "root", "turn": {"id": "new", "status": "completed", "x": 1}}
        )
        data = await client.wait_for_turn("root", turn_id="new", timeout=0.1)
        assert data["turn"]["id"] == "new"
    finally:
        await client.close()



@pytest.mark.asyncio
async def test_notification_handler_failure_is_forwarded_not_swallowed(scripted):
    """A handler's message-limit failure reaches the caller waiting for a turn."""
    client = make_client(scripted)
    await client.start()
    try:
        from inspect_ai.util._limit import LimitExceededError

        async def raising_handler(method: str, params: dict) -> None:
            raise LimitExceededError(type="message", value=2, limit=1)

        client.register_notification_handler("item/completed", raising_handler)

        scripted.notify(
            "item/completed", {"threadId": "root", "turnId": "turn-1",
                               "item": {"type": "agentMessage", "id": "x", "text": "partial"}}
        )

        await asyncio.wait_for(client.failure_event.wait(), 1)
        # The failure slot is set and surfaces from wait boundaries.
        assert isinstance(client.failure, LimitExceededError)
        assert client.failure_event.is_set()
        with pytest.raises(LimitExceededError):
            await client.wait_for_turn("root", "turn-1", timeout=1)
    finally:
        await client.close()



@pytest.mark.asyncio
async def test_server_request_handler_making_client_rpc_does_not_deadlock(scripted):
    """A tool handler can await an RPC while the reader processes its response."""

    async def rpc_tool(params: dict) -> dict:
        result = await client.send_request("external/lookup", params["arguments"], timeout=2.0)
        return {"contentItems": [{"type": "inputText", "text": str(result)}], "success": True}

    client = make_client(scripted)
    client.register_tool_handler("lookup", rpc_tool)
    await client.start()
    try:
        scripted.responses["external/lookup"] = "found-it"

        scripted.server_request(7, "item/tool/call", {"tool": "lookup", "arguments": {"q": 1}, "threadId": "root",
                                                      "turnId": "turn", "callId": "lookup-call"})


        msg = json.loads(await asyncio.wait_for(scripted.client_replies.get(), 1.0))
        assert msg["result"]["success"] is True
        assert "found-it" in msg["result"]["contentItems"][0]["text"]
    finally:
        await client.close()


async def test_request_deadline_includes_stalled_write():
    class StalledWrite(ScriptedAppServer):
        async def write_line(self, line):
            if json.loads(line).get("method") == "stalled":
                await asyncio.Event().wait()
            await super().write_line(line)

    peer = StalledWrite()
    client = CodexClient(transport_factory=lambda: peer)
    await client.start()
    try:
        # The outer bound diagnoses the old unbounded write; the inner deadline
        # is the public request contract under test.
        async with asyncio.timeout(0.5):
            with pytest.raises(TimeoutError):
                await client.send_request("stalled", timeout=0.02)
            assert await client.send_request("initialize", timeout=0.1) == {}
    finally:
        await client.close()


def test_generated_rpc_types_accept_valid_calls_and_reject_wire_mistakes(tmp_path):
    imports = "from inspect_codex_app_server._generated_rpc import CodexRpc\n"
    valid = tmp_path / "valid.py"
    valid.write_text(imports + '''
async def start(rpc: CodexRpc) -> str:
    response = await rpc.turn_start({
        "threadId": "root", "input": [{"type": "text", "text": "Hello"}],
    })
    return response["turn"]["id"]
''')
    invalid = tmp_path / "invalid.py"
    invalid.write_text(imports + '''
async def start(rpc: CodexRpc) -> str:
    await rpc.turn_start({"threadID": "root", "input": []})
    await rpc.turn_start({"threadId": "root", "input": [{"type": "text", "text": 123}]})
    await rpc.turn_start({"threadId": "root", "input": [{"type": "unsupported"}]})
    response = await rpc.turn_start({"threadId": "root", "input": []})
    return response["turnId"]
''')
    command = [sys.executable, "-m", "ty", "check", "--python", sys.executable,
               "--output-format", "concise", "--no-force-exclude"]
    accepted = subprocess.run([*command, str(valid)], capture_output=True, text=True, timeout=30)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    rejected = subprocess.run([*command, str(invalid)], capture_output=True, text=True, timeout=30)
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    # Each independent invalid line must produce its own diagnostic.
    for line in (4, 5, 6, 8):
        assert f"invalid.py:{line}:" in rejected.stdout, rejected.stdout


@pytest.mark.asyncio
async def test_continuation_waits_past_old_start_and_consumes_completed_turn(scripted):
    client = make_client(scripted)
    await client.start()
    try:
        scripted.notify("turn/started", {"threadId": "root", "turn": {"id": "old"}})
        await client.send_request("initialize", timeout=1)
        assert await client.wait_for_next_turn("root", {"old"}, 0) is None
        for method in ("turn/started", "turn/completed"):
            scripted.notify(method, {"threadId": "root", "turn": {"id": "new"}})
        await client.send_request("initialize", timeout=1)
        # Both events may precede the wait without losing the new turn.
        assert await client.wait_for_next_turn("root", {"old"}, 0) == "new"
        assert await client.wait_for_next_turn("root", {"old", "new"}, 0) is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_close_bounds_uncooperative_request_handler(scripted):
    client = make_client(scripted)
    started = asyncio.Event()
    release = asyncio.Event()
    handler_tasks = []

    async def handler(params):
        handler_tasks.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
        return {"success": True, "contentItems": []}

    client.register_tool_handler("uncooperative", handler)
    await client.start()
    try:
        scripted.server_request(1, "item/tool/call", {
            "threadId": "root", "turnId": "turn-1", "callId": "call-1",
            "tool": "uncooperative", "arguments": {},
        })
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(client.close(timeout=0.01), 0.5)
    finally:
        release.set()
        await asyncio.gather(*handler_tasks)
        await client.close()
