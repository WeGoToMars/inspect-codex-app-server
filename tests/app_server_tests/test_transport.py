"""Stdio framing, process ownership, and WebSocket connection boundaries.

Owned subprocesses exercise split output and termination; a local TCP
listener exercises handshake timeout. The native case checks capability
tokens against the pinned Codex WebSocket listener without inference.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap

import pytest

from inspect_codex_app_server.client import (
    CodexClient,
    CodexTransportFailure,
    ExecRemoteJsonRpcTransport,
    WebSocketJsonRpcTransport,
)

from .support import LocalSandbox

EXIT_SERVER = textwrap.dedent(
    """
    import json, sys
    for line in sys.stdin:
        if json.loads(line).get("method") == "die":
            sys.exit(3)
    """
)


@pytest.mark.asyncio
async def test_stdio_transport_frames_partial_chunks_and_roundtrips(tmp_path):
    """Output chunks need not align with messages; framing must handle it."""
    server_script = tmp_path / "server.py"
    server_script.write_text(
        textwrap.dedent(
            """
            import json, sys
            request = json.loads(sys.stdin.readline())
            assert request["method"] == "initialize"
            # Raw UTF-8, rather than JSON's ASCII escaping, exercises split codepoints.
            reply = {"id": request["id"], "result": {"text": "é雪"}}
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\\n")
            sys.stdout.flush()
            """
        )
    )
    # Single-byte reads guarantee that both JSON and multibyte characters split,
    # regardless of how the OS coalesces the server's writes.
    transport = ExecRemoteJsonRpcTransport(
        LocalSandbox(chunk_size=1), [sys.executable, str(server_script)]
    )
    client = CodexClient(transport_factory=lambda: transport)  # type: ignore[arg-type]
    await client.start()
    try:
        result = await asyncio.wait_for(
            client.send_request("initialize", {}, timeout=5.0), 5.0
        )
        assert result == {"text": "é雪"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_failed_kill_rpc_retains_process_until_acknowledgement(monkeypatch):
    """Inspect's public kill suppresses the sandbox failure; our owner cannot."""
    from inspect_ai.util import ExecRemoteProcess, ExecRemoteStreamingOptions
    from inspect_ai.util._sandbox.exec_remote import _KillResult

    proc = ExecRemoteProcess(LocalSandbox(), ["codex"], ExecRemoteStreamingOptions(), 5)
    proc._pid = 123
    transport = ExecRemoteJsonRpcTransport(LocalSandbox(), ["codex"])
    transport._proc = proc
    requests = []

    async def rejected(method, params, result_type):
        requests.append((method, params))
        raise RuntimeError("sandbox kill rejected")

    monkeypatch.setattr(proc, "_rpc", rejected)
    with pytest.raises(RuntimeError, match="sandbox kill rejected"):
        await transport.stop()
    assert transport._proc is proc
    assert not proc._killed

    async def acknowledged(method, params, result_type):
        requests.append((method, params))
        return _KillResult(seq=4, stdout="tail\n", stderr="")

    monkeypatch.setattr(proc, "_rpc", acknowledged)
    await transport.stop()
    assert transport._proc is None
    assert proc._killed
    assert proc._last_seq == 4
    assert requests == [("exec_remote_kill", {"pid": 123, "ack_seq": 0})] * 2
    assert (await anext(proc)).data == "tail\n"


@pytest.mark.asyncio
async def test_server_exit_surfaces_as_transport_failure(tmp_path):
    """A dying app-server must raise a typed failure, not hang the solver."""
    server_script = tmp_path / "server.py"
    server_script.write_text(EXIT_SERVER)
    transport = ExecRemoteJsonRpcTransport(
        LocalSandbox(), [sys.executable, str(server_script)]
    )
    client = CodexClient(transport_factory=lambda: transport)  # type: ignore[arg-type]
    await client.start()
    try:
        await client.send_notification("die")
        await asyncio.wait_for(client.failure_event.wait(), 5.0)
        assert isinstance(client.failure, CodexTransportFailure)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_websocket_stalled_upgrade_is_bounded():
    """A TCP connection without an HTTP upgrade cannot consume the run budget."""
    connected = asyncio.Event()
    closed = asyncio.Event()
    tasks = set()

    async def stall(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            await reader.readuntil(b"\r\n\r\n")
            connected.set()
            await reader.read()
            closed.set()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(stall, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    transport = WebSocketJsonRpcTransport(f"ws://127.0.0.1:{port}", connect_timeout=0.2)
    try:
        with pytest.raises(CodexTransportFailure, match="handshake timed out"):
            await asyncio.wait_for(transport.start(), 2)
        assert connected.is_set()
        await asyncio.wait_for(closed.wait(), 2)
    finally:
        await transport.stop()
        server.close()
        await server.wait_closed()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.native
async def test_real_websocket_handshake_rejects_wrong_token(
    evaluation_workspace, native_binary, unused_tcp_port,
):
    home = evaluation_workspace / "codex-home"
    home.mkdir()
    token = "a" * 64
    token_path = home / "capability-token"
    token_path.write_text(token)
    url = f"ws://127.0.0.1:{unused_tcp_port}"
    daemon = ExecRemoteJsonRpcTransport(
        LocalSandbox(), [native_binary, "app-server", "--listen", url,
                         "--ws-auth", "capability-token", "--ws-token-file", str(token_path)],
        environment={"CODEX_HOME": str(home)},
    )
    rejected = CodexClient(transport="websocket", ws_url=url, auth_token="b" * 64)
    client = CodexClient(transport="websocket", ws_url=url, auth_token=token)
    await daemon.start()
    process = daemon._proc._proc
    try:
        with pytest.raises(CodexTransportFailure, match="authentication rejected"):
            await rejected.start()
        await client.start()
        await client.rpc.initialize({
            "clientInfo": {"name": "inspect-ai", "version": "0"},
            "capabilities": {"experimentalApi": True},
        })
        await client.send_notification("initialized")
        models = await client.rpc.model_list({"limit": 5}, timeout=15)
        assert models["data"]
    finally:
        await rejected.close()
        await client.close()
        await daemon.stop()
    assert process.returncode is not None, "The owned native listener must be stopped and reaped"
