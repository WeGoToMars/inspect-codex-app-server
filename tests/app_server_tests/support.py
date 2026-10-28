"""Owned local processes and fixed protocol stimuli; no benchmark policy."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from contextlib import asynccontextmanager
from typing import Any

from aiohttp import web
from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import TaskState

from inspect_codex_app_server.client import CodexClient
from inspect_codex_app_server.container import REQUIRED_CLI_VERSION
from inspect_codex_app_server.mapping import EventStreamMapper


def make_state(
    messages: int = 1,
    metadata: dict[str, Any] | None = None,
    model: str = "codex/test",
) -> TaskState:
    """A minimal TaskState for unit tests; suite conftests pin the model name."""
    return TaskState(
        model=ModelName(model),
        sample_id="test-sample",
        epoch=1,
        input="task input",
        messages=[ChatMessageUser(content="task input")] if messages else [],
        metadata=metadata or {},
    )


def make_mapper(state=None, *, model_name="codex/test") -> EventStreamMapper:
    mapper = EventStreamMapper(model_name=model_name, state=state)
    mapper.root_thread_id = "root"
    return mapper


def model_events(events):
    return [e for e in events if getattr(e, "event", None) == "model"]


class LocalExecProcess:
    """Real local process exposing Inspect's streaming subprocess interface."""

    def __init__(self, proc, chunk_size):
        self._proc = proc
        self._chunk_size = chunk_size
        self._events = self._read_events()

    @property
    def _completed(self):
        return self._proc.returncode is not None

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await anext(self._events)

    async def _read_events(self):
        # These are the event types returned by the pinned Inspect exec_remote API.
        from inspect_ai.util._sandbox.exec_remote import (
            ExecCompleted,
            ExecStderr,
            ExecStdout,
        )

        pending = {asyncio.create_task(stream.read(self._chunk_size)): (stream, event)
                   for stream, event in ((self._proc.stdout, ExecStdout), (self._proc.stderr, ExecStderr))}
        try:
            while pending:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    stream, event = pending.pop(task)
                    data = task.result()
                    if data:
                        pending[asyncio.create_task(stream.read(self._chunk_size))] = (stream, event)
                        yield event(data=data)
            yield ExecCompleted(exit_code=await self._proc.wait())
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def write_stdin(self, data):
        self._proc.stdin.write(data if isinstance(data, bytes) else data.encode())
        await self._proc.stdin.drain()

    async def close_stdin(self):
        self._proc.stdin.close()

    async def kill(self):
        if self._proc.returncode is None:
            self._proc.kill()
        await self._proc.wait()
        await self._events.aclose()


class LocalSandbox:
    def __init__(self, chunk_size=4096):
        self.chunk_size = chunk_size

    async def exec_remote(self, cmd, options=None):
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **(options.env or {})} if options else None,
            cwd=options.cwd if options else None,
        )
        return LocalExecProcess(proc, self.chunk_size)


@asynccontextmanager
async def responses_peer(handler):
    """Own the local HTTP listener even when native setup fails before inference."""
    app = web.Application()
    app.router.add_post("/v1/responses", handler)
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        await runner.cleanup()


def usage_params(thread: str, total: int, last: int | None = None) -> dict[str, Any]:
    d = {"inputTokens": total, "outputTokens": 0, "totalTokens": total}
    token_usage: dict[str, Any] = {"total": d}
    if last is not None:
        token_usage["last"] = {"inputTokens": last, "outputTokens": 0, "totalTokens": last}
    return {"threadId": thread, "tokenUsage": token_usage}


def completed_item(thread: str, item: dict[str, Any]) -> dict[str, Any]:
    return {"threadId": thread, "item": item}


def command_item(item_id: str, command: str = "true", output: str = "ok", exit_code: int = 0) -> dict[str, Any]:
    return {
        "type": "commandExecution",
        "id": item_id,
        "command": command,
        "status": "completed",
        "exitCode": exit_code,
        "aggregatedOutput": output,
    }


class ScriptedAppServer:
    """In-process app-server stand-in used as a CodexTransport.

    Records every client request; replies from ``responses`` and fires the
    matching ``hooks`` so tests can push notifications at the right moment.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.responses: dict[str, Any] = {"initialize": {}, "turn/interrupt": {}}
        self.hooks: dict[str, Any] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.client_replies: asyncio.Queue[str] = asyncio.Queue()
        self.notifications_sent: list[str] = []
        self.stopped = False
        self.responses["thread/read"] = lambda params: {"thread": {"id": params["threadId"]}}

    # -- transport interface ------------------------------------------------
    async def start(self) -> None: ...

    async def write_line(self, line: str) -> None:
        msg = json.loads(line)
        if "id" in msg and "method" in msg:
            method = msg["method"]
            params = msg.get("params", {})
            self.requests.append((method, params))
            handler = self.responses.get(method, {"error": {"code": -32601, "message": f"Unexpected request: {method}"}})
            result = handler(params) if callable(handler) else handler
            if isinstance(result, dict) and "error" in result:
                # JSON-RPC error response (e.g. transient turn/start rejections).
                self.enqueue({"id": msg["id"], "error": result["error"]})
            else:
                self.enqueue({"id": msg["id"], "result": result if result is not None else {}})
            hook = self.hooks.get(method)
            if hook:
                hook(self, params)
        elif "method" in msg:
            self.notifications_sent.append(msg["method"])
        else:
            # A response from the client to one of our server requests; keep it
            # on a side channel so the client reader does not consume it.
            self.client_replies.put_nowait(line)

    async def lines(self):
        while True:
            line = await self.queue.get()
            if line is None:
                return
            yield line

    async def stop(self) -> None:
        self.stopped = True
        self.queue.put_nowait(None)

    # -- helpers -------------------------------------------------------------
    def enqueue(self, msg: dict[str, Any]) -> None:
        self.queue.put_nowait(json.dumps(msg))

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.notifications_sent.append(method)
        self.enqueue({"method": method, "params": params})

    def server_request(self, req_id: int, method: str, params: dict[str, Any]) -> None:
        self.enqueue({"id": req_id, "method": method, "params": params})



def make_client(scripted: ScriptedAppServer) -> CodexClient:
    return CodexClient(transport_factory=lambda: scripted)


def host_run(command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=120, **kwargs)


def project_resources(project):
    label = f"label=com.docker.compose.project={project}"
    return {
        kind: host_run(["docker", *arguments, "--filter", label]).stdout.strip()
        for kind, arguments in (
            ("containers", ["ps", "-aq"]),
            ("networks", ["network", "ls", "-q"]),
            ("volumes", ["volume", "ls", "-q"]),
        )
    }



def patch_session(monkeypatch: Any, transport: Any, env: Any, on_client: Any = None) -> None:
    """Run sessions on a scripted transport in a fake sandbox, without provisioning a container."""
    import inspect_codex_app_server.session as implementation

    async def resolve(env: Any, name: str) -> str:
        return "owned-test-container"

    async def provision(**kwargs: Any) -> tuple[str, str, str]:
        return "/usr/local/bin/codex", "/isolated-codex-home", REQUIRED_CLI_VERSION

    def client(**kwargs: Any) -> CodexClient:
        if on_client is not None:
            on_client(kwargs)
        return CodexClient(transport_factory=lambda: transport, receipt_reader=kwargs["receipt_reader"])

    monkeypatch.setattr(implementation, "sandbox", lambda *args: env)
    monkeypatch.setattr(implementation, "resolve_service_container", resolve)
    monkeypatch.setattr(implementation, "ensure_container_codex", provision)
    monkeypatch.setattr(implementation, "CodexClient", client)


def configure_common(scripted: ScriptedAppServer) -> None:
    scripted.responses["initialize"] = {}
    scripted.responses["thread/start"] = {"thread": {"id": "root-thread"}}
    scripted.responses["thread/inject_items"] = {}
    scripted.responses["skills/list"] = {"data": []}
    scripted.responses["thread/goal/set"] = {"goal": {"status": "active"}}
    scripted.responses["thread/goal/get"] = {"goal": {"status": "active"}}
    scripted.responses["model/list"] = {"data": [{"id": "test", "isDefault": True, "supportedReasoningEfforts": [{"reasoningEffort": "medium"}]}]}
    scripted.responses["turn/start"] = {"turn": {"id": "turn-1"}}
    scripted.responses["turn/interrupt"] = {}

