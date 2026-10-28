"""Asynchronous JSON-RPC client for the Codex App Server.

Handles transport communication (via container exec_remote or WebSocket),
request/response correlation, incoming server request dispatch, and turn/lifecycle
events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, AsyncIterator, Awaitable, Callable, Coroutine, Literal, Protocol

from inspect_ai.util import message_limit

from . import _generated_protocol as p
from ._generated_rpc import CodexRpc
from .compat import (
    ExecCompleted,
    ExecRemoteStreamingOptions,
    ExecStderr,
    ExecStdout,
    LimitExceededError,
    TerminateSampleError,
    kill_process,
    process_events,
)
from .policy import ReceiptReader, truncate_goal_objective

logger = logging.getLogger(__name__)

ApprovalDecision = Literal["accept", "decline", "terminate"]

# Native wire values for (accept, decline, terminate): v2 item approvals use
# Command/FileChangeApprovalDecision; legacy requests use ReviewDecision.
_APPROVAL_DECISIONS = {
    "item/commandExecution/requestApproval": ("accept", "decline", "cancel"),
    "item/fileChange/requestApproval": ("accept", "decline", "cancel"),
    "execCommandApproval": ("approved", "denied", "abort"),
    "applyPatchApproval": ("approved", "denied", "abort"),
}


_TURN_REQUESTS = frozenset({"item/tool/call", "item/commandExecution/requestApproval",
                            "item/fileChange/requestApproval"})


async def _stop_tasks(tasks: list[asyncio.Task[None]], timeout: float) -> set[asyncio.Task[None]]:
    """Cancel tasks and wait for them; returns those still running at the timeout."""
    for task in tasks:
        task.cancel()
    if not tasks:
        return set()
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    for task in done:
        if not task.cancelled():
            task.exception()
    return pending


class JsonRpcError(Exception):
    """Exception raised when a JSON-RPC error response is received."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(f"JSON-RPC Error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class CodexTransportFailure(RuntimeError):
    """Terminal failure of the client<->app-server connection."""


class _RequestError(Exception):
    """Internal control-flow: answer a server request with a JSON-RPC error."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class _JsonLineFramer:
    """Accumulates transport chunks and yields complete newline-delimited JSON lines."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: str | bytes) -> list[str]:
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._buffer.extend(data)
        lines: list[str] = []
        while True:
            idx = self._buffer.find(b"\n")
            if idx < 0:
                break
            raw = bytes(self._buffer[:idx])
            del self._buffer[: idx + 1]
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                lines.append(text)
        return lines

    def flush(self) -> list[str]:
        remainder = self._buffer.decode("utf-8", errors="replace").strip()
        self._buffer.clear()
        return [remainder] if remainder else []


class CodexTransport(Protocol):
    """Owned, bounded transport carrying newline-delimited JSON-RPC."""

    async def start(self) -> None: ...

    async def write_line(self, line: str) -> None: ...

    def lines(self) -> AsyncIterator[str]: ...

    async def stop(self) -> None: ...

    async def finish(self) -> None:
        await self.stop()


class ExecRemoteJsonRpcTransport(CodexTransport):
    """Run the app-server inside the sandbox via ``exec_remote`` with open stdin.

    The returned ``ExecRemoteProcess`` is the owned process handle: stdout is
    streamed back to us, stdin is writable for requests, and ``kill()``
    terminates the server inside the container without pattern-matching
    process names.
    """

    def __init__(
        self,
        env: Any,
        command: list[str],
        *,
        cwd: str | None = None,
        user: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self._env = env
        self._command = command
        self._cwd = cwd
        self._user = user
        self._environment = environment
        self._proc: Any = None
        self._process_lock = asyncio.Lock()
        self._output_ready = asyncio.Event()

    async def start(self) -> None:
        options = ExecRemoteStreamingOptions(
            stdin_open=True,
            poll_interval=0.1,
            cwd=self._cwd,
            user=self._user,
            env=self._environment or None,
        )
        self._proc = await self._env.exec_remote(self._command, options)

    async def write_line(self, line: str) -> None:
        if self._proc is None:
            raise CodexTransportFailure("exec_remote transport not started")
        try:
            async with self._process_lock:
                await self._proc.write_stdin(line + "\n")
                self._output_ready.set()
        except Exception as e:
            raise CodexTransportFailure(f"failed writing to app-server stdin: {e}") from e

    async def lines(self) -> AsyncIterator[str]:
        if self._proc is None:
            raise CodexTransportFailure("exec_remote transport not started")
        framer = _JsonLineFramer()
        try:
            async for event in process_events(self._proc, self._process_lock, self._output_ready):
                if isinstance(event, ExecStdout):
                    for line in framer.feed(event.data):
                        yield line
                elif isinstance(event, ExecStderr):
                    text = event.data if isinstance(event.data, str) else event.data.decode("utf-8", "replace")
                    if text.strip():
                        logger.debug("[codex stderr] %s", text.strip())
                elif isinstance(event, ExecCompleted):
                    for line in framer.flush():
                        yield line
                    return
            # Yield any remaining lines flushed from the buffer after the stream ends.
            for line in framer.flush():
                yield line
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise CodexTransportFailure(f"exec_remote stream failed: {e}") from e

    async def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        async with self._process_lock:
            await kill_process(proc)
            self._output_ready.set()
        self._proc = None

    async def finish(self) -> None:
        if self._proc is not None:
            async with self._process_lock:
                await self._proc.close_stdin()
                self._output_ready.set()


class WebSocketJsonRpcTransport(CodexTransport):
    """Connect to an app-server WebSocket listener with bearer authentication."""

    def __init__(self, ws_url: str, auth_token: str | None = None, connect_timeout: float = 10.0) -> None:
        self._ws_url = ws_url
        self._auth_token = auth_token
        self._connect_timeout = connect_timeout
        self._session: Any = None
        self._ws: Any = None

    async def start(self) -> None:
        import aiohttp  # imported on use: only the websocket transport needs it

        headers = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        self._session = aiohttp.ClientSession()
        last_error: Exception | None = None
        try:
            async with asyncio.timeout(self._connect_timeout):
                while True:
                    try:
                        self._ws = await self._session.ws_connect(
                            self._ws_url,
                            headers=headers if headers else None,
                            max_msg_size=32 * 1024 * 1024,
                        )
                        return
                    except aiohttp.WSServerHandshakeError as exc:
                        if exc.status in (401, 403):
                            raise CodexTransportFailure("Codex WebSocket authentication rejected") from exc
                        last_error = exc
                    except aiohttp.ClientConnectionError as exc:
                        last_error = exc
                    await asyncio.sleep(0.5)
        except TimeoutError as exc:
            raise CodexTransportFailure(
                f"Could not connect to codex WebSocket at {self._ws_url}: {last_error or 'handshake timed out'}"
            ) from exc
        finally:
            if self._ws is None:
                await self._session.close()
                self._session = None

    async def write_line(self, line: str) -> None:
        if self._ws is None or self._ws.closed:
            raise CodexTransportFailure("Cannot write to app-server: WebSocket closed")
        try:
            await self._ws.send_str(line)
        except Exception as e:
            raise CodexTransportFailure(f"WebSocket send failed: {e}") from e

    async def lines(self) -> AsyncIterator[str]:
        if self._ws is None:
            raise CodexTransportFailure("websocket transport not started")
        from aiohttp import WSMsgType

        try:
            async for ws_msg in self._ws:
                if ws_msg.type == WSMsgType.TEXT:
                    yield ws_msg.data
                elif ws_msg.type == WSMsgType.ERROR:
                    raise CodexTransportFailure(f"WebSocket error: {self._ws.exception()}")
                elif ws_msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.CLOSING):
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise CodexTransportFailure(f"WebSocket stream failed: {e}") from e

    async def stop(self) -> None:
        ws, self._ws = self._ws, None
        session, self._session = self._session, None
        for handle, close in ((ws, "close"), (session, "close")):
            if handle is not None:
                try:
                    await handle.close()
                except Exception as e:
                    logger.debug("Error closing websocket handle: %s", e)


class CodexClient:
    """Asynchronous client communicating with `codex app-server` over JSON-RPC."""

    def __init__(
        self,
        transport: str = "stdio",
        *,
        env: Any = None,
        command: list[str] | None = None,
        cwd: str | None = None,
        user: str | None = None,
        environment: dict[str, str] | None = None,
        ws_url: str | None = None,
        auth_token: str | None = None,
        transport_factory: Callable[[], CodexTransport] | None = None,
        receipt_reader: ReceiptReader | None = None,
    ) -> None:
        if transport_factory is None:
            if transport == "stdio":
                if env is None or not command:
                    raise ValueError("stdio transport requires env and command")
                transport_factory = lambda: ExecRemoteJsonRpcTransport(  # noqa: E731
                    env, command, cwd=cwd, user=user, environment=environment
                )
            elif transport == "websocket":
                if not ws_url:
                    raise ValueError("websocket transport requires ws_url")
                transport_factory = lambda: WebSocketJsonRpcTransport(ws_url, auth_token)  # noqa: E731
            else:
                raise ValueError(f"Unsupported transport: {transport}")

        self.receipt_reader = receipt_reader
        self.root_answers: dict[str, str] = {}
        self.completed_root_answer = ""
        self._transport_factory = transport_factory
        self._transport: CodexTransport | None = None

        self._request_counter = 0
        self._pending_requests: dict[int, asyncio.Future[Any]] = {}
        self._notification_handlers: dict[
            str, list[Callable[[str, dict[str, Any]], Coroutine[Any, Any, None] | None]]
        ] = {}
        self._tool_handlers: dict[
            str, Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]] | dict[str, Any]]
        ] = {}
        self._approval_handler: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None] | None] | None = None
        # Decides native command/file approvals when an Inspect approver is active.
        self.approval_decider: Callable[[str, dict[str, Any]], Awaitable[ApprovalDecision]] | None = None

        # Track turn state by (thread_id, turn_id) to avoid stale completions.
        self._turn_data: dict[tuple[str, str], dict[str, Any]] = {}
        self._turn_events: dict[tuple[str, str], asyncio.Event] = {}
        self._thread_seq: dict[str, int] = {}
        self._thread_event: dict[str, asyncio.Event] = {}
        self._turn_started_seq: dict[str, int] = {}
        self._turn_started_event: dict[str, asyncio.Event] = {}
        self._started_turns: dict[str, list[str]] = {}
        self.active_turn: dict[str, str | None] = {}

        # Root thread tracking and submission state.
        self.root_thread_id: str | None = None
        self.submitted_answer: str | None = None
        self.submitted_turn_id: str | None = None
        self.submit_event = asyncio.Event()
        self._receipts: set[tuple[str | None, str | None]] = set()

        # Store the first fatal exception from transport, reader, or handlers.
        self._failure: BaseException | None = None
        self.failure_event = asyncio.Event()

        self._reader_task: asyncio.Task[None] | None = None
        self._request_tasks: set[asyncio.Task[None]] = set()
        self._turn_requests: dict[tuple[str, str], set[asyncio.Task[None]]] = {}
        self._interrupted_turns: set[tuple[str, str]] = set()
        self._account_update_seq = 0
        self.rpc = CodexRpc(self)
        self._running = False
        self._closing = False
        self.evidence: Any = None
        self.shutdown: dict[str, Any] = {"stream_drained": False, "forced": False, "failures": []}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def register_notification_handler(
        self,
        method: str,
        handler: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None] | None],
    ) -> None:
        """Register a handler for incoming server notifications.

        Handlers run inline in the reader so item ordering is preserved, and
        any exception they raise is forwarded to the failure slot instead of
        being logged and discarded.
        """
        self._notification_handlers.setdefault(method, []).append(handler)

    def register_tool_handler(
        self,
        tool_name: str,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]] | dict[str, Any]],
    ) -> None:
        """Register a handler for dynamic tool calls requested by the server."""
        self._tool_handlers[tool_name] = handler

    def register_approval_handler(
        self,
        handler: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None] | None],
    ) -> None:
        """Register a callback invoked (for transcript evidence) on approval requests.

        The callback's return value does not influence the decision: the
        response is decided by this client's explicit policy.
        """
        self._approval_handler = handler

    def set_root_thread(self, thread_id: str | None) -> None:
        self.root_thread_id = thread_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._transport = self._transport_factory()
        await self._transport.start()
        self._reader_task = asyncio.create_task(self._reader_loop(), name="codex-reader")

    async def close(self, timeout: float = 5.0) -> None:
        """Tear down the connection and the owned app-server process, bounded."""
        self._closing = True
        deadline = asyncio.get_running_loop().time() + timeout
        transport = self._transport
        reader, self._reader_task = self._reader_task, None
        stopped = transport is None
        if transport is not None:
            try:
                async with asyncio.timeout(max(0, timeout - min(5.0, timeout * 0.5))):
                    finish = getattr(transport, "finish", transport.stop)
                    await finish()
                    if reader is not None:
                        await asyncio.shield(reader)
            except Exception as e:
                self.shutdown["forced"] = True
                self.shutdown["failures"].append(f"graceful close: {type(e).__name__}")
            finally:
                try:
                    async with asyncio.timeout(max(0, deadline - asyncio.get_running_loop().time())):
                        await transport.stop()
                        stopped = True
                        if reader is not None:
                            await asyncio.shield(reader)
                except Exception as e:
                    self.shutdown["failures"].append(f"transport close: {type(e).__name__}")
        self._running = False
        self.shutdown["transport_stopped"] = stopped
        if stopped:
            self._transport = None
        self._fail_all_pending("Codex client closed")
        if reader is not None and not reader.done():
            self.shutdown["reader_abandoned"] = True
            reader.cancel()
            # Cancellation cannot extend the absolute cleanup deadline.
            await asyncio.wait([reader], timeout=max(0, deadline - asyncio.get_running_loop().time()))

        pending = await _stop_tasks(list(self._request_tasks), max(0, deadline - asyncio.get_running_loop().time()))
        if pending:
            logger.warning("%d Codex request handlers did not stop within %.1fs", len(pending), timeout)
            self.shutdown["failures"].append(f"{len(pending)} request handlers did not stop")

    # ------------------------------------------------------------------
    # Failure propagation
    # ------------------------------------------------------------------

    def _set_failure(self, exc: BaseException) -> None:
        if self._failure is None:
            self._failure = exc
            self.failure_event.set()
        self._fail_all_pending(f"codex client failure: {exc}")

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def clear_submission(self) -> None:
        """Withdraw the accepted submission so the run can continue (e.g. an incorrect attempt)."""
        self.submitted_answer = None
        self.submitted_turn_id = None
        self.submit_event.clear()

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _fail_all_pending(self, reason: str) -> None:
        for fut in list(self._pending_requests.values()):
            if not fut.done():
                fut.set_exception(CodexTransportFailure(reason))
        self._pending_requests.clear()

    # ------------------------------------------------------------------
    # Reader + dispatch
    # ------------------------------------------------------------------

    async def _reader_loop(self) -> None:
        assert self._transport is not None
        try:
            async for line in self._transport.lines():
                if not self._running:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    if self.evidence:
                        self.evidence.record("in", {"invalidJson": line})
                    logger.warning("Invalid JSON received from codex app-server: %.200s", line)
                    continue
                if self.evidence:
                    self.evidence.record("in", msg)
                await self._dispatch_message(msg)
            self.shutdown["stream_drained"] = True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            failure = CodexTransportFailure(f"reader loop failed: {e}")
            failure.__cause__ = e
            self._set_failure(failure)
            return
        finally:
            if self._running and not self._closing:
                self._set_failure(CodexTransportFailure("codex app-server stream closed"))
                self._running = False

    async def _dispatch_message(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        method = msg.get("method")

        # 1. Server request: handle concurrently so handlers making RPCs don't block the reader.
        if req_id is not None and method:
            params = msg.get("params", {})
            # Bridged tools and approvals awaiting Inspect belong to their turn.
            key = (params["threadId"], params["turnId"]) if method in _TURN_REQUESTS else None
            task = asyncio.create_task(
                self._serve_server_request(req_id, method, params),
                name=f"codex-request-{method}",
            )
            self._request_tasks.add(task)
            task.add_done_callback(self._request_tasks.discard)
            if key is not None:
                self._turn_requests.setdefault(key, set()).add(task)
                task.add_done_callback(lambda finished, key=key: self._finish_turn_request(key, finished))
            return

        # 2. Response to a client request.
        if req_id is not None:
            fut = self._pending_requests.pop(req_id, None)
            if fut and not fut.done():
                if msg.get("error"):
                    err = msg["error"]
                    fut.set_exception(
                        JsonRpcError(
                            code=err.get("code", -32000),
                            message=err.get("message", "Unknown error"),
                            data=err.get("data"),
                        )
                    )
                else:
                    fut.set_result(msg.get("result"))
            return

        # 3. Server notification.
        if method:
            await self._handle_server_notification(method, msg.get("params", {}))

    async def _handle_server_notification(self, method: str, params: dict[str, Any]) -> None:
        logger.debug("Received server notification: %s", method)
        if method == "account/updated":
            self._account_update_seq += 1
        if method == "turn/started":
            thread_id = params.get("threadId")
            turn = params.get("turn") or {}
            turn_id = turn.get("id") if isinstance(turn, dict) else None
            if thread_id:
                self.active_turn[thread_id] = turn_id
                if turn_id and turn_id not in self._started_turns.setdefault(thread_id, []):
                    self._started_turns[thread_id].append(turn_id)
                self._turn_started_seq[thread_id] = self._turn_started_seq.get(thread_id, 0) + 1
                self._turn_started_event.setdefault(thread_id, asyncio.Event()).set()

        if method == "turn/completed":
            thread_id = params.get("threadId")
            turn = params.get("turn") or {}
            turn_id = turn.get("id") if isinstance(turn, dict) else None
            if thread_id and turn_id:
                key = (thread_id, turn_id)
                self._turn_data[key] = params
                self._turn_events.setdefault(key, asyncio.Event()).set()
                self._thread_seq[thread_id] = self._thread_seq.get(thread_id, 0) + 1
                self._thread_event.setdefault(thread_id, asyncio.Event()).set()
                if self.active_turn.get(thread_id) == turn_id:
                    self.active_turn[thread_id] = None
                if thread_id == self.root_thread_id and turn.get("status") == "completed":
                    # Capture completed root answer from the protocol stream.
                    self.completed_root_answer = self.root_answers.get(turn_id, "")

        if method == "item/completed":
            item = params.get("item") or {}
            if (item.get("type") == "subAgentActivity" and item.get("kind") == "started"
                    and "thread/metadata/read" in self._notification_handlers):
                task = asyncio.create_task(self._read_child_metadata(item["agentThreadId"]))
                self._request_tasks.add(task)
                task.add_done_callback(self._request_tasks.discard)
            if item.get("type") == "agentMessage" and params.get("threadId") == self.root_thread_id:
                turn_id = params.get("turnId")
                if turn_id is not None and item.get("phase") in (None, "final_answer"):
                    self.root_answers[turn_id] = item.get("text", "")
            answer = self.receipt_reader(item, self.root_thread_id, params.get("threadId")) if self.receipt_reader else None
            receipt = (params.get("threadId"), item.get("id"))
            # A redelivered receipt cannot re-arm a submission that was withdrawn.
            if answer and receipt not in self._receipts:
                self._receipts.add(receipt)
                self.submitted_answer = answer
                self.submitted_turn_id = params.get("turnId")
                self.submit_event.set()

        # Run notification handlers sequentially to preserve item ordering.
        for handler in self._notification_handlers.get(method, []):
            try:
                with message_limit(None) if self._closing or self._failure else nullcontext():
                    res = handler(method, params)
                    if asyncio.iscoroutine(res):
                        await res
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Error in notification handler for %s: %s", method, e)
                self._set_failure(e)

        if (method == "account/updated" and "account/metadata/read" in self._notification_handlers
                and not self._closing and self._failure is None):
            # Refresh account details via account/read concurrently to avoid blocking the reader.
            task = asyncio.create_task(self._read_account_metadata(self._account_update_seq))
            self._request_tasks.add(task)
            task.add_done_callback(self._request_tasks.discard)

    async def _read_account_metadata(self, revision: int) -> None:
        try:
            result = await self.rpc.account_read({"refreshToken": False}, timeout=10.0)
            if revision == self._account_update_seq:
                await self._handle_server_notification("account/metadata/read", dict(result))
        except asyncio.CancelledError:
            raise
        except JsonRpcError as exc:
            if exc.code != -32601:
                self._set_failure(exc)
        except Exception as exc:
            self._set_failure(exc)

    async def _read_child_metadata(self, thread_id: str) -> None:
        try:
            result = await self.rpc.thread_read({"threadId": thread_id, "includeTurns": False})
            await self._handle_server_notification("thread/metadata/read", dict(result))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._closing:
                # A child spawned as the session closes cannot change the run's outcome.
                self.shutdown["failures"].append(f"child metadata {thread_id}: {type(exc).__name__}")
                return
            self._set_failure(exc)

    # ------------------------------------------------------------------
    # Server requests: typed, schema-aware responses
    # ------------------------------------------------------------------

    async def _serve_server_request(self, req_id: Any, method: str, params: dict[str, Any]) -> None:
        logger.debug("Received server request: %s (id=%s)", method, req_id)
        try:
            self.raise_if_failed()
            result = await self._answer_server_request(method, params)
        except _RequestError as e:
            await self._send_response_safe(req_id, error={"code": e.code, "message": e.message})
        except LimitExceededError as e:
            self._set_failure(e)
            await self._send_response_safe(req_id, error={"code": -32000, "message": str(e)})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Unexpected error answering server request %s", method)
            self._set_failure(e)
            await self._send_response_safe(
                req_id, error={"code": -32000, "message": f"client handler error: {e}"}
            )
        else:
            await self._send_response_safe(req_id, result=result)

    async def _answer_server_request(self, method: str, params: dict[str, Any]) -> Any:
        if method in _APPROVAL_DECISIONS:
            accept, decline, cancel = _APPROVAL_DECISIONS[method]
            if (params.get("threadId"), params.get("turnId")) in self._interrupted_turns:
                return {"decision": decline}
            if self.approval_decider is None:
                await self._notify_approval_handler(method, params)
                return {"decision": accept}
            decision = await self.approval_decider(method, params)
            if decision == "terminate":
                # Inspect's terminate ends the sample, as it does for bridged tools.
                self._set_failure(TerminateSampleError("Tool call approver requested termination."))
            return {"decision": {"accept": accept, "decline": decline, "terminate": cancel}[decision]}

        if method == "item/permissions/requestApproval":
            await self._notify_approval_handler(method, params)
            # GrantedPermissionProfile with null additional grants: the
            # danger-full-access sandbox policy already permits the request.
            return {"permissions": {"fileSystem": None, "network": None}, "scope": "session"}

        if method == "item/tool/call":
            return await self._answer_tool_call(params)

        if method == "item/tool/requestUserInput":
            # In unattended evaluation, respond with empty answers since no interactive input exists.
            logger.warning("requestUserInput answered with no input (unattended run): %s", params)
            return {"answers": {}}

        if method == "mcpServer/elicitation/request":
            # Decline elicitation requests in unattended evaluation.
            return {"action": "decline"}

        if method == "account/chatgptAuthTokens/refresh":
            raise _RequestError(
                -32000, "Unattended client cannot refresh ChatGPT auth tokens; auth.json is pre-provisioned."
            )

        if method == "attestation/generate":
            raise _RequestError(-32000, "Client did not opt into attestation; refusing to fabricate a token.")

        if method == "currentTime/read":
            # Schema: CurrentTimeReadResponse.
            return {"currentTimeAt": int(time.time())}

        raise _RequestError(-32601, f"Method not supported by this client: {method}")

    async def _notify_approval_handler(self, method: str, params: dict[str, Any]) -> None:
        if self._approval_handler:
            try:
                res = self._approval_handler(method, params)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                logger.debug("Error in approval evidence handler: %s", e)

    async def _answer_tool_call(self, params: dict[str, Any]) -> Any:
        if (params["threadId"], params["turnId"]) in self._interrupted_turns:
            return {
                "contentItems": [{"type": "inputText", "text": "This turn was interrupted; the tool was not executed."}],
                "success": False,
            }
        tool_name = params.get("tool", "")
        handler = self._tool_handlers.get(tool_name)
        if handler is None:
            # Never report success for a tool we did not execute.
            logger.error("No handler registered for dynamic tool %r; reporting failure", tool_name)
            return {
                "contentItems": [
                    {"type": "inputText", "text": f"Tool '{tool_name}' is not available in this harness."}
                ],
                "success": False,
            }
        res = handler(params)
        if asyncio.iscoroutine(res):
            res = await res
        return res

    async def _send_response_safe(self, req_id: Any, result: Any = None, error: Any = None) -> bool:
        try:
            await self.send_response(req_id, result=result, error=error)
            return True
        except Exception as e:
            self._set_failure(CodexTransportFailure(f"Failed sending response for server request {req_id}: {e}"))
            return False

    # ------------------------------------------------------------------
    # Outgoing messages
    # ------------------------------------------------------------------

    async def _send_payload(self, payload: dict[str, Any]) -> None:
        if self._transport is None:
            raise CodexTransportFailure("client is not started")
        if self.evidence:
            self.evidence.record("out", payload)
        try:
            await self._transport.write_line(json.dumps(payload))
        except BaseException as exc:
            if self.evidence:
                self.evidence.record("write", {"requestId": payload.get("id"), "status": "failed", "errorType": type(exc).__name__})
            raise
        else:
            if self.evidence:
                self.evidence.record("write", {"requestId": payload.get("id"), "status": "completed"})

    async def send_request(
        self, method: str, params: Mapping[str, Any] | None = None, timeout: float | None = 60.0,
        *, _allow_failed: bool = False,
    ) -> Any:
        """Send a JSON-RPC request and await its result.

        The pending future is removed on every exit path, including failed
        writes and cancellation, so nothing leaks into ``_pending_requests``.
        """
        # Cleanup may bypass a local limit failure while the reader is alive.
        # Once the stream has died, no RPC can receive an acknowledgement.
        if not _allow_failed or not self._running:
            self.raise_if_failed()
        req_id = self._next_id()
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = fut

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": dict(params) if params else {},
        }
        try:
            # A sandbox write is itself an RPC and may block. The request's
            # deadline covers transport admission and the native reply together.
            async with asyncio.timeout(timeout):
                await self._send_payload(payload)
                return await fut
        except Exception:
            if not _allow_failed:
                self.raise_if_failed()
            raise
        finally:
            self._pending_requests.pop(req_id, None)
            if not fut.done():
                fut.cancel()
            elif not fut.cancelled():
                fut.exception()  # A reader failure may arrive during a blocked write.

    async def send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        await self._send_payload(payload)

    async def send_response(self, req_id: Any, result: Any = None, error: Any = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result if result is not None else {}
        await self._send_payload(payload)

    # ------------------------------------------------------------------
    # Turn waiting (keyed by thread AND turn id)
    # ------------------------------------------------------------------

    def _key_event(self, key: tuple[str, str]) -> asyncio.Event:
        return self._turn_events.setdefault(key, asyncio.Event())

    def turn_data(self, thread_id: str, turn_id: str) -> dict[str, Any] | None:
        return self._turn_data.get((thread_id, turn_id))

    def latest_turn_data(self, thread_id: str) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for (tid, _turn), data in self._turn_data.items():
            if tid == thread_id:
                best = data
        return best

    def thread_seq(self, thread_id: str) -> int:
        return self._thread_seq.get(thread_id, 0)

    def thread_event(self, thread_id: str) -> asyncio.Event:
        return self._thread_event.setdefault(thread_id, asyncio.Event())

    def turn_started_seq(self, thread_id: str) -> int:
        return self._turn_started_seq.get(thread_id, 0)

    def turn_started_event(self, thread_id: str) -> asyncio.Event:
        return self._turn_started_event.setdefault(thread_id, asyncio.Event())

    async def wait_for_next_turn(
        self, thread_id: str, seen: set[str], timeout: float
    ) -> str | None:
        """Consume unobserved starts, including turns that already completed."""
        deadline = asyncio.get_running_loop().time() + timeout
        event = self.turn_started_event(thread_id)
        while True:
            self.raise_if_failed()
            if self.submit_event.is_set():
                return None
            event.clear()
            for turn_id in self._started_turns.get(thread_id, []):
                if turn_id not in seen:
                    return turn_id
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return None
            tasks = [asyncio.create_task(signal.wait()) for signal in
                     (event, self.submit_event, self.failure_event)]
            try:
                done, _ = await asyncio.wait(tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    return None
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_for_turn(
        self,
        thread_id: str,
        turn_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Wait for ``turn/completed`` on a specific thread (and turn, if given).

        Waits race the failure and submit signals, so transport death or a
        submission during the turn wake this coroutine immediately.
        """
        self.raise_if_failed()
        if turn_id is not None:
            key = (thread_id, turn_id)
            if key in self._turn_data:
                return self._turn_data[key]

        start_seq = self.thread_seq(thread_id)
        cond_task = asyncio.create_task(self._turn_condition(thread_id, turn_id, start_seq))
        submit_task = asyncio.create_task(self.submit_event.wait())
        failure_task = asyncio.create_task(self.failure_event.wait())
        try:
            done, _pending = await asyncio.wait(
                [cond_task, submit_task, failure_task],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            self.raise_if_failed()
            if submit_task in done and self.submit_event.is_set():
                return {"turn": {"status": "submitted"}}
            if not done:
                raise TimeoutError("Codex turn wait timed out")
            if turn_id is not None:
                data = self._turn_data.get((thread_id, turn_id))
            elif self.thread_seq(thread_id) != start_seq:
                data = self.latest_turn_data(thread_id)
            else:
                data = None
            if data is None:
                # Woken by failure_event or timeout with no completed turn data.
                self.raise_if_failed()
                raise TimeoutError("Codex turn wait timed out")
            return data
        finally:
            cond_task.cancel()
            submit_task.cancel()
            failure_task.cancel()
            await asyncio.gather(cond_task, submit_task, failure_task, return_exceptions=True)

    async def _turn_condition(self, thread_id: str, turn_id: str | None, start_seq: int) -> None:
        if turn_id is not None:
            await self._key_event((thread_id, turn_id)).wait()
        else:
            while self.thread_seq(thread_id) == start_seq:
                event = self.thread_event(thread_id)
                event.clear()
                await event.wait()

    # ------------------------------------------------------------------
    # Convenience RPCs
    # ------------------------------------------------------------------

    def _finish_turn_request(self, key: tuple[str, str], task: asyncio.Task[None]) -> None:
        requests = self._turn_requests[key]
        requests.discard(task)
        if not requests:
            del self._turn_requests[key]

    async def cancel_turn_tools(self, thread_id: str, turn_id: str, timeout: float = 10.0) -> None:
        """Stop Inspect-owned tools before an interrupted turn can resume."""
        self._interrupted_turns.add((thread_id, turn_id))
        await self._cancel_tools(list(self._turn_requests.get((thread_id, turn_id), ())), timeout)

    async def cancel_tools(self, timeout: float = 10.0) -> None:
        """Stop outstanding tools, including those whose native turn has already ended."""
        self._interrupted_turns.update(self._turn_requests)
        self._interrupted_turns.update((thread, turn) for thread, turn in self.active_turn.items() if turn)
        await self._cancel_tools([task for tasks in self._turn_requests.values() for task in tasks], timeout)

    async def _cancel_tools(self, tasks: list[asyncio.Task[None]], timeout: float) -> None:
        if await _stop_tasks(tasks, timeout):
            # A tool still running could act after the redirect, so nothing may resume.
            failure = CodexTransportFailure(f"Interrupted tools did not stop within {timeout:g}s")
            self._set_failure(failure)
            raise failure

    async def interrupt_turn(self, thread_id: str, turn_id: str | None = None) -> None:
        """Send ``turn/interrupt`` to abort the active turn.

        The turn's bridged tools and approvals are cancelled but not awaited:
        Codex completes an interrupted turn without their responses, and one
        that ignores cancellation must not delay the interrupt. Before
        redirecting the turn, wait for them with ``cancel_turn_tools``.
        """
        turn_id = turn_id or self.active_turn.get(thread_id)
        if turn_id:
            self._interrupted_turns.add((thread_id, turn_id))
            for task in self._turn_requests.get((thread_id, turn_id), ()):
                task.cancel()
        params: p.TurnInterruptParams = {"threadId": thread_id, "turnId": turn_id or ""}
        try:
            await self.rpc.turn_interrupt(params, timeout=10.0, _allow_failed=True)
        except JsonRpcError as exc:
            # If the turn already completed, ignore the "no active turn" error.
            if not (exc.code == -32600 and exc.message == "no active turn to interrupt"
                    and turn_id is not None and self.turn_data(thread_id, turn_id) is not None):
                raise
        logger.info("Successfully interrupted turn %s on thread %s", turn_id, thread_id)

    async def set_goal(
        self,
        thread_id: str,
        objective: str | None = None,
        status: p.ThreadGoalStatus = "active",
    ) -> p.ThreadGoalSetResponse:
        """Set or update the goal for a thread via ``thread/goal/set``.

        Only schema fields are sent (``goalId`` does not exist on the wire).
        """
        params: p.ThreadGoalSetParams = {"threadId": thread_id, "status": status}
        if objective is not None:
            if len(objective) > 4000:
                objective = truncate_goal_objective(objective, 4000)
            params["objective"] = objective
        return await self.rpc.thread_goal_set(params)

    async def get_goal(self, thread_id: str) -> p.ThreadGoalGetResponse:
        """Get the current goal for a thread via ``thread/goal/get``."""
        return await self.rpc.thread_goal_get({"threadId": thread_id})
