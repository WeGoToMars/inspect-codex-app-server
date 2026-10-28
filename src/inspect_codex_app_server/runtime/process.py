"""Sandbox remote process execution and transport utilities."""

import asyncio
from typing import Any, AsyncIterator

from inspect_ai.util import ExecRemoteProcess
from inspect_ai.util._sandbox.exec_remote import ExecCompleted, _KillResult


async def kill_process(process: ExecRemoteProcess) -> None:
    """Terminate a sandbox process and wait for remote acknowledgement."""
    if not isinstance(process, ExecRemoteProcess):
        if not getattr(process, "_killed", False):
            await process.kill()
        return
    if process._pid is None or process._completed:
        return
    result = await process._rpc(
        "exec_remote_kill",
        {"pid": process._pid, "ack_seq": process._last_seq},
        _KillResult,
    )
    process._last_seq = result.seq
    process._enqueue_output(result.stdout, result.stderr)
    process._killed = True


async def process_events(
    process: Any, lock: asyncio.Lock, output_ready: asyncio.Event
) -> AsyncIterator[Any]:
    """Stream remote process stdout/stderr events safely under a shared lock."""
    if not isinstance(process, ExecRemoteProcess):
        async for event in process:
            yield event
        return
    aiter(process)  # Preserve Inspect's single-use iteration contract.
    while True:
        async with lock:
            output_ready.clear()
            if not process._pending_events:
                if process._completed or process._killed:
                    return
                result = await process._poll()
                process._enqueue_output(result.stdout, result.stderr)
                if result.state == "completed":
                    if result.exit_code is None:
                        raise RuntimeError("Server returned completed state without exit_code")
                    process._completed = True
                    process._pending_events.append(ExecCompleted(exit_code=result.exit_code))
                elif result.state == "killed":
                    process._killed = True
            event = process._pending_events.pop(0) if process._pending_events else None
            if event is None and (process._completed or process._killed):
                return
        if event is not None:
            yield event
        else:
            try:
                async with asyncio.timeout(process._poll_interval):
                    await output_ready.wait()
            except TimeoutError:
                pass
