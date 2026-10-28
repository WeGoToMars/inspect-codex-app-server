"""Bridge Inspect tools to the Codex App Server as dynamic tools."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, execute_tools
from inspect_ai.tool import ToolCall

from . import compat
from .container import PINNED_CODEX_VERSION


async def _media_url(value: str, default_mime: str) -> str:
    if value.startswith(("https://", "http://", "data:")):
        return value
    path = Path(value).expanduser()
    data = await asyncio.to_thread(path.read_bytes)
    mime = mimetypes.guess_type(path.name)[0] or default_mime
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def inspect_tool_handler(tool: Any, mapper: Any, review: list[Any] | None = None) -> Any:
    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        thread_id = params["threadId"]
        call_id = params["callId"]
        call = ToolCall(
            id=call_id, function=params["tool"], arguments=params["arguments"]
        )
        message = ChatMessageAssistant(content="", tool_calls=[call])
        # Snapshot the current thread's conversation history for tool execution.
        history = mapper.thread_messages(thread_id)
        # Track the tool call to prevent duplicate transcript events.
        mapper.inspect_tool_calls.add((thread_id, call_id))
        with compat.native_thread_span(mapper.thread_span(thread_id)):
            executed = await execute_tools([*history, message], [tool], review=review)
        mapper.inspect_exchanges[(thread_id, call_id)] = (message, executed.messages)
        compat.check_limits()
        result = next(
            m
            for m in executed.messages
            if isinstance(m, ChatMessageTool) and m.tool_call_id == call_id
        )
        items: list[dict[str, str]] = []
        if result.error:
            items.append({"type": "inputText", "text": result.error.message})
        for returned in executed.messages:
            if returned is not result:
                # Include handoff dialogue from delegated tool execution.
                items.append({"type": "inputText", "text": f"[Inspect handoff {returned.role}]"})
            if isinstance(returned.content, str):
                items.append({"type": "inputText", "text": returned.content})
                continue
            for block in returned.content:
                if block.type == "reasoning":
                    continue  # A handoff agent's reasoning is not part of its reply.
                if block.type == "text":
                    items.append({"type": "inputText", "text": block.text})
                elif block.type == "image":
                    items.append(
                        {
                            "type": "inputImage",
                            "imageUrl": await _media_url(block.image, "image/png"),
                        }
                    )
                elif block.type == "audio":
                    items.append(
                        {
                            "type": "inputAudio",
                            "audioUrl": await _media_url(block.audio, "audio/wav"),
                        }
                    )
                else:
                    # Report unsupported content types back to Codex.
                    return {
                        "success": False,
                        "contentItems": [
                            {
                                "type": "inputText",
                                "text": f"Codex {PINNED_CODEX_VERSION} cannot represent Inspect {block.type} tool content.",
                            }
                        ],
                    }
        return {"success": result.error is None, "contentItems": items}

    return handler
