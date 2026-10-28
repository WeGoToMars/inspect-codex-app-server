"""Pinned Codex history replay and audio input handling.

A local scripted provider captures message order, image content, and
audio bytes from the real app-server. Catalog controls establish when
Codex forwards audio or replaces it with an unsupported-content notice.
"""

import base64
import json

import pytest
from aiohttp import web
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentText,
)
from inspect_ai.tool import ToolCall

from inspect_codex_app_server.client import CodexClient, ExecRemoteJsonRpcTransport
from inspect_codex_app_server.session import _history_items

from .support import LocalSandbox, responses_peer

PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJ"
       "RU5ErkJggg==")
HISTORY = [
    ChatMessageSystem(content="HISTORY_SYSTEM"),
    ChatMessageUser(content=[ContentText(text="HISTORY_USER"), ContentImage(image=PNG)]),
    ChatMessageAssistant(content="HISTORY_ASSISTANT", tool_calls=[ToolCall(id="h1", function="lookup",
                                                                            arguments={"key": 1})]),
    ChatMessageTool(content="HISTORY_TOOL_RESULT", tool_call_id="h1", function="lookup"),
]


@pytest.mark.native
@pytest.mark.parametrize("replay", [True, False])
async def test_native_injected_history_precedes_the_first_turn(evaluation_workspace, native_binary, replay):
    bodies = []

    async def respond(request):
        bodies.append(await request.json())
        events = [
            {"type": "response.created", "response": {"id": "r1"}},
            {"type": "response.output_item.done", "item": {
                "type": "message", "role": "assistant", "id": "m1", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "done"}]}},
            {"type": "response.completed", "response": {"id": "r1", "usage": {
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                "input_tokens_details": None, "output_tokens_details": None}}},
        ]
        return web.Response(text="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
                            content_type="text/event-stream")

    items, _ = await _history_items(HISTORY)
    async with responses_peer(respond) as base_url:
        home = evaluation_workspace / "native-home"
        home.mkdir()
        (home / "config.toml").write_text(
            f'model="gpt-5.4"\nmodel_provider="scripted"\n[model_providers.scripted]\nname="scripted"\n'
            f'base_url="{base_url}"\nwire_api="responses"\nsupports_websockets=false\n'
            'requires_openai_auth=false\nrequest_max_retries=0\nstream_max_retries=0\n'
        )
        native = ExecRemoteJsonRpcTransport(LocalSandbox(), [native_binary, "app-server", "--stdio"], cwd=str(home),
                                            environment={"CODEX_HOME": str(home)})
        client = CodexClient(transport_factory=lambda: native)
        try:
            await client.start()
            await client.rpc.initialize({"clientInfo": {"name": "test", "version": "0"},
                                         "capabilities": {"experimentalApi": True}})
            await client.send_notification("initialized")
            thread = await client.rpc.thread_start({"cwd": str(home), "approvalPolicy": "never",
                                                    "sandbox": "danger-full-access"})
            thread_id = thread["thread"]["id"]
            if replay:
                await client.rpc.thread_inject_items({"threadId": thread_id, "items": items})
            turn = await client.rpc.turn_start({"threadId": thread_id, "input": [
                {"type": "text", "text": "CURRENT_INPUT", "text_elements": []}]})
            await client.wait_for_turn(thread_id=thread_id, turn_id=turn["turn"]["id"], timeout=60)
        finally:
            await client.close(timeout=20)

    sent = [{key: value for key, value in item.items() if key != "id"} for item in bodies[0]["input"]]
    current = next(i for i, item in enumerate(sent) if "CURRENT_INPUT" in json.dumps(item))
    if replay:
        # This oracle is independent of _history_items: omitted history or
        # images must fail even if Codex accepts the shortened injected list.
        expected = [
            {"type": "message", "role": "developer", "content": [
                {"type": "input_text", "text": "HISTORY_SYSTEM"}]},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "HISTORY_USER"},
                {"type": "input_image", "image_url": PNG, "detail": "auto"}]},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "HISTORY_ASSISTANT"}]},
            {"type": "function_call", "call_id": "h1", "name": "lookup", "arguments": '{"key": 1}'},
            {"type": "function_call_output", "call_id": "h1", "output": "HISTORY_TOOL_RESULT"},
        ]
        assert sent[current - len(expected):current] == expected
    else:
        assert "HISTORY_" not in json.dumps(sent)


CLIP = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
DATA_URL = "data:audio/wav;base64," + base64.b64encode(CLIP).decode()


@pytest.mark.native
@pytest.mark.parametrize("audio_supported", [True, False])
async def test_native_audio_input_follows_listed_modalities(evaluation_workspace, native_binary, audio_supported):
    bodies = []

    async def respond(request):
        bodies.append(await request.json())
        events = [
            {"type": "response.created", "response": {"id": "r1"}},
            {"type": "response.output_item.done", "item": {
                "type": "message", "role": "assistant", "id": "m1", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "done"}]}},
            {"type": "response.completed", "response": {"id": "r1", "usage": {
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                "input_tokens_details": None, "output_tokens_details": None}}},
        ]
        return web.Response(text="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
                            content_type="text/event-stream")

    async with responses_peer(respond) as base_url:
        home = evaluation_workspace / "native-home"
        home.mkdir()
        (home / "clip.wav").write_bytes(CLIP)
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
        (home / "config.toml").write_text(
            f'model="scripted-contract"\nmodel_provider="scripted"\nmodel_catalog_json="{catalog}"\n'
            f'[model_providers.scripted]\nname="scripted"\n'
            f'base_url="{base_url}"\nwire_api="responses"\nsupports_websockets=false\n'
            'requires_openai_auth=false\nrequest_max_retries=0\nstream_max_retries=0\n'
        )
        native = ExecRemoteJsonRpcTransport(LocalSandbox(), [native_binary, "app-server", "--stdio"], cwd=str(home),
                                            environment={"CODEX_HOME": str(home)})
        client = CodexClient(transport_factory=lambda: native)
        try:
            await client.start()
            await client.rpc.initialize({"clientInfo": {"name": "test", "version": "0"},
                                         "capabilities": {"experimentalApi": True}})
            await client.send_notification("initialized")
            listed = (await client.rpc.model_list({"limit": 100}))["data"]
            thread = await client.rpc.thread_start({"cwd": str(home), "approvalPolicy": "never",
                                                    "sandbox": "danger-full-access"})
            thread_id = thread["thread"]["id"]
            turn = await client.rpc.turn_start({"threadId": thread_id, "input": [
                {"type": "text", "text": "Transcribe.", "text_elements": []},
                {"type": "audio", "url": DATA_URL},
                {"type": "localAudio", "path": str(home / "clip.wav")},
            ]})
            await client.wait_for_turn(thread_id=thread_id, turn_id=turn["turn"]["id"], timeout=60)
        finally:
            await client.close(timeout=20)

    entry, = [m for m in listed if m["id"] == "scripted-contract"]
    assert ("audio" in entry["inputModalities"]) is audio_supported
    user = [item for item in bodies[0]["input"] if item.get("role") == "user"][-1]
    forwarded = [part["audio_url"] for part in user["content"] if part["type"] == "input_audio"]
    if audio_supported:
        assert forwarded == [DATA_URL, DATA_URL]
    else:
        assert forwarded == [] and "audio content omitted" in json.dumps(user)
