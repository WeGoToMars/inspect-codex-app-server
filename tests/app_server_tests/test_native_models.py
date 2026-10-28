"""Pinned Codex model discovery, turn settings, and serving-model reroutes.

Run the real app-server executable and inspect its catalog and requests to a
local scripted provider, including hidden models, reasoning, and output schemas.
"""

import json

import pytest
from aiohttp import web

from inspect_codex_app_server.client import CodexClient, ExecRemoteJsonRpcTransport

from .support import LocalSandbox, responses_peer


async def listing(client: CodexClient, include_hidden: bool) -> list[str]:
    ids, cursor = [], None
    while True:
        page = await client.rpc.model_list(
            {"limit": 100, "includeHidden": include_hidden, **({"cursor": cursor} if cursor else {})})
        ids.extend(model["id"] for model in page["data"])
        cursor = page.get("nextCursor")
        if not cursor:
            return ids


async def started(native_binary: str, home, model: str | None) -> tuple[CodexClient, dict]:
    (home / "config.toml").write_text(f'model="{model}"\n' if model else "")
    native = ExecRemoteJsonRpcTransport(LocalSandbox(), [native_binary, "app-server", "--stdio"], cwd=str(home),
                                        environment={"CODEX_HOME": str(home)})
    client = CodexClient(transport_factory=lambda: native)
    await client.start()
    await client.rpc.initialize({"clientInfo": {"name": "test", "version": "0"},
                                 "capabilities": {"experimentalApi": True}})
    await client.send_notification("initialized")
    thread = await client.rpc.thread_start({"cwd": str(home), "approvalPolicy": "never",
                                            "sandbox": "danger-full-access"})
    return client, dict(thread)


@pytest.mark.native
async def test_native_catalog_lists_configured_hidden_models_only_when_asked(evaluation_workspace, native_binary):
    home = evaluation_workspace / "native-home"
    home.mkdir()
    client, _ = await started(native_binary, home, None)
    try:
        visible, everything = await listing(client, False), await listing(client, True)
    finally:
        await client.close(timeout=20)
    hidden = next(model for model in everything if model not in visible)

    client, thread = await started(native_binary, home, hidden)
    try:
        assert (thread["model"], thread["modelProvider"]) == (hidden, "openai")
        assert hidden in await listing(client, True)
        assert hidden not in await listing(client, False)
    finally:
        await client.close(timeout=20)


@pytest.mark.native
@pytest.mark.parametrize("tier", ["flex", None])
async def test_native_turn_settings_reach_the_model(evaluation_workspace, native_binary, tier):
    bodies = []

    async def respond(request):
        bodies.append(await request.json())
        number = len(bodies)
        events = [
            {"type": "response.created", "response": {"id": f"r{number}"}},
            {"type": "response.output_item.done", "item": {
                "type": "message", "role": "assistant", "id": f"m{number}", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "done"}]}},
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
        native = ExecRemoteJsonRpcTransport(
            LocalSandbox(),
            [native_binary, "-c", 'model_reasoning_summary="concise"', "app-server", "--stdio"],
            cwd=str(home), environment={"CODEX_HOME": str(home)},
        )
        client = CodexClient(transport_factory=lambda: native)
        try:
            await client.start()
            await client.rpc.initialize({"clientInfo": {"name": "test", "version": "0"},
                                         "capabilities": {"experimentalApi": True}})
            await client.send_notification("initialized")
            thread = await client.rpc.thread_start({"cwd": str(home), "approvalPolicy": "never",
                                                    "sandbox": "danger-full-access"})
            thread_id = thread["thread"]["id"]
            schema = {"type": "object", "properties": {"answer": {"type": "string"}},
                      "required": ["answer"], "additionalProperties": False}
            for params in ({"serviceTier": tier} if tier else {}, {}):
                turn = await client.rpc.turn_start({"threadId": thread_id, "input": [
                    {"type": "text", "text": "go", "text_elements": []}],
                    "effort": "high", "outputSchema": schema, **params})
                await client.wait_for_turn(thread_id=thread_id, turn_id=turn["turn"]["id"], timeout=60)
        finally:
            await client.close(timeout=20)

    assert [body.get("service_tier") for body in bodies] == [tier, tier]
    for body in bodies:
        assert body["reasoning"] == {"effort": "high", "summary": "concise"}
        assert body["text"]["format"]["schema"] == schema


@pytest.mark.native
@pytest.mark.parametrize("serving_model", ["gpt-5.4-safety", "gpt-5.4"])
async def test_native_reroute_is_reported_once_per_turn(evaluation_workspace, native_binary, serving_model):
    async def respond(request):
        await request.json()
        response = {"id": "r1", "model": serving_model}
        events = [
            {"type": "response.created", "response": response},
            {"type": "response.output_item.done", "item": {
                "type": "message", "role": "assistant", "id": "m1", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "done"}]}},
            {"type": "response.completed", "response": dict(response, usage={
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                "input_tokens_details": None, "output_tokens_details": None})},
        ]
        return web.Response(text="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
                            content_type="text/event-stream", headers={"OpenAI-Model": serving_model})

    reroutes = []
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
        client.register_notification_handler("model/rerouted", lambda method, params: reroutes.append(params))
        try:
            await client.start()
            await client.rpc.initialize({"clientInfo": {"name": "test", "version": "0"},
                                         "capabilities": {"experimentalApi": True}})
            await client.send_notification("initialized")
            thread = await client.rpc.thread_start({"cwd": str(home), "approvalPolicy": "never",
                                                    "sandbox": "danger-full-access"})
            thread_id = thread["thread"]["id"]
            turn = await client.rpc.turn_start({"threadId": thread_id, "input": [
                {"type": "text", "text": "go", "text_elements": []}]})
            turn_id = turn["turn"]["id"]
            await client.wait_for_turn(thread_id=thread_id, turn_id=turn_id, timeout=60)
        finally:
            await client.close(timeout=20)

    if serving_model == "gpt-5.4":
        assert reroutes == []
        return
    assert [{key: r.get(key) for key in ("threadId", "turnId", "fromModel", "toModel")} for r in reroutes] == [
        {"threadId": thread_id, "turnId": turn_id, "fromModel": "gpt-5.4", "toModel": "gpt-5.4-safety"}]
    assert reroutes[0]["reason"] == "highRiskCyberActivity"
