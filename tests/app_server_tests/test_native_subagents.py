"""Pinned Codex child reuse, role settings, ancestry, and usage recovery.

Local scripted responses drive real orchestration. Verify accepted task input
and reports, role-specific model settings, and rollout totals scoped to the root
session.
"""

import asyncio
import json
import shutil

import pytest
from aiohttp import web

from inspect_codex_app_server import CodexSubagent
from inspect_codex_app_server.client import CodexClient, ExecRemoteJsonRpcTransport
from inspect_codex_app_server.evidence import NativeEvidence
from inspect_codex_app_server.mapping import EventStreamMapper
from inspect_codex_app_server.subagent import role_config, role_overrides

from .support import LocalSandbox, make_state, responses_peer


@pytest.mark.native
async def test_native_child_reuse(
    evaluation_workspace, captured_events, usage_contexts, native_binary
):
    requests = []
    completed = asyncio.Queue()

    async def respond(request):
        data = await request.json()
        requests.append(data)
        n = len(requests)
        rid = f"resp-{n}"
        child_request = any(item.get("type") == "agent_message" and item.get("recipient") == "/root/worker"
                            for item in data["input"])
        root_number = sum(not any(item.get("type") == "agent_message" and item.get("recipient") == "/root/worker"
                                  for item in body["input"]) for body in requests)
        if not child_request and root_number == 4:
            item = {
                "type": "function_call",
                "call_id": "followup-call",
                "name": "followup_task",
                "namespace": "collaboration",
                "arguments": json.dumps(
                    {"target": "/root/worker", "message": "FOLLOWUP_SENTINEL"}
                ),
            }
        elif not child_request and root_number == 2:
            item = {
                "type": "function_call",
                "call_id": "spawn-call",
                "name": "spawn_agent",
                "namespace": "collaboration",
                "arguments": json.dumps(
                    {
                        "task_name": "worker",
                        "message": "DELEGATE_TASK_SENTINEL",
                        "fork_turns": "all",
                    }
                ),
            }
        else:
            item = {
                "type": "message",
                "role": "assistant",
                "id": f"msg-{n}",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": (
                    f"CHILD_REPORT_{n - root_number}" if child_request else f"ROOT_REPORT_{root_number}")}],
            }
        tokens = 100 if n == 1 else (1 if not child_request and root_number == 2 else 10)
        events = [
            {"type": "response.created", "response": {"id": rid}},
            {"type": "response.output_item.done", "item": item},
            {
                "type": "response.completed",
                "response": {
                    "id": rid,
                    "usage": {
                        "input_tokens": tokens,
                        "output_tokens": 0,
                        "total_tokens": tokens,
                        "input_tokens_details": None,
                        "output_tokens_details": None,
                    },
                },
            },
        ]
        return web.Response(
            text="".join(
                f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
            ),
            content_type="text/event-stream",
        )

    async with responses_peer(respond) as base_url:
        home = evaluation_workspace / "native-home"
        home.mkdir()
        (home / "config.toml").write_text(f"""model="gpt-5.4"
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
    """)
        binary = native_binary
        native = ExecRemoteJsonRpcTransport(
            LocalSandbox(),
            [binary, "-c", "features.multi_agent_v2=true", "app-server", "--stdio"],
            cwd=str(home),
            environment={"CODEX_HOME": str(home)},
        )
        client = CodexClient(transport_factory=lambda: native)
        state = make_state(model="codex/gpt-5.4")
        mapper = EventStreamMapper("gpt-5.4", state)
        for method, handler in (
            ("item/started", mapper.on_item_started),
            ("thread/started", mapper.on_thread_metadata),
            ("thread/metadata/read", mapper.on_thread_metadata),
            ("thread/settings/updated", mapper.on_thread_metadata),
            ("rawResponseItem/completed", mapper.on_raw_item),
            ("rawResponse/completed", mapper.on_response_completed),
            ("thread/tokenUsage/updated", mapper.on_token_usage),
            ("turn/completed", mapper.on_turn_completed),
        ):
            client.register_notification_handler(method, handler)

        def item(method, params):
            mapper.on_item_completed(method, params)
            if (
                params["item"]["type"] == "subAgentActivity"
                and params["item"].get("kind") == "completed"
            ):
                completed.put_nowait(params)

        client.register_notification_handler("item/completed", item)
        try:
            await client.start()
            await client.send_request(
                "initialize",
                {
                    "clientInfo": {"name": "inspect-ai", "version": "2.0.0"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            await client.send_notification("initialized")
            root = (
                await client.send_request(
                    "thread/start", {"cwd": str(home), "experimentalRawEvents": True}
                )
            )["thread"]["id"]
            mapper.root_thread_id = root
            for text in ("ROOT_TASK", "SPAWN_NOW"):
                turn = await client.send_request(
                    "turn/start",
                    {
                        "threadId": root,
                        "input": [{"type": "text", "text": text, "text_elements": []}],
                    },
                )
                await client.wait_for_turn(root, turn["turn"]["id"], 10)
            first = await asyncio.wait_for(completed.get(), 10)
            child = first["item"]["agentThreadId"]
            assert not mapper.subagents[child].completed
            assert mapper.subagents[child].prompt == "DELEGATE_TASK_SENTINEL"
            turn = await client.send_request(
                "turn/start",
                {
                    "threadId": root,
                    "input": [
                        {"type": "text", "text": "FOLLOWUP_NOW", "text_elements": []}
                    ],
                },
            )
            await client.wait_for_turn(root, turn["turn"]["id"], 10)
            second = await asyncio.wait_for(completed.get(), 10)
            assert second["item"]["agentThreadId"] == child
            await client.close()
            mapper.finalize_state(state, "")
            assert state.output.usage.total_tokens == 151
            sub = mapper.subagents[child]
            assert sub.parent_thread_id == root and sub.model_name == "gpt-5.4", (
                mapper.thread_metadata
            )
            assert sub.tool_event.result == "CHILD_REPORT_1\n\nCHILD_REPORT_2"
            assert any(message.text == "FOLLOWUP_SENTINEL" for message in sub.messages)
            assert sub.completed

            for owner, expected in ((root, 151), ("another-root", 0)):
                evidence = NativeEvidence(f"rollouts-{owner}")
                try:
                    shutil.copytree(home / "sessions", evidence.directory / "native" / "sessions")
                    recovered_state = make_state(model="codex/gpt-5.4")
                    recovered = EventStreamMapper("gpt-5.4", recovered_state)
                    recovered.root_thread_id = owner
                    recovered.tearing_down = True
                    await evidence.reconcile(recovered)
                    recovered.finalize_state(recovered_state, "")
                    assert recovered_state.output.usage.total_tokens == expected
                finally:
                    evidence.close(shutdown={"stream_drained": True}, accounting={})
        finally:
            await client.close()


ROLE = CodexSubagent(name="reviewer", description="ROLE_DESCRIPTION reviews changes.",
                     prompt='Say "ROLE_PROMPT".\nBe strict.', model="gpt-role-model", reasoning_effort="high")


def sse(response_id: str, item: dict) -> web.Response:
    events = [
        {"type": "response.created", "response": {"id": response_id}},
        {"type": "response.output_item.done", "item": item},
        {"type": "response.completed", "response": {"id": response_id, "usage": {
            "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
            "input_tokens_details": None, "output_tokens_details": None}}},
    ]
    return web.Response(text="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
                        content_type="text/event-stream")


def is_child(body: dict) -> bool:
    return any(i.get("type") == "agent_message" and str(i.get("recipient", "")).startswith("/root/")
               for i in body["input"])


@pytest.mark.native
@pytest.mark.parametrize("agent_type", ["reviewer", None])
async def test_native_child_runs_with_its_declared_role(evaluation_workspace, native_binary, agent_type):
    bodies = []
    child_requested = asyncio.Event()

    async def respond(request):
        body = await request.json()
        bodies.append(body)
        if is_child(body):
            child_requested.set()
        if len(bodies) == 1:
            arguments = {"task_name": "review", "message": "CHILD_TASK", "fork_turns": "none"}
            if agent_type:
                arguments["agent_type"] = agent_type
            return sse("r1", {"type": "function_call", "call_id": "c1", "namespace": "collaboration",
                              "name": "spawn_agent", "arguments": json.dumps(arguments)})
        return sse(f"r{len(bodies)}", {"type": "message", "role": "assistant", "id": f"m{len(bodies)}",
                                       "phase": "final_answer", "content": [{"type": "output_text", "text": "done"}]})

    async with responses_peer(respond) as base_url:
        home = evaluation_workspace / "native-home"
        home.mkdir()
        (home / "config.toml").write_text(
            f'model="gpt-5.4"\nmodel_provider="scripted"\n[model_providers.scripted]\nname="scripted"\n'
            f'base_url="{base_url}"\nwire_api="responses"\nsupports_websockets=false\n'
            'requires_openai_auth=false\nrequest_max_retries=0\nstream_max_retries=0\n'
        )
        (home / "reviewer.toml").write_text(role_config(ROLE))
        command = [native_binary, "-c", "features.multi_agent_v2=true",
                   *role_overrides(ROLE, str(home / "reviewer.toml")), "app-server", "--stdio"]
        native = ExecRemoteJsonRpcTransport(LocalSandbox(), command, cwd=str(home),
                                            environment={"CODEX_HOME": str(home)})
        client = CodexClient(transport_factory=lambda: native)
        children = []
        client.register_notification_handler("item/completed", lambda method, params: children.extend(
            [params["item"]["agentThreadId"]] if params["item"].get("type") == "subAgentActivity" else []))
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
            await client.wait_for_turn(thread_id=thread_id, turn_id=turn["turn"]["id"], timeout=60)
            # The child runs concurrently and may not have asked the model yet.
            await asyncio.wait_for(child_requested.wait(), 60)
            child = (await client.rpc.thread_read({"threadId": children[0], "includeTurns": False}))["thread"]
        finally:
            await client.close(timeout=20)

    spawn, = [tool for namespace in bodies[0]["tools"] if namespace.get("name") == "collaboration"
              for tool in namespace["tools"] if tool["name"] == "spawn_agent"]
    assert "reviewer: {\nROLE_DESCRIPTION reviews changes.\n" in spawn["parameters"]["properties"]["agent_type"][
        "description"]
    child_body, = [body for body in bodies if is_child(body)]
    applied = (child_body["model"], (child_body.get("reasoning") or {}).get("effort"),
               'Say \\"ROLE_PROMPT\\".\\nBe strict.' in json.dumps(child_body), child.get("agentRole"))
    if agent_type:
        assert applied == ("gpt-role-model", "high", True, "reviewer")
    else:
        assert applied[:3] == ("gpt-5.4", None, False)
