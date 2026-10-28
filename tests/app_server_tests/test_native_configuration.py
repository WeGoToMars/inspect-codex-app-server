"""Pinned Codex configuration, skill discovery, and startup diagnostics.

Check valid overrides and warnings, installed skills and search tools,
skill and MCP load failures, and reconnect notices. The real executable
uses temporary homes and local scripted providers and MCP processes.
"""

import json
import subprocess

import pytest
from aiohttp import web
from inspect_ai.tool import Skill

from inspect_codex_app_server import CodexSubagent
from inspect_codex_app_server.client import CodexClient, ExecRemoteJsonRpcTransport
from inspect_codex_app_server.container import DEFAULT_CONFIG_TOML
from inspect_codex_app_server.subagent import role_config, role_overrides

from .support import LocalSandbox, responses_peer

# The overrides a run passes (policy, summary, search, compaction, multi-agent, effort, roles).
ADAPTER_OVERRIDES = [
    "-c", 'approval_policy="never"', "-c", 'sandbox_mode="danger-full-access"',
    "-c", 'model_reasoning_summary="concise"', "-c", 'web_search="live"',
    "-c", "model_auto_compact_token_limit=100000", "-c", "features.multi_agent_v2=true",
    "-c", "features.multi_agent_v2.max_concurrent_threads_per_session=4", "-c", 'model_reasoning_effort="high"',
]


@pytest.mark.native
@pytest.mark.parametrize("extra, warned", [([], False), (["-c", "unknown_top_level_key=1"], True)])
async def test_native_config_warnings(evaluation_workspace, native_binary, extra, warned):
    home = evaluation_workspace / "native-home"
    home.mkdir()
    (home / "config.toml").write_text(DEFAULT_CONFIG_TOML)
    # Text outside the BMP has no TOML escape, so it must reach the files literally.
    role = CodexSubagent(name="reviewer", description="Reviews changes 🔎.", prompt="Be strict 🚀.")
    (home / "reviewer.toml").write_text(role_config(role))
    command = [native_binary, *ADAPTER_OVERRIDES, *role_overrides(role, str(home / "reviewer.toml")), *extra,
               "app-server", "--stdio"]
    native = ExecRemoteJsonRpcTransport(LocalSandbox(), command, cwd=str(home), environment={"CODEX_HOME": str(home)})
    client = CodexClient(transport_factory=lambda: native)
    warnings = []
    client.register_notification_handler("configWarning", lambda method, params: warnings.append(params["summary"]))
    try:
        await client.start()
        await client.rpc.initialize({"clientInfo": {"name": "test", "version": "0"},
                                     "capabilities": {"experimentalApi": True}})
        await client.send_notification("initialized")
        await client.rpc.thread_start({"cwd": str(home), "approvalPolicy": "never", "sandbox": "danger-full-access"})
    finally:
        await client.close(timeout=20)

    if warned:
        assert len(warnings) == 1 and "unknown_top_level_key" in warnings[0]
    else:
        assert warnings == []


@pytest.mark.native
@pytest.mark.parametrize("mode, offered", [("live", True), ("disabled", False)])
async def test_native_session_offers_installed_skills_and_web_search(evaluation_workspace, native_binary, mode,
                                                                     offered):
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
        skill = Skill(name="unit-convert", description="SKILL_SENTINEL converts units.", instructions="Convert.")
        (home / "skills" / skill.name).mkdir(parents=True)
        (home / "skills" / skill.name / "SKILL.md").write_text(skill.skill_md())
        (home / "config.toml").write_text(
            f'model="gpt-5.4"\nmodel_provider="scripted"\n[model_providers.scripted]\nname="scripted"\n'
            f'base_url="{base_url}"\nwire_api="responses"\nsupports_websockets=false\n'
            'requires_openai_auth=false\nrequest_max_retries=0\nstream_max_retries=0\n'
        )
        native = ExecRemoteJsonRpcTransport(
            LocalSandbox(), [native_binary, "-c", f'web_search="{mode}"', "app-server", "--stdio"],
            cwd=str(home), environment={"CODEX_HOME": str(home)})
        client = CodexClient(transport_factory=lambda: native)
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
        finally:
            await client.close(timeout=20)

    body, = bodies
    assert "SKILL_SENTINEL" in json.dumps(body), "The installed skill reaches the model"
    assert any(t.get("type") == "web_search" for t in body["tools"]) == offered


SERVER = '''import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" in msg:
        assert msg["method"] in {"initialize", "tools/list"}, msg["method"]
        result = {"protocolVersion": msg["params"]["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "ok", "version": "0"}} if msg["method"] == "initialize" else {"tools": []}
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
'''


def sse(*events):
    return web.Response(text="".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
                        content_type="text/event-stream")


async def session(home, base_url, binary, body, notifications=()):
    (home / "config.toml").write_text(
        f'model="gpt-5.4"\nmodel_provider="scripted"\n[model_providers.scripted]\nname="scripted"\n'
        f'base_url="{base_url}"\nwire_api="responses"\nsupports_websockets=false\n'
        'requires_openai_auth=false\nrequest_max_retries=0\nstream_max_retries=1\n'
        f'[mcp_servers.ok]\ncommand="python3"\nargs=["{home / "server.py"}"]\n'
        '[mcp_servers.broken]\ncommand="/nonexistent/server"\n'
    )
    (home / "server.py").write_text(SERVER)
    native = ExecRemoteJsonRpcTransport(LocalSandbox(), [binary, "app-server", "--stdio"], cwd=str(home),
                                        environment={"CODEX_HOME": str(home)})
    client = CodexClient(transport_factory=lambda: native)
    seen = []
    for method in notifications:
        client.register_notification_handler(method, lambda method, params: seen.append((method, params)))
    try:
        await client.start()
        await client.rpc.initialize({"clientInfo": {"name": "test", "version": "0"},
                                     "capabilities": {"experimentalApi": True}})
        await client.send_notification("initialized")
        thread = await client.rpc.thread_start({"cwd": str(home), "approvalPolicy": "never",
                                                "sandbox": "danger-full-access"})
        result = await body(client, thread["thread"]["id"])
    finally:
        await client.close(timeout=20)
    return result, seen


@pytest.mark.native
async def test_native_reports_unloadable_skills_failed_mcp_servers_and_retries(evaluation_workspace, native_binary):
    requests = []

    async def respond(request):
        requests.append(await request.json())
        created = {"type": "response.created", "response": {"id": f"r{len(requests)}"}}
        if len(requests) == 1:
            return sse(created)  # Closed before response.completed: Codex reconnects.
        return sse(created, {"type": "response.output_item.done", "item": {
            "type": "message", "role": "assistant", "id": "m", "phase": "final_answer",
            "content": [{"type": "output_text", "text": "done"}]}},
            {"type": "response.completed", "response": {"id": "r", "usage": {
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                "input_tokens_details": None, "output_tokens_details": None}}})

    home = evaluation_workspace / "native-home"
    for name, text in (("good", "---\nname: good\ndescription: Fine.\n---\nBody.\n"), ("bad", "No frontmatter.\n")):
        (home / "skills" / name).mkdir(parents=True)
        (home / "skills" / name / "SKILL.md").write_text(text)
    workspace = evaluation_workspace / "workspace"
    (workspace / ".codex" / "skills" / "bad").mkdir(parents=True)
    (workspace / ".codex" / "skills" / "bad" / "SKILL.md").write_text("No frontmatter.\n")
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)

    async def body(client, thread_id):
        listed = await client.rpc.skills_list({"cwds": [str(workspace)]})
        turn = await client.rpc.turn_start({"threadId": thread_id, "input": [
            {"type": "text", "text": "go", "text_elements": []}]})
        await client.wait_for_turn(thread_id=thread_id, turn_id=turn["turn"]["id"], timeout=60)
        return listed

    async with responses_peer(respond) as base_url:
        listed, seen = await session(home, base_url, native_binary, body,
                                     ("mcpServer/startupStatus/updated", "error"))

    entry, = listed["data"]
    assert sorted(error["path"] for error in entry["errors"]) == sorted(
        [str(home / "skills" / "bad" / "SKILL.md"), str(workspace / ".codex" / "skills" / "bad" / "SKILL.md")])
    assert "good" in [skill["name"] for skill in entry["skills"]]
    final = {params["name"]: params["status"] for method, params in seen
             if method == "mcpServer/startupStatus/updated" and params["status"] != "starting"}
    assert final == {"ok": "ready", "broken": "failed"}
    assert [params["willRetry"] for method, params in seen if method == "error"] == [True]
