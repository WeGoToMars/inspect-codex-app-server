"""Pinned Codex in an Inspect Docker sandbox, through the saved evaluation log.

A local scripted provider drives tool validation, root completion, accounting,
evidence export, and container cleanup through the production solver.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path

import pytest
from inspect_ai import Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import match
from inspect_ai.solver import chain, solver
from inspect_ai.tool import tool
from inspect_ai.util import ExecRemoteStreamingOptions, sandbox

from inspect_codex_app_server import codex_app_server
from inspect_codex_app_server.compat import ExecStdout
from inspect_codex_app_server.container import CODEX_HOME

from .support import host_run, project_resources

pytestmark = pytest.mark.containers


async def test_native_solver_tools_accounting_and_cleanup(evaluation_workspace, native_binary):
    host_run(["docker", "image", "inspect", "python:3.13-slim"])
    requests = []

    peer_script = r"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
requests = []
with open('/tmp/provider-requests.json', 'w') as stream: json.dump(requests, stream)
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_POST(self):
        assert self.path == '/v1/responses', self.path
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        requests.append(body)
        with open('/tmp/provider-requests.json', 'w') as stream: json.dump(requests, stream)
        number = len(requests)
        assert number <= 3, 'unexpected provider retry'
        item = ({'type':'function_call', 'call_id':f'probe-{number}', 'name':'inspect_probe',
                 'arguments':json.dumps({'value':'invalid' if number == 1 else 7})}
                if number <= 2 else {'type':'message','id':f'final-message-{number}','role':'assistant',
                                    'content':[{'type':'output_text','text':'NATIVE_ACCEPTED_ANSWER'}]})
        events = [
            {'type':'response.created','response':{'id':f'response-{number}'}},
            {'type':'response.output_item.done','item':item},
            {'type':'response.completed','response':{'id':f'response-{number}','usage':{
                'input_tokens':100,'output_tokens':10,'total_tokens':110,
                'input_tokens_details':{'cached_tokens':80},'output_tokens_details':None}}},
        ]
        data = ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events).encode()
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)
server = HTTPServer(('127.0.0.1',18765), Handler)
print('PEER_READY', flush=True)
server.serve_forever()
"""
    compose = evaluation_workspace / "compose.json"
    compose.write_text(
        json.dumps(
            {
                "services": {
                    "default": {
                        "image": "python:3.13-slim",
                        "command": ["sleep", "infinity"],
                        "working_dir": "/app",
                        "user": "root",
                        "x-default": True,
                    }
                }
            }
        )
    )
    projects = set()
    executed = []

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    # A valid 2x2 RGB PNG, constructed from the PNG chunk specification.
    pixels = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    )
    pixels += chunk(
        b"IDAT", zlib.compress((b"\0" + bytes((11, 33, 55)) * 2) * 2)
    ) + chunk(b"IEND", b"")
    image = "data:image/png;base64," + base64.b64encode(pixels).decode()
    local_image = evaluation_workspace / "input.png"
    local_image.write_bytes(pixels)
    # Use production's pinned resolver: the operator's PATH may contain a
    # newer Codex, which must not replace the version under acceptance.
    host_binary = Path(native_binary)
    helper = host_binary.parent / "codex-code-mode-host"
    with helper.open("rb") as stream:
        helper_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    copied_helper = []

    @tool
    def inspect_probe():
        async def probe(value: int):
            """Return a typed probe value and a tiny image.

            Args:
                value: Integer probe value.
            """
            executed.append(value)
            return [
                ContentText(text=f"INSPECT_VALUE_{value}"),
                ContentImage(image=image),
            ]

        return probe

    @solver
    def configure_provider():
        async def configure(state, generate):
            from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment

            env = sandbox("default")
            projects.add(env.as_type(DockerSandboxEnvironment)._project.name)
            await env.write_file("/tmp/provider.py", peer_script)
            process = await env.exec_remote(
                ["python3", "-u", "/tmp/provider.py"], ExecRemoteStreamingOptions()
            )
            ready = False
            async with asyncio.timeout(10):
                async for event in process:
                    if isinstance(event, ExecStdout) and "PEER_READY" in event.data:
                        ready = True
                        break
            assert ready, "provider exited before readiness"
            await env.exec(["mkdir", "-p", CODEX_HOME])
            # This local provider needs no account and must not contact the
            # operator's connected-app catalog during a deterministic check.
            await env.write_file(f"{CODEX_HOME}/auth.json", "{}")
            await env.write_file(
                f"{CODEX_HOME}/config.toml",
                """model="gpt-5.4"
model_provider="scripted_contract"
[features]
apps=false
[model_providers.scripted_contract]
name="Scripted contract"
base_url="http://127.0.0.1:18765/v1"
wire_api="responses"
supports_websockets=false
requires_openai_auth=false
request_max_retries=0
stream_max_retries=3
""",
            )
            state.tools = [inspect_probe()]
            # Add the host path after Inspect's dataset image conversion so
            # production staging, rather than dataset preprocessing, owns it.
            state.messages[0].content.append(ContentImage(image=str(local_image)))
            state.messages[0].content[0] = ContentText(text=f"{objective}\n\nREQUIRED_DOCS_SENTINEL")
            return state

        return configure

    @solver
    def execute_native():
        native = codex_app_server(
            model="gpt-5.4", reasoning_effort="low", codex_path=str(host_binary), timeout_sec=60,
        )

        async def execute(state, generate):
            try:
                return await native(state, generate)
            finally:
                digest = await sandbox("default").exec([
                    "sha256sum", "/usr/local/bin/codex-code-mode-host",
                ])
                assert digest.returncode == 0, digest.stderr
                copied_helper.append(digest.stdout.split()[0])
                requests.extend(
                    json.loads(
                        await sandbox("default").read_file("/tmp/provider-requests.json")
                    )
                )
                (evaluation_workspace / "provider-requests.json").write_text(
                    json.dumps(requests)
                )

        return execute

    objective = "FULL_OBJECTIVE_SENTINEL_" + "a" * 9000
    task = Task(
        dataset=[
            Sample(
                input=[ChatMessageUser(content=[
                    ContentText(text="TASK_SENTINEL"), ContentImage(image=image),
                ])],
                target="NATIVE_ACCEPTED_ANSWER",
            )
        ],
        solver=chain(configure_provider(), execute_native()),
        scorer=match(),
        sandbox=("docker", str(compose)),
    )
    try:
        logs = await eval_async(
            task,
            model="codex/gpt-5.4",
            log_dir=str(evaluation_workspace / "logs"),
            max_sandboxes=1,
            sandbox_cleanup=True,
        )
        log = read_eval_log(logs[0].location)
        assert copied_helper == [helper_digest]
        assert log.status == "success", log.error
        sample = log.samples[0]
        assert next(iter(sample.scores.values())).value == "C"
        assert sample.error is None, sample.error
        assert sample.output.completion == "NATIVE_ACCEPTED_ANSWER"
        assert sample.metadata["codex_run"]["completion"] == "turn"
        assert executed == [7]  # Schema rejection never invokes the Python tool.
        tools = [
            event
            for event in sample.events
            if event.event == "tool" and event.function == "inspect_probe"
        ]
        assert len(tools) == 2 and tools[0].error and tools[1].error is None
        assert len(requests) == 3
        tool_result = next(
            item for item in requests[2]["input"]
            if item.get("type") == "function_call_output" and item.get("call_id") == "probe-2"
        )
        assert "INSPECT_VALUE_7" in json.dumps(tool_result["output"])
        images = [
            block["image_url"]
            for block in tool_result["output"]
            if block.get("type") == "input_image"
        ]
        assert len(images) == 1, tool_result
        encoded = base64.b64decode(images[0].split(",", 1)[1])
        assert encoded.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"))
        assert objective in json.dumps(requests[0]["input"])
        assert "REQUIRED_DOCS_SENTINEL" in json.dumps(requests[0]["input"])
        assert sum(
            block.get("type") == "input_image"
            for item in requests[0]["input"]
            for block in item.get("content", [])
        ) == 2, requests[0]["input"]
        assert requests[0]["reasoning"]["effort"] == "low"
        usage = sample.output.usage
        assert usage.total_tokens == 330
        assert usage.input_tokens == 60
        assert usage.input_tokens_cache_read == 240
        evidence = sample.metadata["codex_native_evidence"]
        directory = Path(evidence["directory"])
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["closed"] and not manifest["failures"], manifest
        assert manifest["shutdown"]["stream_drained"], manifest
        assert any(path.startswith("native/sessions") for path in manifest["exports"])
        protocol = [
            json.loads(line)
            for line in (directory / "protocol.jsonl").read_text().splitlines()
        ]
        assert any(
            row["message"].get("method") == "rawResponse/completed" for row in protocol
        )
        for project in projects:
            assert not any(project_resources(project).values())
    finally:
        for project in projects:
            if any(project_resources(project).values()):
                host_run(
                    [
                        "docker",
                        "compose",
                        "-p",
                        project,
                        "-f",
                        str(compose),
                        "down",
                        "--volumes",
                        "--remove-orphans",
                    ]
                )
