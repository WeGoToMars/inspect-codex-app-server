"""Generate Python wire types and the used RPC surface from pinned native schemas.

Run with --check in CI; run without it after deliberately updating the Codex pin.
Only the adapter's supported request methods are selected by hand. Their field
definitions, transitive dependencies, and responses come from native schemas.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version as package_version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from inspect_codex_app_server.container import REQUIRED_CLI_VERSION, find_host_codex

REQUEST_METHODS = (
    "initialize", "account/read", "account/rateLimits/read", "model/list", "thread/start",
    "thread/read", "thread/resume", "thread/goal/set", "thread/goal/get", "turn/start", "turn/steer",
    "turn/interrupt", "thread/inject_items", "skills/list",
)
PACKAGE = Path(__file__).resolve().parents[1]
DESTINATION = PACKAGE / "src" / "inspect_codex_app_server"


def references(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = set()
        if "$ref" in value:
            ref = value["$ref"]
            if not ref.startswith("#/definitions/"):
                raise ValueError(f"Unsupported external schema reference: {ref}")
            found.add(ref.removeprefix("#/definitions/"))
        for item in value.values():
            found.update(references(item))
        return found
    if isinstance(value, list):
        return set().union(*(references(item) for item in value))
    return set()


def generate(binary: Path, directory: Path) -> dict[str, str]:
    version = subprocess.run([str(binary), "--version"], check=True, capture_output=True,
                             text=True, timeout=10).stdout.strip()
    if version != REQUIRED_CLI_VERSION:
        raise ValueError(f"Expected {REQUIRED_CLI_VERSION}; found {version}")
    subprocess.run(
        [str(binary), "app-server", "generate-json-schema", "--experimental", "--out", str(directory)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    # Initialization still lives in v1; all session APIs use v2 definitions.
    definitions = json.loads((directory / "codex_app_server_protocol.schemas.json").read_text())["definitions"]
    definitions.update(json.loads((directory / "codex_app_server_protocol.v2.schemas.json").read_text())["definitions"])
    definitions = json.loads(json.dumps(definitions).replace('"#/definitions/v2/', '"#/definitions/'))
    requests = json.loads((directory / "ClientRequest.json").read_text())
    variants = {item["properties"]["method"]["enum"][0]: item for item in requests["oneOf"]}
    methods = []
    pending = set()
    for method in sorted(REQUEST_METHODS):
        params_schema = variants[method]["properties"]["params"]
        # Optional params are an anyOf of one reference and null.
        refs = [option for option in params_schema.get("anyOf", [params_schema]) if "$ref" in option]
        if len(refs) != 1:
            raise ValueError(f"Unsupported parameter schema for {method}: {params_schema}")
        params = refs[0]["$ref"].removeprefix("#/definitions/")
        # The selected native APIs name their pair FooParams/FooResponse.
        # Refuse an unpaired method rather than generating an Any return type.
        response = params.removesuffix("Params") + "Response"
        if params not in definitions or response not in definitions:
            raise ValueError(f"No native parameter/response pair for {method}: {params}, {response}")
        methods.append((method, params, response))
        pending.update((params, response))
    selected = {}
    while pending:
        name = pending.pop()
        if name not in selected:
            selected[name] = definitions[name]
            pending.update(references(definitions[name]) - selected.keys())
    schema = {"$schema": "http://json-schema.org/draft-07/schema#", "title": "CodexProtocol",
              "type": "object", "definitions": dict(sorted(selected.items()))}
    schema_text = json.dumps(schema, sort_keys=True, indent=2) + "\n"
    schema_path = directory / "protocol.json"
    schema_path.write_text(schema_text)
    generated = directory / "_generated_protocol.py"
    subprocess.run([
        sys.executable, "-m", "datamodel_code_generator", "--input", str(schema_path),
        "--input-file-type", "jsonschema", "--output-model-type", "typing.TypedDict",
        "--target-python-version", "3.11", "--use-standard-collections", "--use-union-operator",
        "--disable-timestamp", "--formatters", "builtin", "--no-allow-remote-refs",
        "--output", str(generated),
    ], check=True, timeout=60)
    digest = hashlib.sha256(schema_text.encode()).hexdigest()
    header = (f"# Generated from {version}; schema SHA-256: {digest}\n"
              f"# Generator: datamodel-code-generator {package_version('datamodel-code-generator')}\n"
              "# Regenerate with scripts/generate_protocol.py; do not edit.\n")
    rpc = '''from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from . import _generated_protocol as p


class RequestSender(Protocol):
    async def send_request(
        self, method: str, params: Mapping[str, Any] | None = None,
        timeout: float | None = 60.0, *, _allow_failed: bool = False,
    ) -> Any: ...


class CodexRpc:
    """Statically typed native calls; received JSON is not runtime-validated."""

    def __init__(self, sender: RequestSender) -> None:
        self._sender = sender
'''
    for method, params, response in methods:
        name = method.replace("/", "_")
        rpc += f'''
    async def {name}(
        self, params: p.{params}, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.{response}:
        return cast(p.{response}, await self._sender.send_request(
            "{method}", params, timeout=timeout, _allow_failed=_allow_failed,
        ))
'''
    return {"_generated_protocol.py": header + generated.read_text(), "_generated_rpc.py": header + rpc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if checked-in bindings differ")
    args = parser.parse_args()
    binary = asyncio.run(find_host_codex())
    with TemporaryDirectory(prefix="codex-protocol-") as directory:
        outputs = generate(Path(binary), Path(directory))
    stale = []
    for name, content in outputs.items():
        target = DESTINATION / name
        if args.check:
            if not target.exists() or target.read_text() != content:
                stale.append(name)
        else:
            target.write_text(content)
    if stale:
        parser.exit(1, f"Stale native bindings: {', '.join(stale)}; run scripts/generate_protocol.py\n")
    print("Native protocol bindings are current." if args.check else "Generated native protocol bindings.")


if __name__ == "__main__":
    main()
