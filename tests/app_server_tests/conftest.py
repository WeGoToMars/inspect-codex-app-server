"""Own every test writer and native process independently of the host repo."""

import asyncio
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from inspect_ai.model._model import (
    model_usage_context_var,
    sample_model_usage_context_var,
)

from inspect_codex_app_server import compat
from inspect_codex_app_server.container import REQUIRED_CLI_VERSION, find_host_codex

from .support import ScriptedAppServer, make_client, patch_session


@pytest.fixture(autouse=True)
def evaluation_workspace(request, monkeypatch):
    with TemporaryDirectory(prefix="inspect-codex-test-") as directory:
        workspace = Path(directory)
        monkeypatch.setenv("INSPECT_LOG_DIR", str(workspace / "logs"))
        monkeypatch.setenv("CODEX_NATIVE_LOG_DIR", str(workspace / "native"))
        try:
            yield workspace
        finally:
            retain = request.config.getoption("--keep-failed-artifacts")
            if retain and getattr(request.node, "test_failed", False):
                destination = Path(retain).expanduser().resolve() / workspace.name
                shutil.copytree(workspace, destination)
                (destination / "test-node-id.txt").write_text(request.node.nodeid + "\n")
                print(f"Retained test artifacts: {destination}")


@pytest.fixture
def tmp_path(evaluation_workspace):
    path = evaluation_workspace / "files"
    path.mkdir()
    return path


@pytest.fixture
def captured_events(monkeypatch):
    events = []
    monkeypatch.setattr(compat, "emit_event", events.append)
    monkeypatch.setattr(compat, "update_event", events.append)
    return events


@pytest.fixture
def usage_contexts():
    sample = sample_model_usage_context_var.set({})
    task = model_usage_context_var.set({})
    try:
        yield sample_model_usage_context_var, model_usage_context_var
    finally:
        sample_model_usage_context_var.reset(sample)
        model_usage_context_var.reset(task)


@pytest.fixture
def scripted():
    return ScriptedAppServer()


@pytest.fixture
def client(scripted):
    return make_client(scripted)


@pytest.fixture
def solver_env(monkeypatch, scripted):
    class Sandbox:
        async def exec(self, *args, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        async def write_file(self, *args, **kwargs):
            pass

    env = Sandbox()
    patch_session(monkeypatch, scripted, env)
    return env


@pytest.fixture
def native_binary():
    # Resolve the same pinned paired bundle as production, never whatever
    # auto-updated binary happens to be first on PATH.
    binary = asyncio.run(find_host_codex())
    version = subprocess.run([str(binary), "--version"], check=True,
                             capture_output=True, text=True, timeout=10).stdout.strip()
    assert version == REQUIRED_CLI_VERSION, version
    return str(binary)
