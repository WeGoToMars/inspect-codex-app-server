"""Pinned binary provisioning, explicit paths, and bounded installation.

Owned npm stubs exercise real subprocess installation and concurrency;
scripted sandbox replies cover image selection without Docker.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from inspect_codex_app_server import container

NPM_STUB = """\
#!/bin/sh
echo "$$" >> "$NPM_CALL_LOG"
echo "invoked" >> "$NPM_CALL_LOG"
if [ -n "$NPM_PID_FILE" ]; then echo $$ > "$NPM_PID_FILE"; fi
{body}
"""

# Creates the exact cache layout download_pinned_codex expects, including the
# paired code-mode-host executable the discovery contract requires.
NPM_INSTALL_BODY = """\
mkdir -p "$NPM_CACHE_DIR/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin"
printf '#!/bin/sh\\n' > "$NPM_CACHE_DIR/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex"
cp "$NPM_CACHE_DIR/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex" \\
   "$NPM_CACHE_DIR/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex-code-mode-host"
chmod +x "$NPM_CACHE_DIR/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/"*
"""


@pytest.fixture
def stub_npm(tmp_path, monkeypatch):
    """Install an ``npm`` shell stub earlier on PATH and isolate HOME.

    Returns the stub script path so a test can swap in a different body
    (e.g. a deliberately stalled install) before calling the code under test.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "npm"
    script.write_text(NPM_STUB.format(body=NPM_INSTALL_BODY))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.delenv("CODEX_PATH", raising=False)
    monkeypatch.setenv("NPM_CALL_LOG", str(tmp_path / "npm-calls.log"))
    monkeypatch.setenv("NPM_CACHE_DIR", str(tmp_path / "home" / ".cache" / "codex-pinned" / container.PINNED_CODEX_VERSION))
    return script


async def test_stalled_install_reports_timeout_stops_child_and_keeps_loop_responsive(
    stub_npm, monkeypatch
):
    """A hung npm install must surface its timeout while the event loop keeps
    serving other work, and the install subprocess must be gone afterwards."""
    workdir = Path(os.environ["NPM_CALL_LOG"]).parent
    stub_npm.write_text(NPM_STUB.format(body="sleep 30\n"))
    pid_file = workdir / "npm.pid"
    monkeypatch.setenv("NPM_PID_FILE", str(pid_file))
    monkeypatch.setattr(container, "CODEX_INSTALL_TIMEOUT_SEC", 0.3)

    responsive = asyncio.Event()

    async def observe_running_child():
        while not pid_file.exists() or not pid_file.read_text().strip():
            await asyncio.sleep(0.01)
        os.kill(int(pid_file.read_text()), 0)
        responsive.set()

    observer = asyncio.create_task(observe_running_child())
    started = time.monotonic()
    try:
        async with asyncio.timeout(5):
            with pytest.raises(RuntimeError, match="did not finish within"):
                await container.download_pinned_codex()
    finally:
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
    elapsed = time.monotonic() - started

    # Another coroutine ran while the real subprocess was still alive.
    assert responsive.is_set()
    assert elapsed < 5, elapsed
    # The stalled child (and its sleep) were killed and reaped.
    assert pid_file.is_file()
    child_pid = int(pid_file.read_text().strip())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("stalled npm subprocess survived the install timeout")


async def test_concurrent_resolution_installs_pinned_codex_once(stub_npm, monkeypatch):
    """Two samples resolving the binary at once must produce one npm install;
    the loser of the install lock re-checks the shared cache and reuses it."""
    workdir = Path(os.environ["NPM_CALL_LOG"]).parent
    pid_file = workdir / "npm.pid"
    release_file = workdir / "release-install"
    monkeypatch.setenv("NPM_PID_FILE", str(pid_file))
    monkeypatch.setenv("NPM_RELEASE_FILE", str(release_file))
    stub_npm.write_text(NPM_STUB.format(
        body='while [ ! -f "$NPM_RELEASE_FILE" ]; do sleep 0.01; done\n' + NPM_INSTALL_BODY
    ))
    second_started = asyncio.Event()

    async def second_sample():
        second_started.set()
        return await container.download_pinned_codex()

    tasks = [asyncio.create_task(container.download_pinned_codex())]
    try:
        async with asyncio.timeout(5):
            while not pid_file.exists():
                await asyncio.sleep(0.01)
            tasks.append(asyncio.create_task(second_sample()))
            await second_started.wait()
            release_file.touch()
            target = await asyncio.gather(*tasks)
    finally:
        release_file.touch()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert target[0] == target[1]
    assert target[0].is_file() and os.access(target[0], os.X_OK)
    companion = target[0].parent / "codex-code-mode-host"
    assert companion.is_file(), "the paired executable must exist beside the binary"
    assert await container.download_pinned_codex() == target[0]
    calls = Path(os.environ["NPM_CALL_LOG"]).read_text().strip().splitlines()
    assert len(calls) == 2, f"expected exactly one npm invocation (two log lines): {calls}"


# The adapter's own earlier copy is not the image's install.
@pytest.mark.parametrize("image_bin, image_version, copied", [
    ("/opt/bin/codex", container.REQUIRED_CLI_VERSION, False), ("/opt/bin/codex", "codex-cli 0.1.0", True),
    (container.IN_CONTAINER_BIN, container.REQUIRED_CLI_VERSION, True)])
async def test_image_with_the_pinned_release_keeps_it(tmp_path, monkeypatch, image_bin, image_version, copied):
    from types import SimpleNamespace

    class Container:
        async def exec(self, cmd, user=None):
            if cmd[0] == "sh":
                return SimpleNamespace(returncode=0, stdout=f"{image_bin}\n{image_version}\n", stderr="")
            out = container.REQUIRED_CLI_VERSION if cmd[-1] == "--version" else ""
            return SimpleNamespace(returncode=1 if cmd[0] == "test" else 0, stdout=out, stderr="")

        async def write_file(self, path, contents):
            if path.endswith("provenance.json"):
                provenance.append(json.loads(contents))

    copies, provenance = [], []

    async def run_host(cmd):
        copies.append(cmd)
        return 0, b"", b""

    async def host_codex(custom_path=None):
        bundle = tmp_path / "codex"
        for path in (bundle, tmp_path / "codex-code-mode-host"):
            path.write_text("#!/bin/sh\n")
        return bundle

    monkeypatch.setattr(container, "_run_host", run_host)
    monkeypatch.setattr(container, "find_host_codex", host_codex)
    monkeypatch.setattr(container, "PINNED_CODEX_SOURCE_COMMIT", "1" * 40)
    monkeypatch.delenv("CODEX_PATH", raising=False)
    monkeypatch.delenv("CODEX_AUTH_PATH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    binary, _, _ = await container.ensure_container_codex(Container(), "owned-container")
    assert (binary, bool(copies)) == ((container.IN_CONTAINER_BIN, True) if copied else ("/opt/bin/codex", False))
    assert provenance[0]["source"] == ("host" if copied else "image")
    assert provenance[0]["source_commit"] == "1" * 40


async def test_empty_selection_variables_select_nothing(stub_npm, monkeypatch):
    monkeypatch.setenv("CODEX_PATH", "")
    monkeypatch.setenv("CODEX_AUTH_PATH", "")
    assert await container.find_host_codex() == await container.download_pinned_codex()
    assert container.find_host_auth() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["codex_binary_path", "auth_path"])
async def test_invalid_explicit_bootstrap_selection_cannot_fall_back(tmp_path, selection):
    from inspect_codex_app_server.container import ensure_container_codex

    class UntouchedEnvironment:
        async def exec(self, *args, **kwargs):
            pytest.fail("Invalid explicit selection must fail before container mutation")

    with pytest.raises(ValueError, match="Selected Codex"):
        await ensure_container_codex(UntouchedEnvironment(), "owned-container", **{selection: tmp_path / "missing"})


@pytest.fixture
def dependency_updater():
    import runpy

    script = Path(__file__).resolve().parents[2] / "scripts/update_dependencies.py"
    return runpy.run_path(str(script))


def test_calendar_versions_increase_within_a_day_and_reset_on_the_next(dependency_updater):
    """Multiple releases must stay unique, and a clock rollback must not downgrade."""
    from datetime import date

    next_version = dependency_updater["next_calendar_version"]
    first = next_version("2026.10.28", date(2026, 10, 28))
    assert first == "2026.10.28.1"
    second = next_version(first, date(2026, 10, 28))
    assert second == "2026.10.28.2"
    next_day = next_version(second, date(2026, 10, 29))
    assert next_day == "2026.10.29"
    with pytest.raises(ValueError, match="later than"):
        next_version(next_day, date(2026, 10, 28))


@pytest.mark.parametrize("trigger", ["inspect", "codex", "manual"])
def test_release_artifacts_identify_the_tested_dependency_pair(
    tmp_path, monkeypatch, dependency_updater, trigger,
):
    """Dependency updates and manual releases must publish the exact tested pair."""
    import re
    import runpy
    import shutil
    import sys
    import tomllib

    from packaging.version import Version

    source = Path(__file__).resolve().parents[2]
    package = tmp_path / "package"
    # Exercise the real package files instead of recreating their source format.
    files = [Path("pyproject.toml"), Path("README.md"), Path("src/inspect_codex_app_server/container.py")]
    for relative in files:
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    manifest = tomllib.loads((package / "pyproject.toml").read_text())["project"]
    old_inspect, = [d.removeprefix("inspect-ai==") for d in manifest["dependencies"] if d.startswith("inspect-ai==")]
    expected = {"inspect": old_inspect, "codex": container.PINNED_CODEX_VERSION}
    # A newer stable registry version; no candidate binary is downloaded here.
    if trigger != "manual":
        expected[trigger] = "999.0.0"
    commit = "2" * 40 if trigger == "codex" else container.PINNED_CODEX_SOURCE_COMMIT
    main = dependency_updater["main"]
    monkeypatch.setitem(main.__globals__, "PACKAGE", package)
    monkeypatch.setitem(main.__globals__, "latest_versions", lambda: (expected["inspect"], expected["codex"]))
    monkeypatch.setitem(main.__globals__, "codex_source_commit", lambda version: "2" * 40)
    output = tmp_path / "candidate"
    argv = ["update_dependencies.py", "--output-dir", str(output)]
    release_argv = [*argv, "--release"] if trigger == "manual" else argv
    before = {relative: (package / relative).read_bytes() for relative in files}
    monkeypatch.setattr(sys, "argv", [*release_argv, "--dry-run"])
    main()
    assert {relative: (package / relative).read_bytes() for relative in files} == before

    monkeypatch.setattr(sys, "argv", release_argv)
    main()
    plan = json.loads((output / "candidate.json").read_text())
    project = tomllib.loads((package / "pyproject.toml").read_text())["project"]
    native = runpy.run_path(str(package / "src/inspect_codex_app_server/container.py"))
    assert plan["changed"] is True
    assert (plan["inspect"], plan["codex"]) == (expected["inspect"], expected["codex"])
    assert f"inspect-ai=={expected['inspect']}" in project["dependencies"]
    assert native["PINNED_CODEX_VERSION"] == expected["codex"]
    assert native["PINNED_CODEX_SOURCE_COMMIT"] == plan["codex_source_commit"] == commit
    assert Version(project["version"]) > Version(manifest["version"])
    assert plan["tag"] == f"v{project['version']}"
    description = (output / "notes.md").read_text()
    assert expected["inspect"] in description
    assert expected["codex"] in description
    links = re.findall(r"\[[^\]]+\]\((https://[^)]+)\)", description)
    assert f"https://github.com/UKGovernmentBEIS/inspect_ai/releases/tag/{expected['inspect']}" in links
    assert f"https://github.com/openai/codex/releases/tag/rust-v{expected['codex']}" in links
    assert f"@openai/codex@{expected['codex']}" in (package / "README.md").read_text()

    # Repeated or stale registry results must not change pins or create another release.
    released = {relative: (package / relative).read_bytes() for relative in files}
    monkeypatch.setitem(main.__globals__, "latest_versions", lambda: (old_inspect, container.PINNED_CODEX_VERSION))
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert json.loads((output / "candidate.json").read_text())["changed"] is False
    assert {relative: (package / relative).read_bytes() for relative in files} == released



def test_upstream_discovery_excludes_yanked_versions_and_resolves_annotated_tags(monkeypatch, dependency_updater):
    latest = dependency_updater["latest_versions"]
    source_commit = dependency_updater["codex_source_commit"]
    api = "https://api.github.com/repos/openai/codex"
    responses = {
        dependency_updater["PYPI"]: {"releases": {
            "0.3.277": [{"yanked": False}], "0.3.278": [{"yanked": True}],
            "0.3.999a1": [{"yanked": False}], "0.3.888": [],
        }},
        dependency_updater["NPM"]: {"version": "0.166.0"},
        f"{api}/releases/tags/rust-v0.166.0": {"draft": False, "prerelease": False},
        f"{api}/git/ref/tags/rust-v0.166.0": {"object": {"type": "tag", "sha": "3" * 40}},
        f"{api}/git/tags/" + "3" * 40: {"object": {"type": "commit", "sha": "2" * 40}},
    }
    monkeypatch.setitem(latest.__globals__, "read_json", responses.__getitem__)
    assert latest() == ("0.3.277", "0.166.0")
    assert source_commit("0.166.0") == "2" * 40
    responses[dependency_updater["NPM"]]["version"] = "0.167.0-alpha.1"
    with pytest.raises(ValueError, match="stable"):
        latest()
    responses[dependency_updater["NPM"]] = {"version": "0.166.0", "deprecated": "withdrawn"}
    with pytest.raises(ValueError, match="deprecated"):
        latest()
    responses[f"{api}/releases/tags/rust-v0.166.0"]["prerelease"] = True
    with pytest.raises(ValueError, match="published stable"):
        source_commit("0.166.0")
