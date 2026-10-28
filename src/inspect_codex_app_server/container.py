"""Docker sandbox setup and binary management for the Codex App Server.

Handles host discovery or download of the Codex binary, container provisioning,
dedicated CODEX_HOME configuration, and credential staging.
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import hashlib
import json
import logging
import os
import shutil
import signal
from pathlib import Path

from inspect_ai.util import SandboxEnvironment

logger = logging.getLogger(__name__)

# Under .cache, which Inspect checkpoints never back up, so credentials stay out of them.
CODEX_HOME = "/root/.cache/codex-inspect"
# Written when the Codex home has no config.toml. The sandbox mode is passed per
# run (sandbox_mode); the pinned release ignores a [sandbox] table with a warning.
DEFAULT_CONFIG_TOML = (
    'approvals_reviewer = "user"\n\n'
    "[shell_environment_policy]\n"
    'inherit = "all"\n\n'
    "[features]\n"
    "multi_agent = true\n\n"
)
IN_CONTAINER_BIN = "/usr/local/bin/codex"
IN_CONTAINER_CODE_MODE_HOST = "/usr/local/bin/codex-code-mode-host"
PINNED_CODEX_VERSION = "0.165.0"
PINNED_CODEX_SOURCE_COMMIT = "360319d8dee62d87e2d59fc0c21805d79519cfed"
REQUIRED_CLI_VERSION = f"codex-cli {PINNED_CODEX_VERSION}"
# Timeouts for background npm installation and executable version probes.
CODEX_INSTALL_TIMEOUT_SEC = 300.0
CODEX_VERSION_PROBE_TIMEOUT_SEC = 10.0

# Coordinate concurrent installations so only one worker downloads to the cache.
_install_locks: dict[str, asyncio.Lock] = {}


def _install_lock(version: str) -> asyncio.Lock:
    return _install_locks.setdefault(version, asyncio.Lock())


async def _run_host_bounded(cmd: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    """Run a host subprocess asynchronously, killing the process group on timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            await proc.wait()
        raise
    return proc.returncode or 0, stdout, stderr


def _pinned_cache_target() -> tuple[Path, Path]:
    cache_dir = Path.home() / ".cache" / "codex-pinned" / PINNED_CODEX_VERSION
    target = (
        cache_dir
        / "node_modules"
        / "@openai"
        / "codex-linux-x64"
        / "vendor"
        / "x86_64-unknown-linux-musl"
        / "bin"
        / "codex"
    )
    return cache_dir, target


async def download_pinned_codex() -> Path:
    """Download the pinned Codex CLI binary package via npm if not already cached."""
    cache_dir, target = _pinned_cache_target()
    if target.is_file() and os.access(target, os.X_OK):
        return target

    async with _install_lock(PINNED_CODEX_VERSION):
        if target.is_file() and os.access(target, os.X_OK):
            return target
        logger.info("Downloading pinned Codex %s via npm...", PINNED_CODEX_VERSION)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["npm", "install", "--prefix", str(cache_dir), f"@openai/codex@{PINNED_CODEX_VERSION}"]
        try:
            code, _, stderr = await _run_host_bounded(cmd, CODEX_INSTALL_TIMEOUT_SEC)
        except TimeoutError as exc:
            raise RuntimeError(
                f"Pinned Codex {PINNED_CODEX_VERSION} download did not finish within "
                f"{CODEX_INSTALL_TIMEOUT_SEC:.0f}s; the npm install was terminated. "
                "Install codex manually or specify CODEX_PATH."
            ) from exc
        if code != 0:
            raise RuntimeError(
                f"Failed to download pinned Codex {PINNED_CODEX_VERSION} via npm: "
                f"{stderr.decode(errors='replace')}"
            )
        if not (target.is_file() and os.access(target, os.X_OK)):
            raise FileNotFoundError(f"Pinned Codex executable missing after install at {target}")
        return target


async def find_host_codex(custom_path: str | Path | None = None) -> Path:
    """Locate the Codex binary on the host system, downloading it if not present."""
    # An empty environment variable selects nothing, as if it were unset.
    selected = custom_path if custom_path is not None else os.environ.get("CODEX_PATH") or None
    if selected is not None:
        path = Path(selected).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Selected Codex executable is not an executable file: {path}")
        return path

    home = Path.home()
    # 1. Check known directories for the specific pinned release
    pinned_candidates: list[Path] = [
        home
        / ".cache"
        / "codex-pinned"
        / PINNED_CODEX_VERSION
        / "node_modules"
        / "@openai"
        / "codex-linux-x64"
        / "vendor"
        / "x86_64-unknown-linux-musl"
        / "bin"
        / "codex",
        home
        / ".codex"
        / "packages"
        / "standalone"
        / "releases"
        / f"{PINNED_CODEX_VERSION}-x86_64-unknown-linux-musl"
        / "bin"
        / "codex",
        home
        / ".codex"
        / "packages"
        / "app-server-daemon"
        / "releases"
        / f"{PINNED_CODEX_VERSION}-x86_64-unknown-linux-musl"
        / "bin"
        / "codex",
    ]

    releases_dir = home / ".codex/packages/standalone/releases"
    if releases_dir.is_dir():
        for release_bin in glob.glob(str(releases_dir / f"*{PINNED_CODEX_VERSION}*/bin/codex")):
            pinned_candidates.append(Path(release_bin).resolve())

    for candidate in pinned_candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            companion = candidate.parent / "codex-code-mode-host"
            if companion.is_file():
                resolved = candidate.resolve()
                logger.info("Found pinned Codex binary at: %s (resolved: %s)", candidate, resolved)
                return resolved

    # 2. Check general candidates (PATH, .local, etc.) only if their version matches pinned
    general_candidates: list[Path] = []
    which_path = shutil.which("codex")
    if which_path:
        general_candidates.append(Path(which_path).resolve())
    general_candidates.append(home / ".local/bin/codex")
    general_candidates.append(home / ".codex/packages/standalone/current/bin/codex")

    for candidate in general_candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            companion = candidate.parent / "codex-code-mode-host"
            if companion.is_file():
                try:
                    code, stdout, stderr = await _run_host_bounded(
                        [str(candidate), "--version"], CODEX_VERSION_PROBE_TIMEOUT_SEC
                    )
                    ver = (stdout or stderr).decode(errors="replace").strip()
                    if code == 0 and ver == REQUIRED_CLI_VERSION:
                        resolved = candidate.resolve()
                        logger.info("Found matching host Codex binary at: %s (resolved: %s)", candidate, resolved)
                        return resolved
                except Exception:
                    pass

    # 3. If no pinned binary was found on host, download it automatically
    downloaded = await download_pinned_codex()
    logger.info("Downloaded pinned Codex binary at: %s", downloaded)
    return downloaded


def find_host_auth(custom_path: str | Path | None = None) -> Path | None:
    """Find Codex auth.json on the host system."""
    candidates: list[Path] = []

    selected = custom_path if custom_path is not None else os.environ.get("CODEX_AUTH_PATH") or None
    if selected is not None:
        path = Path(selected).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ValueError(f"Selected Codex auth path is not a readable file: {path}")
        return path

    candidates.append(Path.home() / ".codex/auth.json")

    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            logger.info("Found host Codex auth at: %s (resolved: %s)", candidate, resolved)
            return resolved

    logger.warning("No host Codex auth.json found (looked at ~/.codex/auth.json and explicit paths)")
    return None


async def _run_host(cmd: list[str]) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await proc.communicate()
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, stdout, stderr


async def resolve_service_container(env: SandboxEnvironment, service: str = "main") -> str:
    """Resolve the selected sandbox through its public connection API."""
    connection = await env.connection()
    container = getattr(connection, "container", None)
    if not container:
        raise RuntimeError(
            f"Codex bootstrap requires a Docker connection for sandbox '{service}'"
        )
    return container


async def get_container_ip(container_id: str) -> str:
    """Retrieve the bridge IP address of the container."""
    code, stdout, stderr = await _run_host(
        [
            "docker", "inspect",
            "--format", "{{json .NetworkSettings.Networks}}",
            container_id,
        ]
    )
    if code != 0:
        raise RuntimeError(f"Could not inspect container {container_id}: {stderr.decode()}")
    try:
        networks = json.loads(stdout.decode())
        for _name, net in networks.items():
            ip = (net or {}).get("IPAddress")
            if ip:
                return ip
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse network settings for {container_id}: {e}") from e
    raise RuntimeError(f"Container {container_id} has no IP address on any network")


async def ensure_container_codex(
    env: SandboxEnvironment,
    container_id: str,
    codex_binary_path: str | Path | None = None,
    auth_path: str | Path | None = None,
) -> tuple[str, str, str | None]:
    """Ensure the codex binary and credentials exist inside the container.

    Returns ``(in_container_bin, codex_home, version)``.
    """
    # Validate explicit selections before any container mutation.
    selected_bin = await find_host_codex(codex_binary_path) if codex_binary_path is not None or os.environ.get("CODEX_PATH") else None
    selected_auth = find_host_auth(auth_path) if auth_path is not None or os.environ.get("CODEX_AUTH_PATH") else None
    # 1. Binary provisioning: If the container image already has the pinned release installed
    # (allowing Docker to resolve the platform architecture, e.g. arm64), keep it.
    # Otherwise, copy the host binary bundle into the container.
    found = await env.exec(["sh", "-c", "command -v codex && codex --version"], user="root")
    image_bin, _, image_version = found.stdout.strip().partition("\n")
    prebuilt = (selected_bin is None and found.returncode == 0 and image_bin != IN_CONTAINER_BIN
                and image_version.strip() == REQUIRED_CLI_VERSION)
    in_container_bin = image_bin if prebuilt else IN_CONTAINER_BIN
    executables: list[tuple[Path, str]] = []
    if not prebuilt:
        host_bin = selected_bin or await find_host_codex()
        host_cm = host_bin.parent / "codex-code-mode-host"
        if not host_cm.is_file():
            raise FileNotFoundError(f"Missing Codex companion executable beside selected binary: {host_cm}")
        executables = [(host_bin, IN_CONTAINER_BIN), (host_cm, IN_CONTAINER_CODE_MODE_HOST)]
        for source, target in executables:
            code, _, cp_err = await _run_host(["docker", "cp", "-L", str(source), f"{container_id}:{target}"])
            if code:
                raise RuntimeError(f"Failed to copy {source.name} into container: {cp_err.decode()}")
            await env.exec(["chmod", "+x", target], user="root")

    # 2. Dedicated CODEX_HOME + auth
    await env.exec(["mkdir", "-p", CODEX_HOME], user="root")
    force_auth = selected_auth is not None
    res_auth = await env.exec(["test", "-f", f"{CODEX_HOME}/auth.json"], user="root")
    if res_auth.returncode != 0 or force_auth:
        host_auth = selected_auth or find_host_auth()
        if host_auth:
            logger.info("Copying Codex auth to %s:%s/auth.json", container_id, CODEX_HOME)
            code, _, cp_err = await _run_host(
                ["docker", "cp", "-L", str(host_auth), f"{container_id}:{CODEX_HOME}/auth.json"]
            )
            if code != 0:
                raise RuntimeError(f"Failed to copy required Codex auth into container: {cp_err.decode()}")
            else:
                await env.exec(["chmod", "600", f"{CODEX_HOME}/auth.json"], user="root")
        else:
            logger.warning("No Codex auth available; app-server may fail to authenticate")

    # 3. Supply defaults only when the dedicated home has no configuration.
    config_exists = await env.exec(["test", "-f", f"{CODEX_HOME}/config.toml"], user="root")
    if config_exists.returncode != 0:
        await env.write_file(f"{CODEX_HOME}/config.toml", DEFAULT_CONFIG_TOML.encode("utf-8"))

    # 4. Verify execution and capture the version actually installed
    verify_res = await env.exec([in_container_bin, "--version"], user="root")
    if verify_res.returncode != 0:
        raise RuntimeError(
            f"Codex binary verification in container failed (exit {verify_res.returncode}): "
            f"{verify_res.stdout} {verify_res.stderr}"
        )
    version = (verify_res.stdout or verify_res.stderr or "").strip()
    if version != REQUIRED_CLI_VERSION:
        raise ValueError(f"Codex integration requires {REQUIRED_CLI_VERSION}; installed: {version!r}")
    logger.info("Codex in container verified: %s (CODEX_HOME=%s)", version, CODEX_HOME)
    def digest(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    provenance = {
        "version": version, "source_commit": PINNED_CODEX_SOURCE_COMMIT,
        "source": "image" if prebuilt else "host",
        "executables": [{"host_path": str(path), "container_path": target,
                         "sha256": await asyncio.to_thread(digest, path)}
                        for path, target in executables] or [{"container_path": in_container_bin}],
    }
    await env.write_file(f"{CODEX_HOME}/provenance.json", json.dumps(provenance).encode())
    return in_container_bin, CODEX_HOME, version or None
