"""Prepare a compatibility release for updated dependencies or manual publication.

The workflow owns locking, protocol generation, tests, and publication. This
script only discovers releases and updates the package metadata and native pin.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from packaging.version import Version

PACKAGE = Path(__file__).resolve().parents[1]
PYPI = "https://pypi.org/pypi/inspect-ai/json"
NPM = "https://registry.npmjs.org/@openai%2fcodex/latest"
CODEX_API = "https://api.github.com/repos/openai/codex"


def read_json(url: str) -> dict[str, Any]:
    headers = {"User-Agent": "inspect-codex-app-server-release"}
    if url.startswith("https://api.github.com/") and os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def stable_version(value: str) -> Version:
    if not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError(f"Expected a stable major.minor.patch version, got {value!r}")
    return Version(value)


def latest_versions() -> tuple[str, str]:
    inspect = read_json(PYPI)
    releases = [
        Version(version)
        for version, files in inspect["releases"].items()
        if re.fullmatch(r"\d+\.\d+\.\d+", version)
        and any(not file.get("yanked", False) for file in files)
    ]
    codex = read_json(NPM)
    if codex.get("deprecated"):
        raise ValueError(f"Latest Codex release is deprecated: {codex['deprecated']}")
    return str(max(releases)), str(stable_version(codex["version"]))


def codex_source_commit(version: str) -> str:
    stable_version(version)
    release = read_json(f"{CODEX_API}/releases/tags/rust-v{version}")
    if release["draft"] or release["prerelease"]:
        raise ValueError(f"Codex {version} is not a published stable GitHub release")
    obj = read_json(f"{CODEX_API}/git/ref/tags/rust-v{version}")["object"]
    while obj["type"] == "tag":
        obj = read_json(f"{CODEX_API}/git/tags/{obj['sha']}")["object"]
    if obj["type"] != "commit" or not re.fullmatch(r"[0-9a-f]{40}", obj["sha"]):
        raise ValueError(f"Codex {version} tag does not resolve to a source commit")
    return obj["sha"]


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


def next_calendar_version(current: str, release_date: date) -> str:
    version = Version(current)
    parts = version.release
    if (
        version.epoch or version.pre or version.dev is not None
        or version.post is not None or version.local or len(parts) not in (3, 4)
    ):
        raise ValueError("Expected a calendar version YYYY.M.D or YYYY.M.D.N")
    current_date = date(*parts[:3])
    if current_date > release_date:
        raise ValueError(f"Current package date {current_date} is later than release date {release_date}")
    base = f"{release_date.year}.{release_date.month}.{release_date.day}"
    if current_date == release_date:
        return f"{base}.{parts[3] + 1 if len(parts) == 4 else 1}"
    return base


def prepare_update(
    package: Path, inspect: str, codex: str, *, dry_run: bool = False, release: bool = False
) -> dict[str, Any]:
    manifest_path = package / "pyproject.toml"
    manifest_text = manifest_path.read_text()
    manifest = tomllib.loads(manifest_text)["project"]
    inspect_pin, = [d for d in manifest["dependencies"] if d.startswith("inspect-ai==")]
    old_inspect = inspect_pin.removeprefix("inspect-ai==")
    container_path = package / "src/inspect_codex_app_server/container.py"
    container = container_path.read_text()
    old_codex, = re.findall(r'^PINNED_CODEX_VERSION = "([^"]+)"$', container, re.MULTILINE)
    old_commit, = re.findall(r'^PINNED_CODEX_SOURCE_COMMIT = "([0-9a-f]{40})"$', container, re.MULTILINE)
    inspect = str(max(stable_version(inspect), stable_version(old_inspect)))
    codex = str(max(stable_version(codex), stable_version(old_codex)))
    changed = release or (inspect, codex) != (old_inspect, old_codex)
    next_version = (
        next_calendar_version(manifest["version"], datetime.now(timezone.utc).date())
        if changed else manifest["version"]
    )
    commit = codex_source_commit(codex) if codex != old_codex else old_commit
    updates = {
        manifest_path: replace_once(
            replace_once(manifest_text, inspect_pin, f"inspect-ai=={inspect}"),
            f'version = "{manifest["version"]}"', f'version = "{next_version}"',
        ),
        container_path: replace_once(
            replace_once(container, f'PINNED_CODEX_VERSION = "{old_codex}"', f'PINNED_CODEX_VERSION = "{codex}"'),
            f'PINNED_CODEX_SOURCE_COMMIT = "{old_commit}"', f'PINNED_CODEX_SOURCE_COMMIT = "{commit}"',
        ),
    }
    if codex != old_codex:
        readme = package / "README.md"
        updates[readme] = replace_once(readme.read_text(), f"@openai/codex@{old_codex}", f"@openai/codex@{codex}")
    # Validate all replacements before touching any tracked file.
    if changed and not dry_run:
        for path, text in updates.items():
            path.write_text(text)
    return {
        "changed": changed,
        "version": next_version,
        "tag": f"v{next_version}",
        "prerelease": Version(next_version).is_prerelease,
        "inspect": inspect,
        "previous_inspect": old_inspect,
        "codex": codex,
        "previous_codex": old_codex,
        "codex_source_commit": commit,
    }


def release_notes(plan: dict[str, Any]) -> str:
    return (
        f"Compatibility release {plan['version']}.\n\n"
        "Tested with this exact dependency pair:\n\n"
        f"- Inspect AI (`inspect-ai`): [{plan['inspect']}]"
        f"(https://github.com/UKGovernmentBEIS/inspect_ai/releases/tag/{plan['inspect']}).\n"
        f"- Codex CLI (`@openai/codex`): [{plan['codex']}]"
        f"(https://github.com/openai/codex/releases/tag/rust-v{plan['codex']}).\n\n"
        f"Codex source: `{plan['codex_source_commit']}`.\n\n"
        "Passed protocol validation, lint, type checking, the default test suite, "
        "and Docker acceptance using local mock model responses.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report the candidate without editing tracked files")
    parser.add_argument("--release", action="store_true", help="Prepare a new release even when dependency pins are unchanged")
    parser.add_argument("--output-dir", type=Path, help="Write release metadata and notes for subsequent workflow jobs")
    args = parser.parse_args()
    inspect, codex = latest_versions()
    plan = prepare_update(PACKAGE, inspect, codex, dry_run=args.dry_run, release=args.release)
    print(json.dumps(plan, indent=2))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "candidate.json").write_text(json.dumps(plan, indent=2) + "\n")
        (args.output_dir / "notes.md").write_text(release_notes(plan))
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            for key, value in plan.items():
                output.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")


if __name__ == "__main__":
    main()
