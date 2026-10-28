"""Native Codex agent roles, offered to the model through ``spawn_agent``."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

ROLE_NAME = re.compile(r"[a-z0-9]([a-z0-9_-]*[a-z0-9])?")


@dataclass(frozen=True)
class CodexSubagent:
    """A role Codex can spawn child agents with (deepagent's ``Subagent``, natively).

    Codex lists each role and its description in ``spawn_agent``'s
    ``agent_type``; a child spawned with the role runs with its prompt, model
    and reasoning effort. Children share the run's bridged tools and skills.
    """

    name: str
    """Role name the model selects (lowercase letters, digits, ``-`` and ``_``)."""

    description: str
    """When to use the role, shown to the model."""

    prompt: str | None = None
    """Instructions for the role's children (Codex developer instructions)."""

    model: str | None = None
    """Model for the role's children (defaults to the parent's)."""

    reasoning_effort: str | None = None
    """Reasoning effort for the role's children."""

    token_limit: int | None = None
    """Tokens each child of the role may use; its turn is interrupted beyond it."""

    def __post_init__(self) -> None:
        if not ROLE_NAME.fullmatch(self.name):
            raise ValueError(f"Invalid subagent name {self.name!r}: use lowercase letters, digits, '-' and '_'")
        if not self.description.strip():
            raise ValueError(f"Subagent {self.name!r} requires a description")
        if self.token_limit is not None and self.token_limit <= 0:
            raise ValueError(f"Subagent {self.name!r} token_limit must be positive")


def toml_string(value: str) -> str:
    """A TOML basic string.

    JSON's escapes are TOML's except that TOML has no surrogate pairs, so
    non-ASCII text stays literal; DEL, which JSON leaves raw, must be escaped.
    """
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def role_config(role: CodexSubagent) -> str:
    """The role's Codex config file (TOML)."""
    settings = {"developer_instructions": role.prompt, "model": role.model,
                "model_reasoning_effort": role.reasoning_effort}
    return "".join(f"{key} = {toml_string(value)}\n" for key, value in settings.items() if value is not None)


def role_overrides(role: CodexSubagent, config_file: str) -> list[str]:
    """``-c`` overrides declaring the role."""
    return ["-c", f"agents.{role.name}.description={toml_string(role.description)}",
            "-c", f"agents.{role.name}.config_file={toml_string(config_file)}"]
