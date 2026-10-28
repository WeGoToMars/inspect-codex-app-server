"""Inspect native Codex authentication status and enforce required auth modes."""

from collections.abc import Mapping
from typing import Any


def account_summary(response: Mapping[str, Any]) -> dict[str, Any]:
    """Extract non-sensitive account metadata from an account/read response."""
    account = response.get("account") or {}
    return {
        "type": account.get("type"),
        "email": account.get("email") if account.get("type") == "chatgpt" else None,
        "plan_type": account.get("planType"),
        "requires_openai_auth": response.get("requiresOpenaiAuth"),
        "observed": True,
    }


def require_authentication(expected: str, summary: dict[str, Any]) -> None:
    """Verify that the detected authentication mode matches the required mode."""
    if expected == "existing":
        return
    wanted = "chatgpt" if expected == "chatgpt" else "apiKey"
    if summary.get("type") != wanted or summary.get("requires_openai_auth") is not True:
        raise ValueError(
            f"Required {expected} authentication, but native Codex reports "
            f"{summary.get('type') or 'no verified account'} "
            f"(requiresOpenaiAuth={summary.get('requires_openai_auth')!r}). "
            "Configure native Codex credentials/provider before running; no billing fallback is allowed."
        )
