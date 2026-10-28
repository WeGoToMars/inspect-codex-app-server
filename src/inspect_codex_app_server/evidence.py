"""Manage raw protocol logs and native session evidence artifacts."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import shortuuid

from .container import _run_host


class NativeEvidence:
    def __init__(self, sample_id: str) -> None:
        base = Path(
            os.environ.get("CODEX_NATIVE_LOG_DIR")
            or Path(os.environ.get("INSPECT_LOG_DIR", "logs")) / "codex_native"
        )
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(sample_id))[:80]
        self.directory = (base / f"{label}-{shortuuid.uuid()}").resolve()
        self.directory.mkdir(parents=True)
        self._stream = (self.directory / "protocol.jsonl").open("w", encoding="utf-8")
        self._sequence = 0
        self._methods: dict[tuple[str, Any], str] = {}
        self.metadata: dict[str, Any] = {
            "directory": str(self.directory),
            "protocol": "protocol.jsonl",
            "manifest": "manifest.json",
            "retention": "full",
            "truncated": False,
            "closed": False,
            "provider_inputs": "unobserved",
            "exports": [],
            "failures": [],
            "outgoing_messages": "write attempts; transport completion is recorded separately",
        }

    def record(self, direction: str, message: dict[str, Any]) -> None:
        method = message.get("method")
        req_id = message.get("id")
        if direction in ("in", "out"):
            request_side = "client" if direction == "out" else "server"
            if method and req_id is not None:
                self._methods[(request_side, req_id)] = method
            elif req_id is not None:
                method = self._methods.pop(
                    ("server" if direction == "out" else "client", req_id), None
                )
        redacted = bool(method and method.startswith("account/"))
        payload = _redact_auth(message) if redacted else message
        self._sequence += 1
        self._stream.write(
            json.dumps(
                {
                    "sequence": self._sequence,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "direction": direction,
                    "message": payload,
                    "auth_fields_redacted": redacted,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._stream.flush()

    async def export(self, env: Any, container_id: str, codex_home: str) -> None:
        # Copy allowed session artifacts (never export credentials or provider config).
        paths = [
            "provenance.json",
            "sessions",
            "archived_sessions",
            "state_5.sqlite",
            "thread_history_1.sqlite",
            "goals_1.sqlite",
        ]
        for relative in paths:
            for name in (
                [relative, relative + "-wal", relative + "-shm"]
                if relative.endswith(".sqlite")
                else [relative]
            ):
                source = f"{codex_home}/{name}"
                result = await env.exec(["test", "-e", source], user="root")
                if result.returncode != 0:
                    continue
                destination = self.directory / "native" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                code, _, _ = await _run_host(
                    ["docker", "cp", "-L", f"{container_id}:{source}", str(destination)]
                )
                if code:
                    self.metadata["failures"].append(f"export failed: {name}")
                else:
                    self.metadata["exports"].append(f"native/{name}")

    async def reconcile(self, mapper: Any) -> None:
        """Recover response token usage records from native session rollout logs."""
        self.metadata["reconciled_usage_records"] = 0
        self.metadata["rollout_reconciliation"] = "running"
        owned_threads = mapper.thread_ids
        partial = False
        try:
            for path in (self.directory / "native").rglob("*.jsonl"):
                try:
                    # Decode each line independently in case the trailing write was truncated.
                    with path.open("rb") as stream:
                        for number, line in enumerate(stream, 1):
                            await asyncio.sleep(0)  # Keep the cleanup deadline enforceable.
                            try:
                                row = json.loads(line)
                                if row.get("type") != "token_usage_record":
                                    continue
                                payload = row["payload"]
                                # A child's records carry its root's session id even when
                                # none of its notifications arrived; earlier runs sharing
                                # the native home carry their own.
                                if (payload["thread_id"] not in owned_threads
                                        and payload.get("session_id") != mapper.root_thread_id):
                                    continue
                                response = {
                                    "threadId": payload["thread_id"],
                                    "turnId": payload["turn_id"],
                                    "responseId": payload["response_id"],
                                    "evidenceSource": "native-rollout",
                                    "usage": _native_usage(payload["usage"]),
                                }
                                total = _native_usage(payload["thread_token_usage"])
                            except (ValueError, KeyError) as exc:
                                partial = True
                                self.metadata["failures"].append(
                                    f"rollout reconciliation {path.name}:{number}: {exc}"
                                )
                                continue
                            mapper.on_response_completed("rollout/token_usage_record", response)
                            mapper.on_token_usage(
                                "rollout/thread_token_usage",
                                {"threadId": payload["thread_id"], "tokenUsage": {"total": total}},
                            )
                            self.metadata["reconciled_usage_records"] += 1
                except (OSError, UnicodeError) as exc:
                    partial = True
                    self.metadata["failures"].append(
                        f"rollout reconciliation {path.name}: {exc}"
                    )
        except asyncio.CancelledError:
            self.metadata["rollout_reconciliation"] = "interrupted"
            raise
        else:
            self.metadata["rollout_reconciliation"] = "partial" if partial else "complete"

    def close(self, *, shutdown: dict[str, Any], accounting: dict[str, Any]) -> None:
        self._stream.close()
        self.metadata.update(
            closed=True, events=self._sequence, shutdown=shutdown, accounting=accounting
        )
        self.metadata["truncated"] = not shutdown.get("stream_drained", False)
        (self.directory / "manifest.json").write_text(
            json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8"
        )


def _redact_auth(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[redacted]"
            if key.lower()
            in {
                "apikey",
                "api_key",
                "accesstoken",
                "refreshtoken",
                "authorization",
                "idtoken",
            }
            else _redact_auth(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_auth(item) for item in value]
    return value


def _native_usage(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        camel: usage[snake]
        for snake, camel in (
            ("input_tokens", "inputTokens"),
            ("cached_input_tokens", "cachedInputTokens"),
            ("cache_write_input_tokens", "cacheWriteInputTokens"),
            ("output_tokens", "outputTokens"),
            ("reasoning_output_tokens", "reasoningOutputTokens"),
            ("total_tokens", "totalTokens"),
        )
        if snake in usage
    }
