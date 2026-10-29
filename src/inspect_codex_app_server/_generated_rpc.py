# Generated from codex-cli 0.166.0; schema SHA-256: 70a8140b0e785292d3da5da1f14c9d8e3f1a2fb0446963eaaa1e3c4e10cee4bb
# Generator: datamodel-code-generator 0.86.0
# Regenerate with scripts/generate_protocol.py; do not edit.
from __future__ import annotations

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

    async def account_rateLimits_read(
        self, params: p.GetAccountRateLimitsParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.GetAccountRateLimitsResponse:
        return cast(p.GetAccountRateLimitsResponse, await self._sender.send_request(
            "account/rateLimits/read", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def account_read(
        self, params: p.GetAccountParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.GetAccountResponse:
        return cast(p.GetAccountResponse, await self._sender.send_request(
            "account/read", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def initialize(
        self, params: p.InitializeParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.InitializeResponse:
        return cast(p.InitializeResponse, await self._sender.send_request(
            "initialize", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def model_list(
        self, params: p.ModelListParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.ModelListResponse:
        return cast(p.ModelListResponse, await self._sender.send_request(
            "model/list", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def skills_list(
        self, params: p.SkillsListParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.SkillsListResponse:
        return cast(p.SkillsListResponse, await self._sender.send_request(
            "skills/list", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def thread_goal_get(
        self, params: p.ThreadGoalGetParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.ThreadGoalGetResponse:
        return cast(p.ThreadGoalGetResponse, await self._sender.send_request(
            "thread/goal/get", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def thread_goal_set(
        self, params: p.ThreadGoalSetParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.ThreadGoalSetResponse:
        return cast(p.ThreadGoalSetResponse, await self._sender.send_request(
            "thread/goal/set", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def thread_inject_items(
        self, params: p.ThreadInjectItemsParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.ThreadInjectItemsResponse:
        return cast(p.ThreadInjectItemsResponse, await self._sender.send_request(
            "thread/inject_items", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def thread_read(
        self, params: p.ThreadReadParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.ThreadReadResponse:
        return cast(p.ThreadReadResponse, await self._sender.send_request(
            "thread/read", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def thread_resume(
        self, params: p.ThreadResumeParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.ThreadResumeResponse:
        return cast(p.ThreadResumeResponse, await self._sender.send_request(
            "thread/resume", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def thread_start(
        self, params: p.ThreadStartParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.ThreadStartResponse:
        return cast(p.ThreadStartResponse, await self._sender.send_request(
            "thread/start", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def turn_interrupt(
        self, params: p.TurnInterruptParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.TurnInterruptResponse:
        return cast(p.TurnInterruptResponse, await self._sender.send_request(
            "turn/interrupt", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def turn_start(
        self, params: p.TurnStartParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.TurnStartResponse:
        return cast(p.TurnStartResponse, await self._sender.send_request(
            "turn/start", params, timeout=timeout, _allow_failed=_allow_failed,
        ))

    async def turn_steer(
        self, params: p.TurnSteerParams, *, timeout: float | None = 60.0,
        _allow_failed: bool = False,
    ) -> p.TurnSteerResponse:
        return cast(p.TurnSteerResponse, await self._sender.send_request(
            "turn/steer", params, timeout=timeout, _allow_failed=_allow_failed,
        ))
