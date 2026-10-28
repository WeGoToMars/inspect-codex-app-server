# Package test gates

## Admission

Extend an existing scenario first. Add a case only when it catches a named defect
existing tests miss, with an independently derived expected result. For regressions,
verify failure against the broken behavior when practical; report if unverified.

## Required assertions by change

| Changed behavior | Gate |
| --- | --- |
| Accounting / evidence | Duplicate wire and rollout records count once; unrelated threads contribute zero. Resume after 25 tokens, consume 5: recorded total 30, newly charged tokens 5. |
| Completion / errors | Child answers cannot replace the root answer. Provider/transport failures persist as execution errors; refusals and limits retain distinct outcomes. |
| Interruption | Interrupt active native and bridged work: tools stop before redirect. Operator wait exceeds remaining deadline: redirect still resumes. |
| Shutdown | Stall a real process, close within a bounded timeout, verify its PID disappears and the final protocol record survives. |
| Checkpoints | Fresh native home resumes the same thread with history, tools, and accurate usage. Scoring-only resume starts no Codex process. |
| Native options / RPC | Installed pinned binary demonstrates the new assumption, with valid and invalid controls where applicable. A scripted app-server peer alone is insufficient. |

Preserve or extend the affected row's evidence. Different ordering, ownership,
or failure mechanisms justify separate cases; different numbers or model labels
on the same path do not.

## Checks before merge

Run from this package directory:

| Change | Required commands |
| --- | --- |
| Code / tests | `uv run pytest -q`; `uv run ruff check .`; `uv run ty check` |
| Provisioning, transport, lifecycle, checkpoints; every release | Also `uv run pytest -q -m containers` |
| Codex/Inspect pin, RPC schema, generator | All above plus `uv run python scripts/generate_protocol.py --check`; regenerate bindings |
| Documentation only | Check changed links and commands |

Default pytest includes native tests and excludes containers. Missing pinned-binary
or Docker prerequisites must fail visibly. Report required checks left unrun.

## Reject / prune

Remove duplicate helper, forwarding, call-count, source-layout, and presentation
checks unless they protect a distinct external contract. Keep behavior-preserving
refactors assertion-neutral. No coverage quotas or mandatory tests per edit.

Peers reject unexpected RPCs and never reproduce adapter accounting/recovery.
Use real processes/files for lifecycle claims; own temporary homes, logs, tasks,
and containers. Synchronize on events; bound teardown. No flaky-test retries,
user-artifact cleanup, live credentials, or paid inference in correctness gates.
