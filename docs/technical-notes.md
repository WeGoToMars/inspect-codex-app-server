# Runtime and compatibility notes

Technical notes on adapter architecture, runtime behavior, and compatibility between Inspect AI and the Codex App Server. For installation steps and quickstart examples, see the [README](../README.md).

## Provisioning and authentication

### Execution environment and sandboxing
Codex runs as `root` inside an Inspect Docker sandbox with native `danger-full-access`. The container boundary isolates the host from code execution and agent tool actions. By default, the adapter communicates with the app server over `stdio` using Inspect's remote process transport. Alternatively, setting the transport to `websocket` connects directly to the container's Codex WebSocket server, which requires the host to have network reachability to the container's bridge IP.

### Codex binary resolution
When launching a container, the adapter provisions a Codex binary using the following order of precedence:

1. **Host bundle override (`codex_path` or `CODEX_PATH`)**: An explicit path always takes precedence and is copied into the container.
2. **Pre-installed container binary**: Reused if the container already has a Codex binary whose version matches the pinned release, provided it is installed outside `/usr/local/bin/codex` (which is reserved for adapter-managed copies). Installing to `/opt/codex`, as demonstrated in the README, avoids host-bundle downloads and enables container reuse across architectures like ARM64.
3. **Automatic host download**: If no explicit binary or matching image installation is found, the adapter automatically discovers and downloads the pinned release on the host. This fallback is limited to Linux x86_64 and requires an adjacent `codex-code-mode-host` binary.

### Authentication and credentials
Host credentials are read from `~/.codex/auth.json` by default, or from the path specified by `auth_path` or `CODEX_AUTH_PATH`. The adapter copies this file into `/root/.cache/codex-inspect/auth.json` inside the sandbox with `0600` permissions. Existing credentials in the container are preserved unless an explicit path is provided. If your credentials are stored only in an OS keyring (such as macOS Keychain), they cannot be copied from disk; see the official [Codex authentication documentation](https://learn.chatgpt.com/docs/auth#login-caching) to export or cache credentials to an `auth.json` file.

When using `"chatgpt"` or `"api_key"` authentication modes, the adapter verifies that the container's authenticated account status matches the requested mode and connects to an OpenAI provider. The `"existing"` mode accepts whatever credentials and account configuration are present in the container without validation.

### Configuration, privacy, and checkpoints
Codex reads its container settings from `/root/.cache/codex-inspect/config.toml`. The adapter generates default settings if this file is missing, but does not import `config.toml` from the host.

Exported artifacts omit `auth.json` and `config.toml`. Native JSON-RPC logs automatically redact known authentication fields while preserving account identifiers and tool call payloads; however, arbitrary secrets printed to stdout or stderr by tool executions are not automatically redacted. Inspect's checkpoint system excludes `.cache` directories by default, ensuring credentials in `/root/.cache/codex-inspect/` are excluded from sandbox snapshots and backups. (Container images that still use the legacy `/root/.codex-inspect` directory should be updated to `/root/.cache/codex-inspect`).

## Inspect feature mapping

Inspect initializes and manages the evaluation task and agent span, while Codex drives inference, context window management, and native tools.

| Inspect feature | `codex_agent` behavior |
|---|---|
| `prompt` | Passed as Codex developer instructions. |
| `tools` (`Tool`, `ToolDef`, MCP sources) | Bridged dynamically to Codex as native tools. |
| `submit` / `attempts` | When `submit=True`, only the root `submit` tool supplies the final answer. Incorrect submissions trigger `incorrect_message` steering prompts until attempts run out. |
| `on_continue` | When paired with `submit`, a custom string replaces the continuation prompt, or a callback decides whether to continue after each turn. |
| `retry_refusals` | Retries turns blocked by model policy; persistent refusals finish with `content_filter`, while exhausted context windows finish with `model_length`. |
| `approval` / `review` | Routes native command and file-change approvals through Inspect's approver; reviews apply to bridged tools. |
| `todo_write` | Mapped to Codex's native `update_plan` and rendered as an Inspect plan checklist. |
| `skills` | Installed into the container as native Codex skills. |
| `web_search` | Configures the native search mode (`"disabled"`, `"cached"`, `"indexed"`, `"live"`); `None` leaves the default unchanged. |
| `memory` | Supported by passing Inspect's `memory()` tool in `tools`. |
| `subagents` | Declares child agent roles via `CodexSubagent`, exposed to the root agent through `spawn_agent`. |
| `compaction` / `truncation` / `max_depth` | Managed internally by Codex rather than exposed through Inspect. Set `auto_compact_token_limit` to configure the compaction threshold. |

A subagent's `token_limit` applies to each child agent's cumulative lifetime usage, including replayed turns. The adapter monitors usage and interrupts active turns when a subagent reaches its limit; because accounting and interruption occur asynchronously, token counts may slightly exceed the configured limit before execution halts. When `submit=True` is enabled, the agent must make an authoritative submission to produce a valid score, though runs can still exit without submitting if they encounter a limit (timeout, tokens, or messages) or reach an inactive goal state.

## Model settings and conversation input

### Model settings and catalog validation
Explicit reasoning parameters on `codex_agent` or `codex_app_server` (`reasoning_effort` and `reasoning_summary`) override any values in the evaluation's `GenerateConfig`. Structured output is supported by mapping `GenerateConfig.response_schema` to Codex's turn schema, while any unsupported `GenerateConfig` parameters are recorded under `codex_unsupported_settings` in sample metadata.

When using Codex's built-in OpenAI provider, the adapter validates requested root and subagent models, reasoning effort levels, and the root service tier against Codex's internal model catalog before starting. Custom providers bypass this validation check. Note that catalog presence only verifies that the model name and settings are recognized by Codex; it does not check whether your account has sufficient quota or permissions to call it.

### Conversation replay and input types
When continuing an existing conversation or resuming from earlier turns, the adapter reconstructs native session history. User text, user images, assistant tool calls, and text tool results are replayed into history, while system messages become developer instructions and assistant reasoning blocks are omitted. Any unrecognized content blocks in previous messages are rejected prior to inference. If the replayed history ends with a user message, Codex starts a new turn immediately; otherwise, the adapter replays the full history and issues a continuation prompt.

Current user inputs support text, images, and local audio files or audio data URLs. Remote audio URLs are rejected, and audio inputs require a model that explicitly advertises audio support in the native catalog. Any unsupported content blocks in the current turn are omitted and recorded in `codex_unsupported_settings`.

If OpenAI reroutes a request due to safety policies (signaled via a `model/rerouted` notification), the event is recorded in `codex_safety`. Subsequent token usage is credited to the reported serving model until that thread begins its next turn.

## Limits and lifecycle policies

Inspect token, message, and overall run time limits (`timeout_sec`) are fully supported. However, cost limits and model-generation turn limits are rejected at startup because native Codex does not expose authoritative cost calculations or raw model-call counts. The configured `timeout_sec` covers the entire run, including container provisioning and setup. Time spent waiting on quota holds or operator input is credited back to the budget so it does not penalize model execution time, while teardown runs under a separate, bounded timer. Transient errors during turn initialization are retried automatically with exponential backoff.

### Custom lifecycle policies (`RunPolicy`)
For advanced control over completion conditions, rate-limit pauses, and time-based prompts, pass a `RunPolicy` to the solver or agent:

```python
from inspect_codex_app_server import RunPolicy

policy = RunPolicy(
    completion="goal",
    quota_hold_sec=300.0,
    quota_poll_sec=30.0,
    checkpoint_fractions=(0.5, 0.8),
    checkpoint_message=lambda fraction, elapsed, limit: (
        f"{fraction * 100:.0f}% of time budget elapsed ({elapsed:.0f}s / {limit:.0f}s). "
        "Please summarize progress and provide your final answer."
    ),
)
```

The policy configures three primary behaviors:

- **Completion mode**: `"turn"` finishes after the first successful root turn using its final answer. `"goal"` allows the agent to continue across turns until the native goal completes, with an optional `terminal_nudge` to prompt for a final answer before an unsuccessful exit. `"submission"` requires an authoritative `receipt_reader` to extract the final answer from an item or tool output.
- **Quota holds**: When Codex raises `usageLimitExceeded`, `quota_hold_sec` defines how long the adapter will pause and wait while polling `account/rateLimits/read` every `quota_poll_sec`. Time spent waiting is credited back to the run's time budget. If the overall run deadline expires during a hold, the run exits cleanly.
- **Steering reminders**: `checkpoint_fractions` schedules prompt injections at specific fractions of the time limit (for example, at 50% and 80%) to warn the agent about its remaining budget and prompt it to wrap up. These runtime reminders are purely conversational and are distinct from Inspect's persisted disk checkpoints.

## Operator interruption and checkpoint resume

### Operator interruption
Each run communicates through an Inspect agent channel. When an operator sends an interrupt, the adapter pauses the active goal and stops in-flight turns across the root agent and any running subagents. Active bridged tools are allowed to finish executing before the operator's follow-up message redirects the run. Any time spent waiting for operator input is credited back to the task timeout. Messages sent while the agent is executing start the next turn, and all operator interactions are recorded as `InterruptEvent` records and logged in metadata under `codex_operator_interventions`.

### Checkpoints and session resume
Inspect checkpoints capture conversation history, agent outputs, and adapter token accounting. At each turn boundary, Codex session files are synced to `~/.codex-inspect-checkpoint` in the sandbox user's home directory. If you configure custom sandbox backup paths, ensure this directory is included.

Resuming a sample restores the mirrored session state and continues the root thread from where it left off; subagents are not automatically re-spawned upon resume. Resuming an evaluation solely for scoring does not start a Codex process.

Restored token counters and an internal response ledger prevent previously billed usage from being charged again after resume. However, because the pinned Codex release only emits raw streaming events on `thread/start` and not on `thread/resume`, live per-response usage metrics and delegated prompt reconstruction are unavailable after resuming. Lifetime token accounting continues from restored counters and native session rollout logs, and this limitation is recorded as `raw_events_unavailable_after_resume` in `codex_unsupported_settings`. Checkpoints that lack accounting state cannot resume execution.

## Transcript and evidence scope

`ModelEvent` records in Inspect transcripts represent reconstructed agent activity. Because Codex manages model calls internally, these records do not contain low-level provider wire details such as exact HTTP payloads, API call timings, or internal generation parameters. Full native JSON-RPC protocol logs and session databases are retained separately in the run artifacts for auditing and debugging.

The adapter deduplicates response records between live wire events and session rollout files using thread and response identifiers. When granular per-response data is unavailable, native aggregate lifetime reports provide total token accounting. Any export anomalies—such as partial rollouts or interrupted reconciliation passes—are logged directly in the sample's evidence manifest; the adapter never fabricates synthetic provider requests.
