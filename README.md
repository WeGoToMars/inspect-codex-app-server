# inspect-codex-app-server

An [Inspect](https://inspect.aisi.org.uk/) solver and agent that run native Codex sessions inside Docker sandboxes via the Codex App Server JSON-RPC protocol.

Codex manages inference, tool execution, and subagent orchestration, while Inspect AI manages evaluation, scoring, tool governance, and token, message, and time limits.

## Key Features

- **Use Your ChatGPT Subscription**: Run evaluations directly against your ChatGPT subscription plan (or API key) using native Codex authentication.
- **Docker Sandbox Execution**: Runs Codex as root with full access inside Docker sandboxes, including filesystem tools, bash commands, and subagents.
- **Inspect Agent**: `codex_agent()` composes with `as_solver()`, `as_tool()`, `handoff()` and `run()`, with react-style `submit`, `attempts`, `on_continue` and `retry_refusals`.
- **Inspect Tool Bridging**: Exposes Inspect tools (`use_tools`, MCP tool sources) directly to Codex as dynamic tools with validation, approval and review handling.
- **Inspect Approvals**: Routes Codex's native command and file-change approval requests to the active Inspect approver.
- **Native Skills, Web Search and Roles**: Installs Inspect skills as Codex skills, sets the native web search mode, and declares agent roles for Codex's child agents.
- **Operator Interruption and Checkpoints**: Responds to operator interrupts and messages through Inspect's agent channel, and resumes from Inspect checkpoints.
- **Session & Quota Management**: Handles rate limits, quota pauses (`quota_hold_sec`), transient turn-start retries with backoff, and token budgeting.
- **Transcript & Usage Tracking**: Reconstructs model activity, tool events, and subagent spans into Inspect transcripts with live metrics.
- **Evidence & Artifact Capture**: Retains protocol transcripts, session databases, and execution manifests for debugging and auditability.

## Prerequisites & Installation

- Python 3.11+
- Docker (with Inspect Docker sandbox support)
- Node.js and npm when a compatible pinned Codex bundle is unavailable.

For an npm-equipped container image, preinstall with
`RUN npm install -g --prefix /opt/codex @openai/codex@0.166.0` and add
`/opt/codex/bin` to `PATH`. See [provisioning constraints](docs/technical-notes.md#provisioning-and-authentication).

Install the package:

```bash
pip install -e .
```

## Authentication

Run `codex login` on your host and provide its `~/.codex/auth.json`, or select
another file with `CODEX_AUTH_PATH` / `auth_path`.

Supported authentication modes:
- `"chatgpt"`: Uses ChatGPT subscription credentials.
- `"api_key"`: Uses an OpenAI API key.
- `"existing"`: Uses the container configuration and available credentials without enforcing an authentication mode.

Credentials are copied into `/root/.cache/codex-inspect/auth.json` inside the
container. See [configuration, credential storage, and logging details](docs/technical-notes.md#provisioning-and-authentication).

## Quickstart

Define a task using `codex_app_server`:

```python
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_codex_app_server import codex_app_server

@task
def arithmetic():
    return Task(
        dataset=[Sample(input="What is 19 + 23? Reply with the number only.", target="42")],
        solver=codex_app_server(
            authentication="chatgpt",
            timeout_sec=600,
            max_child_agents=1,
        ),
        scorer=match(),
        sandbox=("docker", str(Path(__file__).with_name("compose.yaml"))),
    )
```

Run evaluation:

```bash
inspect eval examples/arithmetic.py --model codex/default
```

## Inspect Agent

`codex_agent()` runs the same native session as an Inspect agent, so it composes like `react()`:

```python
from inspect_ai.agent import as_solver, as_tool
from inspect_codex_app_server import CodexSubagent, codex_agent

coder = codex_agent(
    name="coder",
    description="Solves programming tasks in the sandbox.",
    prompt="Work carefully and verify your answer.",
    submit=True,          # accept answers from the root submit tool
    attempts=3,           # score each submission; steer onward when incorrect
    subagents=[CodexSubagent(name="reviewer", description="Reviews changes.", token_limit=200_000)],
)

solver = as_solver(coder)   # or as_tool(coder), handoff(coder), run(coder, ...)
```

See the [Inspect feature mapping](docs/technical-notes.md#inspect-feature-mapping)
and [model settings and input formats](docs/technical-notes.md#model-settings-and-conversation-input)
for compatibility details.

### Interruption and checkpoints

Operator interrupts stop native work and wait for follow-up input. Inspect
checkpoints can resume the root thread; scoring-only resume starts no Codex
process. See [interruption and resume notes](docs/technical-notes.md#operator-interruption-and-checkpoint-resume).

## Solver Options

The `codex_app_server` solver accepts the following configuration parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `authentication` | `"existing"` \| `"chatgpt"` \| `"api_key"` | `"existing"` | Required authentication mode. |
| `timeout_sec` | `float` | `600.0` | Run budget including setup; teardown runs separately. |
| `max_child_agents` | `int` | `4` | Maximum concurrent subagents per sample. |
| `developer_instructions` | `str \| None` | `None` | Thread developer instructions. |
| `base_instructions` | `str \| None` | `None` | Replacement for Codex's base instructions. |
| `reasoning_effort` | `str \| None` | `None` | Model reasoning effort (defaults to the evaluation's `GenerateConfig`). |
| `reasoning_summary` | `str \| None` | `None` | Reasoning summary detail (defaults to the evaluation's `GenerateConfig`). |
| `retry_refusals` | `int \| None` | `None` | Policy-blocked turns to continue before finishing with `content_filter`. |
| `skills` | `list[str \| Path \| Skill] \| None` | `None` | Inspect skills installed as Codex skills. |
| `web_search` | `"disabled"` \| `"cached"` \| `"indexed"` \| `"live"` \| `None` | `None` | Native web search mode. |
| `subagents` | `list[CodexSubagent] \| None` | `None` | Native agent roles for Codex's child agents. |
| `service_tier` | `str \| None` | `None` | Service tier requested on root turns (e.g. `"priority"`, `"flex"`). |
| `cwd` | `str` | `"/app"` | Working directory inside the container. |
| `transport` | `"stdio"` \| `"websocket"` | `"stdio"` | Transport protocol to communicate with the app server. |
| `policy` | `RunPolicy \| None` | `None` | Custom execution lifecycle and completion policy. |
| `codex_path` | `str \| None` | `None` | Host binary override (`CODEX_PATH` also works), taking precedence over image reuse. |
| `auth_path` | `str \| None` | `None` | Explicit path to the host `auth.json`. |

## Custom Lifecycle Policies (`RunPolicy`)

`RunPolicy` configures completion rules, quota waits, setup hooks, and timed
steering reminders. See the [lifecycle example and limits](docs/technical-notes.md#limits-and-lifecycle-policies).

## Development & Testing

The default suite includes real pinned Codex tests using local scripted model
responses, without credentials or paid inference. Docker acceptance additionally
requires a running Docker daemon and `docker pull python:3.13-slim`.

```bash
# Install development dependencies
uv sync --group dev

# Run the default suite, including native tests; excludes Docker acceptance
uv run pytest -q

# Run only tests that launch the real pinned Codex executable
uv run pytest -q -m native

# Run Docker acceptance separately
uv run pytest -q -m containers

# Run linter and type checker
uv run ruff check .
uv run ty check

# Validate generated JSON-RPC protocol schemas
uv run python scripts/generate_protocol.py --check
```

## Automated compatibility releases

The [test workflow](.github/workflows/test.yml) checks pushes and pull requests, while
the [release workflow](.github/workflows/release.yml) checks upstream releases hourly
and publishes passing candidates to GitHub and PyPI. See [release setup, publication, and retry notes](docs/releases.md).

## License

[MIT](LICENSE)
