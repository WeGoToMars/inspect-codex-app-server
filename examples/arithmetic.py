"""Small Inspect task. Running this consumes native Codex account usage."""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match

from inspect_codex_app_server import codex_app_server


@task
def arithmetic():
    return Task(
        dataset=[Sample(input="What is 19 + 23? Reply with the number only.", target="42")],
        solver=codex_app_server(authentication="chatgpt", max_child_agents=1),
        scorer=match(),
        sandbox=("docker", str(Path(__file__).with_name("compose.yaml"))),
    )
