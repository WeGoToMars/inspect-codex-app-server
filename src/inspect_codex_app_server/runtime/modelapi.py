"""ModelAPI placeholder registration for Codex solver routing."""

from __future__ import annotations

from typing import Any

from inspect_ai.model import ModelAPI, modelapi


def register_placeholder_modelapi(name: str) -> None:
    """Register a placeholder ModelAPI under ``name`` (e.g. ``"codex"``)."""

    @modelapi(name=name)
    def _placeholder_model_provider():
        """Placeholder ModelAPI allowing Inspect eval_set to resolve codex/* model names."""

        class PlaceholderModelAPI(ModelAPI):
            async def generate(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError(
                    f"Direct ModelAPI generation is not used; "
                    f"execution is delegated to the {name} agent server."
                )

        return PlaceholderModelAPI
