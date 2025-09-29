"""Query skill leveraging the existing text generation flow."""

from __future__ import annotations

from typing import Mapping, Optional

from torch import Tensor

from .base import SkillSpec

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime


class QuerySkill(SkillSpec):
    """Default skill emitting plain text answers."""

    def __init__(self) -> None:
        super().__init__(name="query")

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        prompt: str,
        *,
        image: Optional[object] = None,
        image_crops: Optional[object] = None,
        options: Optional[Mapping[str, object]] = None,
    ) -> Tensor:
        # Query uses the existing runtime helper to assemble the prompt.
        return runtime.build_prompt_tokens(prompt)
