"""Core skill interfaces shared across inference flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from torch import Tensor

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime


@dataclass(frozen=True)
class SkillSpec:
    """Declarative description of a skill's prompt and decoding behaviour."""

    name: str

    # ------------------------------------------------------------------
    # Prompt construction

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        prompt: str,
        *,
        image: Optional[object] = None,
        image_crops: Optional[object] = None,
        options: Optional[Mapping[str, object]] = None,
    ) -> Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Text formatting

    def decode_tokens(
        self,
        runtime: "MoondreamRuntime",
        tokens: Sequence[int],
    ) -> str:
        if not tokens:
            return ""
        return runtime.tokenizer.decode(list(tokens))

    def stream_text(
        self,
        runtime: "MoondreamRuntime",
        tokens: Sequence[int],
    ) -> str:
        return self.decode_tokens(runtime, tokens)


class SkillRegistry:
    """Lookup table for skills with a default entry."""

    def __init__(self, skills: Iterable[SkillSpec]) -> None:
        self._skills: Dict[str, SkillSpec] = {}
        self._default: Optional[str] = None
        for spec in skills:
            name = spec.name
            if name in self._skills:
                raise ValueError(f"Duplicate skill registered: {name}")
            self._skills[name] = spec
            if self._default is None:
                self._default = name
        if self._default is None:
            raise ValueError("SkillRegistry requires at least one skill")

    # ------------------------------------------------------------------

    @property
    def default(self) -> SkillSpec:
        return self._skills[self._default]  # type: ignore[index]

    def resolve(self, skill: Optional[str | SkillSpec]) -> SkillSpec:
        if isinstance(skill, SkillSpec):
            return skill
        if skill is None:
            return self.default
        try:
            return self._skills[skill]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown skill '{skill}'") from exc

    def add(self, spec: SkillSpec) -> None:
        if spec.name in self._skills:
            raise ValueError(f"Skill '{spec.name}' already registered")
        self._skills[spec.name] = spec
