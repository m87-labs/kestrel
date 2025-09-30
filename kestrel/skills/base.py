"""Core skill interfaces shared across inference flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from torch import Tensor

if False:  # pragma: no cover - type-checking imports
    from kestrel.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest
    from kestrel.moondream.runtime import Token


@dataclass(frozen=True)
class SkillSpec:
    """Declarative description of a skill's prompt and decoding behaviour."""

    name: str

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        prompt: str,
        *,
        image: Optional[object] = None,
        image_crops: Optional[object] = None,
    ) -> Tensor:
        raise NotImplementedError

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        *,
        context: Optional[object] = None,
    ) -> "SkillState":
        raise NotImplementedError


@dataclass(slots=True)
class DecodeStep:
    """Raw token emission from the runtime decode loop."""

    token: "Token"
    position: int
    phase: str = "answer"
    hidden: Optional[Tensor] = None


@dataclass(slots=True)
class SkillFinalizeResult:
    """Final materialisation of a skill-driven request."""

    text: str
    tokens: List["Token"]
    extras: Dict[str, object] = field(default_factory=dict)


class SkillState:
    """Per-request controller that interprets decode steps for a skill."""

    def __init__(self, spec: SkillSpec, request: "GenerationRequest") -> None:
        self.spec = spec
        self.request = request
        self._tokens: List["Token"] = []

    # ------------------------------------------------------------------

    def on_prefill(self, runtime: "MoondreamRuntime") -> None:
        """Hook invoked once prefill completes."""
        return None

    def consume_step(
        self,
        runtime: "MoondreamRuntime",
        step: DecodeStep,
    ) -> None:
        raise NotImplementedError

    def finalize(
        self,
        runtime: "MoondreamRuntime",
        *,
        reason: str,
    ) -> SkillFinalizeResult:
        raise NotImplementedError

    # ------------------------------------------------------------------

    def append_token(self, token: "Token") -> None:
        self._tokens.append(token)

    @property
    def tokens(self) -> Sequence["Token"]:
        return self._tokens

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    def allowed_token_ids(self, runtime: "MoondreamRuntime") -> Optional[Sequence[int]]:
        """Optional per-skill restriction on the next sampled token ids."""
        return None

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
