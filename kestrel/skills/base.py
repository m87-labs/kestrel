"""The model-agnostic skill contract used by the Kestrel kernel.

Defines what a *skill* is — independent of any model: the ``SkillSpec``
behavior, per-request ``SkillState``, the ``SkillRegistry``, and the value
types they exchange with the kernel. Concrete skills live with their model.
"""


from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

if False:  # pragma: no cover - type-checking imports
    import numpy as np
    from kestrel.models.moondream.runtime import MoondreamRuntime
    from kestrel.scheduler.types import GenerationRequest
    from kestrel.models.moondream.runtime import Token


# Moondream's autoregressive serving defaults. These are AR sampling
# config (token sampling), not kernel config — they live with the AR
# skills, which are the model's per-capability units. query/caption use
# AR_DEFAULT_TEMPERATURE; detect/point/segment override to greedy (0.0).
AR_DEFAULT_TEMPERATURE = 0.2
AR_DEFAULT_TOP_P = 0.9
AR_DEFAULT_MAX_NEW_TOKENS = 768


@dataclass(frozen=True, slots=True)
class SkillSettings:
    """Sampling params shared by **autoregressive** skills.

    Temperature / top_p / max_tokens are token-sampling knobs — they apply
    to AR decoding, not to single-pass models (a segmentation forward has
    no temperature). So this is an AR-skill helper, not a universal
    contract: AR skills call :func:`parse_settings` inside their own
    ``build_request`` with their per-capability defaults. Single-pass
    capabilities read whatever their model defines from the raw payload
    and ignore this entirely.
    """

    temperature: float
    top_p: float
    max_tokens: int


@dataclass(frozen=True, slots=True)
class BuiltRequest:
    """What a skill's ``build_request`` hands back to the engine.

    Carries the assembled per-capability ``request_context`` plus the
    sampling params the skill resolved (the engine threads these into the
    scheduler). The skill owns all of it — token budget included
    (detect/point derive ``max_new_tokens`` from ``max_objects``).
    """

    request_context: object
    max_new_tokens: int
    temperature: float
    top_p: float
    # Media the skill extracted from its own prompt (e.g. an image carried
    # inside OpenAI chat messages). When set, the engine sends this through
    # the image pipeline instead of the top-level ``image`` argument; ``None``
    # leaves any caller-supplied ``image`` in force.
    image: "Optional[np.ndarray | bytes]" = None


def parse_settings(
    settings: Optional[Mapping[str, object]],
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> SkillSettings:
    """Extract + validate AR sampling params from a raw settings map.

    Helper for autoregressive skills. The skill supplies its own
    per-capability defaults (e.g. greedy ``temperature=0.0`` for
    detect/point/segment, ``0.2`` for query/caption); this applies any
    overrides from ``settings`` and validates the AR sampling envelope.
    """
    if settings is not None:
        if "temperature" in settings:
            temperature = float(settings["temperature"])  # type: ignore[arg-type]
        if "top_p" in settings:
            top_p = float(settings["top_p"])  # type: ignore[arg-type]
        if "max_tokens" in settings:
            max_tokens = int(settings["max_tokens"])  # type: ignore[arg-type]
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in the range (0, 1]")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return SkillSettings(temperature=temperature, top_p=top_p, max_tokens=max_tokens)


@dataclass(frozen=True)
class SkillSpec:
    """Declarative description of a skill's prompt and decoding behaviour.

    The skill is the model's implementation of one capability. It owns
    that capability's *input contract*: ``build_request`` validates the
    raw call inputs and assembles the request object the scheduler runs.
    The kernel stays model-agnostic — it never builds or imports a
    model's request types; it just calls ``build_request`` and forwards
    the result.
    """

    name: str

    def build_request(
        self,
        image: "Optional[np.ndarray | bytes]",
        prompt: "Mapping[str, object]",
        settings: "Optional[Mapping[str, object]]",
    ) -> "BuiltRequest":
        """Validate raw inputs and build this capability's request.

        Both ``prompt`` and ``settings`` are raw, model-defined maps — the
        seam carries no model assumptions. ``prompt`` is the per-capability
        payload (e.g. ``{"object": ...}`` for detect/point/segment,
        ``{"question": ..., "reasoning": ...}`` for query, ``{"length":
        ...}`` for caption). AR skills parse ``settings`` with
        :func:`parse_settings`; a single-pass capability reads whatever its
        model defines. Returns a :class:`BuiltRequest` (the request_context
        plus resolved sampling params). Raises ``ValueError`` on invalid
        input.
        """
        raise NotImplementedError

    def prompt_text(self, request_context: object) -> str:
        """A human-readable label for this request (logs/metrics).

        Defaults to empty; skills override to surface the salient input
        (the question, the object name, …). Not behavior-bearing — the
        kernel only stores it on the request as a label.
        """
        return ""

    def build_prompt_tokens(
        self,
        runtime: "MoondreamRuntime",
        request_context: object,
    ) -> Sequence["Token"]:
        raise NotImplementedError

    def create_state(
        self,
        runtime: "MoondreamRuntime",
        request: "GenerationRequest",
        request_context: object,
    ) -> "SkillState":
        raise NotImplementedError


@dataclass(slots=True)
class DecodeStep:
    """Raw token emission from the runtime decode loop."""

    token: "Token"
    position: int
    phase: str = "answer"


@dataclass(slots=True)
class SkillFinalizeResult:
    """Final materialisation of a skill-driven request."""

    text: str
    tokens: List["Token"]
    output: Dict[str, object] = field(default_factory=dict)


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

    def suppressed_token_ids(self, runtime: "MoondreamRuntime") -> Optional[Sequence[int]]:
        """Optional per-skill token ids to suppress (set logits to -inf).

        Complement of allowed_token_ids: these tokens are forced to -inf
        rather than being the only ones kept.
        """
        return None

    @property
    def emits_spatial_tokens(self) -> bool:
        """Whether this skill can emit/consume a coord/size token *right now*.

        Drives the runtime's ``post_sample`` spatial-head gate: the coord/size
        decode head (a ~12.6MB matvec + several sample launches per step) only
        produces values the skill actually consumes when the sampled id can be
        ``coord_id`` / ``size_id``. Text-only phases (a plain query answer, a
        caption) never consume those values, so the head is pure waste there and
        the runtime skips it. This mirrors "the mode the masks already encode":
        the same grammar state that gates the coord/size masks gates the compute.

        Default ``False`` (text-only). Spatial skills (point/detect/segment) and
        the grounded-reasoning phase of a query override this. The property is
        host-side (no GPU sync) and phase-aware where the skill's grammar is
        stateful, so gating on it is byte-identical for every skill that can
        consume a coord/size value.
        """
        return False

    def stop_token_ids(self, runtime: "MoondreamRuntime") -> Optional[Sequence[int]]:
        """Optional per-skill token ids that end generation.

        The scheduler stops a sequence when its last token matches the
        model's ``eos_id``; a skill returns extra ids here to also stop on a
        capability-specific terminator. The chat skill uses this for a
        turn-end token (e.g. a ChatML-style ``<|im_end|>``) that differs
        from the model's ``eos_id``.
        """
        return None

    # Streaming -------------------------------------------------------

    def pop_stream_delta(self, runtime: "MoondreamRuntime") -> Optional[str]:
        """Return newly available human-readable text for streaming clients."""

        return None

class SkillRegistry:
    """Maps a model's capability names to their skills.

    May be empty: a model with no autoregressive skills (e.g. a single-pass
    model) registers none and advertises its tasks via the runtime instead.
    """

    def __init__(self, skills: Iterable[SkillSpec]) -> None:
        self._skills: Dict[str, SkillSpec] = {}
        for spec in skills:
            if spec.name in self._skills:
                raise ValueError(f"Duplicate skill registered: {spec.name}")
            self._skills[spec.name] = spec

    def names(self) -> tuple[str, ...]:
        """Registered skill names, in registration order."""
        return tuple(self._skills)

    def resolve(self, skill: str) -> SkillSpec:
        try:
            return self._skills[skill]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown skill '{skill}'") from exc
