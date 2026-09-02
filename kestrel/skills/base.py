"""The model-agnostic skill contract used by the Kestrel kernel.

Defines what a *skill* is — independent of any model: the ``SkillSpec``
behavior, per-request ``SkillState``, the ``SkillRegistry``, and the value
types they exchange with the kernel. Concrete skills live with their model.
"""


from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

if False:  # pragma: no cover - type-checking imports
    import numpy as np
    from kestrel.runtime import AutoregressiveRuntime
    from kestrel.runtime.tokens import Token
    from kestrel.scheduler.types import GenerationRequest


# Default autoregressive sampling settings. Individual model skills may
# override them when their prompt or training contract requires it.
AR_DEFAULT_TEMPERATURE = 0.2
AR_DEFAULT_TOP_P = 0.9
AR_DEFAULT_MAX_NEW_TOKENS = 768


class CapabilityInvoker(Protocol):
    """Submit one ordinary leaf request for the orchestrated capability."""

    async def __call__(
        self,
        prompt: Mapping[str, object],
        *,
        image: "Optional[np.ndarray | bytes]" = None,
        settings: Optional[Mapping[str, object]] = None,
    ) -> object: ...


class CapabilityOrchestrator(Protocol):
    """Model-owned composition of ordinary requests for one capability.

    The orchestrator never enters the scheduler. It receives a narrowly scoped
    ``invoke`` callback that submits one ordinary request for the same skill,
    so windowing, retries, and result aggregation can remain model policy while
    every leaf request keeps the normal admission, batching, and lifecycle.
    """

    async def run(
        self,
        invoke: CapabilityInvoker,
        *,
        image: "Optional[np.ndarray | bytes]",
        prompt: Mapping[str, object],
        settings: Optional[Mapping[str, object]],
    ) -> object: ...


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
    # Optional encoder-side payload. Unlike ``image``, it is not expanded into
    # decoder-prefix positions and never contributes to decoder KV length. The
    # runtime owns its asynchronous preprocessing and per-sequence device state.
    encoder_input: object | None = None
    # A skill can require selected-token log probabilities for its own final
    # result semantics (for example, transcript confidence). The engine enables
    # the existing scheduler logprob path even when the caller did not request
    # the private diagnostic setting.
    capture_logprobs: bool = False


@dataclass(frozen=True, slots=True)
class PreparedSkillPrompt:
    """Runtime-resolved decoder prompt and exact generated-token budget.

    Most skills return their ordinary prompt tokens and unchanged budget via
    :meth:`SkillSpec.prepare_prompt`. A skill whose prompt depends on the
    runtime tokenizer may override that hook and return an updated immutable
    request context alongside the exact budget required after tokenization.
    """

    request_context: object
    tokens: Sequence["Token"]
    max_new_tokens: int


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

    def orchestrator(self) -> Optional[CapabilityOrchestrator]:
        """Return an optional outer request orchestrator for this capability.

        Most skills are one scheduler request and inherit ``None``. A model
        whose customer operation is a dependent sequence of ordinary requests
        may return an orchestrator without teaching the engine or scheduler its
        domain-specific policy.
        """

        return None

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
        runtime: "AutoregressiveRuntime",
        request_context: object,
    ) -> Sequence["Token"]:
        raise NotImplementedError

    def prepare_prompt(
        self,
        runtime: "AutoregressiveRuntime",
        request_context: object,
        max_new_tokens: int,
    ) -> PreparedSkillPrompt:
        """Resolve runtime-tokenizer-dependent prompt state before admission."""

        return PreparedSkillPrompt(
            request_context=request_context,
            tokens=tuple(self.build_prompt_tokens(runtime, request_context)),
            max_new_tokens=max_new_tokens,
        )

    def create_state(
        self,
        runtime: "AutoregressiveRuntime",
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
    logprob: float | None = None


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

    def on_prefill(self, runtime: "AutoregressiveRuntime") -> None:
        """Hook invoked once prefill completes."""
        return None

    def consume_step(
        self,
        runtime: "AutoregressiveRuntime",
        step: DecodeStep,
    ) -> None:
        raise NotImplementedError

    def finalize(
        self,
        runtime: "AutoregressiveRuntime",
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

    def allowed_token_ids(self, runtime: "AutoregressiveRuntime") -> Optional[Sequence[int]]:
        """Optional per-skill restriction on the next sampled token ids."""
        return None

    def suppressed_token_ids(self, runtime: "AutoregressiveRuntime") -> Optional[Sequence[int]]:
        """Optional per-skill token ids to suppress (set logits to -inf).

        Complement of allowed_token_ids: these tokens are forced to -inf
        rather than being the only ones kept.
        """
        return None

    def stop_token_ids(self, runtime: "AutoregressiveRuntime") -> Optional[Sequence[int]]:
        """Optional per-skill token ids that end generation.

        The scheduler stops a sequence when its last token matches the
        model's ``eos_id``; a skill returns extra ids here to also stop on a
        capability-specific terminator. The chat skill uses this for a
        turn-end token (e.g. a ChatML-style ``<|im_end|>``) that differs
        from the model's ``eos_id``.
        """
        return None

    # Streaming -------------------------------------------------------

    def pop_stream_delta(self, runtime: "AutoregressiveRuntime") -> Optional[str]:
        """Return newly available human-readable text for streaming clients."""

        return None

    def pop_reasoning_stream_delta(
        self, runtime: "AutoregressiveRuntime"
    ) -> Optional[str]:
        """Return newly available reasoning text for streaming clients."""

        return None

    def pop_stream_output(
        self,
        runtime: "AutoregressiveRuntime",
    ) -> Optional[Mapping[str, object]]:
        """Return one append-only streaming update.

        Text skills inherit the existing ``pop_stream_delta`` behavior. A
        capability that streams another payload, such as synthesized PCM, can
        override this hook without teaching the scheduler about that modality.
        """

        text = self.pop_stream_delta(runtime)
        reasoning = self.pop_reasoning_stream_delta(runtime)
        output = {}
        if text:
            output["text"] = text
        if reasoning:
            output["reasoning"] = reasoning
        return output or None


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
