"""ModelHandle — the per-model view onto the engine.

``engine.model(id)`` returns a ``ModelHandle`` bound to one model. It is
the primary customer surface: one generic class (no per-model subclass)
exposing the capability vocabulary as uniform ``verb(**prompt)`` methods
plus a ``run(task, inputs)`` escape hatch and ``tasks`` / ``supports``
introspection. The method names are the typed surface; their inputs are
entirely model-defined — including the media payload (``image=`` for a
vision model, ``audio=``/``video=``/``text=`` for others), which the handle
does not privilege, so the surface is modality-agnostic. The bound model
validates the prompt. Capability availability is a runtime check — every
method exists on the class, but calling one a model doesn't serve raises a
clear error (the deliberate "models are data, capabilities are the typed
surface" trade: type count scales with capabilities, not models).

The handle holds no resources; it forwards to the engine, pinning the
model id so callers bind a model once instead of repeating ``model=``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kestrel.runtime import ExecutionShape

if TYPE_CHECKING:
    from kestrel.engine.core import InferenceEngine
    from kestrel.engine._types import EngineResult, EngineStream, ModelStream


class ModelHandle:
    """A capability-bearing view onto one registered model."""

    def __init__(self, engine: "InferenceEngine", model_id: str) -> None:
        self._engine = engine
        self._model = model_id

    @property
    def model_id(self) -> str:
        return self._model

    # -- introspection ------------------------------------------------

    @property
    def tasks(self) -> tuple[str, ...]:
        """Capability names this model serves."""
        return self._engine._tasks_for(self._model)

    def supports(self, task: str) -> bool:
        return task in self.tasks

    def _require(self, task: str) -> None:
        if task not in self.tasks:
            raise ValueError(
                f"Model {self._model!r} does not support {task!r} "
                f"(supports: {', '.join(self.tasks) or 'none'})"
            )

    def _require_default_ar(self, task: str) -> None:
        """Guard the autoregressive verbs.

        The engine's typed verbs (query/caption/...) currently route only
        to the default autoregressive model — they take no model id. So a
        handle for any *other* model must not silently misroute through
        them. Until per-model AR routing exists, fail loud instead.

        Only the default-model gate is checked here (it needs no started
        engine); the engine verb + the model's skill validate that the
        model actually serves ``task``.
        """
        if self._model != self._engine._default_model:
            raise NotImplementedError(
                f"{task!r} on {self._model!r} is not yet routable: the "
                "autoregressive verbs target the default model "
                f"({self._engine._default_model!r}). Per-model AR routing "
                "is a future step."
            )

    # -- generic escape hatch -----------------------------------------

    async def run(self, task: str, inputs: Any) -> "EngineResult":
        """Run an arbitrary ``task`` on this model's single-pass lane.

        The escape hatch for single-pass models, and what the capability
        verbs dispatch to for that shape. Autoregressive models go through
        the skill path instead — ``engine.run`` would reject them, so reject
        here with a clearer message.

        Starts the engine first: a co-hosted model's runtime (and thus its
        tasks/shape) only exists once built, so the task check below must run
        against a started engine, not raise "unknown model" pre-start.
        """
        await self._engine._ensure_started()
        self._require(task)
        runtime = self._engine._runtimes.get(self._model)
        if runtime is not None and runtime.execution_shape is not (
            ExecutionShape.SINGLE_PASS
        ):
            raise ValueError(
                f"run() is for single-pass models; {self._model!r} is "
                f"{runtime.execution_shape.value} — use its typed verbs."
            )
        return await self._engine.run(self._model, task, inputs)

    async def stream(self, task: str, /, **initial_prompt: Any) -> "ModelStream":
        """Start a stateful streaming session on this model.

        This is the generic escape hatch for execution shapes where the
        caller drives model state forward with chunks/frames. It is
        separate from autoregressive token streaming, which remains a
        per-request delivery mode selected by capability prompts such as
        ``stream=True``.
        """
        await self._engine._ensure_started()
        self._require(task)
        runtime = self._engine._runtimes.get(self._model)
        if runtime is None:
            raise ValueError(f"Unknown model {self._model!r}")
        if runtime.execution_shape is not ExecutionShape.STREAMING:
            raise ValueError(
                f"stream() is for streaming models; {self._model!r} is "
                f"{runtime.execution_shape.value}"
            )
        return await self._engine.stream(self._model, task, initial_prompt)

    # -- capability vocabulary ----------------------------------------
    #
    # One task is one method; its inputs are model-defined. Every verb is
    # therefore uniform — ``verb(**prompt)`` — and the whole prompt, media
    # payload included (``image=``/``audio=``/...), is interpreted and
    # validated by the bound model: its skill for an autoregressive model,
    # its runtime for a single-pass one. The handle stays model-agnostic and
    # only forwards. The method *names* are the typed surface (autocomplete +
    # ``tasks``/``supports``); the inputs are not, by design — a model serving
    # the same capability may accept different inputs (and different
    # modalities), so a fixed signature couldn't capture them.

    async def _capability(
        self,
        task: str,
        prompt: dict[str, Any],
    ) -> "EngineResult | EngineStream | ModelStream":
        """Route one capability call by the bound model's execution shape.

        A single-pass model interprets the whole prompt in its forward pass
        (via ``run``); an autoregressive model validates and builds it
        through the model's skill. For the AR path, the engine-level concerns
        are separated from the model prompt: ``settings`` (sampling) is lifted
        out as its own argument, and ``image`` is pulled out for the image
        pipeline; ``stream`` selects streaming delivery but stays in the
        prompt, since the skill reads it to configure its streaming state.

        Starts the engine first so shape dispatch sees a built runtime: a
        co-hosted single-pass model isn't in ``_runtimes`` until startup, and
        without it this would misroute to the autoregressive path.
        """
        await self._engine._ensure_started()
        runtime = self._engine._runtimes.get(self._model)
        if runtime is not None and (
            runtime.execution_shape is ExecutionShape.SINGLE_PASS
        ):
            return await self.run(task, prompt)
        if runtime is not None and runtime.execution_shape is ExecutionShape.STREAMING:
            return await self.stream(task, **prompt)
        self._require_default_ar(task)
        settings = prompt.pop("settings", None)
        stream = bool(prompt.get("stream", False))
        image = prompt.pop("image", None)
        return await self._engine._run_skill(
            task, image=image, prompt=prompt, settings=settings, stream=stream
        )

    async def query(
        self, **prompt: Any
    ) -> "EngineResult | EngineStream | ModelStream":
        return await self._capability("query", prompt)

    async def caption(
        self, **prompt: Any
    ) -> "EngineResult | EngineStream | ModelStream":
        return await self._capability("caption", prompt)

    async def detect(
        self, **prompt: Any
    ) -> "EngineResult | EngineStream | ModelStream":
        return await self._capability("detect", prompt)

    async def point(
        self, **prompt: Any
    ) -> "EngineResult | EngineStream | ModelStream":
        return await self._capability("point", prompt)

    async def segment(
        self, **prompt: Any
    ) -> "EngineResult | EngineStream | ModelStream":
        return await self._capability("segment", prompt)

    def __repr__(self) -> str:
        return f"ModelHandle(model_id={self._model!r}, tasks={self.tasks})"


__all__ = ["ModelHandle"]
