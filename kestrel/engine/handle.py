"""ModelHandle — the per-model view onto the engine.

``engine.model(id)`` returns a ``ModelHandle`` bound to one model. It is
the primary customer surface: one generic class (no per-model subclass)
exposing the capability vocabulary as typed methods plus a
``run(task, inputs)`` escape hatch and ``tasks`` / ``supports``
introspection. Capability availability is a runtime check — every method
exists on the class, but calling one a model doesn't serve raises a clear
error (the deliberate "models are data, capabilities are the typed
surface" trade: type count scales with capabilities, not models).

The handle holds no resources; it forwards to the engine, pinning the
model id so callers bind a model once instead of repeating ``model=``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np

from kestrel.runtime import ExecutionShape

if TYPE_CHECKING:
    from kestrel.engine.core import InferenceEngine
    from kestrel.engine._types import EngineResult, EngineStream


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
        """
        self._require(task)
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

        The escape hatch for single-pass models. Autoregressive models use
        the typed verbs (``query`` etc.), not ``run`` — ``engine.run``
        would reject them, so reject here with a clearer message.
        """
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

    # -- autoregressive capability vocabulary -------------------------
    #
    # Thin forwarders to the engine's typed verbs. Each gates on the model
    # serving the capability AND on it being the default AR model (the
    # verbs don't yet route by model id), then delegates.
    #
    # These re-declare the verbs' keyword defaults, so they must stay in
    # sync with InferenceEngine's — a mismatch silently changes behavior
    # for the same call through the handle (e.g. query's reasoning=True).
    # test_model_handle::test_handle_verb_defaults_match_engine guards this.

    async def query(
        self,
        image: Optional[np.ndarray | bytes] = None,
        question: Optional[str] = None,
        reasoning: bool = True,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> "EngineResult | EngineStream":
        # Parameter order/kind mirrors InferenceEngine.query exactly so a
        # positional call (img, q, reasoning, refs, ...) forwards unchanged.
        self._require_default_ar("query")
        return await self._engine.query(
            image=image,
            question=question,
            reasoning=reasoning,
            spatial_refs=spatial_refs,
            stream=stream,
            settings=settings,
        )

    async def caption(
        self,
        image: Optional[np.ndarray | bytes],
        *,
        length: str = "normal",
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> "EngineResult | EngineStream":
        self._require_default_ar("caption")
        return await self._engine.caption(
            image=image, length=length, stream=stream, settings=settings
        )

    async def detect(
        self,
        image: Optional[np.ndarray | bytes],
        object: str,
        settings: Optional[Mapping[str, object]] = None,
    ) -> "EngineResult":
        self._require_default_ar("detect")
        return await self._engine.detect(image, object, settings=settings)

    async def point(
        self,
        image: Optional[np.ndarray | bytes],
        object: str,
        settings: Optional[Mapping[str, object]] = None,
        *,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
    ) -> "EngineResult":
        self._require_default_ar("point")
        return await self._engine.point(
            image, object, settings, spatial_refs=spatial_refs
        )

    async def segment(
        self,
        image: Optional[np.ndarray | bytes],
        object: str,
        *,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
        settings: Optional[Mapping[str, object]] = None,
    ) -> "EngineResult":
        self._require_default_ar("segment")
        return await self._engine.segment(
            image, object, spatial_refs=spatial_refs, settings=settings
        )

    def __repr__(self) -> str:
        return f"ModelHandle(model_id={self._model!r}, tasks={self.tasks})"


__all__ = ["ModelHandle"]
