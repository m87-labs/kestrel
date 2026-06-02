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

    # -- generic escape hatch -----------------------------------------

    async def run(self, task: str, inputs: Any) -> "EngineResult":
        """Run an arbitrary ``task`` on this model (single-pass lane)."""
        self._require(task)
        return await self._engine.run(self._model, task, inputs)

    # -- autoregressive capability vocabulary -------------------------
    #
    # Thin, model-pinned forwarders to the engine's typed verbs. Each
    # gates on the model actually serving the capability, then delegates.

    async def query(
        self,
        image: Optional[np.ndarray | bytes],
        question: str,
        *,
        reasoning: bool = False,
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
    ) -> "EngineResult | EngineStream":
        self._require("query")
        return await self._engine.query(
            image=image,
            question=question,
            reasoning=reasoning,
            stream=stream,
            settings=settings,
            spatial_refs=spatial_refs,
        )

    async def caption(
        self,
        image: Optional[np.ndarray | bytes],
        *,
        length: str = "normal",
        stream: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> "EngineResult | EngineStream":
        self._require("caption")
        return await self._engine.caption(
            image=image, length=length, stream=stream, settings=settings
        )

    async def detect(
        self,
        image: Optional[np.ndarray | bytes],
        object: str,
        *,
        settings: Optional[Mapping[str, object]] = None,
    ) -> "EngineResult":
        self._require("detect")
        return await self._engine.detect(image, object, settings=settings)

    async def point(
        self,
        image: Optional[np.ndarray | bytes],
        object: str,
        settings: Optional[Mapping[str, object]] = None,
        *,
        spatial_refs: Optional[Sequence[Sequence[float]]] = None,
    ) -> "EngineResult":
        self._require("point")
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
        self._require("segment")
        return await self._engine.segment(
            image, object, spatial_refs=spatial_refs, settings=settings
        )

    def __repr__(self) -> str:
        return f"ModelHandle(model_id={self._model!r}, tasks={self.tasks})"


__all__ = ["ModelHandle"]
