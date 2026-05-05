"""Verify ``InferenceEngine`` accepts an externally-built runtime.

Lets two engines (e.g., different model variants) coexist behind one
``KVMemoryPool``: the caller builds each runtime with the shared pool,
then hands the runtime to its engine instead of letting the engine
construct ``MoondreamRuntime`` internally.
"""

from __future__ import annotations

import pytest

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine

from tests.scheduler._fake_runtime import FakeRuntime


def _cpu_cfg() -> RuntimeConfig:
    return RuntimeConfig(device="cpu", model="moondream2")


def test_engine_holds_injected_runtime() -> None:
    """An injected runtime is exposed via ``engine.runtime`` immediately,
    without having to ``await create``."""

    runtime = FakeRuntime(model_name="injected", device="cpu")
    engine = InferenceEngine(_cpu_cfg(), runtime=runtime)
    assert engine.runtime is runtime


def test_engine_rejects_runtime_with_mismatched_device() -> None:
    """Catch wrong-pool wiring at construction rather than at the first
    GPU call."""

    runtime = FakeRuntime(model_name="injected", device="meta")
    with pytest.raises(ValueError, match="does not match"):
        InferenceEngine(_cpu_cfg(), runtime=runtime)


def test_engine_without_runtime_still_unbuilt_at_construction() -> None:
    """Without the kwarg the engine still defers MoondreamRuntime
    construction to ``_initialize`` (current behaviour)."""

    engine = InferenceEngine(_cpu_cfg())
    with pytest.raises(RuntimeError, match="not been started"):
        _ = engine.runtime
