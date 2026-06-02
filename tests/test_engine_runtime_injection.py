"""Verify ``InferenceEngine`` accepts an externally-built runtime.

The ``runtime=`` seam lets a caller (today: tests) build a runtime and
hand it to the engine instead of having the engine construct one via the
model registry. The engine *adopts* an injected runtime: it registers it
under the config's model id and owns its lifecycle, tearing it down on
``shutdown`` like any engine-built runtime. There is no caller-owned
shared-runtime contract — nothing in the codebase shares a runtime across
engines.
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


def test_engine_registers_injected_runtime_under_config_model_id() -> None:
    """The engine is a registry keyed by model id (data from the config),
    not by interrogating the runtime object. An injected runtime lands
    under ``runtime_cfg.model`` and is reachable as the default."""

    runtime = FakeRuntime(model_name="anything", device="cpu")
    engine = InferenceEngine(_cpu_cfg(), runtime=runtime)
    # Keyed by the config's model id ("moondream2"), regardless of what
    # the runtime reports as its own model_name.
    assert engine._default_model == "moondream2"
    assert engine._runtimes == {"moondream2": runtime}
    assert engine.runtime is runtime
