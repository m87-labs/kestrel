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

import asyncio

import pytest

import kestrel.engine.core as engine_core
from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.models.registry import ModelSpec, register, _REGISTRY

from tests.scheduler._fake_runtime import FakeRuntime


def _cpu_cfg() -> RuntimeConfig:
    return RuntimeConfig(device="cpu", model="moondream2")


def test_engine_holds_injected_runtime() -> None:
    """An injected runtime is exposed via ``engine.runtime`` immediately,
    without having to ``await create``."""

    runtime = FakeRuntime(model_name="injected", device="cpu")
    stream = object()
    runtime.compute_stream = stream
    engine = InferenceEngine(_cpu_cfg(), runtime=runtime)
    assert engine.runtime is runtime
    assert engine._compute_stream is stream
    assert engine._kv_pool is runtime.kv_pool


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


def test_engine_cohosted_runtime_reuses_injected_runtime_pool() -> None:
    """An injected default runtime supplies the shared KV pool for later builds."""

    seen: dict[str, object] = {}

    def factory(cfg: RuntimeConfig, **kwargs):
        seen[cfg.model] = kwargs["kv_pool"]
        return FakeRuntime(model_name=cfg.model, device="cpu")

    register(ModelSpec(name="cohosted-ar", runtime=factory))
    try:
        runtime = FakeRuntime(model_name="anything", device="cpu")
        engine = InferenceEngine(_cpu_cfg(), runtime=runtime, models=["cohosted-ar"])
        engine._build_configured_runtimes(max_lora_rank=None)

        assert engine._runtimes["moondream2"] is runtime
        assert seen["cohosted-ar"] is runtime.kv_pool
        assert engine._kv_pool is runtime.kv_pool
    finally:
        _REGISTRY.pop("cohosted-ar", None)


def test_engine_surfaces_scheduler_startup_failure(monkeypatch) -> None:
    class BrokenExecutor:
        def __init__(self, *args, **kwargs) -> None:
            raise TypeError("scheduler constructor mismatch")

    monkeypatch.setattr(engine_core, "AutoregressiveExecutor", BrokenExecutor)
    runtime = FakeRuntime(model_name="anything", device="cpu")
    engine = InferenceEngine(_cpu_cfg(), runtime=runtime)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="scheduler is not running") as exc:
            await engine._initialize()
        assert isinstance(exc.value.__cause__, TypeError)
        assert "constructor mismatch" in str(exc.value.__cause__)

    asyncio.run(run())
