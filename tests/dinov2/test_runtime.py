"""Host-only DINOv2 runtime and registry tests with injected backends."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import numpy as np
import torch
from kestrel.engine import InferenceEngine
from kestrel.models import get_spec, known_models
from kestrel.runtime import ExecutionShape
from kestrel.models.dinov2.factory import create_dinov2_runtime
from kestrel.models.dinov2.model import Dinov2Output
from kestrel.models.dinov2.runtime import (
    Dinov2ExecutableCapability,
    Dinov2Runtime,
)
from kestrel.models.dinov2.weights import (
    DEFAULT_DINOV2_MODEL,
    DEFAULT_DINOV2_REPO_ID,
    DEFAULT_DINOV2_REVISION,
)


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, image: Any) -> torch.Tensor:
        self.calls.append(image)
        return torch.full((1, 3, 224, 224), 0.25, dtype=torch.float32)


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, pixel_values: torch.Tensor) -> Dinov2Output:
        self.calls.append(pixel_values)
        hidden = torch.zeros(
            1,
            257,
            384,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        hidden = hidden + pixel_values.reshape(-1)[0]
        return Dinov2Output(hidden, hidden[:, 0, :])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.ones(1)}


class _FakeCompiled:
    def __init__(self, capability: Dinov2ExecutableCapability) -> None:
        self.capability = capability
        self.calls: list[torch.Tensor] = []
        self.shutdown_calls = 0

    def forward(self, pixel_values: torch.Tensor) -> Dinov2Output:
        self.calls.append(pixel_values)
        hidden = torch.full(
            (1, 257, 384),
            2.0,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        return Dinov2Output(hidden, hidden[:, 0, :])

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _cfg(
    model: str = DEFAULT_DINOV2_MODEL,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        model_path="/unused/model.safetensors",
        device=device,
        dtype=dtype,
        resolved_device=lambda: torch.device(device),
        resolved_dtype=lambda: dtype,
    )


def _runtime(
    *,
    compiled: _FakeCompiled | None = None,
    model: _FakeModel | None = None,
    processor: _FakeProcessor | None = None,
) -> tuple[Dinov2Runtime, _FakeModel, _FakeProcessor]:
    eager = model or _FakeModel()
    image_processor = processor or _FakeProcessor()
    runtime = Dinov2Runtime(
        _cfg(),
        compute_stream=None,
        kv_pool=object(),
        model=eager,
        processor=image_processor,
        compiled_executable=compiled,
    )
    return runtime, eager, image_processor


def _forward(runtime: Dinov2Runtime, task: str, inputs: Any) -> dict[str, torch.Tensor]:
    return runtime.forward(task, (inputs,))[0]


def test_model_specs_register_on_import() -> None:
    assert {DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID} <= set(known_models())
    for name in (DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID):
        spec = get_spec(name)
        assert spec.runtime is create_dinov2_runtime
        assert spec.repo_id is None
        assert spec.filename is None
        assert spec.checkpoint_format is None
        assert spec.tokenizer_id is None
        assert spec.needs_kv_pool is False


def test_registered_factory_owns_the_pinned_snapshot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel()
    processor = _FakeProcessor()
    calls: list[dict[str, Any]] = []

    def fake_load(source: str | Path, **kwargs: Any):
        calls.append({"source": source, **kwargs})
        return SimpleNamespace(
            model=model,
            model_config=object(),
            processor_config=object(),
        )

    monkeypatch.setattr("kestrel.models.dinov2.factory.load_dinov2", fake_load)
    monkeypatch.setattr(
        "kestrel.models.dinov2.factory.Dinov2ImageProcessor",
        lambda _config: processor,
    )
    cfg = _cfg()
    cfg.model_path = None
    runtime = get_spec(DEFAULT_DINOV2_MODEL).runtime(cfg, kv_pool=object())
    try:
        assert runtime.execution_backend == "eager"
        assert runtime.model is model
        assert runtime.processor is processor
        assert calls == [
            {
                "source": DEFAULT_DINOV2_REPO_ID,
                "revision": DEFAULT_DINOV2_REVISION,
                "device": torch.device("cpu"),
                "dtype": torch.float32,
            }
        ]
    finally:
        runtime.shutdown()


def test_default_engine_build_does_not_allocate_paged_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel()
    monkeypatch.setattr(
        "kestrel.models.dinov2.factory.load_dinov2",
        lambda *_args, **_kwargs: SimpleNamespace(
            model=model, model_config=object(), processor_config=object()
        ),
    )
    monkeypatch.setattr(
        "kestrel.models.dinov2.factory.Dinov2ImageProcessor",
        lambda _config: _FakeProcessor(),
    )
    engine = object.__new__(InferenceEngine)
    engine._runtime_cfg = _cfg()
    engine._default_model = DEFAULT_DINOV2_MODEL
    engine._compute_stream = None

    def no_pool():
        raise AssertionError("DINOv2 allocated a KV pool")

    engine._shared_kv_pool = no_pool
    runtime = engine._build_runtime(DEFAULT_DINOV2_MODEL, None)
    try:
        assert runtime.tasks() == ("embed",)
        engine._runtimes = {DEFAULT_DINOV2_MODEL: runtime}
        assert engine._tasks_for(DEFAULT_DINOV2_MODEL) == ("embed",)
    finally:
        runtime.shutdown()


def test_registered_factory_builds_the_shipped_sm90_runtime_without_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel()
    processor = _FakeProcessor()
    stream = object()
    calls: dict[str, Any] = {"loads": 0, "shutdowns": 0}

    def fake_load(source: str | Path, **kwargs: Any):
        calls["loads"] += 1
        calls["load"] = {"source": source, **kwargs}
        return SimpleNamespace(
            model=model,
            model_config="model-config",
            processor_config="processor-config",
        )

    class FakeExecutable:
        def __init__(self, config: Any, state_dict: Any, **kwargs: Any) -> None:
            calls["executable"] = {
                "config": config,
                "state_dict": state_dict,
                **kwargs,
            }
            self.capability = Dinov2ExecutableCapability(
                device=torch.device("cuda:0"),
                dtype=torch.bfloat16,
            )

        def shutdown(self) -> None:
            calls["shutdowns"] += 1

    monkeypatch.setattr(
        "kestrel.models.dinov2.factory.get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr("kestrel.models.dinov2.factory.load_dinov2", fake_load)
    monkeypatch.setattr(
        "kestrel.models.dinov2.factory.Dinov2ImageProcessor",
        lambda config: processor if config == "processor-config" else None,
    )
    monkeypatch.setattr(
        "kestrel.models.dinov2.factory.Dinov2ShippedExecutable", FakeExecutable)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr("kestrel.models.dinov2.runtime.empty_cache", lambda _device: None)

    cfg = _cfg(device="cuda:0", dtype=torch.bfloat16)
    cfg.model_path = None
    runtime = get_spec(DEFAULT_DINOV2_MODEL).runtime(
        cfg,
        compute_stream=stream,
        kv_pool=object(),
    )
    try:
        assert runtime.execution_backend == "compiled"
        assert runtime.model is None
        assert runtime.processor is processor
        assert calls["loads"] == 1
        assert calls["load"] == {
            "source": DEFAULT_DINOV2_REPO_ID,
            "revision": DEFAULT_DINOV2_REVISION,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
        assert calls["executable"]["config"] == "model-config"
        assert calls["executable"]["device"] == torch.device("cuda:0")
        assert calls["executable"]["dtype"] is torch.bfloat16
        assert calls["executable"]["state_dict"].keys() == {"weight"}
        torch.testing.assert_close(
            calls["executable"]["state_dict"]["weight"],
            torch.ones(1),
        )
    finally:
        runtime.shutdown()
    assert calls["shutdowns"] == 1


def test_eager_runtime_accepts_image_and_pixel_values() -> None:
    runtime, model, processor = _runtime()
    try:
        assert runtime.execution_shape is ExecutionShape.SINGLE_PASS
        assert runtime.model_name == DEFAULT_DINOV2_MODEL
        assert runtime.tasks() == ("embed",)
        assert runtime.execution_backend == "eager"
        assert runtime.primary_stream is None
        assert model.eval_called

        image = np.zeros((12, 9, 3), dtype=np.uint8)
        from_image = _forward(runtime, "embed", {"image": image})
        pixels = torch.full((1, 3, 224, 224), 0.5)
        from_pixels = _forward(runtime, "embed", {"pixel_values": pixels})
    finally:
        runtime.shutdown()

    assert processor.calls == [image]
    assert len(model.calls) == 2
    assert model.calls[0].dtype is torch.float32
    assert set(from_image) == {"last_hidden_state", "pooler_output"}
    assert from_image["last_hidden_state"].shape == (1, 257, 384)
    assert from_image["pooler_output"].shape == (1, 384)
    assert torch.equal(
        from_pixels["pooler_output"],
        from_pixels["last_hidden_state"][:, 0, :],
    )


def test_public_outputs_are_fp32_for_bf16_eager_execution() -> None:
    model = _FakeModel()
    runtime = Dinov2Runtime(
        _cfg(dtype=torch.bfloat16),
        model=model,
        processor=_FakeProcessor(),
    )
    try:
        output = _forward(
            runtime, "embed", {"pixel_values": torch.zeros(1, 3, 224, 224)}
        )
    finally:
        runtime.shutdown()

    assert model.calls[0].dtype is torch.bfloat16
    assert output["last_hidden_state"].dtype is torch.float32
    assert output["pooler_output"].dtype is torch.float32
    torch.testing.assert_close(
        output["pooler_output"],
        output["last_hidden_state"][:, 0, :],
        rtol=0.0,
        atol=0.0,
    )


def test_input_and_task_validation_is_strict() -> None:
    runtime, _, _ = _runtime()
    try:
        with pytest.raises(ValueError, match="does not support task"):
            _forward(runtime, "segment", {"pixel_values": torch.zeros(1, 3, 224, 224)})
        with pytest.raises(TypeError, match="mapping"):
            _forward(runtime, "embed", object())
        with pytest.raises(ValueError, match="exactly one"):
            _forward(runtime, "embed", {})
        with pytest.raises(ValueError, match="exactly one"):
            _forward(
                runtime,
                "embed",
                {
                    "image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "pixel_values": torch.zeros(1, 3, 224, 224),
                },
            )
        with pytest.raises(ValueError, match="unsupported embed inputs"):
            _forward(
                runtime,
                "embed",
                {"pixel_values": torch.zeros(1, 3, 224, 224), "normalize": True},
            )
        with pytest.raises(TypeError, match="torch.Tensor"):
            _forward(runtime, "embed", {"pixel_values": np.zeros((1, 3, 224, 224))})
        with pytest.raises(ValueError, match="must have shape"):
            _forward(runtime, "embed", {"pixel_values": torch.zeros(1, 3, 518, 518)})
        with pytest.raises(TypeError, match="floating point"):
            _forward(
                runtime,
                "embed",
                {"pixel_values": torch.zeros(1, 3, 224, 224, dtype=torch.uint8)},
            )
    finally:
        runtime.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        _forward(runtime, "embed", {"pixel_values": torch.zeros(1, 3, 224, 224)})


def test_compiled_backend_selection_uses_declared_capability() -> None:
    matching = _FakeCompiled(
        Dinov2ExecutableCapability(device=torch.device("cpu"), dtype=torch.float32)
    )
    runtime, eager, _ = _runtime(compiled=matching)
    try:
        output = _forward(
            runtime, "embed", {"pixel_values": torch.zeros(1, 3, 224, 224)}
        )
        assert runtime.execution_backend == "compiled"
        assert len(matching.calls) == 1
        assert eager.calls == []
        assert torch.all(output["pooler_output"] == 2)
    finally:
        runtime.shutdown()
        runtime.shutdown()
    assert matching.shutdown_calls == 1

    mismatched = _FakeCompiled(
        Dinov2ExecutableCapability(device=torch.device("cpu"), dtype=torch.bfloat16)
    )
    with pytest.warns(RuntimeWarning, match="serving the eager fallback"):
        fallback, eager, _ = _runtime(compiled=mismatched)
    try:
        _forward(fallback, "embed", {"pixel_values": torch.zeros(1, 3, 224, 224)})
        assert fallback.execution_backend == "eager"
        assert mismatched.calls == []
        assert len(eager.calls) == 1
    finally:
        fallback.shutdown()
    assert mismatched.shutdown_calls == 1


def test_async_preprocessing() -> None:
    runtime, _, processor = _runtime()
    image = object()
    try:
        future = runtime.preprocess_image_async(image)
        assert future.result().shape == (1, 3, 224, 224)
        assert processor.calls == [image]
    finally:
        runtime.shutdown()


def test_generic_handle_run() -> None:
    try:
        from kestrel.engine import InferenceEngine
    except AttributeError as exc:
        # Kestrel 0.4.1 imports its legacy Moondream scheduler while importing the
        # generic handle. Current kernels deliberately no longer carry that private
        # attention symbol; keep the DINOv2 runtime tests runnable in this environment
        # while leaving the handle integration to a compatible engine lane.
        if "prefix_lm_mask_730" not in str(exc):
            raise
        pytest.skip("installed Kestrel engine predates the current kernels")

    engine = object.__new__(InferenceEngine)
    engine._default_model = "ar-default"
    engine._model_ids = ["ar-default", DEFAULT_DINOV2_MODEL]
    engine._runtimes = {
        DEFAULT_DINOV2_MODEL: SimpleNamespace(
            model_name=DEFAULT_DINOV2_MODEL,
            execution_shape=ExecutionShape.SINGLE_PASS,
            tasks=lambda: ("embed",),
        )
    }
    engine._initialized = True
    engine._scheduler_error = None
    captured: dict[str, Any] = {}

    async def run(model: str, task: str, inputs: Any) -> str:
        captured.update(model=model, task=task, inputs=inputs)
        return "OK"

    engine.run = run  # type: ignore[method-assign]
    pixels = torch.zeros(1, 3, 224, 224)
    result = asyncio.run(
        engine.model(DEFAULT_DINOV2_MODEL).run(
            "embed",
            {"pixel_values": pixels},
        )
    )
    assert result == "OK"
    assert captured == {
        "model": DEFAULT_DINOV2_MODEL,
        "task": "embed",
        "inputs": {"pixel_values": pixels},
    }
