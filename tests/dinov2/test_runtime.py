"""Serving contracts for the shipped DINOv2 executable."""

import gc
import sys
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kestrel.engine import InferenceEngine
from kestrel.models import get_spec, known_models
from kestrel.models.dinov2.factory import create_dinov2_runtime
from kestrel.models.dinov2.metadata import DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID
from kestrel.models.dinov2.runtime import Dinov2Runtime
from kestrel.runtime import ExecutionShape


def _cfg(device="cpu", dtype=torch.bfloat16):
    return SimpleNamespace(
        model=DEFAULT_DINOV2_MODEL, model_path=None,
        resolved_device=lambda: torch.device(device), resolved_dtype=lambda: dtype,
    )


class _Executable:
    def __init__(self):
        self.calls = []
        self.close_calls = 0

    def forward(self, pixels):
        self.calls.append(pixels)
        return torch.full((1, 257, 384), 2.0, dtype=torch.float32)

    def close(self):
        self.close_calls += 1


def _processor(image):
    return torch.full((1, 3, 224, 224), .25)


def _runtime(executable=None, processor=_processor):
    return Dinov2Runtime(_cfg(), executable=executable or _Executable(), processor=processor)


def _factory_stubs(monkeypatch, constructor=None):
    state = {"weight": torch.ones(1)}
    calls = []

    def load(source):
        calls.append(source)
        return SimpleNamespace(state_dict=state, model_config="model-config", processor_config="processor-config")

    def create(**kwargs):
        calls.append(kwargs)
        return _Executable()

    monkeypatch.setattr("kestrel.models.dinov2.factory.load_dinov2", load)
    monkeypatch.setattr("kestrel.models.dinov2.factory.Dinov2ImageProcessor", lambda cfg: _processor)
    monkeypatch.setattr("kestrel.models.dinov2.factory.get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr("kestrel.models.dinov2.runtime.empty_cache", lambda device: None)
    monkeypatch.setitem(sys.modules, "kestrel_kernels.megakernel.dinov2",
                        SimpleNamespace(Dinov2MegakernelEncoder=constructor or create))
    return state, calls


def test_specs_register_the_shipped_factory():
    assert {DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID} <= set(known_models())
    for name in (DEFAULT_DINOV2_MODEL, DEFAULT_DINOV2_REPO_ID):
        spec = get_spec(name)
        assert spec.runtime is create_dinov2_runtime
        assert not spec.needs_kv_pool
        assert spec.tokenizer_id is None


@pytest.mark.parametrize("device,dtype,capability", [
    ("cpu", torch.bfloat16, None),
    ("mps", torch.float16, None),
    ("cuda:0", torch.float32, (9, 0)),
    ("cuda:0", torch.float16, (9, 0)),
    ("cuda:0", torch.bfloat16, (8, 0)),
    ("cuda:0", torch.bfloat16, (10, 0)),
    ("cuda:0", torch.bfloat16, (12, 0)),
])
def test_unsupported_targets_fail_before_loading_weights(monkeypatch, device, dtype, capability):
    def no_load(*args, **kwargs):
        raise AssertionError("unsupported target loaded a checkpoint")

    monkeypatch.setattr("kestrel.models.dinov2.factory.load_dinov2", no_load)
    monkeypatch.setattr("kestrel.models.dinov2.factory.get_device_capability", lambda device: capability)
    with pytest.raises(ValueError, match="requires Hopper CUDA and BF16"):
        create_dinov2_runtime(_cfg(device, dtype))


@pytest.mark.parametrize("source", [None, "/models/checkpoint"])
def test_factory_passes_raw_checkpoint_tensors_to_the_encoder(monkeypatch, source):
    state, calls = _factory_stubs(monkeypatch)
    cfg = _cfg("cuda:0")
    cfg.model_path = source
    runtime = create_dinov2_runtime(cfg, compute_stream=object())
    assert calls[0] == (source or DEFAULT_DINOV2_REPO_ID)
    assert calls[1]["state_dict"] is state
    assert calls[1]["config"] == "model-config"
    assert calls[1]["device"] == torch.device("cuda:0")
    assert calls[1]["dtype"] is torch.bfloat16
    executable = runtime.executable
    runtime.shutdown()
    runtime.shutdown()
    assert executable.close_calls == 1
    assert runtime.executable is None


def test_missing_artifact_is_a_startup_failure(monkeypatch):
    def missing(**kwargs):
        raise RuntimeError("no shipped DINOv2 grid")

    _factory_stubs(monkeypatch, missing)
    with pytest.raises(RuntimeError, match="no shipped DINOv2 grid"):
        create_dinov2_runtime(_cfg("cuda:0"))


def test_engine_build_does_not_allocate_paged_kv(monkeypatch):
    _factory_stubs(monkeypatch)
    engine = object.__new__(InferenceEngine)
    engine._runtime_cfg = _cfg("cuda:0")
    engine._default_model = DEFAULT_DINOV2_MODEL
    engine._compute_stream = None
    engine._shared_kv_pool = lambda: pytest.fail("DINOv2 allocated paged KV")
    runtime = engine._build_runtime(DEFAULT_DINOV2_MODEL, None)
    assert runtime.tasks() == ("embed",)
    runtime.shutdown()


def test_image_and_pixels_return_the_same_owned_output_contract():
    executable = _Executable()
    runtime = _runtime(executable)
    assert runtime.execution_shape is ExecutionShape.SINGLE_PASS
    for request in ({"image": np.zeros((8, 9, 3), dtype=np.uint8)},
                    {"pixel_values": torch.zeros(1, 3, 224, 224)}):
        result = runtime.forward("embed", (request,))[0]
        assert set(result) == {"last_hidden_state", "pooler_output"}
        assert result["last_hidden_state"].shape == (1, 257, 384)
        assert result["pooler_output"].shape == (1, 384)
        assert result["pooler_output"].dtype is torch.float32
        assert result["pooler_output"].data_ptr() == result["last_hidden_state"].data_ptr()
    assert all(pixels.dtype is torch.bfloat16 for pixels in executable.calls)
    runtime.shutdown()


@pytest.mark.parametrize("inputs,error", [
    ({}, "exactly one"),
    ({"image": None}, "must not be None"),
    ({"image": object(), "pixel_values": torch.empty(0)}, "exactly one"),
    ({"pixel_values": torch.zeros(1, 3, 224, 224), "normalize": True}, "unsupported"),
    ({"pixel_values": torch.zeros(1, 3, 518, 518)}, "shape"),
    ({"pixel_values": torch.zeros(1, 3, 224, 224, dtype=torch.uint8)}, "floating point"),
    ({"pixel_values": np.zeros((1, 3, 224, 224))}, "torch.Tensor"),
    (object(), "mapping"),
])
def test_invalid_inputs_do_not_launch(inputs, error):
    executable = _Executable()
    runtime = _runtime(executable)
    with pytest.raises((ValueError, TypeError), match=error):
        runtime.forward("embed", (inputs,))
    assert not executable.calls
    runtime.shutdown()


def test_cpu_pixels_convert_before_transfer_and_reject_other_device():
    runtime = _runtime()
    pixels = torch.randn(1, 3, 224, 224).transpose(2, 3)
    prepared = runtime._pixel_values({"pixel_values": pixels})
    assert prepared.dtype is torch.bfloat16 and prepared.is_contiguous()
    torch.testing.assert_close(prepared, pixels.bfloat16())
    with pytest.raises(ValueError, match="model device/dtype"):
        runtime._pixel_values({"pixel_values": torch.empty(1, 3, 224, 224, device="meta")})
    runtime.shutdown()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 3.4e38, 1e100])
def test_nonfinite_or_overflowing_pixels_do_not_launch(value):
    executable = _Executable()
    runtime = _runtime(executable)
    with pytest.raises(ValueError, match="finite"):
        runtime.forward("embed", ({"pixel_values": torch.full((1, 3, 224, 224), value, dtype=torch.float64)},))
    assert not executable.calls
    runtime.shutdown()


@pytest.mark.parametrize("output", ["invalid", torch.zeros(1, 384),
                                    torch.zeros(1, 257, 384, dtype=torch.bfloat16)])
def test_malformed_executable_outputs_are_rejected(output):
    executable = _Executable()
    executable.forward = lambda pixels: output
    runtime = _runtime(executable)
    with pytest.raises((TypeError, RuntimeError)):
        runtime.forward("embed", ({"pixel_values": torch.zeros(1, 3, 224, 224)},))
    runtime.shutdown()


def test_task_batch_and_shutdown_contracts():
    runtime = _runtime()
    with pytest.raises(ValueError, match="does not support task"):
        runtime.forward("detect", ({},))
    with pytest.raises(ValueError, match="one embed request"):
        runtime.forward("embed", ({}, {}))
    owner = weakref.ref(runtime.executable)
    runtime.shutdown()
    gc.collect()
    assert owner() is None
    with pytest.raises(RuntimeError, match="shut down"):
        runtime.forward("embed", ({},))


def test_preprocessing_future_propagates_errors():
    def invalid(image):
        raise ValueError("bad image")

    runtime = _runtime(processor=invalid)
    with pytest.raises(ValueError, match="bad image"):
        runtime.preprocess_image_async(object()).result()
    runtime.shutdown()
