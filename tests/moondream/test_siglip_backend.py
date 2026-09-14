import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kestrel.models.moondream import siglip_backend
from kestrel.models.moondream.runtime import MoondreamRuntime
from kestrel.models.moondream.vision import prepare_crops_from_overlap


def _config():
    return SimpleNamespace(
        enc_n_layers=27,
        enc_dim=1152,
        enc_ff_dim=4304,
        enc_n_heads=16,
        enc_patch_size=14,
        crop_size=378,
        max_crops=12,
        in_channels=3,
    )


def _vision():
    return SimpleNamespace(blocks=[object()] * 27)


def test_non_hopper_keeps_native_without_loading_hopper_backend(monkeypatch):
    monkeypatch.setattr(
        siglip_backend,
        "_create_hopper_encoder",
        lambda **_kwargs: pytest.fail("non-Hopper must not construct the Hopper backend"),
    )

    backend = siglip_backend.create_siglip_backend(
        model_name="moondream3-preview",
        vision=_vision(),
        config=_config(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert backend is None


def test_hopper_selects_once_by_vision_contract(monkeypatch):
    calls = []
    backend = SimpleNamespace(
        crop_counts=tuple(range(2, 14)),
        crop_dtype=torch.uint8,
        encode_crops=lambda crops, tiling: crops,
        close=lambda: None,
    )

    def factory(**kwargs):
        calls.append(kwargs)
        return backend

    monkeypatch.setattr(siglip_backend, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(siglip_backend, "_create_hopper_encoder", factory)

    selected = siglip_backend.create_siglip_backend(
        model_name="any-model-with-this-siglip-contract",
        vision=_vision(),
        config=_config(),
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
    )

    assert selected is backend
    assert len(calls) == 1
    assert calls[0]["model_name"] == "any-model-with-this-siglip-contract"


def test_incomplete_hopper_family_is_closed_and_rejected(monkeypatch):
    closed = []
    backend = SimpleNamespace(
        crop_counts=tuple(range(2, 13)),
        crop_dtype=torch.uint8,
        encode_crops=lambda crops, tiling: crops,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(siglip_backend, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(
        siglip_backend, "_create_hopper_encoder", lambda **_kwargs: backend)

    with pytest.raises(RuntimeError, match="every image crop count 2..13"):
        siglip_backend.create_siglip_backend(
            model_name="moondream3-preview",
            vision=_vision(),
            config=_config(),
            device=torch.device("cuda:0"),
            dtype=torch.bfloat16,
        )

    assert closed == [True]


def test_hopper_requires_raw_uint8_backend(monkeypatch):
    closed = []
    backend = SimpleNamespace(
        crop_counts=tuple(range(2, 14)),
        crop_dtype=torch.bfloat16,
        encode_crops=lambda crops, tiling: crops,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(siglip_backend, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(
        siglip_backend, "_create_hopper_encoder", lambda **_kwargs: backend)

    with pytest.raises(RuntimeError, match="raw uint8"):
        siglip_backend.create_siglip_backend(
            model_name="moondream3-preview",
            vision=_vision(),
            config=_config(),
            device=torch.device("cuda:0"),
            dtype=torch.bfloat16,
        )

    assert closed == [True]


def test_hopper_lifecycle_does_not_recapture_native_vision():
    closed = []
    preprocessor = SimpleNamespace(shutdown=lambda *, wait: closed.append(("pool", wait)))
    backend = SimpleNamespace(close=lambda: closed.append(("backend", True)))
    runtime = MoondreamRuntime.__new__(MoondreamRuntime)
    runtime._vision_backend = backend
    runtime._image_preprocessor = preprocessor
    runtime._use_cuda_graphs = True
    runtime.graph_capture_lock = threading.RLock()
    runtime.device = torch.device("cpu")
    runtime._decode_graphs = SimpleNamespace(clear=lambda: closed.append(("decode", True)))
    runtime._ensure_cuda_graphs_ready = lambda: closed.append(("ready", True))
    runtime._capture_vision_graphs = lambda: pytest.fail(
        "Hopper backend must not recapture the deleted native tower"
    )

    runtime.rebuild_cuda_graphs()
    runtime.shutdown()

    assert closed == [
        ("decode", True),
        ("ready", True),
        ("backend", True),
        ("pool", True),
    ]


def test_hopper_backend_owns_reconstruction_and_projection():
    calls = []
    expected = torch.randn(729, 2048, dtype=torch.bfloat16)
    backend = SimpleNamespace(
        crop_dtype=torch.uint8,
        encode_crops=lambda crops, tiling: calls.append((crops.shape, crops.dtype, tiling))
        or expected,
    )
    runtime = MoondreamRuntime.__new__(MoondreamRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.bfloat16
    runtime._vision_backend = backend
    runtime.config = SimpleNamespace(vision=_config())
    overlap = {
        "crops": np.zeros((2, 378, 378, 3), dtype=np.uint8),
        "tiling": (1, 1),
    }

    actual = runtime.encode_image(None, overlap=overlap)

    assert actual is expected
    assert calls == [(torch.Size((2, 3, 378, 378)), torch.uint8, (1, 1))]


def test_unnormalized_crop_staging_preserves_raw_uint8():
    overlap = {
        "crops": np.full((2, 378, 378, 3), 127, dtype=np.uint8),
        "tiling": (1, 1),
    }

    crops, tiling = prepare_crops_from_overlap(
        overlap, torch.device("cpu"), torch.uint8, normalize=False)

    assert crops.shape == (2, 3, 378, 378)
    assert crops.dtype is torch.uint8
    assert crops.is_contiguous()
    assert crops.unique().tolist() == [127]
    assert tiling == (1, 1)
