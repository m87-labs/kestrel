import threading
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.moondream import siglip_backend
from kestrel.models.moondream.runtime import MoondreamRuntime


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


class _Point:
    name = "hopper"

    def __init__(self, factory):
        self._factory = factory

    def load(self):
        return self._factory


def test_non_hopper_keeps_native_without_loading_plugins(monkeypatch):
    monkeypatch.setattr(
        siglip_backend,
        "entry_points",
        lambda **_kwargs: pytest.fail("non-Hopper must not discover Hopper plugins"),
    )

    backend = siglip_backend.create_siglip_backend(
        model_name="moondream3-preview",
        vision=_vision(),
        config=_config(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert backend is None


def test_hopper_requires_one_complete_all_count_backend(monkeypatch):
    monkeypatch.setattr(siglip_backend, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(siglip_backend, "entry_points", lambda **_kwargs: ())

    with pytest.raises(RuntimeError, match="exactly one installed"):
        siglip_backend.create_siglip_backend(
            model_name="moondream3-preview",
            vision=_vision(),
            config=_config(),
            device=torch.device("cuda:0"),
            dtype=torch.bfloat16,
        )


def test_hopper_selects_once_by_vision_contract(monkeypatch):
    calls = []
    backend = SimpleNamespace(
        crop_counts=tuple(range(1, 14)),
        encode_crops=lambda crops: crops,
        close=lambda: None,
    )

    def factory(**kwargs):
        calls.append(kwargs)
        return backend

    monkeypatch.setattr(siglip_backend, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(
        siglip_backend,
        "entry_points",
        lambda **_kwargs: (_Point(factory),),
    )

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
        crop_counts=tuple(range(1, 13)),
        encode_crops=lambda crops: crops,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(siglip_backend, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(
        siglip_backend,
        "entry_points",
        lambda **_kwargs: (_Point(lambda **_kwargs: backend),),
    )

    with pytest.raises(RuntimeError, match="every crop count 1..13"):
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
