from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from kestrel.models.moondream import weights


class _Text(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(1))
        self.blocks = nn.ModuleList()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text = _Text()


def _fake_safetensors_open(
    tensors: dict[str, torch.Tensor], calls: list[str]
):
    @contextmanager
    def open_weights(path: str):
        assert path.endswith(".safetensors")

        def get_tensor(name: str) -> torch.Tensor:
            calls.append(name)
            return tensors[name]

        get_tensor.keys = lambda: list(tensors)  # type: ignore[attr-defined]
        yield get_tensor

    return open_weights


def test_safetensors_load_fetches_only_assigned_tensors(monkeypatch) -> None:
    tensors = {
        "wanted.a": torch.tensor([1.0]),
        "wanted.b": torch.tensor([2.0]),
        "unused.large": torch.empty(1024),
    }
    calls: list[str] = []
    assigned: list[torch.Tensor] = []

    monkeypatch.setattr(
        weights, "safetensors_open", _fake_safetensors_open(tensors, calls)
    )

    def assign(get_tensor, model) -> None:
        assigned.extend((get_tensor("wanted.a"), get_tensor("wanted.b")))

    monkeypatch.setattr(weights, "_assign_md2_text_weights", assign)
    weights.load_moondream_weights(
        "weights.safetensors",
        _Model(),
        load_vision=False,
        checkpoint_format="md2",
    )

    assert calls == ["wanted.a", "wanted.b"]
    assert [tensor.item() for tensor in assigned] == [1.0, 2.0]


def test_safetensors_tensor_hook_still_visits_normalized_names(monkeypatch) -> None:
    tensors = {
        "wanted._orig_mod.a": torch.tensor([1.0]),
        "wanted.b": torch.tensor([2.0]),
        "metadata.scale": torch.tensor([3.0]),
    }
    calls: list[str] = []
    hooked: list[tuple[str, float]] = []

    monkeypatch.setattr(
        weights, "safetensors_open", _fake_safetensors_open(tensors, calls)
    )
    monkeypatch.setattr(
        weights,
        "_assign_md2_text_weights",
        lambda get_tensor, model: get_tensor("wanted.a"),
    )

    weights.load_moondream_weights(
        "weights.safetensors",
        _Model(),
        load_vision=False,
        checkpoint_format="md2",
        tensor_hook=lambda name, tensor: hooked.append((name, tensor.item())),
    )

    assert hooked == [
        ("wanted.a", 1.0),
        ("wanted.b", 2.0),
        ("metadata.scale", 3.0),
    ]
    assert calls == [
        "wanted._orig_mod.a",
        "wanted.b",
        "metadata.scale",
        "wanted._orig_mod.a",
    ]


def test_safetensors_tensor_hook_predicate_avoids_unselected_reads(
    monkeypatch,
) -> None:
    tensors = {
        "wanted.a": torch.tensor([1.0]),
        "unused.large": torch.empty(1024),
        "metadata.scale": torch.tensor([3.0]),
    }
    calls: list[str] = []
    hooked: list[tuple[str, float]] = []

    monkeypatch.setattr(
        weights, "safetensors_open", _fake_safetensors_open(tensors, calls)
    )
    monkeypatch.setattr(
        weights,
        "_assign_md2_text_weights",
        lambda get_tensor, model: get_tensor("wanted.a"),
    )

    weights.load_moondream_weights(
        "weights.safetensors",
        _Model(),
        load_vision=False,
        checkpoint_format="md2",
        tensor_hook=lambda name, tensor: hooked.append((name, tensor.item())),
        tensor_hook_predicate=lambda name: name == "metadata.scale",
    )

    assert hooked == [("metadata.scale", 3.0)]
    assert calls == ["metadata.scale", "wanted.a"]


def test_safetensors_normalization_collision_fails_closed(monkeypatch) -> None:
    tensors = {
        "model._orig_mod.text.wte": torch.tensor([1.0]),
        "model.text.wte": torch.tensor([2.0]),
    }
    calls: list[str] = []
    monkeypatch.setattr(
        weights, "safetensors_open", _fake_safetensors_open(tensors, calls)
    )

    with pytest.raises(ValueError, match="collide"):
        weights.load_moondream_weights(
            "weights.safetensors",
            _Model(),
            load_vision=False,
            checkpoint_format="md2",
        )

    assert calls == []


def test_real_safetensor_tensors_survive_per_read_handle_close(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "weights.safetensors"
    save_file(
        {
            "wanted.a": torch.tensor([1.0]),
            "wanted.b": torch.tensor([2.0]),
        },
        path,
    )
    assigned: list[torch.Tensor] = []

    def assign(get_tensor, model) -> None:
        assigned.extend(
            (get_tensor("wanted.a").clone(), get_tensor("wanted.b").clone())
        )

    monkeypatch.setattr(weights, "_assign_md2_text_weights", assign)
    weights.load_moondream_weights(
        str(path),
        _Model(),
        load_vision=False,
        checkpoint_format="md2",
    )

    assert [tensor.item() for tensor in assigned] == [1.0, 2.0]
