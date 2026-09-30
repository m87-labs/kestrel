from __future__ import annotations

import copy

import pytest

from kestrel.models.dinov2.config import Dinov2Config, Dinov2ProcessorConfig

from ._fixtures import MODEL_CONFIG, PROCESSOR_CONFIG


def test_pinned_model_config_parses_and_validates() -> None:
    config = Dinov2Config.from_dict(MODEL_CONFIG)
    config.validate_v1()
    assert config.hidden_size == 384
    assert config.intermediate_size == 1536
    assert config.head_dim == 64
    assert config.base_grid_size == 37
    assert config.num_position_embeddings == 1370


def test_pinned_processor_config_parses_and_validates() -> None:
    config = Dinov2ProcessorConfig.from_dict(PROCESSOR_CONFIG)
    config.validate_v1()
    assert (config.crop_height, config.crop_width) == (224, 224)
    assert config.shortest_edge == 256


@pytest.mark.parametrize("kind", ["unknown", "missing", "unsupported"])
def test_model_config_refuses_non_v1_contracts(kind: str) -> None:
    raw = copy.deepcopy(MODEL_CONFIG)
    if kind == "unknown":
        raw["new_architecture_knob"] = True
        with pytest.raises(ValueError, match="unsupported.*fields"):
            Dinov2Config.from_dict(raw)
    elif kind == "missing":
        del raw["patch_size"]
        with pytest.raises(ValueError, match="missing.*patch_size"):
            Dinov2Config.from_dict(raw)
    else:
        raw["hidden_size"] = 768
        with pytest.raises(ValueError, match="hidden_size=768"):
            Dinov2Config.from_dict(raw).validate_v1()


def test_processor_config_refuses_changed_geometry() -> None:
    raw = copy.deepcopy(PROCESSOR_CONFIG)
    raw["crop_size"]["height"] = 518
    with pytest.raises(ValueError, match="crop_height=518"):
        Dinov2ProcessorConfig.from_dict(raw).validate_v1()
