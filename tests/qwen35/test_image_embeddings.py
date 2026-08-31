from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kestrel.models.qwen35.qwen_model import (
    Qwen3_5Model,
    _copy_image_features_into_embeddings,
)
from kestrel.models.qwen35.runtime import Qwen35Runtime


class _PackedVision(torch.nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((), dtype=output.dtype))
        self.output = output

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.output


def test_image_features_preserve_packed_visual_output() -> None:
    packed = torch.randn(5, 3, dtype=torch.bfloat16)
    model = object.__new__(Qwen3_5Model)
    torch.nn.Module.__init__(model)
    model.visual = _PackedVision(packed)

    result = model.get_image_features(
        torch.zeros(2, 3, dtype=torch.bfloat16),
        bilinear_indices=torch.empty(0),
        bilinear_weights=torch.empty(0),
        position_ids=torch.empty(0),
        cu_seqlens=torch.empty(0),
    )

    assert result is packed


def test_direct_image_feature_copy_preserves_order_without_packing() -> None:
    features = torch.arange(15, dtype=torch.bfloat16).view(3, 5).t()
    assert not features.is_contiguous()
    embeddings = torch.randn(1, 10, 3, dtype=torch.bfloat16)
    expected = embeddings.clone()
    expected[0, 1:3].copy_(features[:2])
    expected[0, 6:9].copy_(features[2:])
    storage = embeddings.untyped_storage().data_ptr()

    _copy_image_features_into_embeddings(
        embeddings,
        features,
        ((1, 3), (6, 9)),
    )

    assert embeddings.untyped_storage().data_ptr() == storage
    assert torch.equal(embeddings, expected)


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("count", "expected"),
        ("device", "could not be converted"),
        ("rank", "must be 2D"),
        ("shape", "expected"),
        ("bounds", "invalid or out of order"),
        ("overlap", "invalid or out of order"),
    ],
)
def test_direct_image_feature_copy_validates_before_writing(
    failure: str,
    match: str,
) -> None:
    features = torch.ones(4, 3, dtype=torch.bfloat16)
    spans = ((1, 3), (5, 7))
    if failure == "count":
        features = torch.ones(3, 3, dtype=torch.bfloat16)
    elif failure == "device":
        features = torch.empty(4, 3, dtype=torch.bfloat16, device="meta")
    elif failure == "rank":
        features = torch.ones(12, dtype=torch.bfloat16)
    elif failure == "shape":
        features = torch.ones(4, 2, dtype=torch.bfloat16)
    elif failure == "bounds":
        spans = ((1, 3), (7, 11))
    elif failure == "overlap":
        spans = ((1, 3), (2, 4))
    embeddings = torch.randn(1, 10, 3, dtype=torch.bfloat16)
    before = embeddings.clone()

    with pytest.raises((TypeError, ValueError), match=match):
        _copy_image_features_into_embeddings(embeddings, features, spans)

    assert torch.equal(embeddings, before)


def test_direct_image_feature_copy_preserves_dtype_conversion_semantics() -> None:
    embeddings = torch.zeros(1, 4, 3, dtype=torch.bfloat16)
    features = torch.tensor(
        [[1.125, 2.25, 3.5], [4.75, 5.0, 6.5]],
        dtype=torch.float32,
    )

    _copy_image_features_into_embeddings(embeddings, features, ((1, 3),))

    torch.testing.assert_close(
        embeddings[0, 1:3],
        features.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )


def test_multimodal_position_ids_return_ordered_packed_image_spans() -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.architecture = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2)
    )
    token_types = np.asarray([0, 1, 1, 0, 1, 1, 1, 1, 0], dtype=np.int64)
    grid = np.asarray([[1, 2, 4], [1, 4, 4]], dtype=np.int64)
    position_ids = np.zeros((3, 1, 12), dtype=np.int64)

    _, spans = runtime._fill_multimodal_position_ids(
        position_ids,
        start=2,
        end=11,
        mm_token_type_ids=token_types,
        image_grid_thw=grid,
    )

    assert spans == ((3, 5), (6, 10))
