from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.runtime import Qwen35Runtime, QwenImageInputs
from kestrel.runtime import ImageMarker, TextToken


def _image_inputs(*, images: int, tokens: int) -> QwenImageInputs:
    grid_width = tokens * 4 // images
    return QwenImageInputs(
        pixel_values=torch.empty((1, 3)),
        image_grid_thw=torch.tensor(
            [[1, 1, grid_width]] * images,
            dtype=torch.long,
        ),
        num_image_tokens=tokens,
    )


def test_qwen_image_kv_length_uses_preprocessed_expansion() -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.image_prefix_length = 4096
    crops = _image_inputs(images=1, tokens=256)

    assert runtime.image_kv_length([TextToken(1)], object(), crops) == 258
    assert runtime.image_kv_length([ImageMarker(0)], object(), crops) == 257
    assert runtime.image_kv_length([TextToken(1)], object(), None) == 4096

    multi = _image_inputs(images=2, tokens=600)
    assert runtime.image_kv_length(
        [ImageMarker(0), ImageMarker(1)],
        [object(), object()],
        multi,
    ) == 602

    inconsistent = _image_inputs(images=1, tokens=256)
    inconsistent.num_image_tokens = 255
    with pytest.raises(ValueError, match="does not match its grid"):
        runtime.image_kv_length([TextToken(1)], object(), inconsistent)


def test_qwen_prepare_sequence_reserves_exact_expanded_length() -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.architecture = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2)
    )
    runtime._chat_image_crops = {}
    runtime._prepare_uncached_sequence = lambda **kwargs: kwargs
    crops = QwenImageInputs(
        pixel_values=torch.empty((1, 3)),
        image_grid_thw=torch.tensor([[1, 32, 32]]),
        num_image_tokens=256,
    )

    prepared = runtime.prepare_sequence(
        [TextToken(1), ImageMarker(0)],
        image=object(),
        image_crops=crops,
        max_new_tokens=1536,
    )

    assert prepared["image_length"] == 258
    assert len(prepared["tokens"]) == 259
    assert prepared["target_length"] == 1795


@pytest.mark.parametrize(
    "markers",
    (
        (ImageMarker(1), ImageMarker(0)),
        (ImageMarker(0), ImageMarker(0)),
    ),
)
def test_qwen_prepare_sequence_rejects_out_of_order_image_markers(
    markers: tuple[ImageMarker, ...],
) -> None:
    runtime = object.__new__(Qwen35Runtime)
    runtime.architecture = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2)
    )
    runtime._chat_image_crops = {}
    runtime._prepare_uncached_sequence = lambda **kwargs: kwargs
    crops = QwenImageInputs(
        pixel_values=torch.empty((2, 3)),
        image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
        num_image_tokens=2,
    )

    with pytest.raises(RuntimeError, match="must appear in image input order"):
        runtime.prepare_sequence(
            list(markers),
            image=[object(), object()],
            image_crops=crops,
        )
