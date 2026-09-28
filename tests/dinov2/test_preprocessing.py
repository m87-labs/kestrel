from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from kestrel.models.dinov2.config import Dinov2ProcessorConfig
from kestrel.models.dinov2.preprocessing import Dinov2ImageProcessor

from ._fixtures import PROCESSOR_CONFIG, synthetic_rgb


@pytest.fixture
def processor() -> Dinov2ImageProcessor:
    return Dinov2ImageProcessor(Dinov2ProcessorConfig.from_dict(PROCESSOR_CONFIG))


_MEAN = torch.tensor(PROCESSOR_CONFIG["image_mean"], dtype=torch.float64).view(1, 3, 1, 1)
_STD = torch.tensor(PROCESSOR_CONFIG["image_std"], dtype=torch.float64).view(1, 3, 1, 1)


def test_uint8_numpy_and_pil_inputs_match(processor: Dinov2ImageProcessor) -> None:
    image = synthetic_rgb()
    numpy_output = processor(image)
    pil_output = processor(Image.fromarray(image))
    assert numpy_output.shape == (1, 3, 224, 224)
    assert numpy_output.dtype is torch.float32
    assert numpy_output.is_contiguous()
    torch.testing.assert_close(numpy_output, pil_output, rtol=0.0, atol=0.0)


def test_grayscale_and_rgba_convert_to_rgb(processor: Dinov2ImageProcessor) -> None:
    grayscale = synthetic_rgb()[..., 0]
    rgb = np.repeat(grayscale[..., None], 3, axis=-1)
    torch.testing.assert_close(processor(grayscale), processor(rgb), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        processor(Image.fromarray(grayscale, mode="L")),
        processor(rgb),
        rtol=0.0,
        atol=0.0,
    )

    alpha = np.full(grayscale.shape, 7, dtype=np.uint8)
    rgba = np.concatenate((synthetic_rgb(), alpha[..., None]), axis=-1)
    torch.testing.assert_close(
        processor(rgba),
        processor(synthetic_rgb()),
        rtol=0.0,
        atol=0.0,
    )


def test_chw_and_hwc_inputs_match(processor: Dinov2ImageProcessor) -> None:
    image = synthetic_rgb()
    torch.testing.assert_close(
        processor(np.moveaxis(image, -1, 0)),
        processor(image),
        rtol=0.0,
        atol=0.0,
    )


def test_fractional_float_input_matches_pil_backend_quantization(
    processor: Dinov2ImageProcessor,
) -> None:
    """Float [0,1] input routes through the same uint8 quantization the PIL backend
    applies (x -> uint8(255x), an exact round trip for every byte value in float32),
    then the standard rescale on top of the restored [0,1] scale -- the Transformers
    double-rescale semantics for pre-scaled floats. The float output must therefore be
    the uint8 output's pixels divided once more by 255 before normalization."""
    image = synthetic_rgb().astype(np.float32) / 255.0
    output = processor(image)
    assert output.shape == (1, 3, 224, 224)
    from_uint8 = processor(synthetic_rgb()).to(torch.float64)
    rescaled_once_more = ((from_uint8 * _STD + _MEAN) / 255.0 - _MEAN) / _STD
    torch.testing.assert_close(
        output, rescaled_once_more.to(torch.float32), rtol=0.0, atol=1e-6
    )


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 3, 4, 5), dtype=np.uint8),
        np.zeros((20, 30, 2), dtype=np.uint8),
        np.full((20, 30, 3), -0.1, dtype=np.float32),
        np.full((20, 30, 3), np.nan, dtype=np.float32),
    ],
)
def test_invalid_numpy_inputs_are_refused(
    processor: Dinov2ImageProcessor,
    image: np.ndarray,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        processor(image)


def test_matches_transformers_pil_backend(processor: Dinov2ImageProcessor) -> None:
    transformers = pytest.importorskip("transformers")
    try:
        reference = transformers.AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small",
            revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
            backend="pil",
            local_files_only=True,
        )
    except Exception as exc:  # offline / not cached: skip cleanly, never error
        pytest.skip(f"dinov2-small processor config unavailable: {exc}")
    rgb = synthetic_rgb()
    inputs = (
        Image.fromarray(rgb),
        rgb,
        rgb.astype(np.float32),
        rgb.astype(np.float32) / 255.0,
    )
    for image in inputs:
        expected = reference(images=image, return_tensors="pt").pixel_values
        torch.testing.assert_close(processor(image), expected, rtol=0.0, atol=0.0)
