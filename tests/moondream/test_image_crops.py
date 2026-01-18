"""Tests for image cropping and preprocessing."""

import numpy as np
import pyvips
import pytest

from kestrel.moondream.config import VisionConfig
from kestrel.moondream.image_crops import (
    OverlapCropOutput,
    overlap_crop_image,
    select_tiling,
)


class TestSelectTiling:
    """Tests for the select_tiling function."""

    def test_small_image_returns_1x1(self):
        """Images smaller than crop_size should return (1, 1) tiling."""
        assert select_tiling(300, 300, 378, 12) == (1, 1)
        assert select_tiling(378, 378, 378, 12) == (1, 1)
        assert select_tiling(100, 500, 378, 12) == (1, 1)
        assert select_tiling(500, 100, 378, 12) == (1, 1)

    def test_large_square_image(self):
        """Large square images should tile appropriately."""
        tiling = select_tiling(1024, 1024, 378, 12)
        assert tiling[0] >= 1 and tiling[1] >= 1
        assert tiling[0] * tiling[1] <= 12

    def test_landscape_image(self):
        """Landscape images should have more horizontal tiles."""
        tiling = select_tiling(1080, 1920, 378, 12)
        assert tiling[1] >= tiling[0]  # More columns than rows

    def test_portrait_image(self):
        """Portrait images should have more vertical tiles."""
        tiling = select_tiling(1920, 1080, 378, 12)
        assert tiling[0] >= tiling[1]  # More rows than columns

    def test_respects_max_crops(self):
        """Tiling should never exceed max_crops."""
        for max_crops in [1, 4, 9, 12, 16]:
            tiling = select_tiling(4000, 4000, 378, max_crops)
            assert tiling[0] * tiling[1] <= max_crops


class TestOverlapCropImage:
    """Tests for the overlap_crop_image function."""

    @pytest.fixture
    def config(self):
        return VisionConfig()

    def test_numpy_input_basic(self, config):
        """Test with numpy array input."""
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        result = overlap_crop_image(
            image,
            overlap_margin=config.overlap_margin,
            max_crops=config.max_crops,
            base_size=(config.crop_size, config.crop_size),
            patch_size=config.enc_patch_size,
        )

        assert "crops" in result
        assert "tiling" in result
        assert result["crops"].dtype == np.uint8
        assert result["crops"].shape[1] == config.crop_size
        assert result["crops"].shape[2] == config.crop_size
        assert result["crops"].shape[3] == 3

    def test_pyvips_input(self, config):
        """Test with pyvips.Image input."""
        np_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        vips_image = pyvips.Image.new_from_memory(
            np_image.tobytes(), 1024, 1024, 3, "uchar"
        )

        result = overlap_crop_image(
            vips_image,
            overlap_margin=config.overlap_margin,
            max_crops=config.max_crops,
            base_size=(config.crop_size, config.crop_size),
            patch_size=config.enc_patch_size,
        )

        assert result["crops"].dtype == np.uint8
        assert result["crops"].shape[1] == config.crop_size

    def test_num_crops_matches_tiling(self, config):
        """Number of crops should be tiling[0] * tiling[1] + 1 (for global crop)."""
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        result = overlap_crop_image(
            image,
            overlap_margin=config.overlap_margin,
            max_crops=config.max_crops,
            base_size=(config.crop_size, config.crop_size),
            patch_size=config.enc_patch_size,
        )

        expected_num_crops = result["tiling"][0] * result["tiling"][1] + 1
        assert result["crops"].shape[0] == expected_num_crops

    @pytest.mark.parametrize(
        "height,width",
        [
            (1024, 1024),  # Square
            (1920, 1080),  # Landscape HD
            (1080, 1920),  # Portrait HD
            (800, 600),  # Small landscape
            (2048, 1536),  # Large
            (378, 378),  # Exactly crop size
            (300, 300),  # Smaller than crop size
        ],
    )
    def test_various_image_sizes(self, config, height, width):
        """Test with various image dimensions."""
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        result = overlap_crop_image(
            image,
            overlap_margin=config.overlap_margin,
            max_crops=config.max_crops,
            base_size=(config.crop_size, config.crop_size),
            patch_size=config.enc_patch_size,
        )

        # Basic shape checks
        assert result["crops"].shape[1] == config.crop_size
        assert result["crops"].shape[2] == config.crop_size
        assert result["crops"].shape[3] == 3

        # Tiling should respect max_crops
        assert result["tiling"][0] * result["tiling"][1] <= config.max_crops

    def test_global_crop_is_first(self, config):
        """First crop should be the global (full image) crop."""
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        result = overlap_crop_image(
            image,
            overlap_margin=config.overlap_margin,
            max_crops=config.max_crops,
            base_size=(config.crop_size, config.crop_size),
            patch_size=config.enc_patch_size,
        )

        # Global crop is at index 0
        global_crop = result["crops"][0]
        assert global_crop.shape == (config.crop_size, config.crop_size, 3)

    def test_crops_are_valid_uint8(self, config):
        """All crop values should be valid uint8 (0-255)."""
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        result = overlap_crop_image(
            image,
            overlap_margin=config.overlap_margin,
            max_crops=config.max_crops,
            base_size=(config.crop_size, config.crop_size),
            patch_size=config.enc_patch_size,
        )

        assert result["crops"].min() >= 0
        assert result["crops"].max() <= 255

    def test_deterministic_output(self, config):
        """Same input should produce same output."""
        np.random.seed(42)
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

        result1 = overlap_crop_image(
            image,
            overlap_margin=config.overlap_margin,
            max_crops=config.max_crops,
            base_size=(config.crop_size, config.crop_size),
            patch_size=config.enc_patch_size,
        )

        result2 = overlap_crop_image(
            image,
            overlap_margin=config.overlap_margin,
            max_crops=config.max_crops,
            base_size=(config.crop_size, config.crop_size),
            patch_size=config.enc_patch_size,
        )

        assert result1["tiling"] == result2["tiling"]
        np.testing.assert_array_equal(result1["crops"], result2["crops"])
