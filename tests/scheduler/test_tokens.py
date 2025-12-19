"""Unit tests for token materialization and prompt processing.

These tests do not require CUDA - they only use CPU tensors.
"""

import pytest
import torch

from kestrel.moondream.runtime import TextToken, CoordToken, SizeToken
from kestrel.scheduler.tokens import (
    prompt_with_spatial_tokens,
    render_tokens_from_packed,
)


# Test IDs used in tests (arbitrary values)
COORD_ID = 100
SIZE_ID = 101
TEXT_ID_A = 1
TEXT_ID_B = 2
TEXT_ID_C = 3


class TestPromptWithSpatialTokens:
    """Tests for prompt_with_spatial_tokens function."""

    def test_no_spatial_refs_all_text(self):
        """Prompt with no placeholders and no refs returns text tokens."""
        prompt = torch.tensor([TEXT_ID_A, TEXT_ID_B, TEXT_ID_C])
        result = prompt_with_spatial_tokens(prompt, COORD_ID, SIZE_ID, [])

        assert len(result) == 3
        assert all(isinstance(t, TextToken) for t in result)
        assert [t.token_id for t in result] == [TEXT_ID_A, TEXT_ID_B, TEXT_ID_C]

    def test_single_point_ref(self):
        """Single point reference replaces two coord placeholders."""
        # Prompt: [text, coord, coord, text]
        prompt = torch.tensor([TEXT_ID_A, COORD_ID, COORD_ID, TEXT_ID_B])
        result = prompt_with_spatial_tokens(
            prompt, COORD_ID, SIZE_ID, [[0.5, 0.75]]
        )

        assert len(result) == 4
        assert isinstance(result[0], TextToken)
        assert result[0].token_id == TEXT_ID_A
        assert isinstance(result[1], CoordToken)
        assert result[1].pos == pytest.approx(0.5)
        assert isinstance(result[2], CoordToken)
        assert result[2].pos == pytest.approx(0.75)
        assert isinstance(result[3], TextToken)
        assert result[3].token_id == TEXT_ID_B

    def test_single_bbox_ref(self):
        """Single bbox reference replaces coord and size placeholders."""
        # Prompt: [coord, coord, size]
        prompt = torch.tensor([COORD_ID, COORD_ID, SIZE_ID])
        # bbox: [x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8]
        # center: (0.3, 0.5), size: (0.4, 0.6)
        result = prompt_with_spatial_tokens(
            prompt, COORD_ID, SIZE_ID, [[0.1, 0.2, 0.5, 0.8]]
        )

        assert len(result) == 3
        assert isinstance(result[0], CoordToken)
        assert result[0].pos == pytest.approx(0.3)  # x center
        assert isinstance(result[1], CoordToken)
        assert result[1].pos == pytest.approx(0.5)  # y center
        assert isinstance(result[2], SizeToken)
        assert result[2].width == pytest.approx(0.4)
        assert result[2].height == pytest.approx(0.6)

    def test_multiple_refs(self):
        """Multiple spatial refs are processed in order."""
        # Two points
        prompt = torch.tensor([COORD_ID, COORD_ID, COORD_ID, COORD_ID])
        result = prompt_with_spatial_tokens(
            prompt, COORD_ID, SIZE_ID, [[0.1, 0.2], [0.3, 0.4]]
        )

        assert len(result) == 4
        assert [t.pos for t in result] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_clamps_point_values(self):
        """Point coordinates are clamped to [0, 1]."""
        prompt = torch.tensor([COORD_ID, COORD_ID])
        result = prompt_with_spatial_tokens(
            prompt, COORD_ID, SIZE_ID, [[-0.5, 1.5]]
        )

        assert result[0].pos == pytest.approx(0.0)
        assert result[1].pos == pytest.approx(1.0)

    def test_2d_prompt_tensor(self):
        """Handles 2D prompt tensors by flattening."""
        prompt = torch.tensor([[TEXT_ID_A, TEXT_ID_B]])
        result = prompt_with_spatial_tokens(prompt, COORD_ID, SIZE_ID, [])

        assert len(result) == 2
        assert [t.token_id for t in result] == [TEXT_ID_A, TEXT_ID_B]

    def test_error_wrong_ref_length(self):
        """Raises error for refs with wrong number of values."""
        prompt = torch.tensor([COORD_ID])
        with pytest.raises(ValueError, match="2 .point. or 4 .bbox."):
            prompt_with_spatial_tokens(prompt, COORD_ID, SIZE_ID, [[0.1, 0.2, 0.3]])

    def test_error_mismatched_placeholder_count(self):
        """Raises error when placeholder count doesn't match refs."""
        prompt = torch.tensor([COORD_ID])  # 1 coord placeholder
        with pytest.raises(ValueError, match="Mismatch"):
            # Point requires 2 coord placeholders
            prompt_with_spatial_tokens(prompt, COORD_ID, SIZE_ID, [[0.5, 0.5]])

    def test_error_invalid_bbox_bounds(self):
        """Raises error for bbox with invalid bounds."""
        prompt = torch.tensor([COORD_ID, COORD_ID, SIZE_ID])
        # x_min > x_max
        with pytest.raises(ValueError, match="0<=x_min<=x_max<=1"):
            prompt_with_spatial_tokens(
                prompt, COORD_ID, SIZE_ID, [[0.8, 0.2, 0.2, 0.8]]
            )


class TestRenderTokensFromPacked:
    """Tests for render_tokens_from_packed function."""

    def test_text_tokens(self):
        """Renders text tokens correctly."""
        token_ids = torch.tensor([TEXT_ID_A, TEXT_ID_B, TEXT_ID_C])
        coord_values = torch.zeros((3, 1))
        size_values = torch.zeros((3, 2))

        result = render_tokens_from_packed(
            token_ids, coord_values, size_values,
            coord_id=COORD_ID, size_id=SIZE_ID,
        )

        assert len(result) == 3
        assert all(isinstance(t, TextToken) for t in result)
        assert [t.token_id for t in result] == [TEXT_ID_A, TEXT_ID_B, TEXT_ID_C]

    def test_coord_tokens(self):
        """Renders coord tokens with correct positions."""
        token_ids = torch.tensor([COORD_ID, COORD_ID])
        coord_values = torch.tensor([[0.25], [0.75]])
        size_values = torch.zeros((2, 2))

        result = render_tokens_from_packed(
            token_ids, coord_values, size_values,
            coord_id=COORD_ID, size_id=SIZE_ID,
        )

        assert len(result) == 2
        assert all(isinstance(t, CoordToken) for t in result)
        assert result[0].pos == pytest.approx(0.25)
        assert result[1].pos == pytest.approx(0.75)

    def test_size_tokens(self):
        """Renders size tokens with correct width/height."""
        token_ids = torch.tensor([SIZE_ID])
        coord_values = torch.zeros((1, 1))
        size_values = torch.tensor([[0.3, 0.5]])

        result = render_tokens_from_packed(
            token_ids, coord_values, size_values,
            coord_id=COORD_ID, size_id=SIZE_ID,
        )

        assert len(result) == 1
        assert isinstance(result[0], SizeToken)
        assert result[0].width == pytest.approx(0.3)
        assert result[0].height == pytest.approx(0.5)

    def test_mixed_tokens(self):
        """Renders mixed token types in order."""
        token_ids = torch.tensor([TEXT_ID_A, COORD_ID, SIZE_ID, TEXT_ID_B])
        coord_values = torch.tensor([[0.0], [0.5], [0.0], [0.0]])
        size_values = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.2, 0.4], [0.0, 0.0]])

        result = render_tokens_from_packed(
            token_ids, coord_values, size_values,
            coord_id=COORD_ID, size_id=SIZE_ID,
        )

        assert len(result) == 4
        assert isinstance(result[0], TextToken)
        assert isinstance(result[1], CoordToken)
        assert isinstance(result[2], SizeToken)
        assert isinstance(result[3], TextToken)

        assert result[0].token_id == TEXT_ID_A
        assert result[1].pos == pytest.approx(0.5)
        assert result[2].width == pytest.approx(0.2)
        assert result[2].height == pytest.approx(0.4)
        assert result[3].token_id == TEXT_ID_B

    def test_empty_input(self):
        """Returns empty list for empty input."""
        token_ids = torch.tensor([], dtype=torch.long)
        coord_values = torch.zeros((0, 1))
        size_values = torch.zeros((0, 2))

        result = render_tokens_from_packed(
            token_ids, coord_values, size_values,
            coord_id=COORD_ID, size_id=SIZE_ID,
        )

        assert result == []

    def test_2d_token_ids(self):
        """Handles 2D token_ids by flattening."""
        token_ids = torch.tensor([[TEXT_ID_A], [TEXT_ID_B]])
        coord_values = torch.zeros((2, 1))
        size_values = torch.zeros((2, 2))

        result = render_tokens_from_packed(
            token_ids, coord_values, size_values,
            coord_id=COORD_ID, size_id=SIZE_ID,
        )

        assert len(result) == 2
        assert [t.token_id for t in result] == [TEXT_ID_A, TEXT_ID_B]
