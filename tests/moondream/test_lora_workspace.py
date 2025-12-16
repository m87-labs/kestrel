"""Unit tests for lora_workspace module."""

import pytest
import torch

from kestrel.moondream.config import TextConfig, TextMoeConfig
from kestrel.moondream.lora import LoRA, TextLoRA, TextLoRAConfig
from kestrel.moondream.lora_workspace import (
    DenseLoRALayerWorkspace,
    MoELoRALayerWorkspace,
    TextLoRAWorkspace,
)


# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def dense_only_config() -> TextConfig:
    """TextConfig with no MoE layers (all dense)."""
    return TextConfig(
        dim=64,
        ff_dim=128,
        n_layers=4,
        moe=None,
    )


@pytest.fixture
def moe_config() -> TextConfig:
    """TextConfig with MoE layers starting at layer 2."""
    return TextConfig(
        dim=64,
        ff_dim=128,
        n_layers=6,
        moe=TextMoeConfig(
            num_experts=4,
            start_layer=2,
            experts_per_token=2,
            expert_inner_dim=32,
        ),
    )


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Allocation tests
# -----------------------------------------------------------------------------


class TestWorkspaceAllocation:
    """Tests for TextLoRAWorkspace allocation correctness."""

    def test_dense_only_shapes(self, dense_only_config: TextConfig, device: torch.device):
        """Dense-only workspace has correct tensor shapes."""
        max_slots = 4
        max_rank = 8
        workspace = TextLoRAWorkspace(
            dense_only_config, max_slots, max_rank, device
        )

        assert workspace.max_slots == max_slots
        assert workspace.max_rank == max_rank
        assert workspace.start_layer == dense_only_config.n_layers
        assert len(workspace.dense) == dense_only_config.n_layers
        assert len(workspace.moe) == 0

        for layer in workspace.dense:
            assert layer.up_a.shape == (max_slots, max_rank, dense_only_config.dim)
            assert layer.up_b.shape == (max_slots, dense_only_config.ff_dim, max_rank)
            assert layer.down_a.shape == (max_slots, max_rank, dense_only_config.ff_dim)
            assert layer.down_b.shape == (max_slots, dense_only_config.dim, max_rank)

    def test_moe_shapes(self, moe_config: TextConfig, device: torch.device):
        """MoE workspace has correct tensor shapes for both dense and MoE layers."""
        max_slots = 4
        max_rank = 8
        workspace = TextLoRAWorkspace(moe_config, max_slots, max_rank, device)

        moe_cfg = moe_config.moe
        assert moe_cfg is not None

        assert workspace.start_layer == moe_cfg.start_layer
        assert len(workspace.dense) == moe_cfg.start_layer
        assert len(workspace.moe) == moe_config.n_layers - moe_cfg.start_layer

        # Check dense layer shapes
        for layer in workspace.dense:
            assert layer.up_a.shape == (max_slots, max_rank, moe_config.dim)
            assert layer.up_b.shape == (max_slots, moe_config.ff_dim, max_rank)
            assert layer.down_a.shape == (max_slots, max_rank, moe_config.ff_dim)
            assert layer.down_b.shape == (max_slots, moe_config.dim, max_rank)

        # Check MoE layer shapes with super-expert indexing
        expected_rank_per_expert = max_rank // moe_cfg.experts_per_token
        total_experts = max_slots * moe_cfg.num_experts

        for layer in workspace.moe:
            assert layer.num_experts == moe_cfg.num_experts
            assert layer.up_a.shape == (
                total_experts,
                expected_rank_per_expert,
                moe_config.dim,
            )
            assert layer.up_b.shape == (
                total_experts,
                moe_cfg.expert_inner_dim * 2,
                expected_rank_per_expert,
            )
            assert layer.down_a.shape == (
                total_experts,
                expected_rank_per_expert,
                moe_cfg.expert_inner_dim,
            )
            assert layer.down_b.shape == (
                total_experts,
                moe_config.dim,
                expected_rank_per_expert,
            )

    def test_slot_zero_is_zeros(self, moe_config: TextConfig, device: torch.device):
        """Slot 0 is initialized to zeros."""
        workspace = TextLoRAWorkspace(moe_config, max_slots=4, max_rank=8, device=device)

        for layer in workspace.dense:
            assert torch.all(layer.up_a[0] == 0)
            assert torch.all(layer.up_b[0] == 0)
            assert torch.all(layer.down_a[0] == 0)
            assert torch.all(layer.down_b[0] == 0)

        moe_cfg = moe_config.moe
        assert moe_cfg is not None
        for layer in workspace.moe:
            # Slot 0 experts: indices [0, num_experts)
            assert torch.all(layer.up_a[: moe_cfg.num_experts] == 0)
            assert torch.all(layer.up_b[: moe_cfg.num_experts] == 0)
            assert torch.all(layer.down_a[: moe_cfg.num_experts] == 0)
            assert torch.all(layer.down_b[: moe_cfg.num_experts] == 0)

    def test_max_rank_per_expert_calculation(self, device: torch.device):
        """max_rank_per_expert is correctly computed as max_rank // experts_per_token."""
        config = TextConfig(
            dim=64,
            ff_dim=128,
            n_layers=4,
            moe=TextMoeConfig(
                num_experts=4,
                start_layer=2,
                experts_per_token=2,
                expert_inner_dim=32,
            ),
        )
        workspace = TextLoRAWorkspace(config, max_slots=4, max_rank=8, device=device)
        assert workspace.max_rank_per_expert == 4  # 8 // 2

        workspace2 = TextLoRAWorkspace(config, max_slots=4, max_rank=16, device=device)
        assert workspace2.max_rank_per_expert == 8  # 16 // 2

    def test_max_rank_too_small_for_experts_raises(self, device: torch.device):
        """Raises ValueError if max_rank < experts_per_token."""
        config = TextConfig(
            dim=64,
            ff_dim=128,
            n_layers=4,
            moe=TextMoeConfig(
                num_experts=4,
                start_layer=2,
                experts_per_token=8,
                expert_inner_dim=32,
            ),
        )
        with pytest.raises(ValueError, match="max_lora_rank.*must be >= experts_per_token"):
            TextLoRAWorkspace(config, max_slots=4, max_rank=4, device=device)


# -----------------------------------------------------------------------------
# clear_slot_ tests
# -----------------------------------------------------------------------------


class TestClearSlot:
    """Tests for TextLoRAWorkspace.clear_slot_()."""

    def test_clear_slot_zeros_target(self, moe_config: TextConfig, device: torch.device):
        """clear_slot_ zeros out the specified slot."""
        workspace = TextLoRAWorkspace(moe_config, max_slots=4, max_rank=8, device=device)

        # Fill slot 1 with non-zero values
        for layer in workspace.dense:
            layer.up_a[1].fill_(1.0)
            layer.up_b[1].fill_(1.0)
            layer.down_a[1].fill_(1.0)
            layer.down_b[1].fill_(1.0)

        moe_cfg = moe_config.moe
        assert moe_cfg is not None
        for layer in workspace.moe:
            start = 1 * moe_cfg.num_experts
            end = start + moe_cfg.num_experts
            layer.up_a[start:end].fill_(1.0)
            layer.up_b[start:end].fill_(1.0)
            layer.down_a[start:end].fill_(1.0)
            layer.down_b[start:end].fill_(1.0)

        # Clear slot 1
        workspace.clear_slot_(1)

        # Verify slot 1 is now zero
        for layer in workspace.dense:
            assert torch.all(layer.up_a[1] == 0)
            assert torch.all(layer.up_b[1] == 0)
            assert torch.all(layer.down_a[1] == 0)
            assert torch.all(layer.down_b[1] == 0)

        for layer in workspace.moe:
            start = 1 * moe_cfg.num_experts
            end = start + moe_cfg.num_experts
            assert torch.all(layer.up_a[start:end] == 0)
            assert torch.all(layer.up_b[start:end] == 0)
            assert torch.all(layer.down_a[start:end] == 0)
            assert torch.all(layer.down_b[start:end] == 0)

    def test_clear_slot_preserves_other_slots(
        self, moe_config: TextConfig, device: torch.device
    ):
        """clear_slot_ does not affect other slots."""
        workspace = TextLoRAWorkspace(moe_config, max_slots=4, max_rank=8, device=device)

        # Fill slots 1 and 2
        for slot in [1, 2]:
            for layer in workspace.dense:
                layer.up_a[slot].fill_(float(slot))

        # Clear slot 1
        workspace.clear_slot_(1)

        # Slot 2 should be unchanged
        for layer in workspace.dense:
            assert torch.all(layer.up_a[2] == 2.0)

    def test_clear_slot_zero_raises(self, dense_only_config: TextConfig, device: torch.device):
        """clear_slot_(0) raises ValueError."""
        workspace = TextLoRAWorkspace(
            dense_only_config, max_slots=4, max_rank=8, device=device
        )
        with pytest.raises(ValueError, match="Slot 0 is reserved"):
            workspace.clear_slot_(0)

    def test_clear_slot_out_of_range_raises(
        self, dense_only_config: TextConfig, device: torch.device
    ):
        """clear_slot_ with out-of-range slot raises ValueError."""
        workspace = TextLoRAWorkspace(
            dense_only_config, max_slots=4, max_rank=8, device=device
        )
        with pytest.raises(ValueError, match="out of range"):
            workspace.clear_slot_(4)

        with pytest.raises(ValueError, match="out of range"):
            workspace.clear_slot_(-1)


# -----------------------------------------------------------------------------
# load_slot_ tests
# -----------------------------------------------------------------------------


def create_test_adapter(
    text_config: TextConfig,
    rank: int,
    device: torch.device,
    fill_value: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
) -> LoRA:
    """Create a test LoRA adapter with specified fill value."""
    lora_config = TextLoRAConfig(rank=rank)
    text_lora = TextLoRA(text_config, lora_config, dtype=dtype)
    text_lora.to(device)

    # Fill with test values
    for layer in text_lora.dense:
        layer.up_a.data.fill_(fill_value)
        layer.up_b.data.fill_(fill_value)
        layer.down_a.data.fill_(fill_value)
        layer.down_b.data.fill_(fill_value)

    for layer in text_lora.moe:
        layer.up_a.data.fill_(fill_value)
        layer.up_b.data.fill_(fill_value)
        layer.down_a.data.fill_(fill_value)
        layer.down_b.data.fill_(fill_value)

    return LoRA(text=text_lora, vision=None)


class TestLoadSlot:
    """Tests for TextLoRAWorkspace.load_slot_()."""

    def test_load_slot_copies_dense_weights(
        self, dense_only_config: TextConfig, device: torch.device
    ):
        """load_slot_ correctly copies dense layer weights."""
        max_rank = 8
        workspace = TextLoRAWorkspace(
            dense_only_config, max_slots=4, max_rank=max_rank, device=device
        )
        adapter = create_test_adapter(
            dense_only_config, rank=max_rank, device=device, fill_value=2.5
        )

        workspace.load_slot_(1, adapter)

        # Verify weights were copied
        for layer_idx, layer in enumerate(workspace.dense):
            adapter_layer = adapter.text.get_dense_lora(layer_idx)
            assert adapter_layer is not None
            assert torch.allclose(layer.up_a[1], adapter_layer.up_a)
            assert torch.allclose(layer.up_b[1], adapter_layer.up_b)
            assert torch.allclose(layer.down_a[1], adapter_layer.down_a)
            assert torch.allclose(layer.down_b[1], adapter_layer.down_b)

    def test_load_slot_copies_moe_weights(
        self, moe_config: TextConfig, device: torch.device
    ):
        """load_slot_ correctly copies MoE layer weights."""
        max_rank = 8
        workspace = TextLoRAWorkspace(
            moe_config, max_slots=4, max_rank=max_rank, device=device
        )
        adapter = create_test_adapter(
            moe_config, rank=max_rank, device=device, fill_value=3.0
        )

        workspace.load_slot_(2, adapter)

        moe_cfg = moe_config.moe
        assert moe_cfg is not None

        # Verify MoE weights were copied with correct super-expert indexing
        for moe_idx, layer in enumerate(workspace.moe):
            layer_idx = workspace.start_layer + moe_idx
            adapter_layer = adapter.text.get_moe_lora(layer_idx)
            assert adapter_layer is not None

            for expert_id in range(moe_cfg.num_experts):
                ws_idx = 2 * moe_cfg.num_experts + expert_id
                rank_per_expert = adapter.text.rank_per_expert
                assert torch.allclose(
                    layer.up_a[ws_idx, :rank_per_expert],
                    adapter_layer.up_a[expert_id],
                )
                assert torch.allclose(
                    layer.up_b[ws_idx, :, :rank_per_expert],
                    adapter_layer.up_b[expert_id],
                )
                assert torch.allclose(
                    layer.down_a[ws_idx, :rank_per_expert],
                    adapter_layer.down_a[expert_id],
                )
                assert torch.allclose(
                    layer.down_b[ws_idx, :, :rank_per_expert],
                    adapter_layer.down_b[expert_id],
                )

    def test_load_slot_with_smaller_rank_zero_pads(
        self, dense_only_config: TextConfig, device: torch.device
    ):
        """load_slot_ zero-pads when adapter rank < max_rank."""
        max_rank = 16
        adapter_rank = 8
        workspace = TextLoRAWorkspace(
            dense_only_config, max_slots=4, max_rank=max_rank, device=device
        )
        adapter = create_test_adapter(
            dense_only_config, rank=adapter_rank, device=device, fill_value=1.0
        )

        workspace.load_slot_(1, adapter)

        for layer_idx, layer in enumerate(workspace.dense):
            adapter_layer = adapter.text.get_dense_lora(layer_idx)
            assert adapter_layer is not None

            # First adapter_rank dimensions should match
            assert torch.allclose(layer.up_a[1, :adapter_rank], adapter_layer.up_a)
            assert torch.allclose(layer.up_b[1, :, :adapter_rank], adapter_layer.up_b)

            # Remaining dimensions should be zero
            assert torch.all(layer.up_a[1, adapter_rank:] == 0)
            assert torch.all(layer.up_b[1, :, adapter_rank:] == 0)
            assert torch.all(layer.down_a[1, adapter_rank:] == 0)
            assert torch.all(layer.down_b[1, :, adapter_rank:] == 0)

    def test_load_slot_zero_raises(self, dense_only_config: TextConfig, device: torch.device):
        """load_slot_(0, ...) raises ValueError."""
        workspace = TextLoRAWorkspace(
            dense_only_config, max_slots=4, max_rank=8, device=device
        )
        adapter = create_test_adapter(dense_only_config, rank=8, device=device)

        with pytest.raises(ValueError, match="Slot 0 is reserved"):
            workspace.load_slot_(0, adapter)

    def test_load_slot_out_of_range_raises(
        self, dense_only_config: TextConfig, device: torch.device
    ):
        """load_slot_ with out-of-range slot raises ValueError."""
        workspace = TextLoRAWorkspace(
            dense_only_config, max_slots=4, max_rank=8, device=device
        )
        adapter = create_test_adapter(dense_only_config, rank=8, device=device)

        with pytest.raises(ValueError, match="out of range"):
            workspace.load_slot_(4, adapter)

    def test_load_slot_rank_exceeds_max_raises(
        self, dense_only_config: TextConfig, device: torch.device
    ):
        """load_slot_ raises ValueError if adapter rank > max_rank."""
        workspace = TextLoRAWorkspace(
            dense_only_config, max_slots=4, max_rank=8, device=device
        )
        adapter = create_test_adapter(dense_only_config, rank=16, device=device)

        with pytest.raises(ValueError, match="Adapter rank.*exceeds max_lora_rank"):
            workspace.load_slot_(1, adapter)


