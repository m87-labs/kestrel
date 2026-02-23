"""Unit tests for lora_workspace module."""

import pytest
import torch

from kestrel.moondream.config import TextConfig, TextMoeConfig
from kestrel.moondream.lora import LoRA, TextLoRA, TextLoRAConfig
from kestrel.moondream.lora_workspace import (
    AdapterSlotManager,
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
        # MoE workspace excludes slot 0: size is (max_slots-1) * num_experts
        expected_rank_per_expert = max_rank // moe_cfg.experts_per_token
        total_super_experts = (max_slots - 1) * moe_cfg.num_experts

        for layer in workspace.moe:
            assert layer.num_experts == moe_cfg.num_experts
            assert layer.up_a.shape == (
                total_super_experts,
                expected_rank_per_expert,
                moe_config.dim,
            )
            assert layer.up_b.shape == (
                total_super_experts,
                moe_cfg.expert_inner_dim * 2,
                expected_rank_per_expert,
            )
            assert layer.down_a.shape == (
                total_super_experts,
                expected_rank_per_expert,
                moe_cfg.expert_inner_dim,
            )
            assert layer.down_b.shape == (
                total_super_experts,
                moe_config.dim,
                expected_rank_per_expert,
            )

    def test_slot_zero_is_zeros(self, moe_config: TextConfig, device: torch.device):
        """Slot 0 is initialized to zeros for dense layers.

        Note: MoE workspace excludes slot 0 entirely (handled via sentinel filtering),
        so we only check dense layers here.
        """
        workspace = TextLoRAWorkspace(moe_config, max_slots=4, max_rank=8, device=device)

        for layer in workspace.dense:
            assert torch.all(layer.up_a[0] == 0)
            assert torch.all(layer.up_b[0] == 0)
            assert torch.all(layer.down_a[0] == 0)
            assert torch.all(layer.down_b[0] == 0)

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
            # MoE uses (slot-1) indexing since slot 0 is excluded
            start = (1 - 1) * moe_cfg.num_experts
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
            # MoE uses (slot-1) indexing since slot 0 is excluded
            start = (1 - 1) * moe_cfg.num_experts
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
        # MoE uses (slot-1) indexing since slot 0 is excluded
        for moe_idx, layer in enumerate(workspace.moe):
            layer_idx = workspace.start_layer + moe_idx
            adapter_layer = adapter.text.get_moe_lora(layer_idx)
            assert adapter_layer is not None

            for expert_id in range(moe_cfg.num_experts):
                ws_idx = (2 - 1) * moe_cfg.num_experts + expert_id
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

    def test_moe_rank_not_divisible_by_experts_per_token_raises(
        self, moe_config: TextConfig, device: torch.device
    ):
        """MoE LoRA rank must be divisible by experts_per_token."""
        with pytest.raises(ValueError, match="must be divisible by experts_per_token"):
            create_test_adapter(moe_config, rank=3, device=device)

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


# -----------------------------------------------------------------------------
# AdapterSlotManager tests
# -----------------------------------------------------------------------------


class TestAdapterSlotManager:
    """Tests for AdapterSlotManager."""

    def test_init_creates_free_slots(self):
        """Initialization creates correct number of free slots."""
        manager = AdapterSlotManager(max_slots=5)
        assert manager.max_slots == 5
        assert manager.num_free_slots() == 4  # slots 1-4, slot 0 reserved

    def test_init_requires_at_least_2_slots(self):
        """Raises ValueError if max_slots < 2."""
        with pytest.raises(ValueError, match="max_slots must be >= 2"):
            AdapterSlotManager(max_slots=1)

    def test_acquire_allocates_new_slot(self):
        """acquire() allocates a new slot for unknown adapter."""
        manager = AdapterSlotManager(max_slots=4)
        slot, is_new = manager.acquire("adapter_a")

        assert slot >= 1  # Slot 0 is reserved
        assert is_new is True
        assert manager.refcount(slot) == 1
        assert manager.get_adapter_id(slot) == "adapter_a"
        assert manager.get_slot("adapter_a") == slot
        assert manager.num_free_slots() == 2  # Started with 3 free, used 1

    def test_acquire_reuses_existing_slot(self):
        """acquire() reuses slot and increments refcount for existing adapter."""
        manager = AdapterSlotManager(max_slots=4)

        slot1, is_new1 = manager.acquire("adapter_a")
        slot2, is_new2 = manager.acquire("adapter_a")

        assert slot1 == slot2
        assert is_new1 is True
        assert is_new2 is False
        assert manager.refcount(slot1) == 2

    def test_acquire_different_adapters_get_different_slots(self):
        """Different adapters get different slots."""
        manager = AdapterSlotManager(max_slots=5)

        slot_a, _ = manager.acquire("adapter_a")
        slot_b, _ = manager.acquire("adapter_b")
        slot_c, _ = manager.acquire("adapter_c")

        assert len({slot_a, slot_b, slot_c}) == 3  # All different
        assert manager.num_free_slots() == 1  # Started with 4, used 3

    def test_acquire_raises_when_out_of_slots(self):
        """acquire() raises RuntimeError when no free slots."""
        manager = AdapterSlotManager(max_slots=3)  # Only slots 1, 2 available

        manager.acquire("adapter_a")
        manager.acquire("adapter_b")

        with pytest.raises(RuntimeError, match="Out of LoRA slots"):
            manager.acquire("adapter_c")

    def test_release_decrements_refcount(self):
        """release() decrements refcount."""
        manager = AdapterSlotManager(max_slots=4)

        slot, _ = manager.acquire("adapter_a")
        manager.acquire("adapter_a")  # refcount = 2
        assert manager.refcount(slot) == 2

        manager.release(slot)
        assert manager.refcount(slot) == 1
        assert manager.get_adapter_id(slot) == "adapter_a"  # Still mapped

    def test_release_frees_slot_when_refcount_zero(self):
        """release() returns slot to free pool when refcount hits 0."""
        manager = AdapterSlotManager(max_slots=4)

        slot, _ = manager.acquire("adapter_a")
        initial_free = manager.num_free_slots()

        manager.release(slot)

        assert manager.refcount(slot) == 0
        assert manager.get_adapter_id(slot) is None
        assert manager.get_slot("adapter_a") is None
        assert manager.num_free_slots() == initial_free + 1

    def test_release_slot_zero_is_noop(self):
        """release(0) is a no-op (slot 0 = no LoRA)."""
        manager = AdapterSlotManager(max_slots=4)
        manager.release(0)  # Should not raise

    def test_release_out_of_range_raises(self):
        """release() raises ValueError for out-of-range slots."""
        manager = AdapterSlotManager(max_slots=4)

        with pytest.raises(ValueError, match="out of range"):
            manager.release(4)

        with pytest.raises(ValueError, match="out of range"):
            manager.release(-1)

    def test_release_unallocated_slot_raises(self):
        """release() raises ValueError if slot has no references."""
        manager = AdapterSlotManager(max_slots=4)
        slot, _ = manager.acquire("adapter_a")
        manager.release(slot)  # refcount = 0

        with pytest.raises(ValueError, match="has no references"):
            manager.release(slot)

    def test_slot_can_be_reused_after_release(self):
        """Released slots can be allocated to new adapters."""
        manager = AdapterSlotManager(max_slots=3)  # Only slots 1, 2 available

        slot_a, _ = manager.acquire("adapter_a")
        slot_b, _ = manager.acquire("adapter_b")
        assert manager.num_free_slots() == 0

        # Release adapter_a
        manager.release(slot_a)
        assert manager.num_free_slots() == 1

        # New adapter can use the freed slot
        slot_c, is_new = manager.acquire("adapter_c")
        assert is_new is True
        assert slot_c == slot_a  # Reused the freed slot
        assert manager.get_adapter_id(slot_c) == "adapter_c"

    def test_release_on_error(self):
        """release_on_error() behaves like release()."""
        manager = AdapterSlotManager(max_slots=4)

        slot, _ = manager.acquire("adapter_a")
        initial_free = manager.num_free_slots()

        manager.release_on_error(slot)

        assert manager.refcount(slot) == 0
        assert manager.num_free_slots() == initial_free + 1

    def test_multiple_sequences_sharing_adapter(self):
        """Multiple sequences using the same adapter share one slot."""
        manager = AdapterSlotManager(max_slots=4)

        # Three sequences all using the same adapter
        slot1, is_new1 = manager.acquire("shared_adapter")
        slot2, is_new2 = manager.acquire("shared_adapter")
        slot3, is_new3 = manager.acquire("shared_adapter")

        assert slot1 == slot2 == slot3
        assert is_new1 is True
        assert is_new2 is False
        assert is_new3 is False
        assert manager.refcount(slot1) == 3

        # Release two sequences
        manager.release(slot1)
        manager.release(slot1)
        assert manager.refcount(slot1) == 1
        assert manager.get_adapter_id(slot1) == "shared_adapter"

        # Release the last one
        manager.release(slot1)
        assert manager.refcount(slot1) == 0
        assert manager.get_adapter_id(slot1) is None
