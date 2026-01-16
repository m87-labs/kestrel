"""Multi-slot LoRA workspace for CUDA-graph-compatible mixed-adapter inference.

This module provides fixed-address GPU tensors that hold LoRA weights for multiple
adapters simultaneously. The workspace is allocated once at engine creation and
never reallocated, ensuring CUDA graph stability.

Slot 0 represents "no LoRA". Active adapters are loaded into slots 1..max_slots-1.

Dense vs MoE Workspace Layout
-----------------------------
Dense layers use direct slot indexing: workspace shape is [max_slots, ...] and
slot N maps to index N. Slot 0 contains zeros.

MoE layers use "super-expert" indexing where each (slot, expert) pair is a unique
index.

To maximize slot capacity, MoE workspaces exclude slot 0 entirely:
- Workspace shape: [(max_slots-1) * num_experts, ...]
- Slot N (N >= 1) maps to indices [(N-1)*num_experts, N*num_experts)
- Slot 0 tokens are filtered out via sentinel values in moe_align_block_size
  (expert IDs >= num_super_experts are skipped by the kernel)

This allows max_slots usable adapter slots (1 through max_slots-1).
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .config import TextConfig
from .lora import LoRA


@dataclass(frozen=True)
class DenseLoRALayerWorkspace:
    """Fixed-address LoRA weight buffers for a single dense MLP layer.

    Tensor layout treats slots as "experts" for compatibility with the MoE LoRA
    kernel (top_k=1, topk_weights=1).

    Shapes:
        up_a:   [max_slots, max_rank, d_model]
        up_b:   [max_slots, d_ffn,    max_rank]
        down_a: [max_slots, max_rank, d_ffn]
        down_b: [max_slots, d_model,  max_rank]
    """

    up_a: torch.Tensor
    up_b: torch.Tensor
    down_a: torch.Tensor
    down_b: torch.Tensor


@dataclass
class MoELoRALayerWorkspace:
    """Fixed-address LoRA weight buffers for a single MoE layer.

    Uses flattened "super-expert" indexing with slot 0 excluded (see module docstring).
    Slot N (N >= 1) maps to super-expert indices [(N-1)*num_experts, N*num_experts).

    Shapes:
        up_a:   [(max_slots-1) * num_experts, max_rank_per_expert, d_model]
        up_b:   [(max_slots-1) * num_experts, d_expert * 2,        max_rank_per_expert]
        down_a: [(max_slots-1) * num_experts, max_rank_per_expert, d_expert]
        down_b: [(max_slots-1) * num_experts, d_model,             max_rank_per_expert]

    """

    up_a: torch.Tensor
    up_b: torch.Tensor
    down_a: torch.Tensor
    down_b: torch.Tensor
    num_experts: int
    stream: torch.cuda.Stream | None = None


class TextLoRAWorkspace:
    """Multi-slot LoRA workspace for text model MLP layers.

    Allocates fixed-address GPU tensors for all dense and MoE layers. Slot 0 is
    reserved (always zero) for "no LoRA" sequences.

    Attributes:
        max_slots: Maximum number of concurrent adapters (including slot 0).
        max_rank: Maximum LoRA rank for dense layers.
        max_rank_per_expert: Maximum LoRA rank per expert for MoE layers.
        start_layer: First MoE layer index (dense layers are [0, start_layer)).
        dense: Per-layer workspace for dense MLP layers.
        moe: Per-layer workspace for MoE layers.
    """

    def __init__(
        self,
        text_config: TextConfig,
        max_slots: int,
        max_rank: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        lora_stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Allocate workspace tensors.

        Args:
            text_config: Model configuration (dimensions, layer counts, MoE config).
            max_slots: Number of slots to allocate (including slot 0).
            max_rank: Maximum LoRA rank for dense layers.
            device: Device to allocate tensors on.
            dtype: Data type for workspace tensors.
        """
        self.max_slots = max_slots
        self.max_rank = max_rank
        self.device = device
        self.dtype = dtype
        self.lora_stream = lora_stream

        moe_cfg = text_config.moe
        self.start_layer = moe_cfg.start_layer if moe_cfg else text_config.n_layers

        # Compute max_rank_per_expert for MoE layers
        if moe_cfg is not None:
            self.max_rank_per_expert = max_rank // moe_cfg.experts_per_token
            if self.max_rank_per_expert < 1:
                raise ValueError(
                    f"max_lora_rank ({max_rank}) must be >= experts_per_token "
                    f"({moe_cfg.experts_per_token})"
                )
        else:
            self.max_rank_per_expert = 0

        d_model = text_config.dim
        d_ffn = text_config.ff_dim

        # Allocate dense layer workspaces: [0, start_layer)
        self.dense: list[DenseLoRALayerWorkspace] = []
        for _ in range(self.start_layer):
            workspace = DenseLoRALayerWorkspace(
                up_a=torch.zeros(max_slots, max_rank, d_model, device=device, dtype=dtype),
                up_b=torch.zeros(max_slots, d_ffn, max_rank, device=device, dtype=dtype),
                down_a=torch.zeros(max_slots, max_rank, d_ffn, device=device, dtype=dtype),
                down_b=torch.zeros(max_slots, d_model, max_rank, device=device, dtype=dtype),
            )
            self.dense.append(workspace)

        # Allocate MoE layer workspaces: [start_layer, n_layers)
        # MoE workspaces exclude slot 0 to stay under vLLM's 1024 super-expert limit
        self.moe: list[MoELoRALayerWorkspace] = []
        if moe_cfg is not None:
            num_experts = moe_cfg.num_experts
            d_expert = moe_cfg.expert_inner_dim
            total_super_experts = (max_slots - 1) * num_experts

            for _ in range(text_config.n_layers - self.start_layer):
                workspace = MoELoRALayerWorkspace(
                    up_a=torch.zeros(
                        total_super_experts, self.max_rank_per_expert, d_model,
                        device=device, dtype=dtype
                    ),
                    up_b=torch.zeros(
                        total_super_experts, d_expert * 2, self.max_rank_per_expert,
                        device=device, dtype=dtype
                    ),
                    down_a=torch.zeros(
                        total_super_experts, self.max_rank_per_expert, d_expert,
                        device=device, dtype=dtype
                    ),
                    down_b=torch.zeros(
                        total_super_experts, d_model, self.max_rank_per_expert,
                        device=device, dtype=dtype
                    ),
                    num_experts=num_experts,
                    stream=lora_stream,
                )
                self.moe.append(workspace)

    def dense_layer(self, layer_idx: int) -> Optional[DenseLoRALayerWorkspace]:
        """Get workspace for a dense layer, or None if layer is MoE."""
        if layer_idx < len(self.dense):
            return self.dense[layer_idx]
        return None

    def moe_layer(self, layer_idx: int) -> Optional[MoELoRALayerWorkspace]:
        """Get workspace for a MoE layer, or None if layer is dense."""
        moe_idx = layer_idx - self.start_layer
        if 0 <= moe_idx < len(self.moe):
            return self.moe[moe_idx]
        return None

    def clear_slot_(self, slot: int) -> None:
        """Zero out all weights in a slot.

        Args:
            slot: Slot index to clear. Must be >= 1 (slot 0 is always zero).

        Raises:
            ValueError: If slot is 0 (reserved) or out of range.
        """
        if slot == 0:
            raise ValueError("Slot 0 is reserved and cannot be modified")
        if slot < 0 or slot >= self.max_slots:
            raise ValueError(f"Slot {slot} out of range [1, {self.max_slots})")

        for layer in self.dense:
            layer.up_a[slot].zero_()
            layer.up_b[slot].zero_()
            layer.down_a[slot].zero_()
            layer.down_b[slot].zero_()

        for layer in self.moe:
            # MoE uses (slot-1) indexing since slot 0 is excluded from workspace
            start = (slot - 1) * layer.num_experts
            end = start + layer.num_experts
            layer.up_a[start:end].zero_()
            layer.up_b[start:end].zero_()
            layer.down_a[start:end].zero_()
            layer.down_b[start:end].zero_()

    def load_slot_(self, slot: int, adapter: LoRA) -> None:
        """Load adapter weights into a slot.

        Copies weights from the adapter into the workspace slot. The adapter's rank
        may be smaller than max_rank; unused dimensions are zero-filled.

        Args:
            slot: Slot index to load into. Must be >= 1.
            adapter: LoRA adapter to copy from (must be on same device/dtype).

        Raises:
            ValueError: If slot is 0 or out of range, or adapter rank > max_rank.
        """
        if slot == 0:
            raise ValueError("Slot 0 is reserved and cannot be modified")
        if slot < 0 or slot >= self.max_slots:
            raise ValueError(f"Slot {slot} out of range [1, {self.max_slots})")

        text_lora = adapter.text
        if text_lora.rank > self.max_rank:
            raise ValueError(
                f"Adapter rank ({text_lora.rank}) exceeds max_lora_rank ({self.max_rank})"
            )

        # Clear the slot first (handles rank < max_rank case)
        self.clear_slot_(slot)

        adapter_rank = text_lora.rank

        # Copy dense layer weights
        for layer_idx, layer in enumerate(self.dense):
            adapter_layer = text_lora.get_dense_lora(layer_idx)
            if adapter_layer is None:
                raise ValueError(
                    f"Adapter missing dense LoRA for layer {layer_idx} "
                    f"(expected {len(self.dense)} dense layers)"
                )
            # Copy with rank slicing: adapter may have smaller rank
            layer.up_a[slot, :adapter_rank, :].copy_(adapter_layer.up_a)
            layer.up_b[slot, :, :adapter_rank].copy_(adapter_layer.up_b)
            layer.down_a[slot, :adapter_rank, :].copy_(adapter_layer.down_a)
            layer.down_b[slot, :, :adapter_rank].copy_(adapter_layer.down_b)

        # Copy MoE layer weights
        adapter_rank_per_expert = text_lora.rank_per_expert
        for moe_idx, layer in enumerate(self.moe):
            layer_idx = self.start_layer + moe_idx
            adapter_layer = text_lora.get_moe_lora(layer_idx)
            if adapter_layer is None:
                raise ValueError(
                    f"Adapter missing MoE LoRA for layer {layer_idx} "
                    f"(expected {len(self.moe)} MoE layers starting at {self.start_layer})"
                )

            # MoE uses (slot-1) indexing since slot 0 is excluded from workspace
            for expert_id in range(layer.num_experts):
                ws_idx = (slot - 1) * layer.num_experts + expert_id
                # Copy with rank slicing
                layer.up_a[ws_idx, :adapter_rank_per_expert, :].copy_(
                    adapter_layer.up_a[expert_id]
                )
                layer.up_b[ws_idx, :, :adapter_rank_per_expert].copy_(
                    adapter_layer.up_b[expert_id]
                )
                layer.down_a[ws_idx, :adapter_rank_per_expert, :].copy_(
                    adapter_layer.down_a[expert_id]
                )
                layer.down_b[ws_idx, :, :adapter_rank_per_expert].copy_(
                    adapter_layer.down_b[expert_id]
                )


# -----------------------------------------------------------------------------
# Slot allocator
# -----------------------------------------------------------------------------


class AdapterSlotManager:
    """Manages adapter-to-slot mapping with reference counting.

    Handles slot allocation/reuse for adapters:
    - When an adapter is first requested, allocates a free slot
    - When the same adapter is requested again, reuses the existing slot (refcount++)
    - When a slot's refcount drops to zero, returns it to the free pool

    Slot 0 is reserved for "no LoRA" and is never allocated or refcounted.

    Thread-safety: This class is NOT thread-safe. It should only be called from
    the scheduler thread (single owner pattern).
    """

    def __init__(self, max_slots: int) -> None:
        """Initialize the slot manager.

        Args:
            max_slots: Total number of slots (including slot 0).
        """
        if max_slots < 2:
            raise ValueError("max_slots must be >= 2 (slot 0 is reserved)")

        self._max_slots = max_slots
        self._adapter_to_slot: dict[str, int] = {}
        self._slot_to_adapter: list[str | None] = [None] * max_slots
        self._refcounts: list[int] = [0] * max_slots
        # Free slots as a stack, initialized to [max_slots-1, ..., 2, 1]
        # Slot 0 is never in this list
        self._free_slots: list[int] = list(range(max_slots - 1, 0, -1))

    @property
    def max_slots(self) -> int:
        return self._max_slots

    def acquire(self, adapter_id: str) -> tuple[int, bool]:
        """Acquire a slot for an adapter, allocating if necessary.

        If the adapter is already resident, reuses the existing slot and increments
        the refcount. Otherwise, allocates a new slot from the free pool.

        Args:
            adapter_id: Identifier for the adapter.

        Returns:
            Tuple of (slot, is_new) where:
            - slot: The slot index assigned to this adapter
            - is_new: True if this is a freshly allocated slot (caller must load weights)

        Raises:
            RuntimeError: If no free slots are available.
        """
        # Check if adapter is already resident
        if adapter_id in self._adapter_to_slot:
            slot = self._adapter_to_slot[adapter_id]
            self._refcounts[slot] += 1
            return slot, False

        # Need to allocate a new slot
        if not self._free_slots:
            # Provide diagnostic info for debugging
            active_slots = [
                (slot, self._slot_to_adapter[slot], self._refcounts[slot])
                for slot in range(1, self._max_slots)
                if self._refcounts[slot] > 0
            ]
            raise RuntimeError(
                f"Out of LoRA slots: all {self._max_slots - 1} slots are in use. "
                f"Active slots: {active_slots}"
            )

        slot = self._free_slots.pop()
        self._adapter_to_slot[adapter_id] = slot
        self._slot_to_adapter[slot] = adapter_id
        self._refcounts[slot] = 1
        return slot, True

    def release(self, slot: int) -> None:
        """Release a reference to a slot.

        Decrements the refcount. When it reaches zero, the slot is returned to
        the free pool and the adapter mapping is cleared.

        Args:
            slot: The slot to release.

        Raises:
            ValueError: If slot is 0, out of range, or has no references.
        """
        if slot == 0:
            # Slot 0 is "no LoRA" - no refcounting needed
            return

        if slot < 0 or slot >= self._max_slots:
            raise ValueError(f"Slot {slot} out of range [1, {self._max_slots})")

        if self._refcounts[slot] <= 0:
            raise ValueError(f"Slot {slot} has no references to release")

        self._refcounts[slot] -= 1

        if self._refcounts[slot] == 0:
            # Return slot to free pool
            adapter_id = self._slot_to_adapter[slot]
            if adapter_id is not None:
                del self._adapter_to_slot[adapter_id]
            self._slot_to_adapter[slot] = None
            self._free_slots.append(slot)

    def release_on_error(self, slot: int) -> None:
        """Release a freshly allocated slot after a loading error.

        This is for cleanup when acquire() succeeded but the subsequent load failed.
        Only call this for slots that were just allocated (is_new=True from acquire).

        Args:
            slot: The slot to release.
        """
        # Same as release(), but semantically different (error cleanup vs normal release)
        self.release(slot)

    def get_adapter_id(self, slot: int) -> str | None:
        """Get the adapter_id for a slot, or None if slot is empty/reserved."""
        if slot < 0 or slot >= self._max_slots:
            return None
        return self._slot_to_adapter[slot]

    def get_slot(self, adapter_id: str) -> int | None:
        """Get the slot for an adapter_id, or None if not resident."""
        return self._adapter_to_slot.get(adapter_id)

    def refcount(self, slot: int) -> int:
        """Get the current refcount for a slot."""
        if slot < 0 or slot >= self._max_slots:
            return 0
        return self._refcounts[slot]

    def num_free_slots(self) -> int:
        """Get the number of available slots."""
        return len(self._free_slots)


__all__ = [
    "DenseLoRALayerWorkspace",
    "MoELoRALayerWorkspace",
    "TextLoRAWorkspace",
    "AdapterSlotManager",
]
