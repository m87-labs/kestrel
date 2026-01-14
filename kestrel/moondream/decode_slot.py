"""Per-slot resources for pipelined decoding.

Each DecodeSlot bundles the GPU resources needed for one decode step:
- Pinned host buffers for H2D metadata copies (batch_idx, input_pos, lora_slot_ids)
- Per-slot FA3 paged-KV metadata buffers (page_table, seqused_k)
- GPU staging buffers for sampled outputs
- Forward output buffers (logits, hidden_last) for delayed sampling
- RenderBuffer for D2H copies
- CUDA graph workspace and captured graphs

With two slots, we can pipeline decode steps: while slot A's forward runs on the GPU,
slot B's D2H transfer completes and its outputs are committed on CPU. The slots
alternate in a ping-pong pattern.

Ownership model:
- MoondreamRuntime creates and owns both DecodeSlots
- All slots share a single decode compute stream (invariant I1)
- All slots share a single copy stream (simpler ordering)
- Scheduler receives slot_id and looks up the slot via runtime.decode_slots[slot_id]
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from kestrel.utils import CpuGpuBuffer


@dataclass
class DecodeMetaBuffers:
    """Per-slot pinned host metadata buffers for H2D copies.

    These buffers hold the batch metadata (batch indices, positions, LoRA slots)
    that are copied to GPU at the start of each decode step. Each slot needs its
    own buffers to prevent the CPU from overwriting the pinned source while a
    previous step's H2D copy is still in flight.

    The pinned memory ensures DMA transfers don't require CPU involvement after
    the copy is initiated.
    """

    batch_idx: CpuGpuBuffer  # int64 [max_batch] - sequence batch indices
    input_pos: CpuGpuBuffer  # int32 [max_batch] - token positions
    lora_slot_ids: CpuGpuBuffer  # int32 [max_batch] - LoRA slot assignments


@dataclass
class DecodeSlot:
    """Bundled resources for one ping-pong decode slot.

    A slot is "in use" if referenced by any of:
    - An entry in PipelineState.batch_queue (sampled, awaiting completion)
    - PipelineState.forward_handle (forward dispatched, not yet sampled)
    - PipelineState.completing_step (popped from queue, completion in progress)

    The two-phase completion model (pop_oldest -> on_step_completed) ensures
    a slot is not reused until the scheduler has finished reading from its
    pinned host buffers.

    Attributes:
        slot_id: The slot index (0 or 1).
        meta: Per-slot pinned host buffers for H2D metadata copies.
        render: Per-slot RenderBuffer for D2H output copies.
        compute_stream: Reference to the shared decode compute stream.
            All decode forwards across both slots serialize on this stream
            to preserve sequential token dependencies (invariant I1).

        fa3_page_table: Per-slot page table rows for FA3 paged decode.
        fa3_seqused_k: Per-slot per-sequence KV lengths for FA3 paged decode.

        # GPU staging buffers for sampled outputs (per-slot to avoid clobbering)
        sampled_ids: GPU buffer for sampled token IDs.
        coord_staging: GPU buffer for decoded coord values.
        size_staging: GPU buffer for decoded size values.

        # Forward output buffers (per-slot for delayed sampling in constrained mode)
        logits: GPU buffer for forward output logits.
        hidden_last: GPU buffer for last hidden states (spatial decode).

        # Decode input staging (per-slot, also used as CUDA graph capture buffers)
        decode_token_ids: GPU buffer for decode token inputs.
        decode_coord_values: GPU buffer for decode coord inputs.
        decode_size_values: GPU buffer for decode size inputs.

        # CUDA graphs (per-slot, optional)
        cuda_graphs: Captured CUDA graphs keyed by batch size, or None if disabled.
            Graphs are captured using this slot's buffers (decode_*, meta.*, logits,
            hidden_last), so replay reads/writes directly to slot buffers.
    """

    slot_id: int
    meta: DecodeMetaBuffers
    render: object  # RenderBuffer (import deferred to avoid circular import)
    compute_stream: torch.cuda.Stream

    fa3_page_table: Tensor
    fa3_seqused_k: Tensor

    # GPU staging for sampled outputs
    sampled_ids: Tensor
    coord_staging: Tensor
    size_staging: Tensor

    # Forward outputs (also used as graph output buffers)
    logits: Tensor
    hidden_last: Tensor

    # Decode input staging (also used as graph input buffers)
    decode_token_ids: Tensor
    decode_coord_values: Tensor
    decode_size_values: Tensor

    # CUDA graphs (optional) - captured using this slot's buffers
    cuda_graphs: Optional[dict[int, torch.cuda.CUDAGraph]] = None

    # Pre-allocated events for decode-step synchronization (avoids per-step allocation).
    #
    # - step_done_event: recorded once per step when per-slot staging buffers are ready
    #   for D2H transfer (copy stream waits on this).
    # - commit_done_event: recorded once per step after writes to shared scheduler
    #   buffers (e.g. `_pending_*`) complete; used to safely release/reuse batch indices.
    step_done_event: torch.cuda.Event = None  # type: ignore[assignment]
    commit_done_event: torch.cuda.Event = None  # type: ignore[assignment]


def create_decode_slot(
    slot_id: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_slots: int,
    max_seq_len: int,
    page_size: int,
    vocab_size: int,
    hidden_dim: int,
    coord_dtype: torch.dtype,
    size_dtype: torch.dtype,
    compute_stream: torch.cuda.Stream,
    copy_stream: torch.cuda.Stream,
) -> DecodeSlot:
    """Create a DecodeSlot with all per-slot resources allocated.

    Args:
        slot_id: The slot index (0 or 1).
        device: CUDA device for GPU tensors.
        dtype: Model dtype (e.g., float16, bfloat16).
        kv_dtype: KV cache dtype (may differ if using FP8).
        max_batch_slots: Maximum batch slots for buffer allocation (includes reserved slot 0).
        max_seq_len: Maximum sequence length.
        page_size: KV cache page size.
        vocab_size: Vocabulary size for logits buffer.
        hidden_dim: Hidden dimension for hidden_last buffer.
        coord_dtype: Dtype for coord values.
        size_dtype: Dtype for size values.
        compute_stream: Shared decode compute stream (same for both slots).
        copy_stream: Shared copy stream for D2H transfers.
        use_cuda_graphs: Whether to enable CUDA graph capture for this slot.

    Returns:
        A fully initialized DecodeSlot.
    """
    # Deferred import to avoid circular dependency at module load time.
    # runtime.py -> decode_slot.py -> scheduler/transfer.py -> scheduler.py -> runtime.py
    from kestrel.scheduler.transfer import RenderBuffer

    # Per-slot pinned host buffers for H2D metadata
    meta = DecodeMetaBuffers(
        batch_idx=CpuGpuBuffer(
            max_batch_slots,
            dtype=torch.int64,
            device=device,
            pin_memory=True,
        ),
        input_pos=CpuGpuBuffer(
            max_batch_slots,
            dtype=torch.int32,
            device=device,
            pin_memory=True,
        ),
        lora_slot_ids=CpuGpuBuffer(
            max_batch_slots,
            dtype=torch.int32,
            device=device,
            pin_memory=True,
        ),
    )

    # Per-slot RenderBuffer for D2H copies (shares the copy stream)
    render = RenderBuffer(
        max_batch_slots,
        device,
        coord_dtype=coord_dtype,
        size_dtype=size_dtype,
        copy_stream=copy_stream,
    )

    n_pages = max_seq_len // page_size
    fa3_page_table = torch.empty(
        (max_batch_slots, n_pages),
        dtype=torch.int32,
        device=device,
    )
    fa3_seqused_k = torch.empty(
        (max_batch_slots,),
        dtype=torch.int32,
        device=device,
    )

    # GPU staging for sampled outputs
    sampled_ids = torch.empty(
        (max_batch_slots,),
        dtype=torch.long,
        device=device,
    )
    coord_staging = torch.empty(
        (max_batch_slots, 1),
        dtype=coord_dtype,
        device=device,
    )
    size_staging = torch.empty(
        (max_batch_slots, 2),
        dtype=size_dtype,
        device=device,
    )

    # Forward output buffers
    logits = torch.empty(
        (max_batch_slots, vocab_size),
        dtype=dtype,
        device=device,
    )
    hidden_last = torch.empty(
        (max_batch_slots, hidden_dim),
        dtype=dtype,
        device=device,
    )

    # Decode input staging
    decode_token_ids = torch.empty(
        (max_batch_slots,),
        dtype=torch.long,
        device=device,
    )
    decode_coord_values = torch.empty(
        (max_batch_slots, 1),
        dtype=coord_dtype,
        device=device,
    )
    decode_size_values = torch.empty(
        (max_batch_slots, 2),
        dtype=size_dtype,
        device=device,
    )

    # Pre-allocated events for decode-step synchronization
    step_done_event = torch.cuda.Event()
    commit_done_event = torch.cuda.Event()

    return DecodeSlot(
        slot_id=slot_id,
        meta=meta,
        render=render,
        compute_stream=compute_stream,
        fa3_page_table=fa3_page_table,
        fa3_seqused_k=fa3_seqused_k,
        sampled_ids=sampled_ids,
        coord_staging=coord_staging,
        size_staging=size_staging,
        logits=logits,
        hidden_last=hidden_last,
        decode_token_ids=decode_token_ids,
        decode_coord_values=decode_coord_values,
        decode_size_values=decode_size_values,
        cuda_graphs=None,  # Captured separately after slot creation
        step_done_event=step_done_event,
        commit_done_event=commit_done_event,
    )
