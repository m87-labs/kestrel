"""Per-slot resources for pipelined decoding.

Each DecodeSlot bundles the GPU resources needed for one decode step:
- Pinned host buffers for H2D metadata copies (batch_idx, input_pos, lora_slot_ids)
- Per-slot paged-KV metadata buffers (page table, KV sequence lengths)
- GPU staging buffers for sampled outputs
- Forward output buffers (logits, hidden_last) for delayed sampling
- RenderBuffer for D2H copies
- CUDA graph input/output workspace

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
from typing import Any

import torch
from torch import Tensor

from kestrel.device import make_event
from kestrel.utils import CpuGpuBuffer, PackedBuffer


class DecodeMetaBuffers:
    """Per-slot pinned host metadata buffers for H2D copies.

    The per-step decode inputs the scheduler stages every step — ``batch_idx``,
    ``input_pos`` and ``lora_slot_ids`` — share one :class:`PackedBuffer`, so
    they upload in a single H2D copy (``copy_inputs_to_gpu``) instead of three.
    They are exposed as :class:`PackedField` views, so call sites read/write
    ``.np``/``.cpu``/``.gpu`` exactly as before. The MoE-LoRA ``active_*``
    buffers are staged separately (only on the LoRA path), so they stay
    individual CpuGpuBuffers.

    Each slot needs its own buffers to prevent the CPU from overwriting the
    pinned source while a previous step's H2D copy is still in flight.
    """

    def __init__(self, *, max_batch_slots: int, device: torch.device) -> None:
        self._inputs = PackedBuffer(
            [
                ("batch_idx", (max_batch_slots,), torch.int64),  # sequence batch idx
                ("input_pos", (max_batch_slots,), torch.int32),  # token positions
                ("lora_slot_ids", (max_batch_slots,), torch.int32),  # logical adapter slots
                ("routed_storage_ids", (max_batch_slots,), torch.int32),
            ],
            device=device,
            pin_memory=True,
        )
        # token-major MoE LoRA ids
        self.active_token_ids = CpuGpuBuffer(
            max_batch_slots, dtype=torch.int32, device=device, pin_memory=True
        )
        # active MoE LoRA ids
        self.active_lora_ids = CpuGpuBuffer(
            max_batch_slots, dtype=torch.int32, device=device, pin_memory=True
        )
        # int32 [max_loras + 4]; max_loras == max_batch_slots - 1 (slot 0 == no
        # LoRA), so + 3 here. Layout: [0] active LoRA count, [1] active max rank,
        # [2] active token count, [3:] per-active-LoRA token-start offsets plus
        # one sentinel end offset (so kernels read meta[3+i]..meta[4+i] with no
        # last-route special case).
        self.active_lora_meta = CpuGpuBuffer(
            max_batch_slots + 3, dtype=torch.int32, device=device, pin_memory=True
        )
        self.moe_lora_metadata: Any | None = None

    @property
    def batch_idx(self):
        return self._inputs.batch_idx

    @property
    def input_pos(self):
        return self._inputs.input_pos

    @property
    def lora_slot_ids(self):
        return self._inputs.lora_slot_ids

    @property
    def routed_storage_ids(self):
        return self._inputs.routed_storage_ids

    def copy_inputs_to_gpu(self) -> None:
        """Stage the per-row decode metadata to the GPU in one H2D copy."""
        self._inputs.copy_to_gpu()


@dataclass
class DecodeSlot:
    """Bundled resources for one ping-pong decode slot.

    A slot is "in use" if referenced by any of:
    - An entry in PipelineState.batch_queue (sampled, awaiting completion)
    - PipelineState.launch_handle (forward dispatched, not yet sampled)
    - PipelineState.committing_step (popped from queue, completion in progress)

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

        paged_kv_page_table: Per-slot page table rows for paged attention decode.
        paged_kv_seqlens_k: Per-slot per-sequence KV lengths for paged attention decode.

        # GPU staging buffers for sampled outputs (per-slot to avoid clobbering)
        sampled_ids: GPU buffer for sampled token IDs.
        sampled_logprobs: GPU buffer for sampled token logprobs.
        coord_staging: GPU buffer for decoded coord values.
        size_staging: GPU buffer for decoded size values.

        # Forward output buffers (per-slot for delayed sampling in constrained mode)
        logits: GPU buffer for forward output logits.
        hidden_last: GPU buffer for last hidden states (spatial decode).

        # Decode input staging (per-slot, also used as CUDA graph capture buffers)
        decode_token_ids: GPU buffer for decode token inputs.
        decode_coord_values: GPU buffer for decode coord inputs.
        decode_size_values: GPU buffer for decode size inputs.

        # CUDA graph input/output workspace
        These fixed-address buffers are used by DecodeGraphManager capture/replay.
    """

    slot_id: int
    meta: DecodeMetaBuffers
    render: object  # RenderBuffer (import deferred to avoid circular import)
    compute_stream: torch.cuda.Stream | None

    paged_kv_page_table: Tensor
    paged_kv_seqlens_k: Tensor

    # GPU staging for sampled outputs
    sampled_ids: Tensor
    sampled_logprobs: Tensor
    # GPU staging for Moondream's per-step spatial decode (post_sample
    # writes here, then a D2H copy lands in coord_cpu/size_cpu). Owned
    # by the runtime; the scheduler never touches them.
    coord_staging: Tensor  # [max_batch_slots, 1]
    size_staging: Tensor   # [max_batch_slots, 2]
    coord_cpu: Tensor      # pinned host [max_batch_slots, 1]
    size_cpu: Tensor       # pinned host [max_batch_slots, 2]
    aux_done_event: torch.cuda.Event  # signals coord/size D2H complete

    # Forward outputs (also used as graph output buffers)
    logits: Tensor
    hidden_last: Tensor

    # Decode input staging (also used as graph input buffers).
    decode_token_ids: Tensor
    decode_coord_values: Tensor  # [max_batch_slots, 1]
    decode_size_values: Tensor   # [max_batch_slots, 2]

    # Pre-allocated events for decode-step synchronization (avoids per-step allocation).
    #
    # - step_done_event: recorded once per step when per-slot staging buffers are ready
    #   for D2H transfer (copy stream waits on this).
    # - commit_done_event: recorded once per step after writes to shared scheduler
    #   buffers (e.g. `_pending_*`) complete; used to safely release/reuse batch indices.
    step_done_event: torch.cuda.Event = None  # type: ignore[assignment]
    commit_done_event: torch.cuda.Event = None  # type: ignore[assignment]

    # Constrained-decode mask: a per-row boolean "disallow" mask over the
    # vocabulary (True => force this token's logit to -inf). Built from
    # skill_state right after commit and uploaded async on the copy stream
    # during the forward; sampling waits on ``mask_ready_event`` then applies it
    # with a single masked_fill_, so the per-step mask H2D never blocks the
    # compute stream and there is no per-row apply loop.
    disallow_mask: CpuGpuBuffer = None  # type: ignore[assignment]
    mask_ready_event: torch.cuda.Event = None  # type: ignore[assignment]


def create_decode_slot(
    slot_id: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_slots: int,
    kv_cache_pages: int,
    vocab_size: int,
    hidden_dim: int,
    coord_dtype: torch.dtype,
    size_dtype: torch.dtype,
    compute_stream: torch.cuda.Stream | None,
    copy_stream: torch.cuda.Stream | None,
) -> DecodeSlot:
    """Create a DecodeSlot with all per-slot resources allocated.

    Args:
        slot_id: The slot index (0 or 1).
        device: CUDA device for GPU tensors.
        dtype: Model dtype (e.g., float16, bfloat16).
        max_batch_slots: Maximum batch slots for buffer allocation (includes reserved slot 0).
        kv_cache_pages: Total number of KV cache pages.
        vocab_size: Vocabulary size for logits buffer.
        hidden_dim: Hidden dimension for hidden_last buffer.
        coord_dtype: Dtype for coord values.
        size_dtype: Dtype for size values.
        compute_stream: Shared decode compute stream (same for both slots).
        copy_stream: Shared copy stream for D2H transfers.

    Returns:
        A fully initialized DecodeSlot.
    """
    # Deferred import to avoid circular dependency at module load time.
    # runtime.py -> decode_slot.py -> scheduler/transfer.py -> scheduler.py -> runtime.py
    from kestrel.scheduler.transfer import RenderBuffer

    # Per-slot pinned host buffers for H2D metadata. batch_idx/input_pos/
    # lora_slot_ids are packed for a single per-step H2D copy.
    meta = DecodeMetaBuffers(max_batch_slots=max_batch_slots, device=device)

    # Per-slot RenderBuffer for D2H copies of sampled ids + logprobs.
    # Moondream's coord/size aux values use a separate runtime-owned
    # D2H pipeline (coord_cpu/size_cpu + aux_done_event below).
    render = RenderBuffer(
        max_batch_slots,
        device,
        copy_stream=copy_stream,
    )

    paged_kv_page_table = torch.empty(
        (max_batch_slots, kv_cache_pages),
        dtype=torch.int32,
        device=device,
    )
    paged_kv_seqlens_k = torch.empty(
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
    sampled_logprobs = torch.empty(
        (max_batch_slots,),
        dtype=torch.float32,
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
    # Pinned host buffers for the runtime-owned coord/size D2H copy.
    # Pinned-memory hint is CUDA-only (MPS hits a dispatch stub even
    # with device='cpu').
    pin = device.type == "cuda"
    coord_cpu = torch.empty(
        (max_batch_slots, 1),
        dtype=coord_dtype,
        device="cpu",
        pin_memory=pin,
    )
    size_cpu = torch.empty(
        (max_batch_slots, 2),
        dtype=size_dtype,
        device="cpu",
        pin_memory=pin,
    )
    aux_done_event = make_event(device, enable_timing=False, blocking=False)

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

    # Pre-allocated events for decode-step synchronization. ``make_event``
    # returns a real ``torch.cuda.Event`` on CUDA, a no-op stand-in on MPS
    # (where the decode path serializes via the single implicit stream).
    step_done_event = make_event(device)
    commit_done_event = make_event(device)

    # Per-row boolean disallow mask over the vocabulary (True => -inf). Fixed
    # shape [batch, vocab], so a step's constraint can never overflow it (no
    # synchronous-H2D fall-back), and the apply is a single masked_fill_ with no
    # per-row loop. copy_to_gpu(batch) ships only the active rows each step.
    disallow_mask = CpuGpuBuffer(
        max_batch_slots,
        vocab_size,
        dtype=torch.bool,
        device=device,
        pin_memory=True,
    )
    mask_ready_event = make_event(device, enable_timing=False, blocking=False)

    return DecodeSlot(
        slot_id=slot_id,
        meta=meta,
        render=render,
        compute_stream=compute_stream,
        paged_kv_page_table=paged_kv_page_table,
        paged_kv_seqlens_k=paged_kv_seqlens_k,
        sampled_ids=sampled_ids,
        sampled_logprobs=sampled_logprobs,
        coord_staging=coord_staging,
        size_staging=size_staging,
        coord_cpu=coord_cpu,
        size_cpu=size_cpu,
        aux_done_event=aux_done_event,
        logits=logits,
        hidden_last=hidden_last,
        decode_token_ids=decode_token_ids,
        decode_coord_values=decode_coord_values,
        decode_size_values=decode_size_values,
        step_done_event=step_done_event,
        commit_done_event=commit_done_event,
        disallow_mask=disallow_mask,
        mask_ready_event=mask_ready_event,
    )
