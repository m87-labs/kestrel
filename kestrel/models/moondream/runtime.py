"""Moondream runtime with paged KV cache and optional image prefixes."""


import contextlib
import functools
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Sequence, cast

import warnings
import threading

import numpy as np
import torch
from torch import Tensor


from kestrel_kernels import get_runtime
from kestrel.config import RuntimeConfig
from kestrel.device import NoopEvent, empty_cache, get_device_capability, make_event, make_stream, set_device, stream_context
from kestrel.kv_cache import KVMemoryPool, PageTable, PagedKVCache
from kestrel.utils import CpuGpuBuffer
from kestrel.prefix_cache import (
    CacheNamespace,
    CacheToken,
    ImageToken,
    MatchResult,
    RadixPrefixCache,
    TreeNode,
)
from kestrel.runtime.decode_graph import DecodeGraphManager
from kestrel.runtime import (
    CoordToken,
    ExecutionShape,
    ImageMarker,
    PrefillClassification,
    PreparedSequence,
    RuntimeDecodeResult,
    SequenceState,
    SizeToken,
    TextToken,
    Token,
)
from kestrel.runtime.sampling import SamplingHooks
from kestrel.runtime.state import _CacheLookupResult
from kestrel.scheduler.spatial import compute_spatial_values

# ``kestrel.scheduler.tokens`` imports CoordToken/SizeToken/TextToken
# back from this module, so importing ``render_tokens_from_packed`` at
# top level here creates a partial-init cycle when ``tokens`` is loaded
# first. Imported lazily inside ``materialize_tokens`` below.

from kestrel.models.registry import get_spec

from .config import MoondreamConfig
from .image_preprocessor import ImagePreprocessor
from .model import MoondreamModel
from .weights import load_moondream_weights
from .text import (
    lm_head,
    text_decoder,
    text_encoder,
)
from .vision import (
    prepare_crops,
    prepare_crops_from_overlap,
    vision_encoder,
    vision_projection,
)
from .lora import LoRA
from .lora_workspace import AdapterSlotManager, TextLoRAWorkspace
from .image_crops import OverlapCropOutput, reconstruct_from_crops
from .region import (
    build_region_module,
    build_spatial_decode_tables,
    encode_coordinate,
    encode_size,
)
from .tokenizer import load_tokenizer
from ...seg_refiner import SegmentRefiner, _HAS_SEG_DEPS
from ...dense_lora import DenseLoRATorchMLPScratch, create_mlp_scratch
from .decode_slot import DecodeSlot, create_decode_slot


DEFAULT_MAX_TOKENS = 768
DEFAULT_INITIAL_DECODE_RESERVE_TOKENS = 16


@contextlib.contextmanager
def _disable_parameter_initialization():
    """Temporarily skip default parameter init during model construction.

    Runtime weights are loaded from checkpoint immediately after construction,
    so random reset_parameters work is redundant startup overhead.
    """

    patches = (
        (torch.nn.Linear, "reset_parameters"),
        (torch.nn.LayerNorm, "reset_parameters"),
        (torch.nn.Embedding, "reset_parameters"),
    )
    originals: list[tuple[type[torch.nn.Module], str, object]] = []
    for cls, method_name in patches:
        original = getattr(cls, method_name, None)
        if original is not None:
            originals.append((cls, method_name, original))
            setattr(cls, method_name, lambda self: None)
    try:
        yield
    finally:
        for cls, method_name, original in originals:
            setattr(cls, method_name, original)


def _count_image_markers(tokens) -> int:
    """Number of ImageMarker sentinels in a prompt-token sequence (one per
    multi-image chat image; each expands to image_prefix_length positions)."""
    return sum(1 for t in tokens if isinstance(t, ImageMarker))


class PrefillScratch:
    """Pre-allocated scratch buffers for prefill to avoid per-layer allocations."""

    def __init__(self, max_tokens: int, config, device: torch.device, dtype: torch.dtype):
        d = config.dim
        n_heads = config.n_heads
        head_dim = d // n_heads
        qkv_dim = d + 2 * (config.n_kv_heads * head_dim)
        n_experts = config.moe.num_experts if config.moe else 0
        tau_out_dim = 2 * n_heads

        self.qkv_out = torch.empty(max_tokens, qkv_dim, dtype=dtype, device=device)
        self.tok_qv_lin = torch.empty(max_tokens, tau_out_dim, dtype=dtype, device=device)
        self.router_logits = torch.empty(max_tokens, n_experts, dtype=dtype, device=device) if n_experts else None

    def ensure_size(self, tokens: int):
        """Grow buffers if needed (rare — only if tokens > initial max_tokens)."""
        if tokens <= self.qkv_out.size(0):
            return
        device, dtype = self.qkv_out.device, self.qkv_out.dtype
        self.qkv_out = torch.empty(tokens, self.qkv_out.size(1), dtype=dtype, device=device)
        self.tok_qv_lin = torch.empty(tokens, self.tok_qv_lin.size(1), dtype=dtype, device=device)
        if self.router_logits is not None:
            self.router_logits = torch.empty(tokens, self.router_logits.size(1), dtype=dtype, device=device)


class PrefillInputStaging:
    """Pinned host staging and persistent device buffers for prefill metadata."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_seq_length: int,
        max_lora_slots: int,
        coord_dtype: torch.dtype,
        size_dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.max_lora_slots = max_lora_slots
        self.max_prefill_tokens = max_batch_size * max_seq_length

        pin = pin_memory and device.type == "cuda"

        def staging_buffer(*size: int, dtype: torch.dtype) -> CpuGpuBuffer:
            return CpuGpuBuffer(
                *size,
                dtype=dtype,
                device=device,
                pin_memory=pin_memory,
                with_numpy=False,
                zero=False,
            )

        def host_buffer(*size: int, dtype: torch.dtype) -> Tensor:
            return torch.empty(*size, dtype=dtype, device="cpu", pin_memory=pin)

        token_shape = (max_batch_size, max_seq_length)
        coord_shape = (max_batch_size, max_seq_length, 1)
        size_shape = (max_batch_size, max_seq_length, 2)

        self.text_ids = staging_buffer(*token_shape, dtype=torch.long)
        self.coord_values = staging_buffer(*coord_shape, dtype=coord_dtype)
        self.size_values = staging_buffer(*size_shape, dtype=size_dtype)

        self.token_positions = {
            kind: staging_buffer(*token_shape, dtype=torch.long)
            for kind in ("text", "coord", "size")
        }

        self.position_template_cpu = torch.arange(max_seq_length, dtype=torch.long)
        self.position_ids = staging_buffer(self.max_prefill_tokens, dtype=torch.long)
        self.slot_batch_idx = staging_buffer(self.max_prefill_tokens, dtype=torch.int64)
        self.batch_idx_cpu = host_buffer(max_batch_size, dtype=torch.int64)
        self.last_token_positions = staging_buffer(max_batch_size, dtype=torch.long)
        self.paged_kv_seqlens_q = staging_buffer(max_batch_size, dtype=torch.int32)
        self.paged_kv_seqlens_k = staging_buffer(max_batch_size, dtype=torch.int32)

        self.lora_slot_ids = staging_buffer(1, dtype=torch.int32)
        self.token_lora_slot_ids_cpu = host_buffer(
            self.max_prefill_tokens, dtype=torch.int32
        )
        self.active_token_ids = staging_buffer(
            self.max_prefill_tokens, dtype=torch.int32
        )
        self.active_lora_ids = staging_buffer(max_lora_slots, dtype=torch.int32)
        self.active_lora_meta = staging_buffer(max_lora_slots + 4, dtype=torch.int32)

    def _check_row(self, row: int) -> None:
        if row < 0 or row >= self.max_batch_size:
            raise ValueError(
                f"Prefill row {row} exceeds max_batch_size={self.max_batch_size}"
            )

    def _check_seq_length(self, length: int, name: str) -> None:
        if length > self.max_seq_length:
            raise ValueError(
                f"{name} length {length} exceeds max_seq_length={self.max_seq_length}"
            )

    @staticmethod
    def _fill_1d_int(values_cpu: Tensor, values: Sequence[int]) -> None:
        for idx, value in enumerate(values):
            values_cpu[idx] = int(value)

    @staticmethod
    def _fill_1d_float(values_cpu: Tensor, values: Sequence[float]) -> None:
        for idx, value in enumerate(values):
            values_cpu[idx] = float(value)

    def stage_text_ids(self, row: int, text_ids: Sequence[int]) -> Tensor:
        self._check_row(row)
        n = len(text_ids)
        self._check_seq_length(n, "text id")
        cpu = self.text_ids.cpu[row, :n]
        self._fill_1d_int(cpu, text_ids)
        gpu = self.text_ids.gpu[row, :n]
        gpu.copy_(cpu, non_blocking=True)
        return gpu.view(1, n)

    def stage_coord_values(self, row: int, coord_values: Sequence[float]) -> Tensor:
        self._check_row(row)
        n = len(coord_values)
        self._check_seq_length(n, "coord")
        cpu_values = self.coord_values.cpu[row, :n, :]
        cpu = cpu_values[:, 0]
        self._fill_1d_float(cpu, coord_values)
        gpu = self.coord_values.gpu[row, :n, :]
        gpu.copy_(cpu_values, non_blocking=True)
        return gpu

    def stage_size_values(
        self, row: int, size_values: Sequence[tuple[float, float]]
    ) -> Tensor:
        self._check_row(row)
        n = len(size_values)
        self._check_seq_length(n, "size")
        cpu = self.size_values.cpu[row, :n, :]
        for idx, (width, height) in enumerate(size_values):
            cpu[idx, 0] = float(width)
            cpu[idx, 1] = float(height)
        gpu = self.size_values.gpu[row, :n, :]
        gpu.copy_(cpu, non_blocking=True)
        return gpu

    def stage_token_positions(
        self,
        row: int,
        positions: Sequence[int],
        *,
        kind: str,
    ) -> Tensor:
        self._check_row(row)
        n = len(positions)
        self._check_seq_length(n, f"{kind} position")
        buffer = self.token_positions.get(kind)
        if buffer is None:  # pragma: no cover - defensive
            raise ValueError(f"Unknown token position kind: {kind}")
        cpu = buffer.cpu[row, :n]
        gpu = buffer.gpu[row, :n]
        self._fill_1d_int(cpu, positions)
        gpu.copy_(cpu, non_blocking=True)
        return gpu

    def stage_batch_metadata(
        self,
        *,
        batch_indices: Sequence[int],
        seq_lens: Sequence[int],
        prompt_lens: Sequence[int],
        position_starts: Sequence[int],
        q_is_padded: bool,
        max_seq_len: int,
        batch_idx_gpu: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor]:
        batch_size = len(batch_indices)
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Prefill batch size {batch_size} exceeds max_batch_size={self.max_batch_size}"
            )
        if not (
            len(seq_lens)
            == len(prompt_lens)
            == len(position_starts)
            == batch_size
        ):
            raise ValueError("Prefill metadata sequences must have matching lengths")
        self._check_seq_length(max_seq_len, "prefill")

        metadata_shape = (batch_size, max_seq_len)
        position_cpu = self.position_ids.cpu_view(metadata_shape)
        slot_batch_idx_cpu = self.slot_batch_idx.cpu_view(metadata_shape)
        position_cpu.zero_()
        slot_batch_idx_cpu.zero_()

        for row, (batch_idx, seq_len, prompt_len, position_start) in enumerate(
            zip(batch_indices, seq_lens, prompt_lens, position_starts)
        ):
            self._check_seq_length(seq_len, "sequence")
            if position_start < 0 or position_start + seq_len > self.max_seq_length:
                raise ValueError(
                    f"Position range [{position_start}, {position_start + seq_len}) "
                    f"exceeds max_seq_length={self.max_seq_length}"
                )
            position_cpu[row, :seq_len].copy_(
                self.position_template_cpu[position_start : position_start + seq_len]
            )
            slot_batch_idx_cpu[row, :seq_len].fill_(int(batch_idx))
            self.batch_idx_cpu[row] = int(batch_idx)
            self.last_token_positions.cpu[row] = int(seq_len - 1)
            self.paged_kv_seqlens_k.cpu[row] = int(prompt_len)
            if q_is_padded:
                self.paged_kv_seqlens_q.cpu[row] = int(seq_len)

        position_ids = self.position_ids.copy_view_to_gpu(metadata_shape)
        slot_batch_idx = self.slot_batch_idx.copy_view_to_gpu(metadata_shape)
        batch_idx_gpu[:batch_size].copy_(
            self.batch_idx_cpu[:batch_size], non_blocking=True
        )
        last_token_positions = self.last_token_positions.copy_to_gpu(batch_size)
        paged_kv_seqlens_k = self.paged_kv_seqlens_k.copy_to_gpu(batch_size)
        paged_kv_seqlens_q = None
        if q_is_padded:
            paged_kv_seqlens_q = self.paged_kv_seqlens_q.copy_to_gpu(batch_size)
        return (
            position_ids,
            slot_batch_idx,
            paged_kv_seqlens_q,
            paged_kv_seqlens_k,
            last_token_positions,
        )

    def stage_lora_slot_ids(self, lora_slot: int) -> Tensor:
        self.lora_slot_ids.cpu[0] = int(lora_slot)
        return self.lora_slot_ids.copy_to_gpu()

    def fill_token_lora_slots(self, count: int, lora_slot: int) -> Tensor:
        if count > self.max_prefill_tokens:
            raise ValueError(
                f"LoRA token count {count} exceeds capacity {self.max_prefill_tokens}"
            )
        slots = self.token_lora_slot_ids_cpu[:count]
        slots.fill_(int(lora_slot))
        return slots

    def lora_metadata_buffers(
        self, *, token_count: int, max_loras: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if token_count > self.max_prefill_tokens:
            raise ValueError(
                f"LoRA token count {token_count} exceeds capacity {self.max_prefill_tokens}"
            )
        if max_loras > self.max_lora_slots:
            raise ValueError(
                f"LoRA count {max_loras} exceeds capacity {self.max_lora_slots}"
            )
        return (
            self.active_token_ids.cpu[:token_count],
            self.active_token_ids.gpu[:token_count],
            self.active_lora_ids.cpu[:max_loras],
            self.active_lora_ids.gpu[:max_loras],
            self.active_lora_meta.cpu[: max_loras + 4],
            self.active_lora_meta.gpu[: max_loras + 4],
        )


@dataclass(slots=True)
class PrefillSlot:
    """Per-prefill slot resources.

    Prefill work runs on the compute stream, but can be pipelined from the CPU
    side (e.g., commit token0 for request A while the GPU runs request B's
    prefill). Slots avoid clobbering per-prefill buffers when multiple prefills
    are in-flight.
    """

    slot_id: int
    batch_idx: Tensor
    # GPU staging + pinned host buffers for Moondream's spatial decode of
    # the first sampled token. Prefill's first token is text for plain
    # query, but skills like point/detect constrain it to coord/size, so
    # the spatial pipeline has to run for prefill too.
    coord_staging: Tensor       # [max_batch_size, 1]
    size_staging: Tensor        # [max_batch_size, 2]
    coord_cpu: Tensor           # pinned host [max_batch_size, 1]
    size_cpu: Tensor            # pinned host [max_batch_size, 2]
    # Type-annotated as the CUDA event for the common case; MPS/CPU paths
    # store ``NoopEvent`` instances at runtime.
    # ``step_done_event`` fires after sampled-id staging AND any post_sample
    # work (spatial decode + pending-pool updates). It's re-recorded
    # inside post_sample so the copy stream waits on the spatial writes.
    step_done_event: "torch.cuda.Event"
    commit_done_event: "torch.cuda.Event"
    aux_done_event: "torch.cuda.Event"   # signals coord/size D2H complete
    scratch: PrefillScratch | None = None
    input_staging: PrefillInputStaging | None = None


class _LayerPagedCache(torch.nn.Module):
    """Adapter that wires :class:`PagedKVCache` into the text blocks."""

    def __init__(
        self,
        page_table: PageTable,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        pool: KVMemoryPool,
        *,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.cache = PagedKVCache(
            page_table,
            n_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            pool=pool,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    def update(
        self,
        pos_ids: Tensor,
        k_val: Tensor,
        v_val: Tensor,
        *,
        slot_mapping: Tensor,
    ):
        if k_val.shape != v_val.shape:
            raise ValueError("k_val and v_val must match shape")

        input_pos = torch.atleast_2d(pos_ids)
        if input_pos.device != k_val.device:
            input_pos = input_pos.to(device=k_val.device)
        if input_pos.shape[0] != 1 and input_pos.shape[0] != k_val.shape[0]:
            raise ValueError(
                f"Unsupported position shape {pos_ids.shape} for batch size {k_val.shape[0]}"
            )

        seq_len = input_pos.shape[1]
        if k_val.shape[1] != seq_len:
            raise ValueError(
                f"KV sequence length {k_val.shape[1]} does not match position tensor {input_pos.shape}"
            )

        return self.cache.update(
            input_pos=input_pos,
            k_val=k_val,
            v_val=v_val,
            slot_mapping=slot_mapping,
        )


class MoondreamRuntime:
    supports_context_clamped_generation = True

    """High-level runtime for paged text-only Moondream inference."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        *,
        max_lora_rank: int | None = None,
        kv_pool: KVMemoryPool,
        compute_stream: torch.cuda.Stream | None,
    ) -> None:
        self._cfg = cfg
        self.device = cfg.resolved_device()
        self.execution_shape = ExecutionShape.AUTOREGRESSIVE
        # Speculative decoding is not wired for Moondream; decode one token per
        # step as before. See kestrel.runtime.spec / the spec-decode design doc.
        self.spec = None
        self.dtype = cfg.resolved_dtype()
        set_device(self.device)
        # Guards CUDA graph capture so other threads avoid device-wide sync during capture.
        self.graph_capture_lock = threading.RLock()

        self._spec = get_spec(cfg.model)
        self.config = MoondreamConfig.from_dict(deepcopy(self._spec.default_config))

        self._kv_layer_k_scales: list[float] | None = None
        self._kv_layer_v_scales: list[float] | None = None

        self.max_seq_length = self.config.text.max_context
        if self.max_seq_length % cfg.page_size != 0:
            raise ValueError("max_seq_length must be divisible by page_size")

        self.page_size = cfg.page_size
        if cfg.max_batch_size < 1:
            raise ValueError(
                "max_batch_size must be at least 1; batch_idx 0 is reserved internally."
            )
        # max_batch_size is the effective user-facing batch capacity.
        # We reserve batch_idx 0 for internal bookkeeping, so allocate +1 slot.
        # We also add +1 extra slot for prefill preparation, allowing prepare to
        # run even at max batch size (the prepared sequence claims the extra slot,
        # then launches when an active sequence completes).
        self.max_batch_size = cfg.max_batch_size
        self.max_batch_slots = cfg.max_batch_size + 2
        n_pages = cfg.kv_cache_pages

        # Create prefix cache if enabled
        self.prefix_cache: RadixPrefixCache | None = None
        if cfg.enable_prefix_cache:
            self.prefix_cache = RadixPrefixCache()

        # Compute stream for model GPU work. The engine passes one shared
        # stream when co-hosting multiple execution shapes.
        self._compute_stream = compute_stream

        self.page_table = PageTable(
            n_pages=n_pages,
            page_size=self.page_size,
            max_batch_size=self.max_batch_slots,
            device=str(self.device),
            prefix_cache=self.prefix_cache,
            h2d_stream=self._compute_stream,
        )
        self._kv_pool = kv_pool
        if self._kv_pool.device != self.device:
            raise ValueError(
                f"kv_pool.device ({self._kv_pool.device}) must match runtime "
                f"device ({self.device})"
            )

        construction_device = torch.device("meta")
        with _disable_parameter_initialization():
            self.model = MoondreamModel(
                self.config,
                dtype=self.dtype,
                device=construction_device,
                setup_caches=False,
            ).eval()
            self.region = build_region_module(
                self.config.region,
                self.dtype,
                device=construction_device,
            )

        # Materialize model storage directly on target device without initializing
        # temporary tensors on CPU.
        self.model.to_empty(device=self.device)
        self.region.to_empty(device=self.device)

        self.image_prefix_length = self.model.vision.pos_emb.shape[1]
        n_layers = self.config.text.n_layers
        captured_k_scales: list[Optional[float]] = [None] * n_layers
        captured_v_scales: list[Optional[float]] = [None] * n_layers

        def _capture_kv_scale(name: str, tensor: torch.Tensor) -> None:
            if not name.startswith("text_model.transformer.h."):
                return
            parts = name.split(".")
            if len(parts) < 6:
                return
            try:
                layer_idx = int(parts[3])
            except ValueError:
                return
            if not (0 <= layer_idx < n_layers):
                return
            if parts[4] != "kv_quantizer":
                return
            target = parts[5]
            value_tensor = tensor.detach()
            if value_tensor.numel() != 1:
                return
            value = float(value_tensor.cpu().item())
            if target == "k_scale":
                captured_k_scales[layer_idx] = value
            elif target == "v_scale":
                captured_v_scales[layer_idx] = value

        load_moondream_weights(
            str(cfg.model_path),
            self.model,
            tensor_hook=_capture_kv_scale,
            region=self.region,
            checkpoint_format=self._spec.checkpoint_format,
        )

        # Build spatial decode tables after region weight loading; otherwise the
        # concatenated weights/biases depend on random init/seed rather than the
        # checkpoint.
        self.spatial_tables = build_spatial_decode_tables(self.region)

        if all(val is not None for val in captured_k_scales) and all(
            val is not None for val in captured_v_scales
        ):
            self._kv_layer_k_scales = [cast(float, val) for val in captured_k_scales]
            self._kv_layer_v_scales = [cast(float, val) for val in captured_v_scales]
        elif any(val is not None for val in captured_k_scales) or any(
            val is not None for val in captured_v_scales
        ):
            warnings.warn(
                "Partial KV scales found in checkpoint; falling back to standard KV cache.",
                stacklevel=2,
            )

        device_cc_major, device_cc_minor = get_device_capability(self.device)
        device_sm = device_cc_major * 10 + device_cc_minor
        fp8_kv_supported_sms = {87, 89, 90, 100, 110, 120}

        if (
            self._kv_layer_k_scales is not None
            and self._kv_layer_v_scales is not None
            and self.page_size == 1
            and hasattr(torch, "float8_e4m3fn")
            and device_sm in fp8_kv_supported_sms
        ):
            self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            if (
                self._kv_layer_k_scales is not None
                and self._kv_layer_v_scales is not None
            ):
                if self.page_size != 1:
                    warnings.warn(
                        "KV scales found in checkpoint but FP8 KV cache currently requires page_size==1; "
                        "falling back to standard KV cache.",
                        stacklevel=2,
                    )
                elif device_sm not in fp8_kv_supported_sms:
                    sm_list = "/".join(f"SM{sm}" for sm in sorted(fp8_kv_supported_sms))
                    warnings.warn(
                        f"KV scales found in checkpoint but FP8 KV cache currently requires {sm_list} "
                        "decode kernels; falling back to standard KV cache.",
                        stacklevel=2,
                    )
            self.kv_cache_dtype = self.dtype

        self.tokenizer = load_tokenizer(
            self._spec.tokenizer_id,
            cfg.tokenizer_path,
        )
        # TokenizerConfig satisfies the PromptTemplate protocol directly.
        self.prompt_template = self.config.tokenizer

        head_dim = self.config.text.dim // self.config.text.n_heads
        self.head_dim = head_dim
        self.layer_caches: list[_LayerPagedCache] = []
        for block in self.model.text.blocks:
            layer_idx = len(self.layer_caches)
            k_scale = None
            v_scale = None
            if (
                self._kv_layer_k_scales is not None
                and self._kv_layer_v_scales is not None
            ):
                k_scale = self._kv_layer_k_scales[layer_idx]
                v_scale = self._kv_layer_v_scales[layer_idx]
            cache = _LayerPagedCache(
                page_table=self.page_table,
                n_kv_heads=self.config.text.n_kv_heads,
                head_dim=head_dim,
                dtype=self.kv_cache_dtype,
                pool=self._kv_pool,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            block.kv_cache = cache
            self.layer_caches.append(cache)

        self.active_sequences: dict[int, SequenceState] = {}
        self._use_cuda_graphs = (
            cfg.enable_cuda_graphs
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )

        # Additional stream for D2H copies.
        # (Primary stream was created earlier for PageTable H2D sync.)
        self._copy_stream = make_stream(self.device)

        # Image preprocessing pool — owned here so other runtimes can
        # plug in their own preprocessing without the engine knowing.
        self._image_preprocessor = ImagePreprocessor()

        # Spatial decode RNG. Pre-refactor the scheduler's sampling RNG
        # was reused; now that spatial decode is runtime-owned we keep
        # our own generator so stochastic point/detect requests don't
        # silently collapse to greedy.
        self._spatial_rng = torch.Generator(device=self.device)
        self._spatial_rng.manual_seed(torch.seed())

        # Vision encoder CUDA graphs
        self._vision_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._vision_input: torch.Tensor | None = None   # [max_crops, 3, 378, 378]
        self._vision_output: torch.Tensor | None = None  # [max_crops, 729, 1152]

        coord_dtype = self.region.coord_features.dtype
        size_dtype = self.region.size_features.dtype
        # Prefill can be pipelined (CPU committing token0 while GPU runs the next prefill).
        # Keep 2 slots to avoid clobbering per-prefill GPU buffers like `batch_idx`.
        # Pre-allocate scratch buffers for prefill to avoid per-layer allocations.
        # Size for typical prefill (1024 tokens). Grows automatically if needed.
        _prefill_scratch_tokens = 1024
        pin = self.device.type == "cuda"
        max_lora_slots = max(0, self.max_batch_slots - 1)
        self._prefill_slots: list[PrefillSlot] = [
            PrefillSlot(
                slot_id=slot_id,
                batch_idx=torch.empty(
                    (self.max_batch_size,), dtype=torch.int64, device=self.device
                ),
                coord_staging=torch.empty(
                    (self.max_batch_size, 1), dtype=coord_dtype, device=self.device,
                ),
                size_staging=torch.empty(
                    (self.max_batch_size, 2), dtype=size_dtype, device=self.device,
                ),
                coord_cpu=torch.empty(
                    (self.max_batch_size, 1), dtype=coord_dtype,
                    device="cpu", pin_memory=pin,
                ),
                size_cpu=torch.empty(
                    (self.max_batch_size, 2), dtype=size_dtype,
                    device="cpu", pin_memory=pin,
                ),
                step_done_event=make_event(self.device, enable_timing=False, blocking=False),
                commit_done_event=make_event(self.device, enable_timing=False, blocking=False),
                aux_done_event=make_event(self.device, enable_timing=False, blocking=False),
                scratch=PrefillScratch(
                    _prefill_scratch_tokens, self.config.text, self.device, self.dtype,
                ) if self.config.text else None,
                input_staging=PrefillInputStaging(
                    max_batch_size=self.max_batch_size,
                    max_seq_length=self.max_seq_length,
                    max_lora_slots=max_lora_slots,
                    coord_dtype=coord_dtype,
                    size_dtype=size_dtype,
                    device=self.device,
                    pin_memory=pin,
                ),
            )
            for slot_id in range(2)
        ]
        self._prefill_slot_free: list[PrefillSlot] = list(reversed(self._prefill_slots))

        self.seg_refiner = (
            SegmentRefiner(self.model.vision, self.config.vision, self.device)
            if _HAS_SEG_DEPS else None
        )

        # Multi-slot LoRA workspace and slot manager.
        # Slot 0 represents "no LoRA". Active adapters are loaded into slots 1+.
        # With max_slots = max_batch_slots, we have max_batch_size usable adapter
        # slots (slot 0 reserved), matching effective batch size.
        self._lora_workspace: TextLoRAWorkspace | None = None
        self._slot_manager: AdapterSlotManager | None = None
        self._dense_lora_decode_scratch: DenseLoRATorchMLPScratch | None = None
        self._max_lora_rank: int | None = max_lora_rank
        if max_lora_rank is not None:
            max_slots = self.max_batch_slots
            self._lora_workspace = TextLoRAWorkspace(
                text_config=self.config.text,
                max_slots=max_slots,
                max_rank=max_lora_rank,
                device=self.device,
                dtype=self.dtype,
            )
            self._slot_manager = AdapterSlotManager(max_slots)
            self._dense_lora_decode_scratch = create_mlp_scratch(
                max_segments=max(1, self.max_batch_size),
                max_segment_len=1,
                max_rank=max_lora_rank,
                d_model=self.config.text.dim,
                d_ffn=self.config.text.ff_dim,
                device=self.device,
                dtype=self.dtype,
            )

        # Create two ping-pong decode slots for pipelined decoding.
        # Each slot has its own staging buffers, paged-KV metadata buffers,
        # and RenderBuffer, but they share the decode compute stream and copy stream.
        vocab_size = self.model.text.lm_head.weight.shape[0]
        hidden_dim = self.model.text.lm_head.weight.shape[1]
        self._decode_slots: list[DecodeSlot] = [
            create_decode_slot(
                slot_id=slot_id,
                device=self.device,
                dtype=self.dtype,
                max_batch_slots=self.max_batch_slots,
                kv_cache_pages=n_pages,
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                coord_dtype=coord_dtype,
                size_dtype=size_dtype,
                compute_stream=self._compute_stream,
                copy_stream=self._copy_stream,
            )
            for slot_id in range(2)
        ]
        self._decode_graphs = DecodeGraphManager[DecodeSlot](
            enabled=self._use_cuda_graphs,
            device=self.device,
            max_batch=self.max_batch_size,
            graph_capture_lock=self.graph_capture_lock,
            compute_stream=self._compute_stream,
            run_forward=self._run_decode_forward,
            prepare_step=self._prepare_decode_graph_step,
            zero_padding=self._zero_decode_graph_padding,
            zero_for_capture=self._zero_decode_graph_capture_buffers,
        )

        # Shared pending coord/size values, indexed by batch_idx. These
        # are the runtime-side equivalent of the scheduler's
        # ``_pending_token_ids`` — written by ``post_sample`` after each
        # decode step and gathered into the next step's slot via
        # ``prepare_decode_inputs``. Owned here (not per-slot) because
        # batch indices outlive any single slot's ping-pong cycle.
        self._pending_coord_values = torch.zeros(
            (self.max_batch_slots, 1),
            dtype=coord_dtype,
            device=self.device,
        )
        self._pending_size_values = torch.zeros(
            (self.max_batch_slots, 2),
            dtype=size_dtype,
            device=self.device,
        )

        # Pre-allocate workspaces unconditionally (needed for both graph and non-graph paths)
        self._preallocate_workspaces()

        if self._use_cuda_graphs:
            self._maybe_release_cuda_allocator_cache()
            self._ensure_cuda_graphs_ready()

        # Allocate vision encoder buffers (always, for consistency)
        self._allocate_vision_buffers()
        self._capture_vision_graphs()

    def _maybe_release_cuda_allocator_cache(self) -> None:
        """Release cached allocator blocks before CUDA graph capture.

        The runtime allocates large KV caches via PyTorch tensors, while some
        graph-captured kernels request memory outside the PyTorch allocator. On
        some systems, PyTorch can keep reclaimed slack in its reserved pool,
        leaving too little driver-visible memory for those raw CUDA allocations.
        """
        if self.device.type != "cuda":
            return
        empty_cache(self.device)

    # ------------------------------------------------------------------
    # Capacity helpers

    def can_reserve(self, total_length: int) -> bool:
        """Return True if a request of ``total_length`` tokens can be admitted."""

        return self.page_table.can_reserve_with_eviction(total_length)

    def initial_reserve_length(self, prompt_length: int, max_length: int) -> int:
        """Return the logical KV length to reserve before prefill launch."""

        if max_length <= prompt_length:
            return max_length
        decode_reserve = max(
            int(self.page_table.page_size),
            DEFAULT_INITIAL_DECODE_RESERVE_TOKENS,
        )
        return min(max_length, prompt_length + decode_reserve)

    def expand_kv_reservation(
        self, state: SequenceState, tokens: int = 1
    ) -> bool:
        """Grow a sequence's KV reservation for an upcoming decode write."""

        target_length = min(state.max_length, state.length + tokens)
        if target_length <= state.length:
            return True
        return self.page_table.reserve(state.batch_idx, target_length)

    def prefill_budget(self) -> tuple[int, int]:
        """Return `(pages, batch_slots)` available for launch-time prefill binding."""

        reclaimable_pages = (
            self.prefix_cache.reclaimable_page_count() if self.prefix_cache is not None else 0
        )
        return (
            self.page_table.pages_available + reclaimable_pages,
            len(self.page_table.free_batch_idx),
        )

    def can_reserve_pages(self, total_length: int) -> bool:
        """Return True if pages are available for ``total_length`` tokens.

        Unlike can_reserve(), this does NOT check batch slot availability.
        Used for prefill preparation which doesn't require a batch slot.
        """
        return self.page_table.can_reserve_pages(total_length)

    @property
    def vocab_size(self) -> int:
        return int(self.config.text.vocab_size)

    @property
    def compute_stream(self) -> torch.cuda.Stream | None:
        """Compute stream used by engine-constructed runtime internals."""
        return self._compute_stream

    @property
    def kv_pool(self) -> KVMemoryPool:
        """KV memory pool owned by the engine/runtime construction path."""
        return self._kv_pool

    @property
    def copy_stream(self) -> torch.cuda.Stream | None:
        """Shared copy stream for D2H transfers."""
        return self._copy_stream

    @property
    def prefill_slots(self) -> list[PrefillSlot]:
        return self._prefill_slots

    @property
    def model_name(self) -> str:
        """Return the model name (e.g., 'moondream2', 'moondream3-preview')."""
        return self._cfg.model

    def skills(self) -> "SkillRegistry":
        """The capabilities this model serves (Runtime protocol).

        Sourced from the model's :class:`~kestrel.models.registry.ModelSpec`
        (``skills=``), the single declaration of the model's skill set, so
        the live runtime and the pre-start engine resolve the same registry.
        """
        return get_spec(self.model_name).skills()

    def tasks(self) -> tuple[str, ...]:
        """Capability names this model serves (Runtime protocol)."""
        return self.skills().names()

    # --- Sampling hooks (kestrel.runtime.sampling.SamplingHooks) ----
    # Moondream's per-step "post-sample" work is a coord/size decode
    # from hidden states. We own all the bytes (per-slot GPU staging +
    # pinned CPU buffers + a shared pending pool indexed by batch_idx)
    # and the D2H pipeline; the scheduler just threads the opaque
    # ``StepHandle`` returned here back into ``materialize_tokens``.
    @functools.cached_property
    def sampling_hooks(self) -> SamplingHooks:
        coord_id = self.config.tokenizer.coord_id
        size_id = self.config.tokenizer.size_id
        spatial_tables = self.spatial_tables
        pending_coord = self._pending_coord_values
        pending_size = self._pending_size_values
        copy_stream = self._copy_stream
        spatial_rng = self._spatial_rng

        def post_sample(
            slot,
            *,
            sampled_ids: Tensor,
            hidden_last: Tensor | None,
            sequences: Sequence,
            batch_idx: Tensor,
            temperatures: Tensor | None,
            top_ps: Tensor | None,
            token_logprobs: Tensor | None,
            ready_event: "torch.cuda.Event",
        ) -> tuple[object, int]:
            if hidden_last is None:
                raise RuntimeError(
                    "Moondream post_sample requires hidden_last for spatial decode"
                )
            batch_size = int(sampled_ids.shape[0])
            coord_out = slot.coord_staging[:batch_size]
            size_out = slot.size_staging[:batch_size]
            compute_spatial_values(
                sampled_ids.view(-1),
                hidden_last,
                [seq.request for seq in sequences],
                spatial_tables,
                temperatures=temperatures,
                top_ps=top_ps,
                token_logprobs=token_logprobs,
                coord_id=coord_id,
                size_id=size_id,
                out_coord=coord_out,
                out_size=size_out,
                rng=spatial_rng,
            )
            # Update the shared pending pool so the next decode step
            # gathers fresh coord/size values for these batch indices.
            pending_coord.index_copy_(0, batch_idx, coord_out)
            pending_size.index_copy_(0, batch_idx, size_out)

            # Re-record ``step_done_event`` on the compute stream after
            # the spatial writes. ``ready_event`` from the scheduler only
            # fenced sampled-id staging; the copy stream is separate from
            # the compute stream, so without this re-record the D2H below
            # can race with ``compute_spatial_values``. The sampled-id
            # D2H the scheduler kicks later anchors on the same event and
            # so picks up the post-spatial fence for free.
            slot.step_done_event.record()
            with stream_context(copy_stream):
                if copy_stream is not None:
                    copy_stream.wait_event(slot.step_done_event)
                slot.coord_cpu[:batch_size].copy_(coord_out, non_blocking=True)
                slot.size_cpu[:batch_size].copy_(size_out, non_blocking=True)
                slot.aux_done_event.record(copy_stream)
            return slot, batch_size

        def materialize_tokens(
            token_ids_cpu: Tensor,
            sequences: Sequence,
            batch_idx: Tensor,
            step_handle: tuple[object, int] | None,
        ) -> list[Token]:
            if step_handle is None:
                # No post_sample ran (e.g., a runtime that opted out).
                # Token can't be coord/size in that case — pure text path.
                return [
                    TextToken(token_id=int(t)) for t in token_ids_cpu.view(-1).tolist()
                ]
            slot, batch_size = step_handle
            slot.aux_done_event.synchronize()
            from kestrel.scheduler.tokens import render_tokens_from_packed
            return render_tokens_from_packed(
                token_ids_cpu,
                slot.coord_cpu[:batch_size],
                slot.size_cpu[:batch_size],
                coord_id=coord_id,
                size_id=size_id,
            )

        def materialize_spec_tokens(
            token_ids_cpu: Tensor,
            sequences: Sequence,
            batch_idx: Tensor,
            side_values: object,
            token_logprobs: list[float] | None = None,
        ) -> list[Token]:
            """Type a spec macro-step's committed run into coord/size tokens.

            Speculative analog of ``materialize_tokens``. A macro-step commits a
            *variable run* of positions per sequence, so the spatial decode the
            non-spec ``post_sample`` does one-position-at-a-time runs here over
            **all** committed positions at once, reading the target's final
            ``hidden_last`` for each committed position out of the macro-step's
            :class:`~kestrel.runtime.spec.SpecSideValues` (rather than the slot's
            ``post_sample`` aux staging, which holds a single step's values).

            ``side_values.hidden`` is ``[num_sequences, K + 1, hidden_dim]`` (the
            verify-block final hidden, indexed by ``sequences`` order); only the
            leading ``n_i`` positions of row ``i`` are committed, where ``n_i`` is
            that sequence's committed-run length. ``token_ids_cpu`` / ``batch_idx``
            are flat over all committed positions (``sequences``-major, contiguous
            per sequence). We slice ``hidden[i, :n_i]`` per sequence, pack the
            committed hiddens into one ``[total, hidden_dim]`` batch, decode
            coord/size in a single ``compute_spatial_values`` call under each
            sequence's sampling knobs, then render the typed tokens.

            ``token_logprobs`` (optional) is the flat per-committed-position
            vocab-token logprob list (parallel to ``token_ids_cpu``) the scheduler
            staged from the spec decoder's verify logits. When supplied, it is
            mutated **in place** so each spatial position's entry gains its
            coord/size head logprob -- mirroring the non-spec ``post_sample`` path,
            where ``compute_spatial_values`` adds the spatial head's selected-bin
            logprob to the vocab logprob in-place. The spec decoder only gathers
            the vocab logprob (it never runs the spatial head), so without this the
            scheduler would under-report logprobs for ``coord_id`` / ``size_id``
            tokens (and the spatial admit token, which routes through this hook).
            ``None`` (no request wanted logprobs) leaves the spatial decode purely
            value-producing, exactly as before.
            """
            from kestrel.runtime.spec import SpecSideValues

            if not isinstance(side_values, SpecSideValues):
                raise TypeError(
                    "materialize_spec_tokens requires a SpecSideValues; got "
                    f"{type(side_values).__name__}"
                )
            seqs = list(sequences)
            flat_ids = token_ids_cpu.view(-1)
            total = int(flat_ids.shape[0])
            if total == 0:
                return []

            # Per-sequence committed-run lengths (``n_i``). Prefer the producer's
            # explicit ``counts``; otherwise recover them from the flat per-token
            # ``batch_idx`` (each active sequence owns a distinct pool row, and the
            # flat layout is sequences-major + contiguous), so the slice of
            # ``hidden[i]`` matches the actually-committed run (which may be capped
            # below ``accept_counts[i] + 1`` by the scheduler's commit cap).
            counts = side_values.counts
            if counts is None:
                flat_rows = batch_idx.view(-1).tolist()
                counts = []
                cursor = 0
                for seq in seqs:
                    row = seq.state.batch_idx
                    n = 0
                    while cursor + n < total and flat_rows[cursor + n] == row:
                        n += 1
                    counts.append(n)
                    cursor += n
            if len(counts) != len(seqs):
                raise ValueError(
                    "materialize_spec_tokens: counts length "
                    f"{len(counts)} != num sequences {len(seqs)}"
                )
            if sum(int(c) for c in counts) != total:
                raise ValueError(
                    "materialize_spec_tokens: committed counts "
                    f"{counts} do not sum to flat token count {total}"
                )

            hidden = side_values.hidden  # [num_sequences, K+1, hidden_dim]
            device = hidden.device
            # Gather the committed final-hidden for every committed position into
            # one packed ``[total, hidden_dim]`` batch (sequences-major), plus the
            # parallel per-position requests / sampling knobs.
            hidden_rows: list[Tensor] = []
            requests: list = []
            temp_parts: list[Tensor] = []
            top_p_parts: list[Tensor] = []
            for i, (seq, n) in enumerate(zip(seqs, counts)):
                n = int(n)
                if n == 0:
                    continue
                hidden_rows.append(hidden[i, :n])
                requests.extend(seq.request for _ in range(n))
                temp_parts.append(side_values.temperatures[i].expand(n))
                top_p_parts.append(side_values.top_ps[i].expand(n))
            hidden_last = torch.cat(hidden_rows, dim=0)  # [total, hidden_dim]
            temperatures = torch.cat(temp_parts, dim=0)
            top_ps = torch.cat(top_p_parts, dim=0)
            token_ids = flat_ids.to(device=device, dtype=torch.long)

            # When the request wants logprobs, seed a packed device buffer with the
            # per-position vocab logprobs (already in the packed sequences-major
            # order ``hidden_last`` is built in -- ``hidden_rows`` and ``flat_ids``
            # share that order, and ``token_logprobs`` is parallel to
            # ``token_ids_cpu`` / ``flat_ids``) so ``compute_spatial_values`` adds
            # each spatial position's coord/size head logprob in-place, exactly like
            # the non-spec ``post_sample`` path. Then copy the combined values back
            # into the caller's flat list so the scheduler stages the combined
            # vocab + spatial-head logprob.
            logprobs_buf: Tensor | None = None
            if token_logprobs is not None:
                if len(token_logprobs) != total:
                    raise ValueError(
                        "materialize_spec_tokens: token_logprobs length "
                        f"{len(token_logprobs)} != flat token count {total}"
                    )
                logprobs_buf = torch.tensor(
                    token_logprobs, device=device, dtype=torch.float32
                )

            coord_out = torch.empty((total, 1), device=device, dtype=torch.float32)
            size_out = torch.empty((total, 2), device=device, dtype=torch.float32)
            compute_spatial_values(
                token_ids,
                hidden_last,
                requests,
                spatial_tables,
                temperatures=temperatures,
                top_ps=top_ps,
                token_logprobs=logprobs_buf,
                coord_id=coord_id,
                size_id=size_id,
                out_coord=coord_out,
                out_size=size_out,
                rng=spatial_rng,
            )
            if logprobs_buf is not None:
                assert token_logprobs is not None
                combined = logprobs_buf.cpu().tolist()
                token_logprobs[:] = combined
            from kestrel.scheduler.tokens import render_tokens_from_packed
            return render_tokens_from_packed(
                token_ids_cpu,
                coord_out.cpu(),
                size_out.cpu(),
                coord_id=coord_id,
                size_id=size_id,
            )

        def prepare_decode_inputs(
            slot: DecodeSlot,
            batch_idx: Tensor,
            batch_size: int,
        ) -> None:
            torch.index_select(
                pending_coord, 0, batch_idx,
                out=slot.decode_coord_values[:batch_size],
            )
            torch.index_select(
                pending_size, 0, batch_idx,
                out=slot.decode_size_values[:batch_size],
            )

        return SamplingHooks(
            post_sample=post_sample,
            materialize_tokens=materialize_tokens,
            materialize_spec_tokens=materialize_spec_tokens,
            prepare_decode_inputs=prepare_decode_inputs,
        )

    # --- Image preprocessing (Runtime protocol) ----------------------
    def preprocess_image_async(self, image):
        """Return a Future for Moondream's overlap-crop preprocessing."""
        return self._image_preprocessor.submit(image, self.config.vision)

    def shutdown(self) -> None:
        """Release runtime resources (Runtime protocol).

        The engine calls this once per runtime on shutdown; for Moondream
        that means tearing down the image-preprocessor thread pool.
        """
        self._image_preprocessor.shutdown(wait=True)

    def acquire_prefill_slot(self, slot_id: int | None = None) -> PrefillSlot:
        if slot_id is None:
            if not self._prefill_slot_free:
                raise RuntimeError("Prefill slot pool exhausted")
            return self._prefill_slot_free.pop()

        if slot_id < 0 or slot_id >= len(self._prefill_slots):
            raise ValueError(f"Invalid prefill_slot_id {slot_id}")

        for idx in range(len(self._prefill_slot_free) - 1, -1, -1):
            slot = self._prefill_slot_free[idx]
            if slot.slot_id == slot_id:
                return self._prefill_slot_free.pop(idx)
        raise RuntimeError(f"Prefill slot {slot_id} is already in use")

    def release_prefill_slot(self, slot: PrefillSlot) -> None:
        self._prefill_slot_free.append(slot)

    @property
    def decode_slots(self) -> list[DecodeSlot]:
        """Two ping-pong decode slots for pipelined decoding."""
        return self._decode_slots

    # ------------------------------------------------------------------
    # Prompt helpers

    @functools.cached_property
    def bos_embed(self) -> Tensor:
        bos = torch.empty((1, 1), device=self.device, dtype=torch.long)
        bos.fill_(self.config.tokenizer.bos_id)
        return text_encoder(bos, self.model.text)

    def _embed_tokens(
        self,
        tokens: Sequence[Token],
        *,
        staging: PrefillInputStaging,
        row: int,
    ) -> Tensor:
        """Embed an in-order prompt (single sequence) into shape (1, L, dim)."""

        if not tokens:
            dim = self.bos_embed.shape[-1]
            return torch.empty((1, 0, dim), device=self.device, dtype=self.dtype)

        length = len(tokens)

        text_pos: list[int] = []
        coord_pos: list[int] = []
        size_pos: list[int] = []
        text_ids: list[int] = []
        coord_vals: list[float] = []
        size_vals: list[tuple[float, float]] = []

        for idx, token in enumerate(tokens):
            if isinstance(token, TextToken):
                text_pos.append(idx)
                text_ids.append(token.token_id)
            elif isinstance(token, CoordToken):
                coord_pos.append(idx)
                coord_vals.append(token.pos)
            elif isinstance(token, SizeToken):
                size_pos.append(idx)
                size_vals.append((token.width, token.height))
            else:  # pragma: no cover - defensive
                raise TypeError(f"Unsupported token type: {type(token)!r}")

        if text_ids:
            ids = staging.stage_text_ids(row, text_ids)
            text_emb = text_encoder(ids, self.model.text)
            if len(text_ids) == length:
                return text_emb
            width = self.bos_embed.shape[-1]
            out = torch.empty((1, length, width), device=self.device, dtype=self.dtype)
            positions = staging.stage_token_positions(row, text_pos, kind="text")
            out.index_copy_(1, positions, text_emb)
        else:
            width = self.bos_embed.shape[-1]
            out = torch.empty((1, length, width), device=self.device, dtype=self.dtype)

        if coord_vals:
            coords = staging.stage_coord_values(row, coord_vals)
            coord_emb = encode_coordinate(coords, self.region)
            positions = staging.stage_token_positions(row, coord_pos, kind="coord")
            out.index_copy_(1, positions, coord_emb.unsqueeze(0))

        if size_vals:
            sizes = staging.stage_size_values(row, size_vals)
            size_emb = encode_size(sizes, self.region)
            positions = staging.stage_token_positions(row, size_pos, kind="size")
            out.index_copy_(1, positions, size_emb.unsqueeze(0))

        return out

    def _embed_packed_token_batch(
        self,
        token_ids: Tensor,
        coord_values: Tensor,
        size_values: Tensor,
    ) -> Tensor:
        """Embed pending decode tokens from packed id/value tensors.

        `coord_values`/`size_values` are meaningful only for rows whose token id is
        the corresponding special token (coord_id/size_id); other rows should be
        zero-filled.
        """

        if token_ids.ndim != 1:
            token_ids = token_ids.view(-1)

        batch = int(token_ids.shape[0])
        if batch == 0:
            dim = self.bos_embed.shape[-1]
            return torch.empty((0, 1, dim), device=self.device, dtype=self.dtype)

        ids = token_ids.to(dtype=torch.long)
        text_emb = text_encoder(ids.view(-1, 1), self.model.text)

        coord_emb = encode_coordinate(coord_values, self.region).unsqueeze(1)
        size_emb = encode_size(size_values, self.region).unsqueeze(1)

        coord_id = self.config.tokenizer.coord_id
        size_id = self.config.tokenizer.size_id
        coord_mask = (ids == coord_id).view(-1, 1, 1)
        size_mask = (ids == size_id).view(-1, 1, 1)

        out = torch.where(coord_mask, coord_emb, text_emb)
        out = torch.where(size_mask, size_emb, out)
        return out

    def encode_image(
        self,
        image: Optional[np.ndarray],
        *,
        overlap: Optional[OverlapCropOutput] = None,
    ) -> Tensor:
        with torch.inference_mode():
            if overlap is not None:
                crops, tiling = prepare_crops_from_overlap(overlap, self.device, self.dtype)
            else:
                if image is None:
                    raise ValueError("image must be provided when overlap is not supplied")
                # Multi-image chat passes images straight through here with no
                # precomputed overlap. The executor decodes/validates the tuple
                # at admission, but normalize defensively in case a caller hands
                # raw bytes straight to encode_image (decode_to_srgb is a no-op
                # for arrays already in sRGB).
                from kestrel.utils.image import decode_to_srgb

                image = decode_to_srgb(image)
                crops, tiling = prepare_crops(image, self.config.vision, self.device, self.dtype)

            batch_size = crops.shape[0]

            # Always use stable buffers for consistency
            self._vision_input[:batch_size].copy_(crops)

            # Use CUDA graph if available, otherwise eager
            if batch_size in self._vision_graphs:
                self._vision_graphs[batch_size].replay()
            else:
                out = vision_encoder(
                    self._vision_input[:batch_size],
                    self.model.vision,
                    self.config.vision,
                )
                self._vision_output[:batch_size].copy_(out)

            outputs = self._vision_output[:batch_size]

            # Rest unchanged: projection, reconstruction
            global_features = outputs[0]
            local = outputs[1:].reshape(
                -1,
                self.config.vision.enc_n_layers,
                self.config.vision.enc_n_layers,
                self.config.vision.enc_dim,
            )
            reconstructed = reconstruct_from_crops(
                local.to(dtype=torch.float32),
                tiling,
                overlap_margin=self.config.vision.overlap_margin,
                patch_size=1,
            )
            reconstructed = reconstructed.to(device=self.device, dtype=outputs.dtype)
            return vision_projection(
                global_features,
                reconstructed,
                self.model.vision,
                self.config.vision,
            )

    # ------------------------------------------------------------------
    # Prefill preparation helpers

    def check_prefix_cache(
        self,
        tokens_list: list[Token],
        image_hash: bytes | None,
        adapter_id: str | None,
    ) -> bool:
        """Check if an image+tokens combo would hit the prefix cache.

        This is a lightweight check that does not acquire locks or map pages.
        Used for early cache lookup to skip crop computation on cache hits.

        Returns True if the cache would hit and cover the full image prefix.
        """
        if self.prefix_cache is None:
            return False
        if image_hash is None:
            return False

        image_kv_length = self.image_prefix_length
        cache_tokens = self._build_cache_tokens(tokens_list, image_hash, image_kv_length)

        namespace = self._cache_namespace(adapter_id, image_hash)
        match = self.prefix_cache.match_prefix(cache_tokens, namespace=namespace)

        # Cache hit must cover at least BOS + full image prefix
        min_hit_length = 1 + image_kv_length
        return match.matched_kv_length >= min_hit_length

    def classify_prefill(
        self,
        prompt_tokens: Sequence[Token],
        *,
        has_image: bool,
        image_hash: bytes | None,
        adapter_id: str | None,
    ) -> PrefillClassification:
        """Return the current prefix-cache classification for a request."""

        tokens_list = list(prompt_tokens)
        num_image_markers = _count_image_markers(tokens_list)
        if num_image_markers > 0:
            image_kv_length = num_image_markers * self.image_prefix_length
            prompt_len = len(tokens_list) - num_image_markers + image_kv_length
        else:
            image_kv_length = self.image_prefix_length if has_image else 0
            prompt_len = len(tokens_list) + image_kv_length
        can_reuse = False
        skip_positions = 0

        # The single-image prefix cache doesn't cover the multi-image marker
        # layout; classify those as full prefill (no reuse).
        if self.prefix_cache is not None and num_image_markers == 0:
            if has_image:
                assert image_hash is not None, (
                    "image_hash must be provided when prefix cache is enabled for image prompts"
                )
            cache_tokens = self._build_cache_tokens(tokens_list, image_hash, image_kv_length)
            namespace = self._cache_namespace(adapter_id, image_hash)
            match = self.prefix_cache.match_prefix(cache_tokens, namespace=namespace)
            can_reuse = match.matched_kv_length > 0
            if image_kv_length > 0 and can_reuse:
                assert match.matched_kv_length >= (1 + image_kv_length), (
                    f"Invariant violated: image namespace hit ({match.matched_kv_length} KV) "
                    f"must include BOS+image ({1 + image_kv_length} KV)"
                )
            if can_reuse:
                skip_positions = min(match.matched_kv_length, prompt_len - 1)

        return PrefillClassification(
            prompt_length=prompt_len,
            skip_positions=skip_positions,
            can_reuse=can_reuse,
            use_prefix_attn=bool(image_kv_length) and not can_reuse,
        )

    def _build_cache_tokens(
        self,
        tokens_list: list[Token],
        image_hash: bytes | None,
        image_kv_length: int,
    ) -> list[CacheToken]:
        """Build cache token sequence: [BOS, ImageToken?, text tokens...]."""
        cache_tokens: list[CacheToken] = []
        if tokens_list:
            cache_tokens.append(tokens_list[0])  # BOS
        if image_hash is not None:
            cache_tokens.append(
                ImageToken(
                    content_hash=int.from_bytes(image_hash[:16], "big"),
                    kv_length_=image_kv_length,
                )
            )
        if len(tokens_list) > 1:
            cache_tokens.extend(tokens_list[1:])
        return cache_tokens

    def _cache_namespace(
        self,
        adapter_id: str | None,
        image_hash: bytes | None,
    ) -> CacheNamespace:
        return CacheNamespace(
            runtime_id=self.model_name,
            lora_id=adapter_id,
            image_hash=(
                int.from_bytes(image_hash[:16], "big")
                if image_hash is not None
                else None
            ),
        )

    def _lookup_prefix_cache(
        self,
        cache_tokens: list[CacheToken],
        adapter_id: str | None,
        image_hash: bytes | None,
        image_kv_length: int,
        prompt_len: int,
        batch_idx: int,
    ) -> _CacheLookupResult:
        """Lookup prefix cache, map cached pages, and acquire temp lock."""
        if self.prefix_cache is None or not cache_tokens:
            return _CacheLookupResult(
                match=None,
                skip_positions=0,
                temp_lock_node=None,
                can_reuse=False,
                namespace=None,
            )

        namespace = self._cache_namespace(adapter_id, image_hash)
        match = self.prefix_cache.match_prefix(cache_tokens, namespace=namespace)
        can_reuse = match.matched_kv_length > 0

        # Invariant: In image namespace, any hit must include BOS+image.
        if image_kv_length > 0 and can_reuse:
            assert match.matched_kv_length >= (1 + image_kv_length), (
                f"Invariant violated: image namespace hit ({match.matched_kv_length} KV) "
                f"must include BOS+image ({1 + image_kv_length} KV)"
            )

        skip_positions = 0
        temp_lock_node: TreeNode | None = None

        if can_reuse:
            # Cap skip_positions to ensure at least one suffix KV position
            skip_positions = min(match.matched_kv_length, prompt_len - 1)

            # Map cached pages
            cached_pages = match.matched_pages[:skip_positions]
            self.page_table.map_pages(batch_idx, 0, cached_pages)

            # Lock matched prefix during prefill
            temp_lock_node = match.last_node
            self.prefix_cache.lock_prefill(temp_lock_node)

        return _CacheLookupResult(
            match=match,
            skip_positions=skip_positions,
            temp_lock_node=temp_lock_node,
            can_reuse=can_reuse,
            namespace=namespace,
        )

    def _prepare_append_prefill_inputs(
        self,
        tokens_list: list[Token],
        skip_positions: int,
        image_kv_length: int,
        prompt_len: int,
        *,
        staging: PrefillInputStaging,
        row: int,
    ) -> tuple[Tensor, int]:
        """Prepare inputs for append prefill (cache hit path)."""
        # Derive suffix tokens from skip_positions
        if image_kv_length > 0:
            # KV layout: [BOS(1)] [Image(image_kv_length)] [Text tokens...]
            prefix_kv = 1 + image_kv_length
            text_kv_cached = skip_positions - prefix_kv
            # tokens_list[0] = BOS, tokens_list[1:] = text after image
            suffix_tokens = tokens_list[1 + text_kv_cached :]
        else:
            suffix_tokens = tokens_list[skip_positions:]

        # Embed suffix tokens
        if suffix_tokens:
            inputs_embeds = self._embed_tokens(suffix_tokens, staging=staging, row=row)
        else:
            # This shouldn't happen due to skip_positions capping
            inputs_embeds = torch.empty(
                (1, 0, self.bos_embed.shape[-1]),
                device=self.device,
                dtype=self.dtype,
            )

        return inputs_embeds, skip_positions

    def _prepare_full_prefill_inputs(
        self,
        tokens_list: list[Token],
        images: Optional[Sequence[Optional[np.ndarray]]],
        image_crops: Optional[OverlapCropOutput],
        prompt_len: int,
        *,
        staging: PrefillInputStaging,
        row: int,
    ) -> tuple[Tensor, int, Optional[Tensor]]:
        """Prepare inputs for full prefill (cache-miss path).

        Returns ``(inputs_embeds, position_start, block_sequence_ids)``. Images
        are spliced in as image-patch embedding blocks: at each ``ImageMarker``
        sentinel (multi-image chat path), or after BOS (single-image query
        path). ``block_sequence_ids`` (int32, one entry per expanded position:
        image-block id >= 0, or -1 for text) drives the block-bidirectional
        attention mask; it is ``None`` when there is no image. BOS joins the
        first image block when the image starts right after it, reproducing the
        retired ``prefix_lm_730`` mask for a single front image.
        """
        marker_positions = [
            i for i, t in enumerate(tokens_list) if isinstance(t, ImageMarker)
        ]

        segments: list[Tensor] = []
        block_ids: list[int] = []

        if marker_positions:
            # Multi-image chat path: images interleaved at ImageMarker sentinels.
            if images is None:
                raise RuntimeError("ImageMarker tokens require images")
            cursor = 0
            for mpos in marker_positions:
                if mpos > cursor:
                    seg = self._embed_tokens(
                        tokens_list[cursor:mpos], staging=staging, row=row
                    )
                    segments.append(seg)
                    block_ids.extend([-1] * int(seg.shape[1]))
                marker = tokens_list[mpos]
                seg = self.encode_image(images[marker.index]).unsqueeze(0)
                segments.append(seg)
                block_ids.extend([marker.index] * int(seg.shape[1]))
                cursor = mpos + 1
            if cursor < len(tokens_list):
                seg = self._embed_tokens(
                    tokens_list[cursor:], staging=staging, row=row
                )
                segments.append(seg)
                block_ids.extend([-1] * int(seg.shape[1]))
        else:
            # Single-image query path: BOS, image, then the rest of the prompt.
            prompt_embed = (
                self._embed_tokens(tokens_list, staging=staging, row=row)
                if len(tokens_list) > 0 else None
            )
            image = images[0] if images else None
            has_image = image is not None or image_crops is not None
            if prompt_embed is not None and len(tokens_list) > 0:
                segments.append(prompt_embed[:, :1, :])  # BOS
                block_ids.append(-1)
            if has_image:
                seg = self.encode_image(image, overlap=image_crops).unsqueeze(0)
                segments.append(seg)
                block_ids.extend([0] * int(seg.shape[1]))
            if prompt_embed is not None and len(tokens_list) > 1:
                segments.append(prompt_embed[:, 1:, :])  # remaining tokens
                block_ids.extend([-1] * (len(tokens_list) - 1))

        if not segments:
            segments = [self.bos_embed]
            block_ids = [-1]

        # BOS joins the first image block when the image starts right after it.
        if len(block_ids) > 1 and block_ids[0] == -1 and block_ids[1] >= 0:
            block_ids[0] = block_ids[1]

        inputs_embeds = torch.cat(segments, dim=1)
        block_sequence_ids = None
        if any(b >= 0 for b in block_ids):
            block_sequence_ids = torch.tensor(
                block_ids, dtype=torch.int32, device=self.device
            )
        return inputs_embeds, 0, block_sequence_ids

    def _finalize_cache_after_prefill(
        self,
        cache_tokens: list[CacheToken],
        cache_result: _CacheLookupResult,
        prompt_len: int,
        batch_idx: int,
        adapter_id: str | None,
        image_hash: bytes | None,
    ) -> tuple[TreeNode | None, int]:
        """Insert into cache and handle lock transfer after prefill."""
        if self.prefix_cache is None or not cache_tokens:
            return None, 0

        full_prompt_cached = (
            cache_result.can_reuse
            and cache_result.match is not None
            and cache_result.match.matched_kv_length >= prompt_len
        )

        if full_prompt_cached:
            # Full prompt was cached - don't insert
            temp_lock_node = cache_result.temp_lock_node
            if temp_lock_node is not None:
                self.prefix_cache.lock(temp_lock_node)
                self.prefix_cache.unlock_prefill(temp_lock_node)
            return temp_lock_node, cache_result.skip_positions

        # Insert prompt into cache
        prompt_pages = self.page_table.get_pages(batch_idx, 0, prompt_len)
        namespace = self._cache_namespace(adapter_id, image_hash)
        insert_result = self.prefix_cache.insert(
            cache_tokens,
            prompt_pages,
            namespace=namespace,
            from_node=cache_result.match.last_node if cache_result.match else None,
            from_token_idx=(
                cache_result.match.matched_token_count if cache_result.match else 0
            ),
            from_page_idx=cache_result.skip_positions,
        )

        # Insert refused or redundant: another sequence inserted the full prompt
        # before we did, or the cache rejected insertion (e.g., because the caller
        # didn't map the cached prefix pages into its own page table).
        #
        # In this case, inserted_pages == 0 and we must NOT transfer the lock to
        # the returned node (we didn't use those pages); keep the existing temp
        # lock if we reused cached prefix pages, otherwise return no lock so our
        # pages can be freed.
        if insert_result.inserted_pages == 0:
            temp_lock_node = cache_result.temp_lock_node
            if temp_lock_node is None:
                return None, 0
            self.prefix_cache.lock(temp_lock_node)
            self.prefix_cache.unlock_prefill(temp_lock_node)
            return temp_lock_node, cache_result.skip_positions

        # Lock transfer
        temp_lock_node = cache_result.temp_lock_node
        if temp_lock_node is None:
            # Miss path
            self.prefix_cache.lock(insert_result.node)
            cache_lock_node = insert_result.node
        else:
            # Hit path
            self.prefix_cache.lock(insert_result.node)
            self.prefix_cache.unlock_prefill(temp_lock_node)
            cache_lock_node = insert_result.node

        cache_owned_page_count = cache_result.skip_positions + insert_result.inserted_pages
        return cache_lock_node, cache_owned_page_count

    # ------------------------------------------------------------------

    def prepare_sequence(
        self,
        prompt_tokens: Sequence[Token],
        *,
        image: Optional[np.ndarray] = None,
        image_crops: Optional[OverlapCropOutput] = None,
        max_new_tokens: Optional[int] = None,
        lora_slot: int = 0,
        image_hash: bytes | None = None,
        adapter_id: str | None = None,
    ) -> PreparedSequence:
        """Prepare a sequence for GPU prefill without launching the prefill forward.

        This performs all CPU-side admission work and KV/prefix-cache setup:
        - validate inputs
        - allocate a batch slot
        - perform prefix cache lookup + map cached pages (if enabled)
        - reserve KV capacity for the prompt plus a small decode window

        The returned PreparedSequence can be launched later via
        `launch_prepared_batch` and finalized via
        `finalize_prepared_sequence_after_prefill`.

        Note: This method is decoupled from PrefillSlot. The slot is acquired
        at launch time, allowing preparation to proceed even when all prefill
        slots are occupied by in-flight prefills.
        """

        # 1. Normalize inputs
        tokens_list = list(prompt_tokens)

        # 2. Validate image/hash consistency
        num_image_markers = _count_image_markers(tokens_list)
        has_image = (
            image is not None or image_crops is not None or num_image_markers > 0
        )
        if not has_image:
            assert image_hash is None, "image_hash must be None when no image is provided"
        else:
            # Image prompts must have at least one text token after BOS to ensure
            # correct suffix slicing on cache hit (text_kv_cached >= 0).
            if len(tokens_list) < 2:
                raise ValueError(
                    "Image prompts must include at least one text token after BOS"
                )
            # The single-image prefix cache (keyed by one image_hash) only applies
            # to the single-image (no-marker) path.
            if num_image_markers == 0 and self.prefix_cache is not None:
                assert image_hash is not None, (
                    "image_hash must be provided when prefix cache is enabled for image prompts"
                )

        # 3. Compute dimensions. ImageMarker tokens (1 each) expand to an
        # image_prefix_length patch block; the single-image path adds one block
        # that is not in the token list.
        if num_image_markers > 0:
            image_kv_length = num_image_markers * self.image_prefix_length
            prompt_len = len(tokens_list) - num_image_markers + image_kv_length
        else:
            image_kv_length = self.image_prefix_length if has_image else 0
            prompt_len = len(tokens_list) + image_kv_length

        max_new = max_new_tokens or DEFAULT_MAX_TOKENS
        target_length = prompt_len + max_new
        if prompt_len > self.max_seq_length:
            raise ValueError(
                f"Prompt length {prompt_len} exceeds max_seq_length={self.max_seq_length}."
            )
        if prompt_len == self.max_seq_length and max_new > 0:
            raise ValueError(
                "Prompt length leaves no room for generation: "
                f"prompt uses {prompt_len} tokens and limit is {self.max_seq_length}."
            )
        target_length = min(target_length, self.max_seq_length)

        # 4. Build cache tokens (skip when prefix cache is disabled, or for the
        # multi-image marker layout which the single-image cache doesn't cover).
        if self.prefix_cache is None or num_image_markers > 0:
            cache_tokens = []
        else:
            cache_tokens = self._build_cache_tokens(
                tokens_list, image_hash, image_kv_length
            )

        # 5. Allocate batch slot
        batch_idx = self.page_table.allocate()

        # 6. Cache lookup (maps pages, acquires temp lock)
        cache_result = self._lookup_prefix_cache(
            cache_tokens, adapter_id, image_hash, image_kv_length, prompt_len, batch_idx
        )

        # 7. Reserve prompt pages plus a small decode window. On cache hit,
        # map_pages() in step 6 already set capacity to skip_positions, so
        # reserve() allocates only the suffix pages needed to reach the initial
        # reservation length. Decode grows this row later as tokens are launched.
        initial_reserve_length = self.initial_reserve_length(prompt_len, target_length)
        try:
            self.page_table.reserve(batch_idx, initial_reserve_length)
        except Exception:
            # Release temp cache lock if held
            if self.prefix_cache is not None and cache_result.temp_lock_node is not None:
                self.prefix_cache.unlock_prefill(cache_result.temp_lock_node)
            # Release batch slot; do NOT free cache-owned mapped prefix pages.
            self.page_table.erase(batch_idx, cache_result.skip_positions)
            raise

        state = SequenceState(
            batch_idx=batch_idx,
            length=prompt_len,
            max_length=target_length,
            prompt_length=prompt_len,
            image_length=image_kv_length,
            last_hidden=None,
            lora_slot=lora_slot,
            cache_tokens=cache_tokens if self.prefix_cache else None,
            cache_lock_node=None,
            # Treat mapped prefix pages as cache-owned so we never free them by accident.
            cache_owned_page_count=cache_result.skip_positions,
            reused_page_count=cache_result.skip_positions,
        )

        return PreparedSequence(
            state=state,
            tokens_list=tokens_list,
            cache_tokens=cache_tokens,
            cache_result=cache_result,
            adapter_id=adapter_id,
            image_hash=image_hash,
        )

    def _build_prefill_inputs_for_prepared(
        self,
        prepared: PreparedSequence,
        *,
        image: Optional[np.ndarray],
        image_crops: Optional[OverlapCropOutput],
        staging: PrefillInputStaging,
        row: int,
    ) -> tuple[Tensor, int, bool]:
        """Build per-sequence prefill inputs for a prepared sequence."""
        tokens_list = prepared.tokens_list
        state = prepared.state
        cache_result = prepared.cache_result
        prompt_len = state.prompt_length or state.length
        image_kv_length = state.image_length

        block_sequence_ids = None
        if cache_result.can_reuse:
            # Append/cache-reuse path: prefix is already cached, attention is
            # plain causal over the new suffix (no image-block mask).
            inputs_embeds, position_start = self._prepare_append_prefill_inputs(
                tokens_list,
                cache_result.skip_positions,
                image_kv_length,
                prompt_len,
                staging=staging,
                row=row,
            )
        else:
            # `image` may be a single array (query path) or an ordered list of
            # images (multi-image chat); normalize to the list the interleaver
            # indexes by ImageMarker.index.
            if image is None:
                images = None
            elif isinstance(image, (list, tuple)):
                images = list(image)
            else:
                images = [image]
            inputs_embeds, position_start, block_sequence_ids = (
                self._prepare_full_prefill_inputs(
                    tokens_list,
                    images,
                    image_crops,
                    prompt_len,
                    staging=staging,
                    row=row,
                )
            )

        use_prefix_attn = bool(image_kv_length) and not cache_result.can_reuse
        return inputs_embeds, position_start, use_prefix_attn, block_sequence_ids

    def launch_prepared_batch(
        self,
        prepared_sequences: Sequence[PreparedSequence],
        prefill_slot: PrefillSlot,
        *,
        images: Sequence[Optional[np.ndarray]] | None = None,
        image_crops_list: Sequence[Optional[OverlapCropOutput]] | None = None,
    ) -> Tensor:
        """Launch GPU prefill for one or more prepared sequences.

        Returns:
            Logits for the first generated token for each sequence, shape [B, vocab].
        """
        batch_size = len(prepared_sequences)
        if batch_size == 0:
            raise ValueError("prepared_sequences must be non-empty")
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Prefill batch size {batch_size} exceeds max_batch_size={self.max_batch_size}"
            )

        if images is None:
            images = [None] * batch_size
        if image_crops_list is None:
            image_crops_list = [None] * batch_size
        if len(images) != batch_size:
            raise ValueError("images length must match prepared_sequences")
        if len(image_crops_list) != batch_size:
            raise ValueError("image_crops_list length must match prepared_sequences")

        input_staging = prefill_slot.input_staging
        if input_staging is None:
            raise RuntimeError("Prefill slot is missing input staging buffers")

        lora_slots = [prepared.state.lora_slot for prepared in prepared_sequences]
        if batch_size > 1 and any(slot != 0 for slot in lora_slots):
            raise NotImplementedError("Batched prefill does not yet support LoRA slots")

        # Keep all GPU work on the compute stream so all callers share identical
        # ordering semantics.
        with stream_context(self._compute_stream):
            per_inputs: list[Tensor] = []
            position_starts: list[int] = []
            per_block_ids: list[Optional[Tensor]] = []
            use_prefix_attn: bool | None = None
            for row, (prepared, image, image_crops) in enumerate(zip(
                prepared_sequences, images, image_crops_list
            )):
                inputs_embeds, position_start, seq_use_prefix_attn, seq_block_ids = (
                    self._build_prefill_inputs_for_prepared(
                        prepared,
                        image=image,
                        image_crops=image_crops,
                        staging=input_staging,
                        row=row,
                    )
                )
                if inputs_embeds.shape[1] == 0:
                    raise RuntimeError("Prefill inputs must contain at least one token")
                if use_prefix_attn is None:
                    use_prefix_attn = seq_use_prefix_attn
                elif use_prefix_attn != seq_use_prefix_attn:
                    raise ValueError(
                        "All sequences in a prefill batch must share use_prefix_attn mode"
                    )
                per_inputs.append(inputs_embeds)
                position_starts.append(position_start)
                per_block_ids.append(seq_block_ids)

            assert use_prefix_attn is not None
            hidden_dim = self.bos_embed.shape[-1]
            seq_lens = [int(t.shape[1]) for t in per_inputs]
            prompt_lens = [
                int(prepared.state.prompt_length or prepared.state.length)
                for prepared in prepared_sequences
            ]
            max_seq_len = max(seq_lens)
            inputs_embeds = torch.zeros(
                (batch_size, max_seq_len, hidden_dim),
                dtype=self.dtype,
                device=self.device,
            )
            q_is_padded = any(seq_len != max_seq_len for seq_len in seq_lens)
            batch_indices = [
                prepared.state.batch_idx for prepared in prepared_sequences
            ]

            (
                position_ids,
                slot_batch_idx,
                paged_kv_seqlens_q,
                paged_kv_seqlens_k,
                last_token_positions,
            ) = input_staging.stage_batch_metadata(
                batch_indices=batch_indices,
                seq_lens=seq_lens,
                prompt_lens=prompt_lens,
                position_starts=position_starts,
                q_is_padded=q_is_padded,
                max_seq_len=max_seq_len,
                batch_idx_gpu=prefill_slot.batch_idx,
            )

            for row, (prepared, embed_row, seq_len) in enumerate(
                zip(prepared_sequences, per_inputs, seq_lens)
            ):
                prompt_len = prompt_lens[row]
                if seq_len > prompt_len:
                    raise AssertionError(
                        f"Prefill input length ({seq_len}) exceeds prompt length ({prompt_len})"
                    )
                inputs_embeds[row, :seq_len, :].copy_(embed_row[0, :, :])

            block_sequence_ids = None
            if use_prefix_attn:
                # Assemble per-row image-block ids into [batch, max_seq_len]; the
                # padded (seqlen-masked) tail stays -1.
                block_sequence_ids = torch.full(
                    (batch_size, max_seq_len), -1,
                    dtype=torch.int32, device=self.device,
                )
                for r, seq_block_ids in enumerate(per_block_ids):
                    if seq_block_ids is not None:
                        block_sequence_ids[r, : int(seq_block_ids.shape[0])].copy_(
                            seq_block_ids
                        )

            # Commit page table rows for all batch indices before forward pass.
            self.page_table.commit_block_table(batch_indices)

            lora_slot = lora_slots[0] if lora_slots else 0
            hidden, logits = self._prefill_impl(
                inputs_embeds,
                None,  # attention_mask
                position_ids,
                batch_idx=slot_batch_idx,
                lora_slot=lora_slot,
                use_prefix_attn=use_prefix_attn,
                block_sequence_ids=block_sequence_ids,
                paged_kv_seqlens_q=paged_kv_seqlens_q,
                paged_kv_seqlens_k=paged_kv_seqlens_k,
                last_token_positions=last_token_positions,
                input_staging=input_staging,
            )

            gather_idx = last_token_positions.to(dtype=torch.long).view(
                batch_size, 1, 1
            )
            gather_idx = gather_idx.expand(batch_size, 1, hidden.shape[-1])
            hidden_last = hidden.gather(1, gather_idx).squeeze(1)
            for row, prepared in enumerate(prepared_sequences):
                prepared.state.last_hidden = hidden_last[row].detach()

        return logits

    def finalize_prepared_sequence_after_prefill(
        self, prepared: PreparedSequence
    ) -> None:
        """Finalize prefix cache state after prefill completes."""

        state = prepared.state
        prompt_len = state.prompt_length or state.length
        cache_lock_node, cache_owned_page_count = self._finalize_cache_after_prefill(
            prepared.cache_tokens,
            prepared.cache_result,
            prompt_len,
            state.batch_idx,
            prepared.adapter_id,
            prepared.image_hash,
        )
        state.cache_lock_node = cache_lock_node
        state.cache_owned_page_count = cache_owned_page_count
        self.active_sequences[state.batch_idx] = state

    def abort_prepared_sequence(self, prepared: PreparedSequence) -> None:
        """Abort a prepared sequence and release its reserved resources."""

        # Release temp cache lock if held.
        if self.prefix_cache is not None and prepared.cache_result.temp_lock_node is not None:
            self.prefix_cache.unlock_prefill(prepared.cache_result.temp_lock_node)

        # Ensure we don't keep an incomplete sequence registered.
        batch_idx = prepared.state.batch_idx
        self.active_sequences.pop(batch_idx, None)

        # Release batch slot; do NOT free cache-owned mapped prefix pages.
        self.page_table.erase(batch_idx, prepared.cache_result.skip_positions)

    def retain_sequence_prefix(
        self,
        state: SequenceState,
        generated_tokens: Sequence[Token],
        *,
        adapter_id: str | None,
        image_hash: bytes | None,
    ) -> None:
        """Make decoded generated KV pages eligible for prefix-cache reuse."""
        if self.prefix_cache is None or not state.cache_tokens:
            return

        prompt_len = state.prompt_length or state.length
        processed_generated = min(
            len(generated_tokens),
            max(0, state.length - prompt_len),
        )
        if processed_generated <= 0:
            return

        target_len = prompt_len + processed_generated
        if target_len <= state.cache_owned_page_count:
            return

        cache_tokens: list[CacheToken] = (
            list(state.cache_tokens) + list(generated_tokens[:processed_generated])
        )
        pages = self.page_table.get_pages(state.batch_idx, 0, target_len)
        namespace = self._cache_namespace(adapter_id, image_hash)
        insert_result = self.prefix_cache.insert(
            cache_tokens,
            pages,
            namespace=namespace,
        )
        if insert_result.inserted_pages == 0:
            # Normal non-retainable case: the token path may already exist with
            # different physical pages. This happens after append-prefill
            # recomputes the last prompt token, or when another branch cached a
            # shared generated prefix first. Keep this sequence's private pages
            # private so release_sequence() frees them normally.
            return

        old_lock_node = state.cache_lock_node
        if insert_result.node is not old_lock_node:
            self.prefix_cache.lock(insert_result.node)
            self.prefix_cache.unlock(old_lock_node)
            state.cache_lock_node = insert_result.node
        state.cache_owned_page_count = target_len

    def release_sequence(self, state: SequenceState) -> None:
        self.active_sequences.pop(state.batch_idx, None)

        # Unlock the cached prefix (exactly one unlock per sequence)
        if self.prefix_cache is not None and state.cache_lock_node is not None:
            self.prefix_cache.unlock(state.cache_lock_node)

        # Release batch slot
        # Don't free cache-owned pages - they belong to the prefix cache tree
        self.page_table.erase(state.batch_idx, state.cache_owned_page_count)

        # Release the adapter slot (no-op if lora_slot == 0)
        self.release_adapter_slot(state.lora_slot)

    # ------------------------------------------------------------------
    # Core forward paths

    def _prefill_impl(
        self,
        inputs_embeds: Tensor,
        attn_mask: Optional[Tensor],
        position_ids: Tensor,
        batch_idx: Tensor,
        lora_slot: int = 0,
        *,
        use_prefix_attn: bool,
        block_sequence_ids: Tensor | None = None,
        paged_kv_seqlens_q: Tensor | None,
        paged_kv_seqlens_k: Tensor,
        last_token_positions: Tensor,
        input_staging: PrefillInputStaging,
    ) -> tuple[Tensor, Tensor]:
        if batch_idx.ndim != 2:
            raise ValueError("batch_idx must be rank-2")
        if batch_idx.shape != position_ids.shape:
            raise ValueError("batch_idx and position_ids must have matching shape")
        batch_rows = batch_idx[:, 0].to(dtype=torch.long)

        batch_size = int(inputs_embeds.shape[0])
        if batch_size > 1 and lora_slot != 0:
            raise NotImplementedError("Batched prefill does not yet support LoRA slots")

        slot_mapping = self.page_table.build_slot_mapping(
            batch_idx=batch_idx, positions=position_ids
        )

        # Build paged attention metadata for prefill.
        paged_kv_page_table = torch.index_select(
            self.page_table.page_table, 0, batch_rows
        )

        # For no-adapter prefill, skip LoRA entirely to avoid redundant work
        M = inputs_embeds.size(0) * inputs_embeds.size(1)
        if lora_slot == 0:
            lora_workspace = None
            lora_slot_ids = None
            moe_lora_metadata = None
        else:
            lora_workspace = self._lora_workspace
            moe_runtime = get_runtime().moe
            max_loras = max(0, lora_workspace.max_slots - 1) if lora_workspace else 0
            active_lora_max_rank = (
                lora_workspace.moe_lora_rank_for_slot(lora_slot)
                if lora_workspace is not None
                else None
            )
            lora_slot_ids = input_staging.stage_lora_slot_ids(lora_slot)
            token_lora_slot_ids_cpu = input_staging.fill_token_lora_slots(
                int(M), lora_slot
            )
            (
                active_token_ids_cpu,
                active_token_ids_gpu,
                active_lora_ids_cpu,
                active_lora_ids_gpu,
                active_lora_meta_cpu,
                active_lora_meta_gpu,
            ) = input_staging.lora_metadata_buffers(
                token_count=int(M), max_loras=max_loras
            )
            moe_lora_metadata = moe_runtime.prepare_lora_metadata(
                lora_slot_ids_cpu=token_lora_slot_ids_cpu,
                active_token_ids_cpu=active_token_ids_cpu,
                active_token_ids_gpu=active_token_ids_gpu,
                active_lora_ids_cpu=active_lora_ids_cpu,
                active_lora_ids_gpu=active_lora_ids_gpu,
                # 3 header ints + one start offset per active LoRA + one sentinel
                # end offset, so kernels can read route i as meta[3+i]..meta[4+i].
                active_lora_meta_cpu=active_lora_meta_cpu,
                active_lora_meta_gpu=active_lora_meta_gpu,
                batch_size=int(M),
                max_loras=max_loras,
                lora_ranks_host=[
                    lora_workspace.moe_lora_rank_for_id(i) for i in range(max_loras)
                ] if lora_workspace else None,
                active_lora_max_rank=active_lora_max_rank,
            )

        # Build scratch buffer pool for prefill to avoid per-layer allocations.
        tc = self.config.text
        head_dim = tc.dim // tc.n_heads
        scratch_pool = {
            (M, tc.dim + 2 * tc.n_kv_heads * head_dim): torch.empty(
                M, tc.dim + 2 * tc.n_kv_heads * head_dim,
                dtype=inputs_embeds.dtype, device=self.device),  # QKV
            (M, 2 * tc.n_heads): torch.empty(
                M, 2 * tc.n_heads,
                dtype=inputs_embeds.dtype, device=self.device),  # tau_wqwv
        }
        if tc.tau_attn:
            qkv_dim = int(tc.dim * (1 + 2 * tc.n_kv_heads / tc.n_heads))
            scratch_pool["gelu"] = torch.empty(
                M, qkv_dim,
                dtype=inputs_embeds.dtype, device=self.device)  # GELU for tau
        if tc.moe and tc.moe.num_experts:
            scratch_pool[(M, tc.moe.num_experts)] = torch.empty(
                M, tc.moe.num_experts,
                dtype=inputs_embeds.dtype, device=self.device)  # router

        hidden = text_decoder(
            inputs_embeds,
            self.model.text,
            attn_mask,
            position_ids,
            self.config.text,
            slot_mapping=slot_mapping,
            mode="prefill",
            use_prefix_attn=use_prefix_attn,
            block_sequence_ids=block_sequence_ids,
            page_table=paged_kv_page_table,
            paged_kv_seqlens_q=paged_kv_seqlens_q,
            paged_kv_seqlens_k=paged_kv_seqlens_k,
            lora_workspace=lora_workspace,
            lora_slot_ids=lora_slot_ids,
            moe_lora_metadata=moe_lora_metadata,
            scratch_pool=scratch_pool,
        )

        batch_size = int(hidden.shape[0])
        gather_idx = last_token_positions.to(dtype=torch.long).view(batch_size, 1, 1)
        gather_idx = gather_idx.expand(batch_size, 1, hidden.shape[-1])
        hidden_last = hidden.gather(1, gather_idx).squeeze(1)
        logits = lm_head(hidden_last.unsqueeze(1), self.model.text)
        return hidden, logits

    def decode_with_slot(self, slot: DecodeSlot, batch_size: int) -> None:
        """Run batched decode forward pass using per-slot resources.

        IMPORTANT: Caller must ensure:
        - Already on slot.compute_stream (via `with torch.cuda.stream()`)
        - Inputs staged in slot buffers (decode_token_ids, decode_coord_values, etc.)
        - Metadata copied to GPU (batch_idx, input_pos, lora_slot_ids)

        This method runs the forward pass and writes results to slot.logits
        and slot.hidden_last.
        """
        self._decode_with_slot(slot, batch_size)

    def _decode_with_slot(self, slot: DecodeSlot, batch_size: int) -> None:
        """Unified decode using slot buffers. Writes results to slot.logits/hidden_last.

        Args:
            slot: DecodeSlot with inputs already staged in its buffers.
            batch_size: Actual number of sequences (before padding).
        """
        self._decode_graphs.run(slot, batch_size)

    def _zero_decode_graph_padding(
        self,
        slot: DecodeSlot,
        batch_size: int,
        graph_batch_size: int,
    ) -> None:
        slot.decode_token_ids[batch_size:graph_batch_size].zero_()
        slot.decode_coord_values[batch_size:graph_batch_size].zero_()
        slot.decode_size_values[batch_size:graph_batch_size].zero_()
        slot.meta.batch_idx.gpu[batch_size:graph_batch_size].zero_()
        slot.meta.input_pos.gpu[batch_size:graph_batch_size].zero_()
        slot.meta.lora_slot_ids.gpu[batch_size:graph_batch_size].zero_()
        slot.meta.lora_slot_ids.cpu[batch_size:graph_batch_size].zero_()

    def _prepare_decode_graph_step(self, slot: DecodeSlot, batch_size: int) -> None:
        # Only host-side LoRA metadata stays live here; the paged-KV metadata is
        # built inside the captured decode graph (see _run_decode_forward).
        if self._lora_workspace is not None:
            self._prepare_decode_lora_metadata(slot, batch_size)

    def _zero_decode_graph_capture_buffers(self, slot: DecodeSlot) -> None:
        # Use batch index 0 for all entries: page-table row 0 is initialized and
        # provides valid memory access patterns during capture warmup/replay.
        slot.decode_token_ids.zero_()
        slot.decode_coord_values.zero_()
        slot.decode_size_values.zero_()
        slot.meta.batch_idx.gpu.zero_()
        slot.meta.input_pos.gpu.zero_()
        slot.meta.input_pos.cpu.zero_()
        slot.meta.lora_slot_ids.gpu.zero_()
        slot.meta.active_token_ids.gpu.zero_()
        slot.meta.active_token_ids.cpu.zero_()
        slot.meta.active_lora_ids.gpu.zero_()
        slot.meta.active_lora_ids.cpu.zero_()
        slot.meta.active_lora_meta.gpu.zero_()
        slot.meta.active_lora_meta.cpu.zero_()
        slot.paged_kv_page_table.zero_()
        slot.paged_kv_seqlens_k.zero_()

    def _prepare_decode_lora_metadata(
        self,
        slot: DecodeSlot,
        batch_size: int,
    ) -> None:
        workspace = self._lora_workspace
        if workspace is None:
            slot.meta.active_token_ids.np[:batch_size] = 0
            slot.meta.active_lora_ids.np[:batch_size] = 0
            slot.meta.active_lora_meta.np[:] = 0
            slot.meta.active_token_ids.copy_to_gpu(batch_size)
            slot.meta.active_lora_ids.copy_to_gpu(batch_size)
            slot.meta.active_lora_meta.copy_to_gpu()
            slot.meta.moe_lora_metadata = None
            return

        max_loras = max(0, workspace.max_slots - 1)
        slot.meta.moe_lora_metadata = get_runtime().moe.prepare_lora_metadata(
            lora_slot_ids_cpu=slot.meta.lora_slot_ids.cpu,
            active_token_ids_cpu=slot.meta.active_token_ids.cpu,
            active_token_ids_gpu=slot.meta.active_token_ids.gpu,
            active_lora_ids_cpu=slot.meta.active_lora_ids.cpu,
            active_lora_ids_gpu=slot.meta.active_lora_ids.gpu,
            active_lora_meta_cpu=slot.meta.active_lora_meta.cpu,
            active_lora_meta_gpu=slot.meta.active_lora_meta.gpu,
            batch_size=batch_size,
            max_loras=max_loras,
            lora_ranks_host=[
                workspace.moe_lora_rank_for_id(i) for i in range(max_loras)
            ],
            fixed_capacity=True,
        )

    def _run_decode_forward(
        self,
        slot: DecodeSlot,
        batch_size: int,
    ) -> None:
        """Run decode forward pass and write results to slot output buffers.

        This is the core forward computation, used by both eager decode and
        CUDA graph replay.

        Args:
            slot: DecodeSlot with inputs in its buffers.
            batch_size: Batch size (may be padded for graph capture).
        """
        embeds = self._embed_packed_token_batch(
            slot.decode_token_ids[:batch_size],
            slot.decode_coord_values[:batch_size],
            slot.decode_size_values[:batch_size],
        )
        batch_idx = slot.meta.batch_idx.gpu[:batch_size]
        # Build the paged-KV metadata here, inside the captured decode graph,
        # rather than eagerly each step. It reads the static (pre-reserved) page
        # table and the engine-written batch_idx/input_pos buffers and writes the
        # slot's fixed metadata buffers, so a graph replay reproduces it
        # bit-identically while folding the launch into the single graph replay
        # (this is already how build_slot_mapping below is handled).
        self.page_table.populate_paged_kv_metadata(
            batch_idx=batch_idx,
            input_pos=slot.meta.input_pos.gpu[:batch_size],
            out_page_table=slot.paged_kv_page_table[:batch_size],
            out_seqused_k=slot.paged_kv_seqlens_k[:batch_size],
        )
        position_ids = slot.meta.input_pos.gpu[:batch_size].to(torch.long).view(-1, 1)
        slot_mapping = self.page_table.build_slot_mapping(
            batch_idx=batch_idx.view(-1, 1),
            positions=position_ids,
        )
        lora_workspace = self._lora_workspace
        lora_slot_ids = None
        moe_lora_metadata = None
        if lora_workspace is not None:
            lora_slot_ids = slot.meta.lora_slot_ids.gpu[:batch_size]
            moe_lora_metadata = slot.meta.moe_lora_metadata

        hidden = text_decoder(
            embeds,
            self.model.text,
            attn_mask=None,
            position_ids=position_ids,
            config=self.config.text,
            mode="decode",
            slot_mapping=slot_mapping,
            page_table=slot.paged_kv_page_table[:batch_size],
            paged_kv_seqlens_k=slot.paged_kv_seqlens_k[:batch_size],
            lora_workspace=lora_workspace,
            lora_slot_ids=lora_slot_ids,
            moe_lora_metadata=moe_lora_metadata,
            dense_lora_scratch=self._dense_lora_decode_scratch,
        )
        logits = lm_head(hidden, self.model.text)

        # Write to slot output buffers (stable addresses for graph capture)
        slot.logits[:batch_size].copy_(logits)
        slot.hidden_last[:batch_size].copy_(hidden[:, 0, :])

    def acquire_adapter_slot(self, adapter_id: str, adapter: LoRA) -> int:
        """Acquire a slot for an adapter, loading weights if necessary.

        Uses the slot manager to either reuse an existing slot (if the adapter is
        already resident) or allocate a new one. If newly allocated, copies the
        adapter weights into the workspace.

        Args:
            adapter_id: Identifier for the adapter (from settings.adapter).
            adapter: The LoRA adapter to load (must be on same device/dtype).

        Returns:
            The slot number assigned to this adapter.

        Raises:
            NotImplementedError: If no adapter provider is configured, vision LoRA
                is provided, or adapter is on wrong device/dtype.
            ValueError: If adapter rank exceeds max_lora_rank.
            RuntimeError: If no free slots are available.
        """
        if self._lora_workspace is None or self._slot_manager is None:
            raise NotImplementedError(
                "Adapter provider is not configured for this runtime."
            )
        if adapter.vision is not None:
            raise NotImplementedError("Vision LoRA is not supported.")

        if adapter.text.rank > self._max_lora_rank:
            raise ValueError(
                f"Adapter rank ({adapter.text.rank}) exceeds max_lora_rank ({self._max_lora_rank})."
            )

        # Require CUDA tensors and exact dtype/device matches.
        try:
            sample_param = next(adapter.text.parameters())
        except StopIteration as exc:  # pragma: no cover - defensive
            raise ValueError("Adapter contains no parameters") from exc
        if sample_param.device != self.device:
            raise NotImplementedError(
                f"Adapter must be on device {self.device}; received {sample_param.device}."
            )
        if sample_param.dtype != self.dtype:
            raise NotImplementedError(
                f"Adapter must have dtype {self.dtype}; received {sample_param.dtype}."
            )

        # Acquire slot (reuse if already resident, allocate if new)
        slot, is_new = self._slot_manager.acquire(adapter_id)

        if is_new:
            try:
                with stream_context(self._compute_stream):
                    self._lora_workspace.load_slot_(slot, adapter)
            except Exception:
                # Rollback on load failure
                self._slot_manager.release_on_error(slot)
                raise

        return slot

    def release_adapter_slot(self, slot: int) -> None:
        """Release a reference to an adapter slot.

        Decrements the slot's refcount. When the last reference is released,
        the slot is returned to the free pool.

        Args:
            slot: The slot to release (0 is a no-op).
        """
        if slot == 0:
            return  # No LoRA, nothing to release

        if self._slot_manager is None:
            return

        self._slot_manager.release(slot)

    def rebuild_cuda_graphs(self) -> None:
        """Reset and recapture CUDA graphs used for decode and vision encoding.

        This is intended for workflows that mutate runtime-owned tensors
        (e.g. weight tying or hot-swapping checkpoints) where a previously
        captured CUDA graph might replay with stale tensor pointers.

        Callers must ensure no CUDA work is in flight (e.g. pause the engine)
        before invoking this method.
        """

        if not self._use_cuda_graphs:
            return

        with self.graph_capture_lock:
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.set_device(self.device)

            # Clear vision graphs
            self._vision_graphs = {}

            self._decode_graphs.clear()

            self._ensure_cuda_graphs_ready()
            self._capture_vision_graphs()

    def _ensure_cuda_graphs_ready(self) -> None:
        """Capture CUDA graphs for all slots using slot buffers directly."""
        if not self._use_cuda_graphs:
            return
        self._decode_graphs.ensure_ready(self._decode_slots)

    def _preallocate_workspaces(self) -> None:
        """Pre-allocate all shared workspaces to ensure stable pointers for CUDA graphs.

        All MoE layers share a single set of workspace buffers since they execute
        sequentially. This reduces memory from O(num_layers * workspace) to O(workspace).
        The buffers are fixed-size; requesting more tokens than allocated raises an error.
        """
        # Pre-allocate vision fused MLP workspace
        # max_tokens = (max_crops + 1) * patches_per_crop
        # The +1 accounts for the global/overview crop added by overlap_crop_image
        vision_cfg = self.config.vision
        patches_per_crop = (vision_cfg.crop_size // vision_cfg.enc_patch_size) ** 2
        max_vision_tokens = (vision_cfg.max_crops + 1) * patches_per_crop
        from kestrel.ops.fused_mlp import preallocate_fused_mlp_workspaces
        preallocate_fused_mlp_workspaces(
            max_num_tokens=max_vision_tokens,
            hidden_dim=vision_cfg.enc_ff_dim,
            device=self.device,
            dtype=self.dtype,
        )

        # Base MoE workspaces are owned by kestrel-kernels MoeHandle
        # preparation. MoE LoRA scratch is reserved when the LoRA workspace is
        # created so CUDA graph capture sees stable intermediate buffers.

    def _allocate_vision_buffers(self) -> None:
        """Allocate stable buffers for vision encoder."""
        config = self.config.vision
        max_batch = config.max_crops + 1  # Support up to max_crops + 1
        patches = (config.crop_size // config.enc_patch_size) ** 2  # 729

        self._vision_input = torch.zeros(
            (max_batch, config.in_channels, config.crop_size, config.crop_size),
            dtype=self.dtype, device=self.device
        )
        self._vision_output = torch.empty(
            (max_batch, patches, config.enc_dim),
            dtype=self.dtype, device=self.device
        )

    def _capture_vision_graphs(self) -> None:
        """Capture CUDA graphs for vision encoder at all crop counts."""
        if not self._use_cuda_graphs:
            return

        config = self.config.vision
        max_batch = config.max_crops + 1

        with self.graph_capture_lock:
            torch.cuda.synchronize(device=self.device)

            # Capture largest batch first, then hint smaller captures to reuse
            # its graph memory pool. This lowers peak graph-reserved memory.
            shared_pool = None
            for batch_size in range(max_batch, 0, -1):
                graph = torch.cuda.CUDAGraph()

                with torch.inference_mode():
                    # Warmup
                    out = vision_encoder(
                        self._vision_input[:batch_size],
                        self.model.vision,
                        config,
                    )
                    self._vision_output[:batch_size].copy_(out)
                    torch.cuda.synchronize(device=self.device)

                    # Capture
                    with torch.cuda.graph(graph, pool=shared_pool):
                        out = vision_encoder(
                            self._vision_input[:batch_size],
                            self.model.vision,
                            config,
                        )
                        self._vision_output[:batch_size].copy_(out)

                if shared_pool is None:
                    shared_pool = graph.pool()
                self._vision_graphs[batch_size] = graph
                torch.cuda.synchronize(device=self.device)


__all__ = ["MoondreamRuntime", "SequenceState", "DEFAULT_MAX_TOKENS"]
