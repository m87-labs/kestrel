# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.

# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128.
# - (hdim_qk, hdim_v) = (192, 128) for Blackwell (i.e. DeepSeek shape)
# - varlen
# - sliding window
# - bwd pass for Ampere (will also run on Hopper/Blackwell, but will be slow)

# Features not supported yet:
# - general SplitKV (except the SM90 page_size==1 decode fused path)
# - tuned block sizes
# - append KV to existing KV cache
# - FP8
# - bwd pass optimized for Hopper/Blackwell

import math
import warnings
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch


@lru_cache(maxsize=None)
def _get_device_capability():
    """Cached device capability check."""
    return torch.cuda.get_device_capability()[0]

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from kestrel_kernels.flash_attn.cute import utils

# Lazy import cache for kernel classes - only imported when JIT compilation is needed.
# These modules are not included in the distributed wheel.
_KERNEL_CLASSES = {}

def _get_kernel_class(name: str):
    """Lazily import kernel class for JIT compilation."""
    if name in _KERNEL_CLASSES:
        return _KERNEL_CLASSES[name]

    try:
        if name == "FlashAttentionForwardSm80":
            from kestrel_kernels.flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
            _KERNEL_CLASSES[name] = FlashAttentionForwardSm80
        elif name == "FlashAttentionForwardSm90":
            from kestrel_kernels.flash_attn.cute.flash_fwd import FlashAttentionForwardSm90
            _KERNEL_CLASSES[name] = FlashAttentionForwardSm90
        elif name == "FlashAttentionDecodeSm90":
            from kestrel_kernels.flash_attn.cute.flash_decode_sm90 import FlashAttentionDecodeSm90
            _KERNEL_CLASSES[name] = FlashAttentionDecodeSm90
        elif name == "FlashAttentionDecodeSm90PersistentSplitFused":
            from kestrel_kernels.flash_attn.cute.flash_decode_sm90_persistent_fused import FlashAttentionDecodeSm90PersistentSplitFused
            _KERNEL_CLASSES[name] = FlashAttentionDecodeSm90PersistentSplitFused
        elif name == "FlashAttentionForwardSm100":
            from kestrel_kernels.flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
            _KERNEL_CLASSES[name] = FlashAttentionForwardSm100
        elif name == "FlashAttentionBackwardPreprocess":
            from kestrel_kernels.flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
            _KERNEL_CLASSES[name] = FlashAttentionBackwardPreprocess
        elif name == "FlashAttentionBackwardSm80":
            from kestrel_kernels.flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
            _KERNEL_CLASSES[name] = FlashAttentionBackwardSm80
        elif name == "FlashAttentionBackwardSm90":
            from kestrel_kernels.flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
            _KERNEL_CLASSES[name] = FlashAttentionBackwardSm90
        elif name == "FlashAttentionBackwardSm100":
            from kestrel_kernels.flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
            _KERNEL_CLASSES[name] = FlashAttentionBackwardSm100
        elif name == "FlashAttentionBackwardPostprocess":
            from kestrel_kernels.flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
            _KERNEL_CLASSES[name] = FlashAttentionBackwardPostprocess
        elif name == "FlashAttentionForwardCombine":
            from kestrel_kernels.flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine
            _KERNEL_CLASSES[name] = FlashAttentionForwardCombine
        else:
            raise ValueError(f"Unknown kernel class: {name}")
    except ImportError as e:
        raise RuntimeError(
            f"JIT compilation requires source install. "
            f"Kernel template '{name}' not available in wheel distribution. "
            f"Install from source for JIT support."
        ) from e

    return _KERNEL_CLASSES[name]

from kestrel_kernels.flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_tensors,
    get_block_sparse_expected_shapes,
    get_block_sparse_expected_shapes_bwd,
)
from kestrel_kernels.precompile import get_cuda_arch, load_precompiled_module

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    assert t.is_cuda, f"{name} must be on CUDA"

def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

# Import mask for hash comparison in precompiled kernel lookup
from kestrel_kernels.flash_attn.cute.mask_definitions import cute_prefix_lm_mask_730

_cute_dtype_to_name = {
    cutlass.BFloat16: "bf16",
    cutlass.Float16: "f16",
}


def _try_load_precompiled_flash_attn(compile_key, use_sm90_decode_fastpath: bool):
    """Try to load a precompiled flash attention kernel if available.

    Returns the loaded kernel function or None if not available.
    """
    # Unpack compile_key fields we care about
    (
        dtype, dtype_kv, head_dim, head_dim_v, qhead_per_kvhead, causal,
        score_mod_hash, mask_mod_hash, use_block_sparsity, n_aux,
        lse_is_none, cu_seqlens_q_none, cu_seqlens_k_none,
        seqused_q_none, seqused_k_none, has_page_table,
        has_window_left, has_window_right, has_sink,
        m_block, n_block, num_threads, num_splits, is_split_kv,
        pack_gqa, cc, paged_kv_non_tma, use_decode_fastpath, intra_wg_overlap,
    ) = compile_key

    # Only precompile for SM90
    if cc != 9:
        return None

    # Check basic requirements for all precompiled variants
    # Note: hash values are False (not None) when the callable is not present
    if score_mod_hash:
        return None
    if use_block_sparsity or n_aux > 0:
        return None
    if has_window_left or has_window_right or has_sink:
        return None
    if qhead_per_kvhead != 1:  # Only MHA for now
        return None

    dtype_name = _cute_dtype_to_name.get(dtype)
    if dtype_name is None:
        return None

    arch = get_cuda_arch()

    if use_sm90_decode_fastpath:
        # Decode variant
        if head_dim not in (64, 128):
            return None

        # Determine KV dtype name
        if dtype_kv == cutlass.Float8E4M3FN:
            dtype_kv_name = "fp8_e4m3"
        elif dtype_kv == cutlass.BFloat16:
            dtype_kv_name = "bf16"
        elif dtype_kv == cutlass.Float16:
            dtype_kv_name = "f16"
        else:
            return None

        # Decode is always causal
        if not causal:
            return None

        filename = f"flash_attn_decode_sm90_{dtype_name}_{dtype_kv_name}_hd{head_dim}_causal_{arch}.so"
    else:
        # Forward variant
        if head_dim not in (64, 72, 128):
            return None
        if head_dim != head_dim_v:
            return None

        # Determine KV dtype name for forward
        # For FP8 paged prefill, dtype_kv is Float8E4M3FN
        if dtype_kv is not None and dtype_kv != dtype:
            if dtype_kv == cutlass.Float8E4M3FN:
                dtype_kv_name = "fp8_e4m3"
            else:
                return None  # Unsupported dtype_kv for forward
            dtype_kv_str = f"_{dtype_kv_name}"
        else:
            dtype_kv_str = ""

        # Check for matching mask
        # Note: mask_mod_hash is False (not None) when no mask is provided
        mask_str = ""
        if mask_mod_hash:
            # Check if it's the prefix_lm_730 mask
            prefix_lm_hash = getattr(cute_prefix_lm_mask_730, "__cute_hash__", None)
            if mask_mod_hash == prefix_lm_hash:
                mask_str = "_prefix_lm_730"
            else:
                return None  # Unknown mask, can't use precompiled

        # Check for standard tile sizes
        if m_block != 128 or n_block != 128 or num_threads != 384:
            return None

        causal_str = "_causal" if causal else ""
        paged_str = "_paged" if paged_kv_non_tma else ""

        filename = f"flash_attn_forward_sm90_{dtype_name}{dtype_kv_str}_hd{head_dim}{causal_str}{mask_str}{paged_str}_{arch}.so"

    # Try to load the module
    mod = load_precompiled_module(filename)
    if mod is None:
        return None

    fn_name = filename.replace(".so", "")
    kernel_fn = getattr(mod, fn_name, None)
    if kernel_fn is None:
        return None

    # Create a wrapper that removes the stream parameter (position 8, 0-indexed)
    # because TVM FFI precompiled kernels don't take stream as a parameter.
    # The runtime call signature is:
    #   (q, k, v, out, lse, scale, k_scale, v_scale, stream, cu_seqlens_q, ...)
    # But precompiled expects:
    #   (q, k, v, out, lse, scale, k_scale, v_scale, cu_seqlens_q, ...)
    def precompiled_wrapper(
        q, k, v, out, lse, softmax_scale, k_scale, v_scale,
        stream,  # This is ignored - TVM FFI uses env stream
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, page_table,
        window_size_left, window_size_right, learnable_sink,
        sparse_tensors, aux_tensors,
    ):
        return kernel_fn(
            q, k, v, out, lse, softmax_scale, k_scale, v_scale,
            cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, page_table,
            window_size_left, window_size_right, learnable_sink,
            sparse_tensors, aux_tensors,
        )

    return precompiled_wrapper


def _try_load_precompiled_persistent_split_fused(
    dtype, dtype_kv, head_dim: int, num_splits: int, split_tokens: int
):
    """Try to load a precompiled persistent split-fused decode kernel if available.

    Returns the loaded kernel function or None if not available.
    """
    arch = get_cuda_arch()

    # Only precompile for SM90 and head_dim 64 (Moondream config)
    if not arch.startswith("sm9"):
        return None
    if head_dim != 64:
        return None

    dtype_name = _cute_dtype_to_name.get(dtype)
    if dtype_name is None:
        return None

    # Determine KV dtype name
    if dtype_kv == cutlass.Float8E4M3FN:
        dtype_kv_name = "fp8_e4m3"
    elif dtype_kv == cutlass.BFloat16:
        dtype_kv_name = "bf16"
    elif dtype_kv == cutlass.Float16:
        dtype_kv_name = "f16"
    else:
        return None

    # Try both sm90 and sm90a suffixes since precompilation might generate either
    arch_variants = [arch]
    if arch == "sm90a":
        arch_variants.append("sm90")
    elif arch == "sm90":
        arch_variants.append("sm90a")

    for try_arch in arch_variants:
        filename = (
            f"flash_attn_decode_persistent_split_fused_sm90_{dtype_name}_"
            f"{dtype_kv_name}_hd{head_dim}_s{num_splits}_t{split_tokens}_{try_arch}.so"
        )

        mod = load_precompiled_module(filename)
        if mod is None:
            continue

        fn_name = filename.replace(".so", "")
        kernel_fn = getattr(mod, fn_name, None)
        if kernel_fn is None:
            continue

        # Create wrapper that removes stream parameter (TVM FFI uses env stream)
        def precompiled_wrapper(
            q, k, v, out, lse, seqused_k, page_table,
            out_partial, lse_partial, split_counters,
            softmax_scale, k_scale, v_scale,
            window_size_left, window_size_right,
            stream,  # Ignored - TVM FFI uses env stream
        ):
            return kernel_fn(
                q, k, v, out, lse, seqused_k, page_table,
                out_partial, lse_partial, split_counters,
                softmax_scale, k_scale, v_scale,
                window_size_left, window_size_right,
            )

        return precompiled_wrapper

    return None


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)

_sm90_decode_persistent_split_fused_counter_cache: dict[
    tuple[int, int, int, int], torch.Tensor
] = {}


def _get_sm90_decode_persistent_split_fused_counters(
    *,
    device: torch.device,
    batch_size: int,
    num_kv_heads: int,
    group_count: int,
) -> torch.Tensor:
    """Get a reusable int32 counter buffer for the fused persistent split path.

    Important: we intentionally do *not* zero the buffer here to avoid capturing a memset
    into CUDA graphs. The fused kernel resets counters to 0 after each replay.
    """
    if device.type != "cuda":
        raise ValueError(f"Expected CUDA device; got {device!r}")
    if device.index is None:
        dev_index = int(torch.cuda.current_device())
        device = torch.device("cuda", dev_index)
    else:
        dev_index = int(device.index)

    batch_size = int(batch_size)
    num_kv_heads = int(num_kv_heads)
    group_count = int(group_count)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive (got {batch_size})")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive (got {num_kv_heads})")
    if group_count <= 0:
        raise ValueError(f"group_count must be positive (got {group_count})")

    shape = (batch_size, num_kv_heads * group_count)
    key = (dev_index, batch_size, num_kv_heads, group_count)
    buf = _sm90_decode_persistent_split_fused_counter_cache.get(key)
    if (
        buf is None
        or buf.device != device
        or buf.dtype != torch.int32
        or tuple(buf.shape) != shape
    ):
        buf = torch.zeros(shape, device=device, dtype=torch.int32)
        _sm90_decode_persistent_split_fused_counter_cache[key] = buf
    return buf


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    paged_kv_non_tma: Optional[bool] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
    """
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    if isinstance(k_scale, torch.Tensor):
        raise TypeError("k_scale must be a float, not a tensor (tensor.item() causes GPU sync)")
    if isinstance(v_scale, torch.Tensor):
        raise TypeError("v_scale must be a float, not a tensor (tensor.item() causes GPU sync)")
    k_scale_val = 1.0 if k_scale is None else float(k_scale)
    v_scale_val = 1.0 if v_scale is None else float(v_scale)
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    assert q.dtype in [torch.float16, torch.bfloat16], "q must be float16 or bfloat16"
    kv_is_fp8 = (k.dtype == torch.float8_e4m3fn) or (v.dtype == torch.float8_e4m3fn)
    if kv_is_fp8:
        if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"FP8 KV cache requires k and v both be float8_e4m3fn (got k={k.dtype}, v={v.dtype})"
            )
        if k_scale is None or v_scale is None:
            raise ValueError("FP8 KV cache requires k_scale and v_scale")
        if k_scale_val <= 0.0 or v_scale_val <= 0.0:
            raise ValueError("k_scale and v_scale must be positive for FP8 KV cache")
    else:
        assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    assert all(
        t is None or t.is_cuda
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        _validate_tensor(out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), out_torch_dtype, device)

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    dtype = torch2cute_dtype_map[q.dtype]
    dtype_kv = cutlass.Float8E4M3FN if kv_is_fp8 else dtype
    compute_capability = (
        _get_device_capability()
        if _compute_capability is None
        else _compute_capability
    )

    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"

    use_block_sparsity = block_sparse_tensors is not None

    # Common sentinel: -1 means "no limit". Treat as None to avoid negative window math.
    if window_size_left is not None and window_size_left < 0:
        window_size_left = None
    if window_size_right is not None and window_size_right < 0:
        window_size_right = None

    if mask_mod is None:
        if causal:
            window_size_right = 0
        local = window_size_left is not None or window_size_right is not None
        if window_size_left is not None or window_size_right is not None:
            if window_size_left is None and window_size_right == 0:
                causal, local = True, False
                window_size_right = None
            else:
                causal, local = False, True
    else:
        causal, local = False, False

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compute_capability == 9:  # TODO: tune block size according to hdim.
        if head_dim == head_dim_v == 128 and not causal and not local and not use_block_sparsity:
            n_block_size = 192

    if compute_capability == 10:
        # TODO: fix the varlen case
        if (
            pack_gqa
            and (128 % qhead_per_kvhead != 0)
            or (cu_seqlens_q is not None or seqused_q is not None)
        ):
            pack_gqa = False
        # TODO: fix GQA + SplitKV + non-varlen
        if pack_gqa and num_splits != 1 and cu_seqlens_q is None:
            pack_gqa = False

    if num_splits < 1:
        max_seqlen_k = seqlen_k if cu_seqlens_k is None else (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
        seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
        seqlen_k_loaded = max_seqlen_k if not local else max(0, min(max_seqlen_k, window_size_right + window_size_left + 1 + m_block_size))
        num_n_blocks = (seqlen_k_loaded + n_block_size - 1) // n_block_size
        num_m_blocks = (seqlen_q_packgqa + m_block_size - 1) // m_block_size
        total_mblocks = batch_size * num_head_kv * num_m_blocks
        num_splits = num_splits_heuristic(
            total_mblocks,
            torch.cuda.get_device_properties(device).multi_processor_count,
            num_n_blocks,
            128,
        )

    # Optional: SM90 decode-only persistent split path (page_size==1).
    #
    # Default behavior: enable only during CUDA graph capture (decode is graph-replayed in
    # production) and only for small batches where the baseline grid under-fills the GPU.
    #
    sm90_persist_oversub = 4
    sm90_persistent_split_enabled = (
        torch.cuda.is_current_stream_capturing()
        and (not requires_grad)
        and compute_capability == 9
        and batch_size <= 4
        and page_table is not None
        and page_size == 1
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is not None
        and seqlen_q == 1
        and head_dim_v == head_dim
        and score_mod is None
        and mask_mod is None
        and block_sparse_tensors is None
        and aux_tensors is None
        and learnable_sink is None
        and head_dim in (64, 128)
    )
    if sm90_persistent_split_enabled:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        # Baseline SM90 decode launches one CTA per (batch, kv_head_group).
        # Try to get kernel class for geometry calculation; if unavailable (wheel install),
        # use a simplified group_count calculation.
        try:
            FlashAttentionDecodeSm90PersistentSplitFused = _get_kernel_class("FlashAttentionDecodeSm90PersistentSplitFused")
            tmp_kernel = FlashAttentionDecodeSm90PersistentSplitFused(
                dtype=dtype,
                dtype_kv=dtype_kv,
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kvhead,
                num_splits=2,
                is_causal=causal,
                is_local=local,
                split_tokens=1,
                persist_oversub=sm90_persist_oversub,
            )
            group_count = tmp_kernel.group_count
        except RuntimeError:
            # Wheel install without JIT templates - use simplified group_count calculation
            # For qhead_per_kvhead=1 (MHA), group_count=1
            # For GQA with small qhead_per_kvhead, group_count is typically 1
            group_count = 1 if qhead_per_kvhead <= 8 else (qhead_per_kvhead + 7) // 8
        head_groups = num_head_kv * group_count
        baseline_blocks = batch_size * head_groups
        # Heuristic: use split-KV when the baseline grid is too small to fill the GPU.
        #
        # For this kernel on H100 we tend to be limited to ~4 CTAs/SM (register + smem),
        # so target at least one full "wave" of work:
        #
        #   target_tasks ~= sm_count * persist_oversub
        target_tasks = int(sm_count) * int(sm90_persist_oversub)
        desired_splits = (target_tasks + baseline_blocks - 1) // max(1, baseline_blocks)
        if desired_splits <= 1:
            sm90_persistent_split_enabled = False
            # Avoid accidentally selecting the unsupported generic SplitKV path on SM90.
            num_splits = 1
        else:
            # Compile with a small fixed max_splits (for graph capture), and let the kernel compute
            # active splits from seqused_k at runtime. This makes CUDA graph replay adapt to the
            # *actual* KV work (instead of being tied to the page_table capacity used during capture).
            if head_dim == 64 and int(qhead_per_kvhead) == 1:
                # MHA decode: baseline is already 1 CTA per head, so splitting too much can be
                # net-negative (extra partial traffic + counter coordination).
                #
                # Target ~1 CTA/SM worth of split work for the common B<=4 decode graphs.
                sm90_max_splits = (int(sm_count) + baseline_blocks - 1) // max(1, baseline_blocks)
                sm90_max_splits = max(2, min(int(sm90_max_splits), 4))
            else:
                sm90_max_splits = max(2, min(int(desired_splits), 6))

            # Split-token heuristic:
            # - Target ~740-1000 decode context lengths (Moondream-ish) and keep split sizes aligned.
            if head_dim == 64 and int(qhead_per_kvhead) == 1:
                # Keep the active split count modest for typical decode lengths.
                sm90_split_tokens = 256 if sm90_max_splits >= 3 else 64
            else:
                sm90_split_tokens = 192 if head_dim == 128 else 64

            out_partial = torch.empty(
                sm90_max_splits,
                *q_batch_seqlen_shape,
                num_head,
                head_dim_v,
                dtype=torch.float32,
                device=device,
            )
            lse_partial = torch.empty(
                sm90_max_splits, *lse_shape, dtype=torch.float32, device=device
            )
            # Counter is per (batch, head_group). Use group_count from earlier.
            split_counters = _get_sm90_decode_persistent_split_fused_counters(
                device=device,
                batch_size=batch_size,
                num_kv_heads=num_head_kv,
                group_count=group_count,
            )
            _flash_attn_sm90_decode_persistent_split_fused(
                q,
                k,
                v,
                out,
                lse,
                page_table=page_table,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
                k_scale=k_scale_val,
                v_scale=v_scale_val,
                causal=causal,
                local=local,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                qhead_per_kvhead=qhead_per_kvhead,
                num_splits=sm90_max_splits,
                split_tokens=sm90_split_tokens,
                persist_oversub=sm90_persist_oversub,
                out_partial=out_partial,
                lse_partial=lse_partial,
                split_counters=split_counters,
            )
            _flash_attn_fwd._debug_last_impl = "sm90_decode_persistent_split_fused"
            return out, lse

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(num_splits, *q_batch_seqlen_shape, num_head, head_dim_v, dtype=torch.float32, device=device)
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    if mask_mod is not None:
        if pack_gqa:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )

    if use_block_sparsity:
        if is_varlen:
            raise NotImplementedError(
                "Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        if pack_gqa:
            raise NotImplementedError(
                "Block sparsity is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )

    inferred_paged_kv_non_tma = paged_kv_non_tma
    if inferred_paged_kv_non_tma is None:
        if compute_capability == 9:
            # For SM90, use the non-TMA (cp.async) path for all paged KV to
            # support arbitrary page sizes.
            inferred_paged_kv_non_tma = page_table is not None
        else:
            # SM100: currently uses a non-TMA path when page_size != 128.
            inferred_paged_kv_non_tma = page_size not in [None, 128]

    use_sm90_decode_fastpath = (
        compute_capability == 9
        and page_table is not None
        and page_size == 1
        and cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is not None
        and seqlen_q == 1
        and head_dim_v == head_dim
        and score_mod is None
        and mask_mod is None
        and block_sparse_tensors is None
        and aux_tensors is None
        and learnable_sink is None
        and not is_split_kv
        and head_dim in (64, 128)
    )
    # FP8 KV cache is supported for:
    # 1. SM90 decode fastpath (paged KV with page_size==1, seqlen_q==1)
    # 2. SM90 paged prefill with paged_kv_non_tma=True
    use_sm90_paged_prefill_fp8 = (
        kv_is_fp8
        and compute_capability == 9
        and page_table is not None
        and inferred_paged_kv_non_tma
    )
    intra_wg_overlap = True
    if kv_is_fp8 and not use_sm90_decode_fastpath and not use_sm90_paged_prefill_fp8:
        raise NotImplementedError(
            "FP8 KV cache is currently supported only for:\n"
            "  1. SM90 decode fastpath (paged KV with page_size==1, seqlen_q==1)\n"
            "  2. SM90 paged prefill with paged_kv_non_tma=True"
        )

    compile_key = (
        dtype,
        dtype_kv,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        len(aux_tensors) if aux_tensors is not None else 0,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        m_block_size,
        n_block_size,
        num_threads,
        num_splits,
        is_split_kv,
        pack_gqa,
        compute_capability,
        inferred_paged_kv_non_tma,
        use_sm90_decode_fastpath,
        intra_wg_overlap,
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        # Try loading precompiled kernel first
        precompiled = _try_load_precompiled_flash_attn(compile_key, use_sm90_decode_fastpath)
        if precompiled is not None:
            _flash_attn_fwd.compile_cache[compile_key] = precompiled
    if compile_key not in _flash_attn_fwd.compile_cache:
        # JIT compile if no precompiled version available
        kernel_type = "decode" if use_sm90_decode_fastpath else "forward"
        warnings.warn(
            f"Flash Attention {kernel_type} kernel not precompiled (hd={head_dim}, "
            f"causal={causal}, paged={inferred_paged_kv_non_tma}, "
            f"mask={'yes' if mask_mod is not None else 'no'}). JIT compiling...",
            stacklevel=3,
        )
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        k_for_cute = k
        v_for_cute = v
        if kv_is_fp8 and (use_sm90_decode_fastpath or use_sm90_paged_prefill_fp8):
            k_for_cute = k.view(torch.uint8)
            v_for_cute = v.view(torch.uint8)
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t)
            for t in (q, k_for_cute, v_for_cute, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        sparse_tensors = None
        if block_sparse_tensors is not None:
            if seqlen_q is None:
                raise ValueError("Block sparsity requires fixed-length sequences (seqlen_q must be known).")
            expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes(
                batch_size, num_head, seqlen_q, seqlen_k,
                m_block_size, n_block_size, compute_capability,
            )
            compile_time_normalized = normalize_block_sparse_tensors(
                block_sparse_tensors,
                expected_count_shape=expected_count_shape,
                expected_index_shape=expected_index_shape,
            )
            sparse_tensors = to_cute_block_sparse_tensors(compile_time_normalized)

        cute_aux_tensors = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_tensor(buf, assumed_align=None, fully_dynamic=True) for buf in aux_tensors]

        if compute_capability == 9:
            if is_split_kv:
                raise NotImplementedError("SplitKV is not supported on SM90.")
            if page_table is not None and not inferred_paged_kv_non_tma:
                assert page_size == n_block_size, (
                    "paged KV TMA path on SM90 requires page_size == n_block_size "
                    f"(got page_size={page_size}, n_block_size={n_block_size})"
                )
            if use_sm90_decode_fastpath:
                tile_size_per_bdx = 2 if kv_is_fp8 and qhead_per_kvhead == 1 else 4
                FlashAttentionDecodeSm90 = _get_kernel_class("FlashAttentionDecodeSm90")
                fa_fwd = FlashAttentionDecodeSm90(
                    dtype,
                    head_dim,
                    qhead_per_kvhead,
                    dtype_kv=dtype_kv,
                    is_causal=causal,
                    is_local=local,
                    tile_size_per_bdx=tile_size_per_bdx,
                )
            else:
                # fa_fwd = FlashAttentionForwardSm80(
                FlashAttentionForwardSm90 = _get_kernel_class("FlashAttentionForwardSm90")
                fa_fwd = FlashAttentionForwardSm90(
                    dtype,
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead,
                    dtype_kv=dtype_kv if use_sm90_paged_prefill_fp8 else None,
                    is_causal=causal,
                    is_local=local,
                    pack_gqa=pack_gqa,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    num_stages=2,
                    num_threads=num_threads,
                    Q_in_regs=False,
                    intra_wg_overlap=intra_wg_overlap,
                    mma_pv_is_rs=True,
                    mask_mod=mask_mod,
                    score_mod=score_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=inferred_paged_kv_non_tma,
                )
        elif compute_capability == 10:
            FlashAttentionForwardSm100 = _get_kernel_class("FlashAttentionForwardSm100")
            fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                is_persistent=not causal
                    and not local
                    and cu_seqlens_q is None
                    and seqused_q is None
                    and not is_split_kv,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                paged_kv_non_tma=inferred_paged_kv_non_tma,
                is_varlen_q=cu_seqlens_q is not None
                    or seqused_q is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x"
            )
        # TODO: check @can_implement
        if use_sm90_decode_fastpath:
            compiled = cute.compile(
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                k_scale_val,
                v_scale_val,
                current_stream,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                learnable_sink_tensor,
                sparse_tensors,
                cute_aux_tensors,
                options="--enable-tvm-ffi",
            )
        else:
            compiled = cute.compile(
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                k_scale_val,
                v_scale_val,
                current_stream,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                learnable_sink_tensor,
                sparse_tensors,
                cute_aux_tensors,
                options="--enable-tvm-ffi",
            )
        _flash_attn_fwd.compile_cache[compile_key] = compiled

    # Expand block sparse tensors to match actual head count (may be broadcast from 1)
    normalized_block_sparse_tensors = None
    if block_sparse_tensors is not None:
        expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes(
            batch_size, num_head, seqlen_q, seqlen_k,
            m_block_size, n_block_size, compute_capability,
        )
        normalized_block_sparse_tensors = normalize_block_sparse_tensors(
            block_sparse_tensors,
            expected_count_shape=expected_count_shape,
            expected_index_shape=expected_index_shape,
        )

    k_for_call = k
    v_for_call = v
    if kv_is_fp8 and (use_sm90_decode_fastpath or use_sm90_paged_prefill_fp8):
        k_for_call = k.view(torch.uint8)
        v_for_call = v.view(torch.uint8)
    if use_sm90_decode_fastpath:
        _flash_attn_fwd.compile_cache[compile_key](
            q,
            k_for_call,
            v_for_call,
            out if not is_split_kv else out_partial,
            lse_partial if is_split_kv else lse,
            softmax_scale,
            k_scale_val,
            v_scale_val,
            current_stream,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            window_size_left,
            window_size_right,
            learnable_sink,
            normalized_block_sparse_tensors,
            aux_tensors,
        )
    else:
        _flash_attn_fwd.compile_cache[compile_key](
            q,
            k_for_call,
            v_for_call,
            out if not is_split_kv else out_partial,
            lse_partial if is_split_kv else lse,
            softmax_scale,
            k_scale_val,
            v_scale_val,
            current_stream,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            window_size_left,
            window_size_right,
            learnable_sink,
            normalized_block_sparse_tensors,
            aux_tensors,
        )
    # Expose last dispatch choice for tests.
    _flash_attn_fwd._debug_last_impl = "sm90_decode" if use_sm90_decode_fastpath else "fwd"
    if is_split_kv:
        lse_partial_bsh = lse_partial.transpose(-1, -2)
        lse_bsh = lse.transpose(-1, -2) if lse is not None else None
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial_bsh,
            out,
            lse_bsh,
            cu_seqlens_q,
            seqused_q,
        )
    return out, lse


_flash_attn_fwd.compile_cache = {}


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compute_capability = _get_device_capability()
    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"

    if compute_capability == 9:
        m_block_size = 80 if not causal else 64
        n_block_size = 128
        num_stages_Q = 2
        num_stages_dO = 2
        num_stages_PdS = 2
        SdP_swapAB = True
        dKV_swapAB = False
        dQ_swapAB = not causal
        AtomLayoutMSdP = 1
        AtomLayoutNdKV = 2
        AtomLayoutMdQ = 1
        cluster_size = 1
        assert window_size_left is None and window_size_right is None, "local not supported yet on 9.x"
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        # TODO: support cluster size 2
        cluster_size = 1
    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        seqlen_k = None
        total_k = k.shape[0]

    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    if causal:
        window_size_right = 0
    local = window_size_left is not None or window_size_right is not None
    if local:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True

    use_block_sparsity = block_sparse_tensors is not None

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (batch_size, num_head, seqlen_q), (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    assert all(
        t is None or t.is_cuda for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    if compute_capability == 10:
        pack_gqa = False # override for now
    if compute_capability != 10:
        assert deterministic is False, "bwd deterministic only supported for sm100 for now"

    if score_mod is not None:
        assert score_mod_bwd is not None, "score_mod_bwd is required when score_mod is provided"
        assert softcap == 0.0, "softcap and score_mod are mutually exclusive (different log2 scaling)"
        assert cu_seqlens_q is None and cu_seqlens_k is None, (
            "varlen + score_mod not supported in bwd yet"
        )
        assert compute_capability == 10, "score_mod in bwd only supported on SM100 for now"

    device = q.device
    out_torch_dtype = q.dtype

    # nb: this could be derived from the block_sparse_tensors but for now we hardcode it to 2
    subtile_factor = 2

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
        seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
        dq_accum = torch.empty(
            batch_size,
            num_head,
            seqlen_q_rounded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum = torch.empty(
            num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device
        )
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    if qhead_per_kvhead > 1:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if cu_seqlens_k is None:
            seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
            num_n_blocks = seqlen_k_rounded // n_block_size
            if cluster_size == 2 and num_n_blocks % cluster_size != 0:
                seqlen_k_rounded = seqlen_k_rounded + n_block_size
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )
        else:
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * n_block_size - 1) // n_block_size * n_block_size
            )
            num_n_blocks = total_k_rounded_padded // n_block_size
            if cluster_size == 2 and num_n_blocks % cluster_size != 0:
                total_k_rounded_padded = total_k_rounded_padded + n_block_size
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if deterministic:
        dQ_semaphore = torch.zeros(batch_size, num_head, seqlen_q_rounded // m_block_size, 1, dtype=torch.int32, device="cuda")
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device="cuda")
        dV_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device="cuda")
    else:
        dK_semaphore = None
        dV_semaphore = None

    # Preprocess kernel: compute (o * dout).sum(dim=-1), lse * log2_e, and zero out dq_accum.
    compile_key_pre = (compute_capability, dtype, head_dim_v, m_block_size, num_threads)
    if compile_key_pre not in _flash_attn_bwd.compile_cache_pre:
        o_tensor, do_tensor = [to_cute_tensor(t) for t in (out, dout)]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
            to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4)
        cu_seqlens_q_tensor, seqused_q_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, seqused_q)
        ]
        FlashAttentionBackwardPreprocess = _get_kernel_class("FlashAttentionBackwardPreprocess")
        fa_bwd_pre = FlashAttentionBackwardPreprocess(
            dtype,
            head_dim_v,
            m_block_size,
            num_threads=num_threads,
        )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache_pre[compile_key_pre] = cute.compile(
            fa_bwd_pre,
            o_tensor,
            do_tensor,
            dpsum_tensor,
            lse_tensor,
            lse_log2_tensor,
            dq_accum_tensor,
            cu_seqlens_q_tensor,
            seqused_q_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flash_attn_bwd.compile_cache_pre[compile_key_pre](
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        cu_seqlens_q,
        seqused_q,
        current_stream,
    )

    # Backward kernel: compute dk, dv, dq_accum.
    if compute_capability == 9:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            softcap != 0.0,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
        )
    else:
        # Hash callables for compile key
        score_mod_hash = utils.hash_callable(score_mod) if score_mod else False
        score_mod_bwd_hash = utils.hash_callable(score_mod_bwd) if score_mod_bwd else False
        mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod else False
        num_aux_tensors = len(aux_tensors) if aux_tensors else 0
        # Convert aux_tensors to cute tensors
        cute_aux_tensors = None
        if aux_tensors is not None:
            cute_aux_tensors = [from_dlpack(buf).mark_layout_dynamic() for buf in aux_tensors]
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            softcap != 0.0,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            deterministic,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            use_block_sparsity,
        )
    num_threads = 384
    if compile_key not in _flash_attn_bwd.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
            to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)
        ]
        if qhead_per_kvhead > 1:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order())
            if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        FlashAttentionBackwardSm80 = _get_kernel_class("FlashAttentionBackwardSm80")
        fa_bwd_sm80 = FlashAttentionBackwardSm80(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            m_block_size,
            n_block_size,
            num_stages_Q,
            num_stages_dO,
            num_threads,
            pack_gqa,
            causal,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs=V_in_regs,
        )
        if compute_capability == 9:
            FlashAttentionBackwardSm90 = _get_kernel_class("FlashAttentionBackwardSm90")
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                m_block_size,
                n_block_size,
                num_stages_Q,
                num_stages_dO,
                num_stages_PdS,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                num_threads,
                V_in_regs=V_in_regs,
            )
        else:
            FlashAttentionBackwardSm100 = _get_kernel_class("FlashAttentionBackwardSm100")
            fa_bwd_obj = FlashAttentionBackwardSm100(
                head_dim,
                head_dim_v,
                is_causal=causal,
                is_local=local,
                qhead_per_kvhead=qhead_per_kvhead,
                # tile_m=m_block_size,
                # tile_n=n_block_size,
                cluster_size=cluster_size,
                # cluster_size=1,
                deterministic=deterministic,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None and len(aux_tensors) > 0,
                subtile_factor=subtile_factor,
            )

        # Block sparse tensors for backward use Q-direction indexing (transposed from forward).
        # sparse_block_size_q = 2*tile_m matches forward's q_stage=2 pipelining.
        sparse_tensors_compile = None
        if block_sparse_tensors is not None and compute_capability == 10:
            expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes_bwd(
                batch_size, num_head, seqlen_q, seqlen_k,
                m_block_size, n_block_size, subtile_factor,
            )
            compile_time_normalized = normalize_block_sparse_tensors(
                block_sparse_tensors,
                expected_count_shape=expected_count_shape,
                expected_index_shape=expected_index_shape,
            )
            sparse_tensors_compile = to_cute_block_sparse_tensors(compile_time_normalized)

        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if qhead_per_kvhead == 1 else dk_accum_tensor,
            dv_tensor if qhead_per_kvhead == 1 else dv_accum_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            None,  # softcap - not yet supported in backward
            window_size_left,
            window_size_right,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            cute_aux_tensors,
            sparse_tensors_compile,
            options="--enable-tvm-ffi",
        )
    normalized_block_sparse_tensors = None
    if block_sparse_tensors is not None and compute_capability == 10:
        expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes_bwd(
            batch_size, num_head, seqlen_q, seqlen_k,
            m_block_size, n_block_size, subtile_factor,
        )
        normalized_block_sparse_tensors = normalize_block_sparse_tensors(
            block_sparse_tensors,
            expected_count_shape=expected_count_shape,
            expected_index_shape=expected_index_shape,
        )

    _flash_attn_bwd.compile_cache[compile_key](
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk if qhead_per_kvhead == 1 else dk_accum,
        dv if qhead_per_kvhead == 1 else dv_accum,
        softmax_scale,
        current_stream,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        None,  # softcap - not yet supported in backward
        window_size_left,
        window_size_right,
        dQ_semaphore,
        dK_semaphore,
        dV_semaphore,
        aux_tensors,
        normalized_block_sparse_tensors,
    )

    num_threads = 256 if compute_capability == 9 else 128
    # Postprocess kernel: convert dq_accum from float32 to dq in bf16/fp16
    compile_key_post = (dtype, head_dim, m_block_size, num_threads, AtomLayoutMdQ, dQ_swapAB)
    if compile_key_post not in _flash_attn_bwd.compile_cache_post:
        dq_accum_tensor = to_cute_tensor(dq_accum)
        dq_tensor = to_cute_tensor(dq)
        cu_seqlens_q_tensor, seqused_q_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, seqused_q)
        ]
        arch = compute_capability * 10
        FlashAttentionBackwardPostprocess = _get_kernel_class("FlashAttentionBackwardPostprocess")
        fa_bwd_post = FlashAttentionBackwardPostprocess(
            dtype, head_dim, arch, m_block_size, num_threads, AtomLayoutMdQ, dQ_swapAB
        )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
            fa_bwd_post,
            dq_accum_tensor,
            dq_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            seqused_q_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flash_attn_bwd.compile_cache_post[compile_key_post](
        dq_accum,
        dq,
        softmax_scale,
        cu_seqlens_q,
        seqused_q,
        current_stream,
    )

    if qhead_per_kvhead > 1:
        # Postprocess kernel: convert dk_accum & dv_accum from float32 to bf16/fp16
        compile_key_post = (dtype, head_dim, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB)
        if compile_key_post not in _flash_attn_bwd.compile_cache_post:
            dk_accum_tensor = to_cute_tensor(dk_accum)
            dk_tensor = to_cute_tensor(dk)
            cu_seqlens_k_tensor, seqused_k_tensor = [
                to_cute_tensor(t, assumed_align=4) if t is not None else None
                for t in (cu_seqlens_k, seqused_k)
            ]
            FlashAttentionBackwardPostprocess = _get_kernel_class("FlashAttentionBackwardPostprocess")
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, head_dim, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB
            )
            # TODO: check @can_implement
            _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post,
                dk_accum_tensor,
                dk_tensor,
                softmax_scale,
                cu_seqlens_k_tensor,
                seqused_k_tensor,
                current_stream,
                options="--enable-tvm-ffi",
            )
        _flash_attn_bwd.compile_cache_post[compile_key_post](
            dk_accum,
            dk,
            softmax_scale,
            cu_seqlens_k,
            seqused_k,
            current_stream,
        )
        compile_key_post = (
            dtype,
            head_dim_v,
            n_block_size,
            num_threads,
            AtomLayoutNdKV,
            dKV_swapAB,
        )
        if compile_key_post not in _flash_attn_bwd.compile_cache_post:
            dv_accum_tensor = to_cute_tensor(dv_accum)
            dv_tensor = to_cute_tensor(dv)
            cu_seqlens_k_tensor, seqused_k_tensor = [
                to_cute_tensor(t, assumed_align=4) if t is not None else None
                for t in (cu_seqlens_k, seqused_k)
            ]
            FlashAttentionBackwardPostprocess = _get_kernel_class("FlashAttentionBackwardPostprocess")
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, head_dim_v, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB
            )
            # TODO: check @can_implement
            _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post,
                dv_accum_tensor,
                dv_tensor,
                cutlass.Float32(1.0),
                cu_seqlens_k_tensor,
                seqused_k_tensor,
                current_stream,
                options="--enable-tvm-ffi",
            )
        _flash_attn_bwd.compile_cache_post[compile_key_post](
            dv_accum,
            dv,
            1.0,
            cu_seqlens_k,
            seqused_k,
            current_stream,
        )

    return dq, dk, dv


_flash_attn_bwd.compile_cache_pre = {}
_flash_attn_bwd.compile_cache = {}
_flash_attn_bwd.compile_cache_post = {}


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        mask_mod: Optional[Callable] = None,
        full_block_cnt: Optional[torch.Tensor] = None,
        full_block_idx: Optional[torch.Tensor] = None,
        mask_block_cnt: Optional[torch.Tensor] = None,
        mask_block_idx: Optional[torch.Tensor] = None,
    ):
        # Only create block sparse tensors if at least one block sparse parameter is provided
        block_sparse_tensors = None
        if any(t is not None for t in [full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx]):
            block_sparse_tensors = BlockSparseTensorsTorch(
                full_block_cnt=full_block_cnt,
                full_block_idx=full_block_idx,
                mask_block_cnt=mask_block_cnt,
                mask_block_idx=mask_block_idx,
            )
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, *((None,) * 20)  # Extra Nones is fine


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        aux_tensors: Optional[list] = None,
    ):
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors
        assert seqused_q == seqused_k == None
        assert ctx.softcap == 0.0
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            deterministic=ctx.deterministic,
        )

        return dq, dk, dv, *((None,) * 20)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    mask_mod: Optional[Callable] = None,
    full_block_cnt: Optional[torch.Tensor] = None,
    full_block_idx: Optional[torch.Tensor] = None,
    mask_block_cnt: Optional[torch.Tensor] = None,
    mask_block_idx: Optional[torch.Tensor] = None,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        mask_mod,
        full_block_cnt,
        full_block_idx,
        mask_block_cnt,
        mask_block_idx,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
):
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        aux_tensors,
    )


def _flash_attn_sm90_decode_persistent_split_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    causal: bool,
    local: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    qhead_per_kvhead: int,
    num_splits: int,
    split_tokens: int,
    persist_oversub: int,
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    split_counters: torch.Tensor,
) -> None:
    """SM90 decode-only persistent split with in-kernel combine (page_size==1, seqlen_q==1)."""
    assert q.is_cuda and k.is_cuda and v.is_cuda and out.is_cuda, "tensors must be on CUDA device"
    assert page_table.is_cuda and seqused_k.is_cuda, "page_table/seqused_k must be on CUDA device"
    assert q.shape[1] == 1 and out.shape[1] == 1, "fused decode requires seqlen_q == 1"
    assert q.dtype == out.dtype, "q/out dtype mismatch"
    assert out_partial.dtype == torch.float32 and lse_partial.dtype == torch.float32, "partials must be fp32"
    assert split_counters.dtype == torch.int32, "split_counters must be int32"
    if lse is not None:
        assert lse.is_cuda and lse.dtype == torch.float32, "lse must be fp32 CUDA tensor"

    dtype = torch2cute_dtype_map[q.dtype]
    kv_is_fp8 = (k.dtype == torch.float8_e4m3fn) or (v.dtype == torch.float8_e4m3fn)
    if kv_is_fp8:
        if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"FP8 KV cache requires k and v both be float8_e4m3fn (got k={k.dtype}, v={v.dtype})"
            )
        if k_scale <= 0.0 or v_scale <= 0.0:
            raise ValueError("k_scale and v_scale must be positive for FP8 KV cache")
    else:
        assert q.dtype == k.dtype == v.dtype, "q/k/v dtype mismatch"
    dtype_kv = cutlass.Float8E4M3FN if kv_is_fp8 else dtype
    head_dim = q.shape[-1]
    assert head_dim in (64, 128), "fused persistent split is only tuned for head_dim 64/128"

    compile_key = (
        dtype,
        dtype_kv,
        head_dim,
        int(qhead_per_kvhead),
        bool(causal),
        bool(local),
        int(num_splits),
        int(split_tokens),
        int(persist_oversub),
        window_size_left is not None,
        window_size_right is not None,
        lse is None,
    )
    if compile_key not in _flash_attn_sm90_decode_persistent_split_fused.compile_cache:
        # Try loading precompiled kernel first
        precompiled = _try_load_precompiled_persistent_split_fused(
            dtype, dtype_kv, head_dim, int(num_splits), int(split_tokens)
        )
        if precompiled is not None:
            _flash_attn_sm90_decode_persistent_split_fused.compile_cache[compile_key] = precompiled
        else:
            # JIT compile if no precompiled version available
            q_tensor = to_cute_tensor(q)
            k_for_cute = k.view(torch.uint8) if kv_is_fp8 else k
            v_for_cute = v.view(torch.uint8) if kv_is_fp8 else v
            k_tensor = to_cute_tensor(k_for_cute)
            v_tensor = to_cute_tensor(v_for_cute)
            out_tensor = to_cute_tensor(out)
            lse_tensor = (
                to_cute_tensor(lse, assumed_align=4) if lse is not None else None
            )
            page_table_tensor = to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            seqused_k_tensor = to_cute_tensor(seqused_k, assumed_align=4, leading_dim=0)
            out_partial_tensor = to_cute_tensor(out_partial, leading_dim=4)
            lse_partial_tensor = to_cute_tensor(lse_partial, assumed_align=4)
            split_counters_tensor = to_cute_tensor(split_counters, assumed_align=4, leading_dim=1)
            current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

            FlashAttentionDecodeSm90PersistentSplitFused = _get_kernel_class("FlashAttentionDecodeSm90PersistentSplitFused")
            fa_fused = FlashAttentionDecodeSm90PersistentSplitFused(
                dtype=dtype,
                dtype_kv=dtype_kv,
                head_dim=head_dim,
                qhead_per_kvhead=int(qhead_per_kvhead),
                num_splits=int(num_splits),
                is_causal=bool(causal),
                is_local=bool(local),
                split_tokens=int(split_tokens),
                persist_oversub=int(persist_oversub),
                tile_size_per_bdx=2 if kv_is_fp8 and int(qhead_per_kvhead) == 1 else 4,
            )
            _flash_attn_sm90_decode_persistent_split_fused.compile_cache[compile_key] = cute.compile(
                fa_fused,
                q_tensor,
                k_tensor,
                v_tensor,
                out_tensor,
                lse_tensor,
                seqused_k_tensor,
                page_table_tensor,
                out_partial_tensor,
                lse_partial_tensor,
                split_counters_tensor,
                softmax_scale,
                k_scale,
                v_scale,
                window_size_left,
                window_size_right,
                current_stream,
                options="--enable-tvm-ffi",
            )

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    k_for_call = k.view(torch.uint8) if kv_is_fp8 else k
    v_for_call = v.view(torch.uint8) if kv_is_fp8 else v
    _flash_attn_sm90_decode_persistent_split_fused.compile_cache[compile_key](
        q,
        k_for_call,
        v_for_call,
        out,
        lse,
        seqused_k,
        page_table,
        out_partial,
        lse_partial,
        split_counters,
        softmax_scale,
        k_scale,
        v_scale,
        window_size_left,
        window_size_right,
        current_stream,
    )


_flash_attn_sm90_decode_persistent_split_fused.compile_cache = {}


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"
    assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    assert out_partial.stride(-1) == 1, "out_partial must be contiguous in the last dimension"
    assert lse_partial.stride(-2) == 1, "lse_partial must be contiguous in the seqlen dimension"
    assert lse_partial.shape == out_partial.shape[:-1]

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    # Validate output tensor shapes and types
    assert out.shape == out_partial.shape[1:], "out shape mismatch"
    if lse is not None:
        assert lse.shape == lse_partial.shape[1:], "lse shape mismatch"
        assert lse.dtype == torch.float32, "lse must be fp32"

    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"

    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    m_block_size = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if m_block_size == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]

    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        m_block_size,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
    )

    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        out_partial_tensor = to_cute_tensor(
            out_partial, leading_dim=4 if not is_varlen else 3
        )
        lse_partial_tensor = to_cute_tensor(
            lse_partial, assumed_align=4, leading_dim=lse_partial.ndim - 2
        )
        out_tensor = to_cute_tensor(out, leading_dim=3 if not is_varlen else 2)
        lse_tensor = (
            to_cute_tensor(lse, assumed_align=4, leading_dim=lse.ndim - 2)
            if lse is not None
            else None
        )

        optional_tensors = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens, seqused, num_splits_dynamic_ptr, semaphore_to_reset)
        ]
        cu_seqlens_tensor, seqused_tensor, num_splits_dynamic_tensor, semaphore_tensor = (
            optional_tensors
        )
        FlashAttentionForwardCombine = _get_kernel_class("FlashAttentionForwardCombine")
        fa_combine = FlashAttentionForwardCombine(
            dtype=dtype,
            dtype_partial=dtype_partial,
            head_dim=head_dim,
            m_block_size=m_block_size,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
        )

        # Check if implementation is supported
        if not fa_combine.can_implement(
            dtype,
            dtype_partial,
            head_dim,
            m_block_size,
            k_block_size,
            log_max_splits,
            num_threads=256,
        ):
            raise RuntimeError(
                "FlashAttention combine kernel cannot be implemented with given parameters"
            )

        _flash_attn_fwd_combine.compile_cache[compile_key] = cute.compile(
            fa_combine,
            out_partial_tensor,
            lse_partial_tensor,
            out_tensor,
            lse_tensor,
            cu_seqlens_tensor,
            seqused_tensor,
            num_splits_dynamic_tensor,
            semaphore_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
    _flash_attn_fwd_combine.compile_cache[compile_key](
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        num_splits_dynamic_ptr,
        semaphore_to_reset,
        current_stream,
    )


_flash_attn_fwd_combine.compile_cache = {}


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        return_lse: Whether to return the combined LSE tensor. Default is True.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype == torch.float32, "out_partial must be fp32 (from accumulation)"
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == (num_splits, total_q, num_heads), (
            "lse_partial shape mismatch for varlen"
        )
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == (num_splits, batch_size, seqlen, num_heads), (
            "lse_partial shape mismatch"
        )

    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype

    # Create output if not provided
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(total_q, num_heads, head_size, dtype=out_dtype, device=device)
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )

    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=device).transpose(
                0, 1
            )
        else:
            lse = torch.empty(
                batch_size, num_heads, seqlen, dtype=torch.float32, device=device
            ).transpose(1, 2)
    else:
        lse = None

    _flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
    )
    return out, lse
