"""CuTe MoE kernel dispatch and invoke functions."""

import os
from typing import Any, Dict, Literal, Tuple

import torch

import cutlass
import cutlass.cute as cute

from kestrel_kernels.cute_moe.config import CuteMoeConfig, get_cute_moe_config, _should_use_wgmma_bf16
from kestrel_kernels.cute_moe.utils import (
    _ensure_cutlass_initialized,
    _set_compiled_kernel_shared_carveout,
    _maybe_set_device_cache_config,
    _load_precompiled_kernel,
    _to_cute_tensor_1d_i32,
    _to_cute_tensor_1d_contig,
    _to_cute_tensor_scalar_i32,
    _to_cute_tensor_2d_contig,
    _to_cute_tensor_2d_contig_u8,
    _to_cute_tensor_3d_last_contig,
    _to_cute_tensor_3d_last_contig_u8,
)
# Note: Kernel template classes (cute_moe_*.py) are lazily imported only when
# JIT compilation is enabled. They are not included in the distributed wheel.
from kestrel_kernels.flash_attn.cute import utils as fa_utils


_CUTE_TOP_K_UP_DECODE = 8

# Cache compiled variants keyed by (kind, config). We only support two decode kernels:
# - up-proj: mul_routed_weight=False, top_k=8
# - down-proj: mul_routed_weight=True, top_k=1
_COMPILE_CACHE: Dict[Tuple[str, CuteMoeConfig, int, int], Any] = {}
_COMPILE_CACHE_FP8: Dict[Tuple[str, CuteMoeConfig], Any] = {}

# Enable JIT compilation for autotuning (set KESTREL_CUTE_MOE_JIT=1)
_ENABLE_JIT = os.environ.get("KESTREL_CUTE_MOE_JIT", "0") == "1"


def _invoke_cute_moe_impl(
    kind: str,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: CuteMoeConfig,
) -> None:
    if A.dtype != torch.bfloat16:
        raise ValueError(f"CuTe fused MoE supports bfloat16 only (got {A.dtype})")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A/B/C must be CUDA tensors")
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    if B.ndim != 3:
        raise ValueError("B must be 3D [E, N, K]")
    if C.ndim != 3:
        raise ValueError("C must be 3D [M, top_k, N]")
    if B.stride(-1) != 1:
        raise ValueError("B must be contiguous in the last dimension (K)")
    if C.stride(-1) != 1:
        raise ValueError("C must be contiguous in the last dimension (N)")
    if B.shape[2] != A.shape[1]:
        raise ValueError("A and B must have the same K dimension")
    if sorted_token_ids.dtype != torch.int32 or expert_ids.dtype != torch.int32:
        raise ValueError("sorted_token_ids and expert_ids must be int32")
    if num_tokens_post_padded.dtype != torch.int32:
        raise ValueError("num_tokens_post_padded must be int32")
    if mul_routed_weight:
        if topk_weights is None:
            raise ValueError("topk_weights is required when mul_routed_weight=True")
    else:
        topk_weights = torch.empty((0,), device=A.device, dtype=A.dtype)
    if topk_weights is None:
        raise ValueError("topk_weights must be set (internal error)")

    # Match Triton's EM shrink for tiny decode batches: avoid launching blocks that will
    # immediately exit due to routing padding. This is safe because `num_tokens_post_padded`
    # is always <= num_valid_tokens * block_m.
    num_valid_tokens = int(A.shape[0]) * int(top_k)
    em_launch = min(int(sorted_token_ids.numel()), num_valid_tokens * int(config.block_m))
    if em_launch < int(sorted_token_ids.numel()):
        sorted_token_ids = sorted_token_ids[:em_launch]
        m_blocks = (em_launch + int(config.block_m) - 1) // int(config.block_m)
        expert_ids = expert_ids[:m_blocks]

    _ensure_cutlass_initialized()
    _maybe_set_device_cache_config()

    # Flatten output to [M_assignments, N] like the Triton kernel expects.
    C2d = C.view(-1, C.shape[-1])

    N_dim = int(B.shape[1])
    K_dim = int(A.shape[1])
    key = (kind, config, N_dim, K_dim)

    if key not in _COMPILE_CACHE:
        if _ENABLE_JIT:
            # JIT compile for autotuning - requires source install
            try:
                from kestrel_kernels.cute_moe.cute_moe_bf16_sm90_warp import _FusedMoeMatmulCuTe
                from kestrel_kernels.cute_moe.cute_moe_bf16_sm90_wgmma import _FusedMoeMatmulCuTeWgmmaBf16
            except ImportError as e:
                raise RuntimeError(
                    "JIT compilation requires source install. "
                    "Install from source for JIT support."
                ) from e

            from cutlass import BFloat16

            op_cls = (
                _FusedMoeMatmulCuTeWgmmaBf16
                if _should_use_wgmma_bf16(config)
                else _FusedMoeMatmulCuTe
            )
            op = op_cls(
                BFloat16,
                config,
                mul_routed_weight=mul_routed_weight,
                top_k=top_k,
                N=N_dim,
                K=K_dim,
            )
            a_cute = _to_cute_tensor_2d_contig(A)
            b_cute = _to_cute_tensor_3d_last_contig(B)
            c_cute = _to_cute_tensor_2d_contig(C2d)
            sorted_cute = _to_cute_tensor_1d_i32(sorted_token_ids)
            expert_cute = _to_cute_tensor_1d_i32(expert_ids)
            post_cute = _to_cute_tensor_scalar_i32(num_tokens_post_padded)
            topk_w_cute = _to_cute_tensor_1d_contig(topk_weights)
            # Use env stream so TVM-FFI auto-picks current CUDA stream (matches precompiled)
            stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            compiled = cute.compile(
                op,
                a_cute,
                b_cute,
                c_cute,
                topk_w_cute,
                sorted_cute,
                expert_cute,
                post_cute,
                stream_fake,
                options="--enable-tvm-ffi",
            )
            _set_compiled_kernel_shared_carveout(compiled)
            _COMPILE_CACHE[key] = compiled
        else:
            # Load precompiled kernel
            from kestrel_kernels.cute_moe.config import _get_cuda_arch

            precompiled = _load_precompiled_kernel(kind, config, N_dim, K_dim)
            if precompiled is not None:
                _COMPILE_CACHE[key] = precompiled
            else:
                arch = _get_cuda_arch()
                raise RuntimeError(
                    "No precompiled kernel for "
                    f"cute_moe(kind={kind}, config={config}, N={N_dim}, K={K_dim}, arch={arch}). "
                    f"Run precompile_cute_moe.py on this architecture to generate it."
                )

    # TVM-FFI handles PyTorch tensor conversion automatically
    _COMPILE_CACHE[key](
        A,
        B,
        C2d,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


def _invoke_cute_moe_fp8_impl(
    kind: str,
    A_fp8_bits: torch.Tensor,
    A_scale: torch.Tensor,
    B_fp8_bits: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: CuteMoeConfig,
) -> None:
    if A_fp8_bits.dtype != torch.uint8:
        raise ValueError(f"Expected FP8 activation bits as uint8 (got {A_fp8_bits.dtype})")
    if A_scale.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"Expected A_scale float16/float32 (got {A_scale.dtype})")
    if B_fp8_bits.dtype != torch.uint8:
        raise ValueError(f"Expected FP8 weight bits as uint8 (got {B_fp8_bits.dtype})")
    if B_scale.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"Expected B_scale float16/float32 (got {B_scale.dtype})")
    if C.dtype != torch.bfloat16:
        raise ValueError(f"CuTe fused MoE FP8 expects bfloat16 C (got {C.dtype})")
    if not (
        A_fp8_bits.is_cuda
        and A_scale.is_cuda
        and B_fp8_bits.is_cuda
        and B_scale.is_cuda
        and C.is_cuda
    ):
        raise ValueError("A_fp8_bits/A_scale/B_fp8_bits/B_scale/C must be CUDA tensors")
    if A_fp8_bits.ndim != 2:
        raise ValueError("A_fp8_bits must be 2D")
    if A_scale.ndim != 1:
        raise ValueError("A_scale must be 1D")
    if B_fp8_bits.ndim != 3:
        raise ValueError("B_fp8_bits must be 3D [E, N, K]")
    if B_scale.ndim != 2:
        raise ValueError("B_scale must be 2D [E, N]")
    if C.ndim != 3:
        raise ValueError("C must be 3D [M, top_k, N]")
    if A_fp8_bits.stride(-1) != 1:
        raise ValueError("A_fp8_bits must be contiguous in the last dimension (K)")
    if B_fp8_bits.stride(-1) != 1:
        raise ValueError("B_fp8_bits must be contiguous in the last dimension (K)")
    if C.stride(-1) != 1:
        raise ValueError("C must be contiguous in the last dimension (N)")
    if A_scale.shape[0] != A_fp8_bits.shape[0]:
        raise ValueError("A_scale must have length matching A_fp8_bits.shape[0]")
    if B_fp8_bits.shape[2] != A_fp8_bits.shape[1]:
        raise ValueError("A_fp8_bits and B_fp8_bits must have the same K dimension")
    if int(A_fp8_bits.shape[1]) % int(config.block_k) != 0:
        raise ValueError("CuTe FP8 kernel requires K divisible by block_k")

    # Config validation is done in CuteMoeConfig.__post_init__
    if B_scale.shape[0] != B_fp8_bits.shape[0] or B_scale.shape[1] != B_fp8_bits.shape[1]:
        raise ValueError("B_scale must have shape [E, N] matching B_fp8_bits")
    if B_scale.stride(-1) != 1:
        raise ValueError("B_scale must be contiguous in the last dimension (N)")
    if sorted_token_ids.dtype != torch.int32 or expert_ids.dtype != torch.int32:
        raise ValueError("sorted_token_ids and expert_ids must be int32")
    if num_tokens_post_padded.dtype != torch.int32:
        raise ValueError("num_tokens_post_padded must be int32")
    if mul_routed_weight:
        if topk_weights is None:
            raise ValueError("topk_weights is required when mul_routed_weight=True")
    else:
        topk_weights = torch.empty((0,), device=C.device, dtype=C.dtype)
    if topk_weights is None:
        raise ValueError("topk_weights must be set (internal error)")

    num_valid_tokens = int(A_fp8_bits.shape[0]) * int(top_k)
    em_launch = min(int(sorted_token_ids.numel()), num_valid_tokens * int(config.block_m))
    if em_launch < int(sorted_token_ids.numel()):
        sorted_token_ids = sorted_token_ids[:em_launch]
        m_blocks = (em_launch + int(config.block_m) - 1) // int(config.block_m)
        expert_ids = expert_ids[:m_blocks]

    _ensure_cutlass_initialized()
    _maybe_set_device_cache_config()

    C2d = C.view(-1, C.shape[-1])

    dtype = cutlass.BFloat16
    fp8_dtype = cutlass.Float8E4M3FN
    N_dim = int(C2d.shape[1])
    K_dim = int(A_fp8_bits.shape[1])
    key = (kind, config, N_dim, K_dim)

    # Ensure scales are float32 without creating copies if already float32
    if A_scale.dtype != torch.float32:
        A_scale = A_scale.to(dtype=torch.float32)
    if B_scale.dtype != torch.float32:
        B_scale = B_scale.to(dtype=torch.float32)

    # Choose between warp-level MMA and WGMMA based on kernel_type config
    use_wgmma_fp8 = config.kernel_type == "wgmma"

    if key not in _COMPILE_CACHE_FP8:
        if _ENABLE_JIT:
            # JIT compile for autotuning - requires source install
            try:
                from kestrel_kernels.cute_moe.cute_moe_fp8_sm90_wgmma import _FusedMoeMatmulCuTeFp8
                from kestrel_kernels.cute_moe.cute_moe_fp8_sm90_warp import _FusedMoeMatmulCuTeWarpFp8
            except ImportError as e:
                raise RuntimeError(
                    "JIT compilation requires source install. "
                    "Install from source for JIT support."
                ) from e

            # Create CuTe tensors only for compilation (to derive tensor signatures)
            a_bits_cute = _to_cute_tensor_2d_contig_u8(A_fp8_bits)
            a_scale_cute = _to_cute_tensor_1d_contig(A_scale)
            b_bits_cute = _to_cute_tensor_3d_last_contig_u8(B_fp8_bits)
            b_scale_cute = _to_cute_tensor_2d_contig(B_scale)
            c_cute = _to_cute_tensor_2d_contig(C2d)
            sorted_cute = _to_cute_tensor_1d_i32(sorted_token_ids)
            expert_cute = _to_cute_tensor_1d_i32(expert_ids)
            post_cute = _to_cute_tensor_scalar_i32(num_tokens_post_padded)
            topk_w_cute = _to_cute_tensor_1d_contig(topk_weights)

            # Use TVM-FFI env stream for automatic stream handling (matches BF16 path)
            stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

            if use_wgmma_fp8:
                op = _FusedMoeMatmulCuTeFp8(
                    dtype, fp8_dtype, config, mul_routed_weight=mul_routed_weight, top_k=top_k,
                    N=N_dim, K=K_dim,
                )
            else:
                op = _FusedMoeMatmulCuTeWarpFp8(
                    dtype, fp8_dtype, config, mul_routed_weight=mul_routed_weight, top_k=top_k,
                    N=N_dim, K=K_dim,
                )
            compiled = cute.compile(
                op,
                a_bits_cute,
                a_scale_cute,
                b_bits_cute,
                b_scale_cute,
                c_cute,
                topk_w_cute,
                sorted_cute,
                expert_cute,
                post_cute,
                stream_fake,
                options="--enable-tvm-ffi",
            )
            _set_compiled_kernel_shared_carveout(compiled)
            _COMPILE_CACHE_FP8[key] = compiled
        else:
            # Load precompiled kernel
            from kestrel_kernels.cute_moe.config import _get_cuda_arch

            precompiled = _load_precompiled_kernel(kind, config, N_dim, K_dim)
            if precompiled is not None:
                _COMPILE_CACHE_FP8[key] = precompiled
            else:
                arch = _get_cuda_arch()
                raise RuntimeError(
                    "No precompiled kernel for "
                    f"cute_moe_fp8(kind={kind}, config={config}, N={N_dim}, K={K_dim}, arch={arch}). "
                    f"Run precompile_cute_moe.py on this architecture to generate it."
                )

    # TVM-FFI handles PyTorch tensor conversion automatically (like BF16 path)
    _COMPILE_CACHE_FP8[key](
        A_fp8_bits,
        A_scale,
        B_fp8_bits,
        B_scale,
        C2d,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


def invoke_cute_moe_up_fp8(
    A_fp8_bits: torch.Tensor,
    A_scale: torch.Tensor,
    B_fp8_bits: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    *,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: CuteMoeConfig | None = None,
) -> None:
    """CuTe fused MoE up-projection with FP8 activations+weights (W8A8).

    If config is None, auto-selects optimal config based on tensor shapes and GPU.
    """
    if config is None:
        # Infer model dimensions from tensor shapes
        # A_fp8_bits: [M, hidden_size], B_fp8_bits: [E, intermediate_size*2, hidden_size]
        num_tokens = A_fp8_bits.shape[0]
        hidden_size = A_fp8_bits.shape[1]
        num_experts = B_fp8_bits.shape[0]
        intermediate_size = B_fp8_bits.shape[1] // 2
        config = get_cute_moe_config(
            "up", num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype="fp8",
        )
    _invoke_cute_moe_fp8_impl(
        "up",
        A_fp8_bits,
        A_scale,
        B_fp8_bits,
        B_scale,
        C,
        topk_weights=None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=_CUTE_TOP_K_UP_DECODE,
        config=config,
    )


def invoke_cute_moe_down_fp8(
    A_fp8_bits: torch.Tensor,
    A_scale: torch.Tensor,
    B_fp8_bits: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: CuteMoeConfig | None = None,
) -> None:
    """CuTe fused MoE down-projection with FP8 activations+weights (W8A8).

    If config is None, auto-selects optimal config based on tensor shapes and GPU.
    """
    if config is None:
        # Infer model dimensions from tensor shapes
        # A_fp8_bits: [M*top_k, intermediate_size], B_fp8_bits: [E, hidden_size, intermediate_size]
        # C: [M, top_k, hidden_size]
        num_tokens = C.shape[0]
        hidden_size = B_fp8_bits.shape[1]
        num_experts = B_fp8_bits.shape[0]
        intermediate_size = B_fp8_bits.shape[2]
        config = get_cute_moe_config(
            "down", num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype="fp8",
        )
    _invoke_cute_moe_fp8_impl(
        "down",
        A_fp8_bits,
        A_scale,
        B_fp8_bits,
        B_scale,
        C,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=config,
    )


def invoke_cute_moe_up(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: CuteMoeConfig | None = None,
) -> None:
    """CuTe fused MoE up-projection (no routed-weight scaling).

    If config is None, auto-selects optimal config based on tensor shapes and GPU.
    """
    if config is None:
        # Infer model dimensions from tensor shapes
        # A: [M, hidden_size], B: [E, intermediate_size*2, hidden_size]
        num_tokens = A.shape[0]
        hidden_size = A.shape[1]
        num_experts = B.shape[0]
        intermediate_size = B.shape[1] // 2  # gate+up are fused
        config = get_cute_moe_config(
            "up", num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    _invoke_cute_moe_impl(
        "up",
        A,
        B,
        C,
        topk_weights=None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=_CUTE_TOP_K_UP_DECODE,
        config=config,
    )


def invoke_cute_moe_down(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: CuteMoeConfig | None = None,
) -> None:
    """CuTe fused MoE down-projection (includes routed-weight scaling).

    If config is None, auto-selects optimal config based on tensor shapes and GPU.
    """
    if config is None:
        # Infer model dimensions from tensor shapes
        # A: [M*top_k, intermediate_size], B: [E, hidden_size, intermediate_size]
        # C: [M, top_k, hidden_size] - use C to get num_tokens since A is expanded
        num_tokens = C.shape[0]
        hidden_size = B.shape[1]
        num_experts = B.shape[0]
        intermediate_size = B.shape[2]
        config = get_cute_moe_config(
            "down", num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    _invoke_cute_moe_impl(
        "down",
        A,
        B,
        C,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=config,
    )
