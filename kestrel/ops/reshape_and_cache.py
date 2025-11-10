"""Triton kernel that reshapes/stores KV-cache into HND layout.

Adapted from vLLM's reshape_and_cache_flash implementation to support
[num_blocks, num_heads, block_size, head_dim] (HND) cache layout without
requiring temporary transposes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform


@triton.jit
def _reshape_and_cache_kernel_hnd(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    k_scale,
    v_scale,
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    head_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    tile_i = tl.program_id(axis=1)
    tile_offsets = tl.arange(0, TILE_SIZE)
    linear = tile_i * TILE_SIZE + tile_offsets
    total_elems = num_heads * head_size
    mask = linear < total_elems

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    head_ids = linear // head_size
    dim_offsets = linear % head_size

    key_load = tl.load(key_ptr + src_key_idx + linear, mask=mask, other=0.0)
    if FP8_KV_CACHE:
        key_tile = (
            key_load
            if key_load.dtype.is_fp8()
            else key_load / tl.load(k_scale)
        )
    else:
        key_tile = key_load

    value_load = tl.load(value_ptr + src_value_idx + linear, mask=mask, other=0.0)
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load

    block_stride_t = tl.full((), block_stride, dtype=tl.int64)
    head_stride_t = tl.full((), head_stride, dtype=tl.int64)
    page_stride_t = tl.full((), page_stride, dtype=tl.int64)

    tgt_idx = (
        block_idx * block_stride_t
        + head_ids * head_stride_t
        + block_offset * page_stride_t
        + dim_offsets
    )

    tl.store(key_cache_ptr + tgt_idx, key_tile, mask=mask)
    tl.store(value_cache_ptr + tgt_idx, value_tile, mask=mask)


def reshape_and_cache_hnd(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[2]
    n = num_heads * head_size

    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    head_stride = key_cache.stride()[1]
    page_stride = key_cache.stride()[2]

    assert (
        kv_cache_dtype == "auto" or kv_cache_dtype.startswith("fp8")
    ), f"unsupported kv_cache_dtype {kv_cache_dtype}"

    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if kv_cache_dtype.startswith("fp8")
        else key_cache.dtype
    )

    if key_cache.dtype != kv_cache_torch_dtype and kv_cache_dtype.startswith("fp8"):
        key_cache = key_cache.view(kv_cache_torch_dtype)
        value_cache = value_cache.view(kv_cache_torch_dtype)

    FP8_KV_CACHE = kv_cache_dtype.startswith("fp8")
    if FP8_KV_CACHE:
        assert kv_cache_torch_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.uint8,
            torch.float8_e4m3fnuz,
        ), "unsupported fp8 cache dtype"

    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    if current_platform.is_rocm() or current_platform.is_xpu():
        num_stages = 4
        num_warps = 8
    else:
        num_stages = 10
        num_warps = 16
        if torch.cuda.get_device_capability(key.device)[0] < 9:
            TILE_SIZE = min(512, TILE_SIZE)

    grid = lambda meta: (int(num_tokens), triton.cdiv(n, meta["TILE_SIZE"]))

    _reshape_and_cache_kernel_hnd[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        head_stride=head_stride,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        FP8_KV_CACHE=FP8_KV_CACHE,
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
