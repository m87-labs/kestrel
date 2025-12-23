"""Correctness tests for reshape_and_cache_flash."""

import pytest
import torch

from kestrel_kernels.kv_cache_write import reshape_and_cache_flash as reshape_and_cache_flash_cuda


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _write_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_k = key_cache.clone()
    ref_v = value_cache.clone()
    slot_mapping_cpu = slot_mapping.cpu()
    for i in range(int(slot_mapping_cpu.numel())):
        slot = int(slot_mapping_cpu[i].item())
        if slot < 0:
            continue
        block = slot // block_size
        offset = slot % block_size
        ref_k[block, :, offset, :] = key[i]
        ref_v[block, :, offset, :] = value[i]
    return ref_k, ref_v


def test_reshape_and_cache_flash_bf16_matches_reference(device: torch.device) -> None:
    torch.manual_seed(0)

    num_tokens = 11
    num_heads = 4
    head_size = 8
    block_size = 4
    num_blocks = 5

    key = torch.randn((num_tokens, num_heads, head_size), device=device, dtype=torch.bfloat16)
    value = torch.randn((num_tokens, num_heads, head_size), device=device, dtype=torch.bfloat16)

    # Include a padded token.
    slot_mapping = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, -1, 8, 9, 10], device=device, dtype=torch.int64
    )

    key_cache = torch.zeros(
        (num_blocks, num_heads, block_size, head_size), device=device, dtype=torch.bfloat16
    )
    value_cache = torch.zeros_like(key_cache)

    # Match runtime: pass a view with the same shape but strided heads.
    key_cache_arg = key_cache.permute(0, 2, 1, 3)
    value_cache_arg = value_cache.permute(0, 2, 1, 3)

    reshape_and_cache_flash_cuda(
        key,
        value,
        key_cache_arg,
        value_cache_arg,
        slot_mapping,
        "auto",
        torch.tensor(1.0, device=device, dtype=torch.float32),
        torch.tensor(1.0, device=device, dtype=torch.float32),
    )

    ref_k, ref_v = _write_reference(
        key, value, key_cache, value_cache, slot_mapping, block_size=block_size
    )
    torch.testing.assert_close(key_cache, ref_k, atol=0, rtol=0)
    torch.testing.assert_close(value_cache, ref_v, atol=0, rtol=0)


def test_reshape_and_cache_flash_bf16_accepts_strided_tokens(
    device: torch.device,
) -> None:
    torch.manual_seed(0)

    num_tokens = 9
    num_heads = 4
    head_size = 8
    block_size = 4
    num_blocks = 4

    key_base = torch.randn(
        (num_tokens * 2, num_heads, head_size),
        device=device,
        dtype=torch.bfloat16,
    )
    value_base = torch.randn_like(key_base)
    key = key_base[::2]
    value = value_base[::2]
    assert not key.is_contiguous()
    assert not value.is_contiguous()

    slot_mapping = torch.tensor(
        [0, 1, 2, 3, 4, -1, 6, 7, 8], device=device, dtype=torch.int64
    )

    key_cache = torch.zeros(
        (num_blocks, num_heads, block_size, head_size), device=device, dtype=torch.bfloat16
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache_arg = key_cache.permute(0, 2, 1, 3)
    value_cache_arg = value_cache.permute(0, 2, 1, 3)

    reshape_and_cache_flash_cuda(
        key,
        value,
        key_cache_arg,
        value_cache_arg,
        slot_mapping,
        "auto",
        torch.tensor(1.0, device=device, dtype=torch.float32),
        torch.tensor(1.0, device=device, dtype=torch.float32),
    )

    ref_k, ref_v = _write_reference(
        key, value, key_cache, value_cache, slot_mapping, block_size=block_size
    )
    torch.testing.assert_close(key_cache, ref_k, atol=0, rtol=0)
    torch.testing.assert_close(value_cache, ref_v, atol=0, rtol=0)


def test_reshape_and_cache_flash_fp8_matches_reference(device: torch.device) -> None:
    torch.manual_seed(0)

    num_tokens = 7
    num_heads = 4
    head_size = 8
    block_size = 4
    num_blocks = 3

    key = (torch.randn((num_tokens, num_heads, head_size), device=device, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    value = (torch.randn((num_tokens, num_heads, head_size), device=device, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    slot_mapping = torch.tensor([0, 1, 3, 4, 6, 7, 9], device=device, dtype=torch.int64)

    k_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    v_scale = torch.tensor(1.0, device=device, dtype=torch.float32)

    key_cache = torch.zeros(
        (num_blocks, num_heads, block_size, head_size),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    value_cache = torch.zeros_like(key_cache)

    key_cache_arg = key_cache.permute(0, 2, 1, 3).view(torch.uint8)
    value_cache_arg = value_cache.permute(0, 2, 1, 3).view(torch.uint8)

    reshape_and_cache_flash_cuda(
        key,
        value,
        key_cache_arg,
        value_cache_arg,
        slot_mapping,
        "fp8_e4m3",
        k_scale,
        v_scale,
    )

    # Reference: compute the exact fp8 bits using torch's conversion.
    ref_key_cache = torch.zeros_like(key_cache)
    ref_value_cache = torch.zeros_like(value_cache)
    slot_mapping_cpu = slot_mapping.cpu()
    for i in range(int(slot_mapping_cpu.numel())):
        slot = int(slot_mapping_cpu[i].item())
        block = slot // block_size
        offset = slot % block_size
        ref_key_cache[block, :, offset, :] = (key[i].float() / k_scale).to(torch.float8_e4m3fn)
        ref_value_cache[block, :, offset, :] = (value[i].float() / v_scale).to(torch.float8_e4m3fn)

    torch.testing.assert_close(
        key_cache.view(torch.uint8),
        ref_key_cache.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        value_cache.view(torch.uint8),
        ref_value_cache.view(torch.uint8),
        atol=0,
        rtol=0,
    )
