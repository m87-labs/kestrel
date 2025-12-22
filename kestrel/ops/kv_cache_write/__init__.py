import torch

from ..jit import cpp_jit


@cpp_jit(function_name="reshape_and_cache_flash")
def reshape_and_cache_flash_cuda(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None: ...


__all__ = ["reshape_and_cache_flash_cuda"]

