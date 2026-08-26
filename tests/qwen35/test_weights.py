import pytest
import torch
from safetensors.torch import save_file

from kestrel.models.qwen35.qwen_loader import (
    _load_sharded_safetensors,
    _qwen_gdn_norm_weight_keys,
    _qwen_rms_norm_weight_keys,
)
from kestrel.models.qwen35.qwen_model import Qwen3_5RMSNormGated


def _gdn_norm_holder(hidden_size: int) -> torch.nn.Module:
    model = torch.nn.Module()
    model.norm = Qwen3_5RMSNormGated(hidden_size)
    return model


def test_gdn_norm_preserves_exact_fp32_checkpoint_weight_under_bf16_default(
    tmp_path,
) -> None:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = _gdn_norm_holder(4)
    finally:
        torch.set_default_dtype(previous_dtype)

    exact_fp32_keys = _qwen_gdn_norm_weight_keys(model)
    assert exact_fp32_keys == {"norm.weight"}
    assert _qwen_rms_norm_weight_keys(model) == set()
    assert model.norm.weight.dtype == torch.float32

    checkpoint = torch.tensor(
        [0.9991, 1.0009, 1.0071, 0.9929], dtype=torch.float32
    )
    assert not torch.equal(
        checkpoint,
        checkpoint.to(torch.bfloat16).to(torch.float32),
    )
    shard = tmp_path / "model.safetensors"
    save_file({"norm.weight": checkpoint}, shard)
    missing, unexpected = _load_sharded_safetensors(
        model,
        tmp_path,
        [shard.name],
        device=torch.device("cpu"),
    )

    assert missing == []
    assert unexpected == []
    assert model.norm.weight.dtype == torch.float32
    torch.testing.assert_close(model.norm.weight, checkpoint, rtol=0, atol=0)


def test_gdn_norm_rejects_rounded_checkpoint_weight(tmp_path) -> None:
    model = _gdn_norm_holder(4)
    shard = tmp_path / "model.safetensors"
    save_file({"norm.weight": torch.full((4,), 0.5, dtype=torch.bfloat16)}, shard)

    with pytest.raises(ValueError, match="requires an FP32 checkpoint tensor"):
        _load_sharded_safetensors(
            model,
            tmp_path,
            [shard.name],
            device=torch.device("cpu"),
        )
    torch.testing.assert_close(
        model.norm.weight,
        torch.ones(4, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
