from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from kestrel.models.qwen35.qwen_config import Qwen3_5VisionConfig
from kestrel.models.qwen35.qwen_model import Qwen3_5VisionBlock


def _qwen08_vision_config() -> Qwen3_5VisionConfig:
    return Qwen3_5VisionConfig(
        depth=12,
        hidden_size=768,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=3072,
        num_heads=12,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=1024,
        num_position_embeddings=2304,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_l4_vision_block_is_invariant_to_duplicate_image_batch_shape() -> None:
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("Requires the L4 Qwen vision path")

    torch.manual_seed(83)
    config = _qwen08_vision_config()
    sequence_length = 2052
    module = Qwen3_5VisionBlock(config).cuda().bfloat16().eval()
    sequence = torch.randn(
        sequence_length,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    angles = torch.randn(
        sequence_length,
        config.hidden_size // config.num_heads,
        device="cuda",
        dtype=torch.float32,
    )

    captures: dict[str, torch.Tensor] = {}
    hooks = []

    def capture(name: str) -> Callable:
        def hook(_module, _args, output) -> None:
            captures[name] = output[:sequence_length].detach().clone()

        return hook

    for name, submodule in (
        ("norm1", module.norm1),
        ("qkv", module.attn.qkv),
        ("attention_projection", module.attn.proj),
        ("attention", module.attn),
        ("norm2", module.norm2),
        ("mlp", module.mlp),
        ("block", module),
    ):
        hooks.append(submodule.register_forward_hook(capture(name)))

    def run(sequence_count: int) -> dict[str, torch.Tensor]:
        captures.clear()
        hidden_states = sequence.repeat(sequence_count, 1)
        cos = angles.cos().repeat(sequence_count, 1)
        sin = angles.sin().repeat(sequence_count, 1)
        cu_seqlens = torch.arange(
            0,
            (sequence_count + 1) * sequence_length,
            sequence_length,
            device="cuda",
            dtype=torch.int32,
        )
        with torch.inference_mode():
            module(hidden_states, cu_seqlens, (cos, sin))
        torch.cuda.synchronize()
        return dict(captures)

    try:
        single_before = run(1)
        packed = run(7)
        single_after = run(1)
    finally:
        for hook in hooks:
            hook.remove()

    assert single_before.keys() == packed.keys() == single_after.keys()
    temporal_differences = {
        name: float((single_after[name].float() - value.float()).abs().max())
        for name, value in single_before.items()
        if not torch.equal(single_after[name], value)
    }
    batch_shape_differences = {
        name: float((packed[name].float() - value.float()).abs().max())
        for name, value in single_before.items()
        if not torch.equal(packed[name], value)
    }
    assert not temporal_differences, temporal_differences
    assert not batch_shape_differences, batch_shape_differences
