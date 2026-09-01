from types import SimpleNamespace

import torch
import torch.nn.functional as F

import kestrel.models.qwen35.qwen_model as model_module
from kestrel.models.qwen35.qwen_model import Qwen3_5VisionPatchMerger
from kestrel.models.qwen35.qwen_model import Qwen3_5VisionPatchEmbed


def test_patch_embed_uses_runtime_linear_without_changing_semantics(
    monkeypatch,
) -> None:
    config = SimpleNamespace(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=2,
        hidden_size=5,
    )
    patch_embed = Qwen3_5VisionPatchEmbed(config).eval()
    inputs = torch.randn(7, 24)
    expected = patch_embed.proj(inputs)
    calls = []

    def runtime_linear(x, weight, bias):
        calls.append((weight, bias))
        return F.linear(x, weight, bias)

    monkeypatch.setattr(model_module, "_kestrel_linear", runtime_linear)

    actual = patch_embed(inputs)

    assert calls == [(patch_embed.proj.weight, patch_embed.proj.bias)]
    torch.testing.assert_close(actual, expected)


def test_patch_merger_uses_runtime_linear_without_changing_semantics(
    monkeypatch,
) -> None:
    config = SimpleNamespace(
        hidden_size=2,
        spatial_merge_size=2,
        out_hidden_size=3,
    )
    merger = Qwen3_5VisionPatchMerger(config).eval()
    inputs = torch.randn(8, config.hidden_size)
    normalized = merger.norm(inputs).view(-1, merger.hidden_size)
    expected = merger.linear_fc2(
        merger.act_fn(merger.linear_fc1(normalized))
    )
    calls = []

    def runtime_linear(x, weight, bias):
        calls.append((weight, bias))
        return F.linear(x, weight, bias)

    monkeypatch.setattr(model_module, "_kestrel_linear", runtime_linear)

    actual = merger(inputs)

    assert len(calls) == 2
    assert calls[0][0] is merger.linear_fc1.weight
    assert calls[0][1] is merger.linear_fc1.bias
    assert calls[1][0] is merger.linear_fc2.weight
    assert calls[1][1] is merger.linear_fc2.bias
    torch.testing.assert_close(actual, expected)
