from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.dflash import DFlashConfig, _DynamicConv, _CandidateSelector


def test_dynamic_convolution_matches_explicit_causal_sum():
    torch.manual_seed(41)
    config = SimpleNamespace(hidden_size=12, conv_group_size=3, conv_kernel_size=3)
    conv = _DynamicConv(config)
    with torch.no_grad():
        conv.base_kernel.normal_()
    hidden = torch.randn(2, 5, 12)
    prepared, dynamic = conv.prepare(hidden)
    kernels = conv.kernel_projection(hidden).reshape(2, 5, 2, 3, 4)
    def reference(value, arm):
        result = torch.zeros_like(value)
        for row in range(5):
            for offset in range(min(3, row + 1)):
                for channel in range(12):
                    scale = (conv.base_kernel[arm, offset, channel]
                             + kernels[:, row, arm, offset, channel // 3])
                    result[:, row, channel] += scale * value[:, row-offset, channel]
        return result
    torch.testing.assert_close(prepared, reference(hidden, 0))
    torch.testing.assert_close(conv.finish(hidden, dynamic), reference(hidden, 1))
    isolated, _ = conv.prepare(hidden[:1])
    torch.testing.assert_close(isolated, prepared[:1])


def test_candidate_selector_uses_anchor_and_selected_predecessor():
    config = SimpleNamespace(hidden_size=2, vocab_size=3, selector_rank=1, selector_top_k=3)
    selector = _CandidateSelector(config)
    with torch.no_grad():
        selector.hidden_projection.weight.fill_(1)
        selector.predecessor_codebook.weight.copy_(torch.tensor([[1.], [-1.], [0.]]))
        selector.successor_codebook.weight.copy_(torch.tensor([[0.], [2.], [-2.]]))
    hidden = torch.ones(2, 3, 2)
    logits = torch.zeros(2, 3, 3)
    result = selector(hidden, logits, torch.tensor([0, 1]))
    assert result.tolist() == [[1, 2, 0], [2, 0, 1]]


@pytest.mark.parametrize("architecture", ["DSparkDraftModel", "UnknownDraftModel"])
def test_unknown_draft_architecture_rejected_before_loading(architecture):
    with pytest.raises(ValueError, match="architecture"):
        DFlashConfig.from_dict({"architectures": [architecture], "dflash_config": {}})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sliding", [False, True])
def test_dflash2_graph_matches_eager_with_changing_anchors(sliding):
    from kestrel.models.qwen35.dflash import DFlashDraftModel, DFlashContextCache, _LayerContextBuffer
    from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession
    torch.manual_seed(92)
    config = DFlashConfig(128, 256, 1, 4, 1, 128, 1e-6, 10000, 8, 0,
                         (0,), ("sliding_attention" if sliding else "full_attention",),
                         8 if sliding else None, causal_override=False,
                         conv_kernel_size=2, conv_group_size=16,
                         selector_rank=16, selector_top_k=4, vocab_size=257)
    model = DFlashDraftModel(config).cuda().bfloat16().requires_grad_(False)
    for layer in model.layers:
        layer.attention_conv.base_kernel.normal_(std=0.1)
        layer.mlp_conv.base_kernel.normal_(std=0.1)
    head = torch.nn.Linear(128, 257, bias=False).cuda().bfloat16().requires_grad_(False)
    caches = []
    for _ in range(2):
        cache = DFlashContextCache(64)
        cache.length = 3
        keys = torch.randn(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
        cache.layers = [_LayerContextBuffer(keys, keys.clone())]
        caches.append(cache)
    session = DFlashDraftGraphSession(model, caches, lm_head=head)
    try:
        for anchors in ([1, 2], [3, 4], [2, 1]):
            noise = [torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in caches]
            features = [value[:, :0] for value in noise]
            positions = [torch.arange(3, 11, device="cuda")[None] for _ in caches]
            anchors = torch.tensor(anchors, device="cuda")
            hidden = torch.cat([model(n, f, p, context_cache=c) for n, f, p, c in
                                zip(noise, features, positions, caches)])
            expected = model.select_tokens(hidden, head, anchors)
            with session.launch(noise, features, positions, anchors) as actual:
                assert torch.equal(actual, expected)
            packed = torch.cat(model.forward_many(noise, features, positions, context_caches=caches))
            torch.testing.assert_close(packed, hidden, atol=0.05, rtol=0.03)
    finally:
        session.shutdown()
