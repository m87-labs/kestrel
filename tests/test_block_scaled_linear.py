import pytest
import torch

from kestrel.ops.block_scaled_linear import BlockScaledLinear


def test_block_scaled_linear_forwards_explicit_activation(monkeypatch):
    import importlib
    source = importlib.import_module("kestrel.ops.block_scaled_linear")
    module = BlockScaledLinear(128, 1024, interleaved_parts=2)
    activation = torch.zeros(1, 128)
    seen = []
    def project(x, weight, scales, **kwargs):
        seen.append(kwargs["gated_activation"])
        return x.new_empty((1, 512 if kwargs["gated_activation"] else 1024))
    monkeypatch.setattr(source, "_block_scaled_linear", project)
    assert module(activation).shape == (1, 1024)
    assert module(activation, gated_activation="silu").shape == (1, 512)
    assert seen == [None, "silu"]


@pytest.mark.parametrize("storage", ["dense", "fp8", "bias", "tail", "noninterleaved"])
def test_qwen_mlp_requests_fused_activation_only_for_block_scaled_storage(storage, monkeypatch):
    import kestrel.models.qwen35.qwen_model as source
    module = source.Qwen3_5MLP.__new__(source.Qwen3_5MLP)
    torch.nn.Module.__init__(module)
    module.intermediate_size = 512
    fp8 = storage == "fp8"
    module.gate_up_proj = (torch.nn.Linear(128, 1024, bias=False) if storage == "dense"
                          else BlockScaledLinear(
                              128, 1024, quantized_rows=896 if storage == "tail" else 1024,
                              interleaved_parts=1 if storage in ("tail", "noninterleaved") else 2,
                              bias=storage == "bias"))
    module.down_proj = torch.nn.Identity()
    seen = []
    def project(x, **kwargs):
        seen.append(kwargs)
        return x.new_zeros((1, 512 if kwargs else 1024))
    monkeypatch.setattr(module.gate_up_proj, "forward", project)
    def separate(out, gate_up, **kwargs):
        assert not fp8
        seen.append(kwargs)
        out.zero_()
    monkeypatch.setattr(source, "_kestrel_gated_activation_into", separate)
    assert module(torch.zeros(1, 128)).shape == (1, 512)
    assert seen == ([{"gated_activation": "silu"}] if fp8 else [
        {}, {"activation": "silu", "layout": "interleaved_i8"}])


@pytest.mark.parametrize("rows,quantized,parts", [(145, 145, 1), (146, 128, 1), (288, 288, 2)])
@pytest.mark.parametrize("batch", [1, 8])
def test_block_scaled_linear_preserves_blocks_and_unquantized_tail(rows, quantized, parts, batch):
    torch.manual_seed(6000)
    module = BlockScaledLinear(136, rows, quantized_rows=quantized, interleaved_parts=parts)
    packed = torch.randn(rows, 136).to(torch.float8_e4m3fn)
    module.weight.data.copy_(packed.view(torch.uint8))
    module.weight_scale_inv.uniform_(0.25, 2.0)
    expected = torch.empty(rows, 136)
    for row in range(quantized):
        part = (row // 8) % 2 if parts == 2 else 0
        logical_row = row // 16 * 8 + row % 8 if parts == 2 else row
        for col in range(136):
            expected[row, col] = packed[row, col].float() * module.weight_scale_inv[
                part, logical_row // 128, col // 128]
    if module.weight_tail is not None:
        module.weight_tail.fill_(1.0078125)  # BF16 value not representable in E4M3.
        expected[quantized:] = module.weight_tail.float()
    torch.testing.assert_close(module.dequantized_weight(torch.float32), expected, rtol=0, atol=0)
    activation = torch.randn(batch, 136)
    torch.testing.assert_close(module(activation), activation @ expected.T)
    assert module.weight.dtype == torch.uint8
    assert module.weight.element_size() == 1


@pytest.mark.parametrize("kwargs", [
    {"in_features": 0, "out_features": 128},
    {"in_features": 128, "out_features": 128, "quantized_rows": 129},
    {"in_features": 128, "out_features": 145, "interleaved_parts": 2},
    {"in_features": 128, "out_features": 256, "quantized_rows": 128, "interleaved_parts": 2},
])
def test_block_scaled_linear_rejects_ambiguous_storage(kwargs):
    with pytest.raises(ValueError):
        BlockScaledLinear(**kwargs)


def test_block_scaled_linear_bias_uses_activation_precision():
    module = BlockScaledLinear(16, 16, bias=True)
    module.weight.data.zero_()
    module.weight_scale_inv.fill_(1)
    module.bias.data.fill_(1.0078125)
    result = module(torch.zeros(2, 16, dtype=torch.bfloat16))
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, module.bias.bfloat16().expand(2, 16))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_block_scaled_linear_dtype_moves_preserve_checkpoint_precision(dtype):
    module = BlockScaledLinear(16, 17, quantized_rows=16)
    module.weight.data.zero_()
    module.weight_scale_inv.fill_(1.234567)
    module.weight_tail.fill_(1.0078125)
    scales = module.weight_scale_inv.clone()
    tail = module.weight_tail.clone()
    module.to(dtype=dtype)
    assert module.weight.dtype == torch.uint8
    assert module.weight_scale_inv.dtype == torch.float32
    assert module.weight_tail.dtype == torch.bfloat16
    torch.testing.assert_close(module.weight_scale_inv, scales, rtol=0, atol=0)
    torch.testing.assert_close(module.weight_tail, tail, rtol=0, atol=0)
