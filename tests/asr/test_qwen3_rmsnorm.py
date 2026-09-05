import torch

from kestrel.models.qwen3_asr import model as qwen_model


def test_rmsnorm_preserves_cast_before_weight(monkeypatch):
    norm = qwen_model.RmsNorm(128, 1e-6).to(dtype=torch.bfloat16)
    value = torch.linspace(-2, 3, 256).reshape(2, 128).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 2, 128))

    def shared_rms(value, unit_weight, eps):
        assert unit_weight.dtype == value.dtype
        assert torch.equal(unit_weight, torch.ones_like(unit_weight))
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + eps
        )
        return normalized.to(value.dtype)

    monkeypatch.setattr(qwen_model, "_rmsnorm", shared_rms)
    normalized = value.float() * torch.rsqrt(
        value.float().square().mean(-1, keepdim=True) + norm.eps
    )
    expected = norm.weight * normalized.to(value.dtype)
    assert torch.equal(norm(value), expected)


def test_rmsnorm_unit_weight_reset_and_moves():
    with torch.device("meta"):
        norm = qwen_model.RmsNorm(128, 1e-6)
    norm.load_state_dict({"weight": torch.ones(128)}, assign=True)
    norm.reset_nonpersistent_buffers()
    assert norm.unit_weight.device.type == "cpu"
    assert torch.equal(norm.unit_weight, torch.ones(128))
    assert set(norm.state_dict()) == {"weight"}
    norm.to(dtype=torch.bfloat16)
    assert norm.unit_weight.dtype == norm.weight.dtype == torch.bfloat16
    norm.to_empty(device="cpu")
    norm.reset_nonpersistent_buffers()
    assert torch.equal(norm.unit_weight, torch.ones_like(norm.weight))
