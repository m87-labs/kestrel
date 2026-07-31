import pytest
import torch

from kestrel.ops import norm as norm_ops
from kestrel.ops.norm import RMSNorm


@pytest.mark.parametrize("dim", [128, 1152, 1536, 5376])
def test_rms_norm_delegates_capability_selection_to_runtime(monkeypatch, dim):
    module = RMSNorm(dim, eps=1e-6)
    hidden_states = torch.randn(2, dim, dtype=torch.bfloat16)
    expected = torch.randn_like(hidden_states)
    calls = []

    def runtime(x, weight, eps):
        calls.append((x, weight, eps))
        return expected

    monkeypatch.setattr(norm_ops, "_rmsnorm", runtime)

    assert module(hidden_states) is expected
    assert len(calls) == 1
    assert calls[0][0] is hidden_states
    assert calls[0][1] is module.weight
    assert calls[0][2] == module.eps


def test_rms_norm_unscaled_weight_is_not_persistent():
    module = RMSNorm(256, with_scale=False)

    assert not isinstance(module.weight, torch.nn.Parameter)
    assert torch.equal(module.weight, torch.ones(256, dtype=torch.float32))
    assert module.state_dict() == {}
