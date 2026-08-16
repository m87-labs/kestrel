from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kestrel.models.moondream import text as text_mod


class _StopAfterTau(RuntimeError):
    pass


def test_tau_attention_uses_moondream_tanh_gelu(monkeypatch) -> None:
    qkv_weight = object()
    tau_weight = object()
    qkv_out = torch.randn((1, 1, 8), dtype=torch.bfloat16)
    captured: dict[str, object] = {}

    def fake_linear(inp, weight, bias=None):
        del bias
        if weight is qkv_weight:
            return qkv_out
        assert weight is tau_weight
        captured["gelu_out"] = inp
        return torch.empty((1, 1, 4), dtype=inp.dtype)

    def fake_gelu(inp, *, approximate):
        captured["gelu_input"] = inp
        captured["approximate"] = approximate
        return inp + 1

    def fake_tau_tail_apply_into(**kwargs):
        captured["tau_qkv"] = kwargs["qkv_out"]
        captured["tau_tok_qv"] = kwargs["tok_qv_lin"]

    def stop_after_tau(*args, **kwargs):
        del args, kwargs
        raise _StopAfterTau

    monkeypatch.setattr(text_mod, "_kestrel_linear", fake_linear)
    monkeypatch.setattr(text_mod.F, "gelu", fake_gelu)
    monkeypatch.setattr(text_mod, "tau_tail_apply_into", fake_tau_tail_apply_into)
    monkeypatch.setattr(text_mod, "rotary_embedding", stop_after_tau)

    module = SimpleNamespace(
        qkv=SimpleNamespace(weight=qkv_weight, bias=None),
        tau={"wqwv": tau_weight},
        _tau_pos_table=torch.ones((1, 2), dtype=torch.bfloat16),
    )

    with pytest.raises(_StopAfterTau):
        text_mod.attn(
            torch.zeros((1, 1, 4), dtype=torch.bfloat16),
            module,
            torch.empty((1, 1, 2)),
            object(),
            None,
            2,
            1,
            torch.tensor([0]),
            slot_mapping=torch.tensor([0]),
        )

    assert captured["gelu_input"] is qkv_out
    assert captured["approximate"] == "tanh"
    torch.testing.assert_close(captured["gelu_out"], qkv_out + 1)
    assert captured["tau_qkv"] is qkv_out
