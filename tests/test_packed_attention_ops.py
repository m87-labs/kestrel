from types import SimpleNamespace

import pytest
import torch

from kestrel.ops import attention


def test_distinct_query_key_boundaries_are_forwarded(monkeypatch):
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 7, 11], dtype=torch.int32)
    q = torch.empty(1, 4, 5, 16)
    k = torch.empty(1, 1, 11, 16)
    v = torch.empty_like(k)

    def forward(query, key, value, **kwargs):
        assert query.shape == (5, 4, 16)
        assert key.shape == value.shape == (11, 1, 16)
        assert kwargs["cu_seqlens_q"] is cu_q
        assert kwargs["cu_seqlens_k"] is cu_k
        return torch.zeros_like(query), None

    monkeypatch.setattr(attention, "get_runtime", lambda: SimpleNamespace(
        attention=SimpleNamespace(flash_attn_fwd=forward)))
    result = attention.dense_attention(q, k, v, scaling=.25, causal=False,
                                      cu_seqlens=cu_q, cu_seqlens_k=cu_k)
    assert result.shape == (1, 5, 4, 16)


@pytest.mark.parametrize("cu_q,cu_k,match", [
    (None, torch.tensor([0, 2], dtype=torch.int32), "require query"),
    (torch.tensor([0, 2], dtype=torch.int32), torch.tensor([0, 2, 3], dtype=torch.int32), "same sequences"),
    (torch.tensor([0, 2], dtype=torch.int32), torch.tensor([0, 2]), "int32"),
])
def test_invalid_key_boundaries_rejected(cu_q, cu_k, match):
    value = torch.empty(1, 1, 2, 16)
    with pytest.raises(ValueError, match=match):
        attention.dense_attention(value, value, value, scaling=.25, causal=False,
                                  cu_seqlens=cu_q, cu_seqlens_k=cu_k)
