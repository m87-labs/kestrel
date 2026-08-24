from types import SimpleNamespace

import torch

from kestrel.models.qwen35 import runtime as qwen_runtime
from kestrel.models.qwen35.qwen_model import Qwen3_5GatedDeltaNet
from kestrel.models.qwen35.runtime import Qwen35Runtime, _PackedPrefillBatch
from kestrel.runtime.tokens import TextToken


def test_builder_binds_topology_from_ordered_host_lengths(monkeypatch) -> None:
    observed: dict[str, object] = {}
    topology_token = object()
    bound_cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    def bind_packed_prefill_topology(*, sequence_lengths, device):
        observed["sequence_lengths"] = sequence_lengths
        observed["device"] = device
        return bound_cu_seqlens, topology_token

    monkeypatch.setattr(
        qwen_runtime,
        "get_runtime",
        lambda: SimpleNamespace(
            gated_delta=SimpleNamespace(
                bind_packed_prefill_topology=bind_packed_prefill_topology
            )
        ),
    )

    runtime = object.__new__(Qwen35Runtime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.page_table = SimpleNamespace(
        n_pages=8,
        page_size=1,
        page_table_cpu=(tuple(range(8)), tuple(range(8, 16))),
        page_table=torch.tensor(
            (tuple(range(8)), tuple(range(8, 16))), dtype=torch.int32
        ),
    )
    prefill_slot = SimpleNamespace(batch_idx=torch.zeros(2, dtype=torch.int64))
    prepared = (
        SimpleNamespace(tokens_list=(TextToken(11), TextToken(12))),
        SimpleNamespace(tokens_list=(TextToken(21), TextToken(22), TextToken(23))),
    )

    packed = runtime._build_packed_prefill_batch(
        prepared,
        prefill_slot=prefill_slot,
        image_crops_list=(None, None),
        batch_indices=(0, 1),
    )

    assert observed == {
        "sequence_lengths": (2, 3),
        "device": torch.device("cpu"),
    }
    assert packed.cu_seq_lens_q is bound_cu_seqlens
    assert packed.sequence_lengths == (2, 3)
    assert packed.topology_token is topology_token


def test_packed_prefill_batch_forwards_host_sequence_lengths() -> None:
    observed: dict[str, object] = {}
    cache = SimpleNamespace(advance_to=lambda length: observed.setdefault("max", length))
    output = SimpleNamespace(last_hidden_state=object(), past_key_values=cache)
    model = SimpleNamespace(model=lambda **kwargs: observed.update(kwargs) or output)
    runtime = object.__new__(Qwen35Runtime)
    runtime.model = model
    runtime._new_cache = lambda: cache

    marker = object()
    packed = _PackedPrefillBatch(
        input_ids=marker,
        cache_position_ids=marker,
        position_ids=marker,
        cu_seq_lens_q=marker,
        sequence_lengths=(544, 137),
        topology_token=marker,
        seq_idx=marker,
        batch_indices=marker,
        max_length=544,
        last_token_offsets=marker,
        paged_kv_page_table=marker,
        paged_kv_seqlens_k=marker,
        slot_mapping=marker,
        rope_deltas=marker,
    )

    hidden, returned_cache = runtime._forward_packed_prefill(packed)

    assert hidden is output.last_hidden_state
    assert returned_cache is cache
    assert observed["sequence_lengths"] == (544, 137)
    assert observed["topology_token"] is marker
    assert observed["max"] == 544


def test_gdn_prefill_forwards_host_sequence_lengths_to_recurrence() -> None:
    observed: dict[str, object] = {}
    layer = SimpleNamespace(
        conv_states=None,
        recurrent_states=None,
        has_previous_state=False,
        _reset_replay_rows=lambda state, indices: None,
    )
    cache = SimpleNamespace(
        layers=[layer],
        has_previous_state=lambda layer_idx: False,
    )
    conv1d = SimpleNamespace(
        weight=torch.ones((1, 1, 1)),
        bias=None,
    )

    def recurrent(*args, **kwargs):
        observed.update(kwargs)
        return torch.zeros((1, 3, 1)), None

    fake = SimpleNamespace(
        layer_idx=0,
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=1,
        head_v_dim=1,
        conv_dim=1,
        conv_kernel_size=1,
        value_dim=1,
        activation="silu",
        A_log=torch.zeros((1,)),
        dt_bias=torch.zeros((1,)),
        conv1d=conv1d,
        in_proj=lambda hidden: torch.zeros((1, 3, 4)),
        supports_packed_gdn=lambda *args: True,
        causal_conv1d_packed=lambda **kwargs: kwargs["x"],
        packed_prefill_prepare=lambda *args: (
            torch.zeros((1, 3, 1, 1)),
            torch.zeros((1, 3, 1, 1)),
            torch.zeros((1, 3, 1, 1)),
            torch.zeros((1, 3, 1)),
            torch.zeros((1, 3, 1)),
        ),
        packed_recurrent_prefill=recurrent,
        norm=lambda value, gate: value,
        out_proj=lambda value: value,
    )

    result = Qwen3_5GatedDeltaNet.forward(
        fake,
        torch.zeros((1, 3, 1)),
        cache_params=cache,
        cu_seq_lens_q=torch.tensor([0, 3], dtype=torch.int32),
        sequence_lengths=(3,),
        topology_token=fake,
        seq_idx=torch.zeros((1, 3), dtype=torch.int32),
    )

    assert result.shape == (1, 3, 1)
    assert observed["sequence_lengths"] == (3,)
    assert observed["topology_token"] is fake
