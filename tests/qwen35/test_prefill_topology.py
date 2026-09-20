from types import SimpleNamespace

import torch

from kestrel.models.qwen35 import runtime as qwen_runtime
from kestrel.models.qwen35 import qwen_model as qwen_model_module
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
    runtime._linear_state_pool = SimpleNamespace(
        bind_prefill_state=lambda _cache: None
    )

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
        image_token_spans=(),
    )

    hidden, returned_cache = runtime._forward_packed_prefill(packed)

    assert hidden is output.last_hidden_state
    assert returned_cache is cache
    assert observed["sequence_lengths"] == (544, 137)
    assert observed["topology_token"] is marker
    assert observed["max"] == 544


def test_launch_prepared_batch_uses_runtime_linear_for_lm_head(monkeypatch) -> None:
    observed: dict[str, object] = {}
    runtime = object.__new__(Qwen35Runtime)
    runtime.max_batch_size = 2
    runtime._chat_image_crops = {}
    runtime.page_table = SimpleNamespace(
        commit_block_table=lambda indices: observed.setdefault(
            "batch_indices", indices
        )
    )
    packed = SimpleNamespace(
        last_token_offsets=torch.tensor([1, 3]),
        batch_indices=torch.tensor([4, 7]),
        rope_deltas=object(),
    )
    runtime._build_packed_prefill_batch = lambda *args, **kwargs: packed
    last_hidden = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
    cache = object()
    runtime._forward_packed_prefill = lambda value: (last_hidden, cache)
    runtime._store_packed_sequence_caches = (
        lambda *args, **kwargs: observed.setdefault("stored", (args, kwargs))
    )
    runtime._record_prefill_slot_done = lambda slot: observed.setdefault(
        "done", slot
    )
    weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    runtime.model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=weight, bias=None)
    )
    prepared = [
        SimpleNamespace(state=SimpleNamespace(batch_idx=4, last_hidden=None)),
        SimpleNamespace(state=SimpleNamespace(batch_idx=7, last_hidden=None)),
    ]
    linear_calls = []

    def runtime_linear(value, linear_weight, bias):
        linear_calls.append((value, linear_weight, bias))
        return torch.nn.functional.linear(value, linear_weight, bias)

    monkeypatch.setattr(qwen_runtime, "_kestrel_linear", runtime_linear)
    prefill_slot = object()

    logits = runtime.launch_prepared_batch(prepared, prefill_slot)

    hidden_rows = last_hidden[0].index_select(0, packed.last_token_offsets)
    assert len(linear_calls) == 1
    torch.testing.assert_close(linear_calls[0][0], hidden_rows)
    assert linear_calls[0][1] is weight
    assert linear_calls[0][2] is None
    torch.testing.assert_close(logits, torch.nn.functional.linear(hidden_rows, weight))
    torch.testing.assert_close(prepared[0].state.last_hidden, hidden_rows[0])
    torch.testing.assert_close(prepared[1].state.last_hidden, hidden_rows[1])
    assert observed["batch_indices"] == [4, 7]
    assert observed["done"] is prefill_slot


def test_gdn_prefill_forwards_topology_to_combined_prefill(monkeypatch) -> None:
    observed: dict[str, object] = {}
    linear_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    layer = SimpleNamespace(
        conv_states=None,
        recurrent_states=torch.empty((1, 1, 1, 1), dtype=torch.bfloat16),
        has_previous_state=False,
    )
    cache = SimpleNamespace(
        layers=[layer],
        has_previous_state=lambda layer_idx: False,
    )
    conv1d = SimpleNamespace(
        weight=torch.ones((1, 1, 1)),
        bias=None,
    )

    def packed_prefill(*args, **kwargs):
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
        in_proj=SimpleNamespace(
            weight=torch.zeros((4, 1)),
            bias=None,
        ),
        supports_packed_gdn=lambda *args: True,
        causal_conv1d_packed=lambda **kwargs: kwargs["x"],
        allocate_packed_gdn_prefill_workspace=lambda *args, **kwargs: object(),
        _prefill_workspace_cache=SimpleNamespace(
            get=lambda *args, **kwargs: object()
        ),
        packed_gated_delta_rule_prefill=packed_prefill,
        norm=lambda value, gate: value,
        out_proj=SimpleNamespace(
            weight=torch.ones((1, 1)),
            bias=None,
        ),
    )

    def runtime_linear(value, weight, bias):
        linear_calls.append((weight, bias))
        return torch.nn.functional.linear(value, weight, bias)

    monkeypatch.setattr(qwen_model_module, "_kestrel_linear", runtime_linear)

    result = Qwen3_5GatedDeltaNet.forward(
        fake,
        torch.zeros((1, 3, 1)),
        cache_params=cache,
        cu_seq_lens_q=torch.tensor([0, 3], dtype=torch.int32),
        sequence_lengths=(3,),
        topology_token=fake,
        seq_idx=torch.zeros((1, 3), dtype=torch.int32),
        gdn_state_indices=torch.tensor([0]),
        gdn_state_indices_allocator_owned=True,
    )

    assert result.shape == (1, 3, 1)
    assert len(linear_calls) == 2
    assert linear_calls[0][0] is fake.in_proj.weight
    assert linear_calls[0][1] is fake.in_proj.bias
    assert linear_calls[1][0] is fake.out_proj.weight
    assert linear_calls[1][1] is fake.out_proj.bias
    assert observed["sequence_lengths"] == (3,)
    assert observed["topology_token"] is fake
