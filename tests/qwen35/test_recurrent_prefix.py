from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.cache import Qwen35InferenceCache
from kestrel.models.qwen35.qwen_model import Qwen3_5GatedDeltaNet


def test_packed_prefix_records_split_independent_histories():
    from kestrel.models.qwen35.qwen_model import _RecurrentPrefixRecord
    module = SimpleNamespace(conv_kernel_size=4)
    qkv = torch.arange(20).reshape(1, 5, 4)
    conv = torch.arange(44).reshape(1, 4, 11)
    state = torch.arange(8).reshape(2, 1, 2, 2)
    indices = torch.tensor([3, 1])
    record = _RecurrentPrefixRecord.capture(module, qkv, qkv, qkv, conv, state, indices)
    indices.zero_()
    first, second = record.split_sequences((2, 3))
    assert torch.equal(first.qkv, qkv[:, :2])
    assert torch.equal(second.qkv, qkv[:, 2:])
    assert torch.equal(first.conv_input, conv[..., :5])
    assert torch.equal(second.conv_input, conv[..., 5:])
    assert torch.equal(first.initial_state, state[:1])
    assert torch.equal(second.initial_state, state[1:])
    assert first.state_indices.tolist() == [3]
    assert second.state_indices.tolist() == [1]
    assert first.state_indices.data_ptr() == record.state_indices.data_ptr()
    assert second.state_indices.data_ptr() == record.state_indices[1:].data_ptr()
    with pytest.raises(ValueError, match="do not match"):
        record.split_sequences((1, 3))


def test_packed_continuation_keeps_convolution_histories_separate():
    observed = {}
    layer = SimpleNamespace(
        conv_states=torch.tensor([[[10., 11., 12.]], [[20., 21., 22.]]]),
        recurrent_states=torch.zeros(2, 1, 1, 1, dtype=torch.bfloat16),
        has_previous_state=True)
    cache = Qwen35InferenceCache(
        config=SimpleNamespace(layer_types=("linear_attention",)), paged_kv=(None,))
    cache.layers = (layer,)
    cache.advance_to(9)
    cache._prefix_source = object()
    cache._prefix_start = 9
    cache._prefix_records = {}
    def conv(**kwargs):
        observed["conv"] = kwargs["x"].clone()
        observed["seq_idx"] = kwargs["seq_idx"].clone()
        return kwargs["x"]
    def recurrence(qkv, *args, **kwargs):
        observed["qkv"] = qkv.clone()
        observed["initial"] = kwargs["initial_state"]
        return torch.zeros(1, 5, 1), None
    projected = torch.zeros(1, 5, 4)
    projected[0, :, 0] = torch.arange(1, 6)
    fake = SimpleNamespace(layer_idx=0, num_k_heads=1, num_v_heads=1,
        head_k_dim=1, head_v_dim=1, conv_dim=1, conv_kernel_size=3, value_dim=1,
        activation="silu", A_log=torch.zeros(1), dt_bias=torch.zeros(1),
        conv1d=SimpleNamespace(weight=torch.ones(1, 1, 3), bias=None),
        in_proj=lambda _: projected, supports_packed_gdn=lambda *args: True,
        causal_conv1d_packed=conv, packed_gated_delta_rule_prefill=recurrence,
        allocate_packed_gdn_prefill_workspace=lambda *args, **kwargs: object(),
        _prefill_workspace_cache=SimpleNamespace(get=lambda *args, **kwargs: object()),
        norm=lambda value, gate: value, out_proj=lambda value: value)
    output = Qwen3_5GatedDeltaNet.forward(fake, torch.zeros(1, 5, 1),
        cache_params=cache, cu_seq_lens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        sequence_lengths=(2, 3), topology_token=object(),
        gdn_state_indices=torch.tensor([1, 0]), gdn_state_indices_allocator_owned=True)
    assert output.shape == (1, 5, 1)
    assert observed["conv"].flatten().tolist() == [11, 12, 1, 2, 21, 22, 3, 4, 5]
    assert observed["seq_idx"].flatten().tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert observed["qkv"].flatten().tolist() == [1, 2, 3, 4, 5]
    assert observed["initial"].shape == (2, 1, 1, 1)
    records = cache._prefix_records[0].split_sequences((2, 3))
    assert records[0].state_indices.tolist() == [1]
    assert records[1].state_indices.tolist() == [0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lengths", [(16, 16), (5, 13)])
def test_native_packed_continuation_matches_independent_sequences(monkeypatch, lengths):
    import kestrel_kernels
    from kestrel_kernels.runtime import get_runtime
    monkeypatch.setattr(kestrel_kernels, "get_runtime", get_runtime)
    torch.manual_seed(318)
    config = SimpleNamespace(hidden_size=1028, linear_num_value_heads=48,
        linear_num_key_heads=16, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6, dense_weight_format=None,
        layer_types=("linear_attention",))
    module = Qwen3_5GatedDeltaNet(config, 0).cuda().to(torch.bfloat16)
    module.A_log.data = module.A_log.data.float()
    module.dt_bias.data = module.dt_bias.data.float()
    module.norm.weight.data = module.norm.weight.data.float()
    module.in_proj = torch.nn.Identity()
    module.out_proj = torch.nn.Identity()
    width = module.conv_dim + module.value_dim + 2 * module.num_v_heads
    x = torch.randn(1, sum(lengths), width, device="cuda", dtype=torch.bfloat16) * .1
    conv = torch.randn(2, module.conv_dim, 4, device="cuda", dtype=torch.bfloat16) * .1
    recurrent = torch.randn(2, 48, 128, 128, device="cuda", dtype=torch.bfloat16) * .01
    def run(values, rows, lens):
        # A real projection allocates exact-T output; sliced B1 inputs retain
        # an irrelevant parent batch stride even after contiguous().
        values = values.reshape(-1, width).clone().view(1, sum(lens), width)
        cache = Qwen35InferenceCache(config=config, paged_kv=(None,))
        cache.layers[0].conv_states = conv[rows].clone()
        cache.layers[0].recurrent_states = recurrent[rows].clone()
        cache.layers[0].has_previous_state = True
        cache.advance_to(31)
        branch = cache.fork_recurrent_state(capture_prefix=True)
        cu, topology = get_runtime().gated_delta.bind_packed_prefill_topology(
            sequence_lengths=lens, device=x.device)
        output = module(values, branch, cu_seq_lens_q=cu, sequence_lengths=lens,
            topology_token=topology, gdn_state_indices=torch.arange(len(lens), device=x.device),
            gdn_state_indices_allocator_owned=True)
        return output, branch
    with torch.inference_mode():
        packed, cache = run(x, slice(None), lengths)
        offset = 0
        for index, length in enumerate(lengths):
            single, reference = run(x[:, offset:offset + length].contiguous(),
                slice(index, index + 1), (length,))
            torch.testing.assert_close(packed[:, offset:offset + length], single, rtol=0, atol=0)
            torch.testing.assert_close(cache.layers[0].recurrent_states[index:index + 1],
                reference.layers[0].recurrent_states, rtol=0, atol=0)
            torch.testing.assert_close(cache.layers[0].conv_states[index:index + 1],
                reference.layers[0].conv_states, rtol=0, atol=0)
            offset += length


@pytest.fixture(autouse=True)
def topology(monkeypatch):
    import kestrel_kernels
    monkeypatch.setattr(kestrel_kernels, "get_runtime", lambda: SimpleNamespace(
        gated_delta=SimpleNamespace(bind_packed_prefill_topology=lambda **kwargs:
                                   (torch.tensor([0, kwargs["sequence_lengths"][0]]), object()))))


def captured():
    source = Qwen35InferenceCache(
        config=SimpleNamespace(layer_types=("linear_attention", "linear_attention")),
        paged_kv=(None, None))
    for layer in source.layers:
        layer.conv_states = torch.ones(1, 8, 4, dtype=torch.bfloat16)
        layer.recurrent_states = torch.ones(2, 2, 4, 4, dtype=torch.bfloat16)
        layer.has_previous_state = True
    source.advance_to(31)
    branch = source.fork_recurrent_state(capture_prefix=True)

    class Record:
        qkv = torch.zeros(1, 16, 8)
        fail = False

        @property
        def replay_geometry(self):
            return (id(self),)

        def replay_into(self, layer, length, cu, topology):
            layer.recurrent_states.fill_(length)
            if self.fail:
                raise RuntimeError("injected replay failure")
            layer.conv_states.fill_(length)

    branch._prefix_records = {index: Record() for index in range(2)}
    branch.advance_to(47)
    return source, branch


def test_partial_commit_isolated_and_consumes_records():
    source, branch = captured()
    result = branch.commit_recurrent_prefix(7)
    assert result.seq_length == 38
    for old, new in zip(source.layers, result.layers):
        assert torch.all(old.recurrent_states == 1)
        assert torch.all(old.conv_states == 1)
        assert torch.all(new.recurrent_states == 7)
        assert torch.all(new.conv_states == 7)
    assert not branch._prefix_records and branch._prefix_source is None
    with pytest.raises(RuntimeError, match="no captured"):
        branch.commit_recurrent_prefix(7)


def test_full_commit_returns_verified_branch_and_releases_source():
    _, branch = captured()
    assert branch.commit_recurrent_prefix(16) is branch
    assert branch._prefix_source is None and not branch._prefix_records


def test_convolution_layout_reuses_only_matching_geometry():
    source, _ = captured()
    first = source.conv_sequence_indices((2, 3), 3, torch.device("cpu"))
    assert first.tolist() == [[0] * 5 + [1] * 6]
    assert source.conv_sequence_indices([2, 3], 3, torch.device("cpu")) is first
    branch = source.fork_recurrent_state()
    assert branch.conv_sequence_indices((2, 3), 3, torch.device("cpu")) is first
    changed = branch.conv_sequence_indices((3, 2), 3, torch.device("cpu"))
    assert changed.tolist() == [[0] * 6 + [1] * 5]
    assert changed is not first
    assert source.conv_sequence_indices((2, 3), 3, torch.device("cpu")) is first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_convolution_layout_does_not_cross_streams():
    source, _ = captured()
    device = torch.device("cuda", torch.cuda.current_device())
    first = source.conv_sequence_indices((2, 3), 3, device)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        second = source.conv_sequence_indices((2, 3), 3, device)
        assert source.conv_sequence_indices((2, 3), 3, device) is second
    stream.synchronize()
    assert second is not first
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_failed_commit_can_retry_without_modifying_source():
    source, branch = captured()
    branch._prefix_records[1].fail = True
    with pytest.raises(RuntimeError, match="injected"):
        branch.commit_recurrent_prefix(3)
    assert branch._prefix_source is source and len(branch._prefix_records) == 2
    assert all(torch.all(layer.recurrent_states == 1) for layer in source.layers)
    branch._prefix_records[1].fail = False
    assert branch.commit_recurrent_prefix(3).seq_length == 34


@pytest.mark.parametrize("length", [0, -1, 17, True, 1.5])
def test_invalid_prefix_rejected(length):
    _, branch = captured()
    with pytest.raises(ValueError, match="prefix length"):
        branch.commit_recurrent_prefix(length)


def test_stale_incomplete_and_unfinished_capture_rejected():
    source, branch = captured()
    source.advance_to(32)
    with pytest.raises(RuntimeError, match="advanced"):
        branch.commit_recurrent_prefix(1)
    _, branch = captured()
    del branch._prefix_records[1]
    with pytest.raises(RuntimeError, match="missing recurrent"):
        branch.commit_recurrent_prefix(1)
    _, branch = captured()
    branch.seq_length = 46
    with pytest.raises(RuntimeError, match="verified sequence"):
        branch.commit_recurrent_prefix(1)


def test_ordinary_fork_drops_capture_ownership():
    _, branch = captured()
    ordinary = branch.fork_recurrent_state()
    assert ordinary._prefix_source is None and not ordinary._prefix_records
    with pytest.raises(ValueError, match="committed"):
        branch.fork_recurrent_state(capture_prefix=True)


def test_duplicate_forward_and_invalid_lengths_rejected_before_kernels():
    _, branch = captured()
    module = SimpleNamespace(layer_idx=0)
    with pytest.raises(RuntimeError, match="one verification forward"):
        Qwen3_5GatedDeltaNet.forward(
            module, torch.zeros(1, 16, 8), branch,
            sequence_lengths=(16,), gdn_state_indices_allocator_owned=True)
    with pytest.raises(ValueError, match="committed sequences"):
        Qwen3_5GatedDeltaNet.forward(
            module, torch.zeros(1, 16, 8), branch,
            sequence_lengths=(8, 7), gdn_state_indices_allocator_owned=True)


def test_captured_indices_survive_caller_metadata_reuse():
    from kestrel.models.qwen35.qwen_model import _RecurrentPrefixRecord

    source, branch = captured()
    indices = torch.tensor([1])

    def native(*args, **kwargs):
        kwargs["final_state"].index_copy_(
            0, kwargs["final_state_indices"], kwargs["initial_state"])

    module = SimpleNamespace(
        head_k_dim=4, head_v_dim=4, conv_kernel_size=4, A_log=None, dt_bias=None,
        allocate_packed_gdn_prefill_workspace=None,
        _prefill_workspace_cache=SimpleNamespace(get=lambda *args, **kwargs: object()),
        packed_gated_delta_rule_prefill=native)
    for index in range(2):
        branch._prefix_records[index] = _RecurrentPrefixRecord.capture(
            module, torch.zeros(1, 16, 8 + index * 8), torch.zeros(1, 16, 2),
            torch.zeros(1, 16, 2), torch.zeros(1, 8, 19),
            torch.full((1, 2, 4, 4), 7., dtype=torch.bfloat16), indices)
    indices.zero_()
    result = branch.commit_recurrent_prefix(3)
    for old, layer in zip(source.layers, result.layers):
        assert torch.all(old.recurrent_states == 1)
        assert torch.all(layer.recurrent_states[0] == 1)
        assert torch.all(layer.recurrent_states[1] == 7)


@pytest.mark.parametrize("value_dim", [4, 6])
@pytest.mark.parametrize("pool_rows", [1, 2])
@pytest.mark.parametrize("use_graph", [False, True])
def test_grouped_replay_keeps_layer_parameters_and_state_rows(monkeypatch, value_dim, pool_rows, use_graph):
    import kestrel_kernels
    from kestrel.models.qwen35.qwen_model import _RecurrentPrefixRecord

    source, branch = captured()
    for cache in (source, branch):
        for layer in cache.layers:
            layer.recurrent_states = torch.ones(pool_rows, 2, value_dim, 4, dtype=torch.bfloat16)
    prepared, recurrences = [], []
    fail = True

    def prepare(qkv, a, b, A_log, dt_bias, *, cu_seqlens,
                sequence_lengths, topology_token, **buffers):
        prepared.append((A_log.tolist(), dt_bias.tolist()))
        assert sequence_lengths == (3, 3) and cu_seqlens.tolist() == [0, 3, 6]
        assert qkv.shape == (1, 6, 8 + 2 * value_dim)
        assert torch.all(qkv[:, :3] == 2) and torch.all(qkv[:, 3:] == 5)
        for value in buffers.values():
            value[:, :3].fill_(A_log[0, 0])
            value[:, 3:].fill_(A_log[1, 0])

    def recurrence(q, k, v, g, beta, cu, **kwargs):
        recurrences.append(kwargs["sequence_lengths"])
        assert kwargs["sequence_lengths"] == (3, 3)
        assert torch.all(q[:, :3] == 2) and torch.all(q[:, 3:] == 5)
        assert beta.dtype == torch.float32
        assert v.shape == (1, 6, 2, value_dim)
        kwargs["final_state"].copy_(kwargs["initial_state"])
        if fail:
            raise RuntimeError("injected grouped recurrence failure")

    def topology(**kwargs):
        lengths = kwargs["sequence_lengths"]
        return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]), object()

    monkeypatch.setattr(kestrel_kernels, "get_runtime", lambda: SimpleNamespace(
        gated_delta=SimpleNamespace(bind_packed_prefill_topology=topology,
            packed_prefill_prepare=prepare,
            packed_recurrent_gated_delta_rule_prefill=recurrence)))
    from contextlib import contextmanager
    from kestrel.models.qwen35.qwen_model import _replay_recurrent_prefix
    class Graph:
        @contextmanager
        def launch(self, *inputs):
            cu, token = topology(sequence_lengths=(3, 3))
            final = _replay_recurrent_prefix(*inputs, cu, token)
            yield (final,)
            # Reuse immediately after the lease: committed caches must own data.
            final.zero_()
    graph = Graph() if use_graph else None
    indices = torch.tensor([pool_rows - 1])
    for index, parameter in enumerate((2, 5)):
        module = SimpleNamespace(head_k_dim=4, head_v_dim=value_dim, conv_kernel_size=4,
                                 A_log=torch.full((2,), parameter, dtype=torch.float32),
                                 dt_bias=torch.full((2,), parameter + 1, dtype=torch.float32))
        branch._prefix_records[index] = _RecurrentPrefixRecord.capture(
            module, torch.full((1, 16, 8 + 2 * value_dim), parameter, dtype=torch.bfloat16),
            torch.zeros(1, 16, 2, dtype=torch.bfloat16),
            torch.zeros(1, 16, 2, dtype=torch.bfloat16), torch.full((1, 8, 19), parameter),
            torch.full((1, 2, value_dim, 4), parameter, dtype=torch.bfloat16), indices)
    indices.zero_()
    with pytest.raises(RuntimeError, match="grouped recurrence failure"):
        branch.commit_recurrent_prefix(3, replay_graph=graph)
    assert branch._prefix_source is source and len(branch._prefix_records) == 2
    assert all(torch.all(layer.recurrent_states == 1) for layer in source.layers)
    fail = False
    prepared.clear()
    recurrences.clear()
    result = branch.commit_recurrent_prefix(3, replay_graph=graph)
    assert prepared == [([[2, 2], [5, 5]], [[3, 3], [6, 6]])]
    assert recurrences == [(3, 3)]
    for old, layer, parameter in zip(source.layers, result.layers, (2, 5)):
        assert torch.all(old.recurrent_states == 1)
        if pool_rows > 1:
            assert torch.all(layer.recurrent_states[0] == 1)
        assert torch.all(layer.recurrent_states[pool_rows - 1] == parameter)
        assert torch.all(layer.conv_states == parameter)
