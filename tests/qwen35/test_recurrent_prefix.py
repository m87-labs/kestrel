from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.cache import Qwen35InferenceCache
from kestrel.models.qwen35.qwen_model import Qwen3_5GatedDeltaNet


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


def test_duplicate_forward_and_packed_batch_rejected_before_kernels():
    _, branch = captured()
    module = SimpleNamespace(layer_idx=0)
    with pytest.raises(RuntimeError, match="one verification forward"):
        Qwen3_5GatedDeltaNet.forward(
            module, torch.zeros(1, 16, 8), branch,
            sequence_lengths=(16,), gdn_state_indices_allocator_owned=True)
    with pytest.raises(ValueError, match="one committed sequence"):
        Qwen3_5GatedDeltaNet.forward(
            module, torch.zeros(1, 16, 8), branch,
            sequence_lengths=(8, 8), gdn_state_indices_allocator_owned=True)


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
        branch._prefix_records[index] = _RecurrentPrefixRecord(
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
def test_grouped_replay_keeps_layer_parameters_and_state_rows(monkeypatch, value_dim):
    import kestrel_kernels
    from kestrel.models.qwen35.qwen_model import _RecurrentPrefixRecord

    source, branch = captured()
    for cache in (source, branch):
        for layer in cache.layers:
            layer.recurrent_states = torch.ones(2, 2, value_dim, 4, dtype=torch.bfloat16)
    prepared, recurrences = [], []
    fail = True

    def prepare(qkv, a, b, A_log, dt_bias, **buffers):
        prepared.append((A_log, dt_bias))
        for value in buffers.values():
            value.fill_(A_log)

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
    indices = torch.tensor([1])
    for index, parameter in enumerate((2, 5)):
        module = SimpleNamespace(head_k_dim=4, head_v_dim=value_dim, conv_kernel_size=4,
                                 A_log=parameter, dt_bias=parameter + 1)
        branch._prefix_records[index] = _RecurrentPrefixRecord(
            module, torch.zeros(1, 16, 8 + 2 * value_dim), torch.zeros(1, 16, 2),
            torch.zeros(1, 16, 2), torch.full((1, 8, 19), parameter),
            torch.full((1, 2, value_dim, 4), parameter, dtype=torch.bfloat16), indices)
    indices.zero_()
    with pytest.raises(RuntimeError, match="grouped recurrence failure"):
        branch.commit_recurrent_prefix(3)
    assert branch._prefix_source is source and len(branch._prefix_records) == 2
    assert all(torch.all(layer.recurrent_states == 1) for layer in source.layers)
    fail = False
    prepared.clear()
    recurrences.clear()
    result = branch.commit_recurrent_prefix(3)
    assert prepared == [(2, 3), (5, 6)] and recurrences == [(3, 3)]
    for old, layer, parameter in zip(source.layers, result.layers, (2, 5)):
        assert torch.all(old.recurrent_states == 1)
        assert torch.all(layer.recurrent_states[0] == 1)
        assert torch.all(layer.recurrent_states[1] == parameter)
        assert torch.all(layer.conv_states == parameter)
