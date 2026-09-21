from types import SimpleNamespace
from contextlib import nullcontext

import pytest
import torch

from kestrel.models.qwen35.spec_decoder import Qwen35DFlashDecoder, _Session
from kestrel.runtime.spec import DraftResult, SpecAdmission
from kestrel.runtime.tokens import TextToken


def decoder():
    obj = Qwen35DFlashDecoder.__new__(Qwen35DFlashDecoder)
    obj._target_graph = None
    obj._replay_graph = None
    obj._draft_graph = None
    obj._draft_graph_enabled = False
    obj._graph_failed = obj._closed = False
    obj._commit_stream = SimpleNamespace(synchronize=lambda: None)
    obj._finalization_stream = nullcontext
    obj._commit_ready = None
    obj._commit_pending = False
    state = SimpleNamespace(batch_idx=1, max_length=100, length=10)
    erased = []
    obj.runtime = SimpleNamespace(page_table=SimpleNamespace(
        free_batch_idx=[2], erase=erased.append), max_seq_length=128, vocab_size=256,
        max_batch_size=2)
    obj.num_lookahead_tokens = 4
    obj.num_speculative_tokens = 3
    obj._sessions = {1: _Session(state, SimpleNamespace(seq_length=10), None, None, 7)}
    obj.propose = lambda ctx: DraftResult(torch.tensor([[8, 9, 10]]))
    obj._propose_many = lambda sessions: [
        [session.bonus, *obj.propose(session).token_ids[0].tolist()] for session in sessions]
    commits = []

    class Verified:
        def __init__(self):
            self._prefix_records = {}

        def commit_recurrent_prefix(self, count, *, replay_graph=None):
            commits.append(count)
            return SimpleNamespace(seq_length=10+count)

    obj._target = lambda tokens, cache, slot, capture, leases=None: ([8, 9, 99, 100], torch.ones(1, 4, 2), Verified())
    obj._target_many = lambda candidates, sessions, leases=None: [
        obj._target(tokens, session.cache, session.state.batch_idx, capture=True)
        for tokens, session in zip(candidates, sessions)]
    return obj, state, commits, erased


@pytest.mark.parametrize("cap,expected", [(None, [8, 9, 99]), (1, [8]), (2, [8, 9]), (4, [8, 9, 99])])
def test_bonus_not_duplicated_and_caps_commit_matching_input_prefix(cap, expected):
    obj, state, commits, _ = decoder()
    result = obj.step([state], commit_caps=[cap])
    assert result.tokens == [expected]
    assert result.accept_counts == [len(expected)-1]
    assert commits == [len(expected)]
    assert obj._sessions[1].bonus == expected[-1]
    assert obj._sessions[1].cache.seq_length == 10+len(expected)
    assert state.length == 10  # Scheduler advances this exactly once.


def test_retire_releases_slot_and_capture_once():
    obj, state, _, erased = decoder()
    obj.retire(state)
    obj.retire(state)
    assert erased == [1]
    assert obj.free_slots == 1 and not obj._sessions


def test_retire_waits_for_pending_state_writes_before_releasing_slot():
    obj, state, _, erased = decoder()
    obj._commit_pending = True
    obj._commit_ready = SimpleNamespace(synchronize=lambda: erased.append("completed"))
    obj.retire(state)
    assert erased == ["completed", 1]


def test_shutdown_releases_replay_graph_after_target_shutdown_error():
    obj, _, _, _ = decoder()
    closed = []
    def target_shutdown():
        closed.append("target")
        raise RuntimeError("injected shutdown failure")
    obj._target_graph = SimpleNamespace(shutdown=target_shutdown)
    obj._replay_graph = SimpleNamespace(shutdown=lambda: closed.append("replay"))
    with pytest.raises(RuntimeError, match="injected shutdown failure"):
        obj.shutdown()
    assert obj._closed and closed == ["target", "replay"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_draft_device_ids_are_ordered_after_stream_lease():
    from contextlib import contextmanager
    obj = Qwen35DFlashDecoder.__new__(Qwen35DFlashDecoder)
    obj.draft = SimpleNamespace(config=SimpleNamespace(block_size=4, mask_token_id=0))
    obj.runtime = SimpleNamespace(device=torch.device("cuda:0"), model=SimpleNamespace(
        lm_head=lambda hidden: torch.cat((hidden, hidden + 1), dim=-1)))
    obj.text = SimpleNamespace(embed_tokens=lambda ids: torch.zeros((*ids.shape, 1), device="cuda"))
    stream = torch.cuda.Stream()
    @contextmanager
    def hidden(*args):
        caller = torch.cuda.current_stream()
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1000000)
            yield torch.zeros(1, 4, 1, device="cuda")
        caller.wait_stream(stream)
    obj._draft_hidden = hidden
    ctx = SimpleNamespace(cache=SimpleNamespace(seq_length=8),
                          draft_cache=SimpleNamespace(length=6), bonus=1,
                          features=torch.zeros(1, 2, 1, device="cuda"))
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        result = obj.propose(ctx)
        assert result.token_ids.device.type == "cuda"
        assert result.token_ids.dtype == torch.int32
        assert result.token_ids.tolist() == [[1, 1, 1]]


def test_target_graph_lease_outlives_commit_and_poison_rejects_retry():
    from contextlib import contextmanager
    obj, state, _, _ = decoder()
    active = []
    original = obj._target
    @contextmanager
    def lease():
        active.append(True)
        try:
            yield
        finally:
            active.pop()
    def target(tokens, cache, slot, capture, leases=None):
        leases.enter_context(lease())
        return original(tokens, cache, slot, capture)
    obj._target = target
    commit = obj.commit_accept
    def checked_commit(ctx):
        assert active == [True]
        commit(ctx)
    obj.commit_accept = checked_commit
    obj.step([state])
    assert not active
    obj._target_graph = SimpleNamespace(shutdown=lambda: None)
    def fail(ctx):
        assert active
        raise RuntimeError('commit failed')
    obj.commit_accept = fail
    with pytest.raises(RuntimeError, match='commit failed'):
        obj.step([state])
    assert not active and obj._graph_failed
    with pytest.raises(RuntimeError, match='graph failed'):
        obj.step([state])
    obj.shutdown()
    assert obj._closed


def test_base_adapter_slot_retirement_is_valid_without_lora_support():
    from kestrel.runtime.uncached_paged import UncachedPagedRuntime

    runtime = UncachedPagedRuntime()
    runtime.release_adapter_slot(0)
    with pytest.raises(NotImplementedError):
        runtime.release_adapter_slot(1)


@pytest.mark.parametrize("kwargs", [{"temperature": .5}, {"image": object()},
                                    {"allowed_token_ids": [1]}, {"suppress_next_token_ids": [1]},
                                    {"top_p": float("nan")}, {"top_p": float("inf")},
                                    {"temperature": float("nan")}])
def test_unsupported_admit_fails_before_allocation(kwargs):
    obj, state, _, erased = decoder()
    obj._sessions.clear()
    with pytest.raises(ValueError):
        obj.admit(state, [TextToken(token_id=1)], **kwargs)
    assert erased == [] and not obj._sessions


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_invalid_cap_fails_without_advancing(cap):
    obj, state, commits, _ = decoder()
    with pytest.raises(ValueError, match="commit cap"):
        obj.step([state], commit_caps=[cap])
    assert commits == []


def test_failed_admission_returns_page_slot():
    obj, state, _, erased = decoder()
    obj._sessions.clear()
    obj.runtime.page_table.allocate = lambda: 1
    obj.runtime.page_table.reserve = lambda *args: True
    obj.runtime.page_table.commit_block_table = lambda *args: None
    obj.runtime._paged_kv = ()
    obj.runtime._linear_state_pool = SimpleNamespace(bind_prefill_state=lambda cache: None)
    obj.text = SimpleNamespace(config=SimpleNamespace(layer_types=()))

    def fail(*args, **kwargs):
        raise RuntimeError("injected target failure")

    obj._target = fail
    with pytest.raises(RuntimeError, match="injected"):
        obj.admit(state, [TextToken(token_id=1)], top_p=.9)
    assert erased == [1] and state.batch_idx == -1 and not obj._sessions


def add_session(obj, slot=2):
    state = SimpleNamespace(batch_idx=slot, max_length=100, length=20)
    obj._sessions[slot] = _Session(state, SimpleNamespace(seq_length=20), None, None, 17)
    return state


def test_concurrent_sessions_preserve_input_order_caps_and_retirement():
    obj, first, commits, erased = decoder()
    second = add_session(obj)
    assert obj.free_slots == 0
    result = obj.step([second, first], commit_caps=[1, 2])
    assert result.tokens == [[8], [8, 9]]
    assert result.accept_counts == [0, 1]
    assert commits == [1, 2]
    assert obj._sessions[2].bonus == 8
    assert obj._sessions[1].bonus == 9
    obj.retire(first)
    obj.retire(first)
    assert erased == [1]
    assert obj._sessions[2].state is second
    obj.step([second])


@pytest.mark.parametrize("invalid", ["duplicate", "foreign", "cap", "mask", "length", "empty"])
def test_all_rows_validated_before_any_proposal(invalid):
    obj, first, commits, _ = decoder()
    second = add_session(obj)
    states = [first, second]
    kwargs = {}
    if invalid == "duplicate":
        states[1] = first
    elif invalid == "foreign":
        states[1] = SimpleNamespace(batch_idx=2)
    elif invalid == "cap":
        kwargs["commit_caps"] = [1, 0]
    elif invalid == "mask":
        kwargs["allowed_token_ids"] = [None, [3]]
    elif invalid == "length":
        kwargs["commit_caps"] = [1]
    else:
        states = []
    proposed = []
    obj.propose = lambda ctx: proposed.append(ctx)
    with pytest.raises(ValueError):
        obj.step(states, **kwargs)
    assert proposed == commits == []
    assert not any(session.failed for session in obj._sessions.values())


@pytest.mark.parametrize("phase", ["propose", "verify", "commit"])
def test_failed_multirow_step_is_fail_stop_until_retirement(phase):
    obj, first, _, erased = decoder()
    second = add_session(obj)
    original = {"propose": obj.propose, "verify": obj._target, "commit": obj.commit_accept}[phase]
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    setattr(obj, {"propose": "propose", "verify": "_target", "commit": "commit_accept"}[phase], fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        obj.step([first, second])
    assert all(session.failed for session in obj._sessions.values())
    assert erased == []
    with pytest.raises(RuntimeError, match="must be retired"):
        obj.step([first])
    obj.retire(first)
    obj.retire(second)
    assert erased == [1, 2] and not obj._sessions


def test_stale_state_cannot_retire_reused_slot():
    obj, stale, _, erased = decoder()
    current = add_session(obj, slot=1)
    with pytest.raises(ValueError, match="different"):
        obj.retire(stale)
    assert erased == [] and obj._sessions[1].state is current


def test_duplicate_admission_fails_before_allocation():
    obj, state, _, erased = decoder()
    with pytest.raises(ValueError, match="already admitted"):
        obj.admit(state, [TextToken(token_id=1)])
    assert erased == []


def test_failed_second_admission_preserves_first_session():
    obj, first, _, erased = decoder()
    original = obj._sessions[1]
    state = SimpleNamespace(batch_idx=-1, max_length=100)
    obj.runtime.page_table.allocate = lambda: 2
    obj.runtime.page_table.reserve = lambda *args: True
    obj.runtime.page_table.commit_block_table = lambda *args: None
    obj.runtime._paged_kv = ()
    obj.runtime._linear_state_pool = SimpleNamespace(bind_prefill_state=lambda cache: None)
    obj.text = SimpleNamespace(config=SimpleNamespace(layer_types=()))

    def fail(*args, **kwargs):
        raise RuntimeError("injected admission failure")

    obj._target = fail
    with pytest.raises(RuntimeError, match="injected"):
        obj.admit(state, [TextToken(token_id=1)])
    assert obj._sessions == {1: original}
    assert original.state is first and not original.failed
    assert erased == [2] and state.batch_idx == -1


@pytest.mark.parametrize("concurrency", [1, 2, 4, 8])
def test_dflash_config_accepts_concurrent_requests(concurrency):
    from kestrel.config import RuntimeConfig

    config = RuntimeConfig(model="Qwen/Qwen3.5-27B-FP8", model_path="unused",
                           draft_model_path="unused", max_batch_size=concurrency,
                           device="cuda", dtype=torch.bfloat16)
    assert config.max_batch_size == concurrency


def test_packed_target_preserves_per_sequence_positions_and_branch_ownership(monkeypatch):
    from dataclasses import dataclass, replace
    import kestrel.models.qwen35.spec_decoder as module
    from kestrel.models.qwen35.cache import Qwen35InferenceCache

    obj, first, _, _ = decoder()
    second = add_session(obj, slot=3)
    sessions = [obj._sessions[3], obj._sessions[1]]
    for session, start in zip(sessions, (7, 13)):
        cache = Qwen35InferenceCache(config=SimpleNamespace(layer_types=("linear_attention",)), paged_kv=(None,))
        cache.seq_length = start
        cache.layers[0].has_previous_state = True
        cache.layers[0].conv_states = torch.full((1, 2, 3), float(start))
        cache.layers[0].recurrent_states = torch.full((1, 1, 1, 1), float(session.state.batch_idx))
        session.cache = cache
    page_table = torch.arange(32).reshape(4, 8)
    obj.runtime.device = torch.device("cpu")
    obj.runtime.page_size = 4
    obj.runtime.page_table.page_table = page_table
    obj.runtime.model = SimpleNamespace(lm_head=lambda hidden: torch.nn.functional.one_hot(
        hidden[..., 0].long(), num_classes=16).float())
    obj.draft = SimpleNamespace(config=SimpleNamespace(target_layer_ids=(0,)))
    monkeypatch.setattr(module, "get_runtime", lambda: SimpleNamespace(gated_delta=SimpleNamespace(
        bind_packed_prefill_topology=lambda **kwargs: (torch.tensor([0, 2, 5]), object()))))

    @dataclass
    class Record:
        state_indices: torch.Tensor
        prefix_context: object | None = None

        def split_sequences(self, lengths):
            return tuple(replace(self, state_indices=self.state_indices[i:i+1]) for i in range(len(lengths)))

    def text(**kwargs):
        assert kwargs["sequence_lengths"] == (2, 3)
        assert kwargs["position_ids"].tolist() == [[7, 8, 13, 14, 15]]
        assert kwargs["paged_kv_seqlens_k"].tolist() == [9, 16]
        assert kwargs["page_table"].equal(page_table[[3, 1]])
        assert kwargs["seq_idx"].tolist() == [[0, 0, 1, 1, 1]]
        expected_slots = [page_table[3, pos // 4].item() * 4 + pos % 4 for pos in (7, 8)]
        expected_slots += [page_table[1, pos // 4].item() * 4 + pos % 4 for pos in (13, 14, 15)]
        assert kwargs["slot_mapping"].tolist() == [expected_slots]
        cache = kwargs["past_key_values"]
        assert cache.layers[0].recurrent_states.flatten().tolist() == [3, 1]
        cache.layers[0].recurrent_states.add_(100)
        cache.layers[0].conv_states.add_(100)
        cache._prefix_records[0] = Record(kwargs["gdn_state_indices"])
        hidden = kwargs["input_ids"][..., None].float()
        return SimpleNamespace(last_hidden_state=hidden, layer_hidden_states=(hidden,))

    obj.text = text
    results = Qwen35DFlashDecoder._target_many(obj, [[2, 3], [4, 5, 6]], sessions)
    for row, (result, session, count) in enumerate(zip(results, sessions, (2, 3))):
        expected, features, branch = result
        slot = session.state.batch_idx
        assert expected == ([2, 3] if row == 0 else [4, 5, 6])
        assert features.shape == (1, count, 1)
        assert branch._prefix_source is session.cache
        assert branch._prefix_start == session.cache.seq_length
        assert branch.seq_length == session.cache.seq_length + count
        assert branch._prefix_records[0].state_indices.tolist() == [0]
        assert branch.layers[0].recurrent_states.shape[0] == 1
        assert branch.layers[0].recurrent_states[0].item() == slot + 100
        assert session.cache.layers[0].recurrent_states[0].item() == slot
        assert torch.all(session.cache.layers[0].conv_states == session.cache.seq_length)
    first_branch = results[0][2].layers[0]
    second_branch = results[1][2].layers[0]
    first_branch.recurrent_states.zero_()
    assert second_branch.recurrent_states.item() == 101
    assert sessions[0].cache.layers[0].recurrent_states.item() == 3
    assert sessions[1].cache.layers[0].recurrent_states.item() == 1


def test_packed_fork_rejects_shared_owner_and_noncompact_state():
    from kestrel.models.qwen35.cache import Qwen35InferenceCache
    cache = Qwen35InferenceCache(
        config=SimpleNamespace(layer_types=("linear_attention",)), paged_kv=(None,))
    cache.seq_length = 3
    cache.layers[0].conv_states = torch.zeros(2, 2, 3)
    cache.layers[0].recurrent_states = torch.zeros(2, 1, 1, 1)
    with pytest.raises(ValueError, match="distinct caches"):
        Qwen35InferenceCache.fork_packed_recurrent_state([cache, cache])
    with pytest.raises(ValueError, match="single-row state"):
        Qwen35InferenceCache.fork_packed_recurrent_state([cache])


def test_admission_forks_only_owned_recurrent_row():
    obj, first, _, erased = decoder()
    state = SimpleNamespace(batch_idx=-1, max_length=100)
    obj.runtime.page_table.allocate = lambda: 2
    obj.runtime.page_table.reserve = lambda *args: True
    obj.runtime.page_table.commit_block_table = lambda *args: None
    obj.runtime._paged_kv = (None,)
    obj.text = SimpleNamespace(config=SimpleNamespace(layer_types=("linear_attention",)))
    pool = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)

    def bind(cache):
        cache.layers[0].recurrent_states = pool

    obj.runtime._linear_state_pool = SimpleNamespace(bind_prefill_state=bind)

    def target(tokens, cache, slot, capture, leases=None):
        assert slot == 2 and not capture
        assert cache.layers[0].recurrent_states.shape == (1, 1, 1, 1)
        assert cache.layers[0].recurrent_states.item() == 2
        branch = cache.fork_recurrent_state()
        branch.layers[0].recurrent_states.add_(100)
        return [3], torch.zeros(1, 1, 1), branch

    obj._target = target
    obj.admit(state, [TextToken(token_id=1)])
    assert pool.flatten().tolist() == [0, 1, 2, 3]
    assert obj._sessions[2].cache.layers[0].recurrent_states.item() == 102
    assert obj._sessions[1].state is first and erased == []


@pytest.mark.parametrize("fail_forward", [False, True])
def test_packed_admission_isolates_slots_lengths_and_failures(monkeypatch, fail_forward):
    obj, existing, _, erased = decoder()
    obj.runtime.max_batch_size = 3
    obj.runtime.device = torch.device("cpu")
    obj.runtime.page_size = 2
    available = [3, 2]
    obj.runtime.page_table.free_batch_idx = available
    obj.runtime.page_table.allocate = lambda: available.pop(0)
    obj.runtime.page_table.reserve = lambda *args: None
    obj.runtime.page_table.commit_block_table = lambda *args: None
    obj.runtime.page_table.page_table = torch.arange(40).reshape(4, 10)
    obj.runtime._paged_kv = (None,)
    obj.text = SimpleNamespace(config=SimpleNamespace(layer_types=("linear_attention",)))
    obj.draft = SimpleNamespace(config=SimpleNamespace(target_layer_ids=(0,)))
    pool = torch.arange(4, dtype=torch.bfloat16).reshape(4, 1, 1, 1)
    obj.runtime._linear_state_pool = SimpleNamespace(bind_prefill_state=lambda cache:
        setattr(cache.layers[0], "recurrent_states", pool))
    obj.runtime.model = SimpleNamespace(lm_head=lambda hidden:
        torch.nn.functional.one_hot(hidden.long().squeeze(-1), num_classes=16).float())
    from kestrel.models.qwen35 import spec_decoder as module
    def topology(*, sequence_lengths, device):
        return torch.tensor([0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
                            device=device, dtype=torch.int32), None
    monkeypatch.setattr(module, "get_runtime", lambda: SimpleNamespace(gated_delta=
        SimpleNamespace(bind_packed_prefill_topology=topology)))
    calls = []
    def verify(leases, **kwargs):
        calls.append(kwargs)
        assert kwargs["sequence_lengths"] == (2, 1)
        assert kwargs["position_ids"].tolist() == [[0, 1, 0]]
        assert kwargs["slot_mapping"].tolist() == [[60, 61, 40]]
        assert kwargs["gdn_state_indices"].tolist() == [0, 1]
        assert kwargs["seq_idx"].tolist() == [[0, 0, 1]]
        if fail_forward:
            raise RuntimeError("injected packed prefill failure")
        layer = kwargs["past_key_values"].layers[0]
        assert not layer.has_previous_state
        layer.recurrent_states.copy_(torch.tensor([2, 3]).reshape(2, 1, 1, 1))
        layer.conv_states = torch.tensor([2, 1]).reshape(2, 1, 1)
        layer.has_previous_state = True
        hidden = kwargs["input_ids"].float().unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden, layer_hidden_states=(hidden,))
    obj._verify = verify
    states = [SimpleNamespace(batch_idx=-1, max_length=100) for _ in range(3)]
    requests = [SpecAdmission(state, [TextToken(token_id=t) for t in tokens], {})
                for state, tokens in zip(states, ([1, 2], [999], [3]))]
    results = obj.admit_many(requests)
    assert len(calls) == 1 and isinstance(results[1], ValueError)
    assert obj._sessions[1].state is existing
    assert pool.flatten().tolist() == [0, 1, 2, 3]
    if fail_forward:
        assert all(isinstance(result, Exception) for result in results)
        assert erased == [3, 2] and all(state.batch_idx == -1 for state in states)
        assert set(obj._sessions) == {1}
    else:
        assert results[0] == (2, None) and results[2] == (3, None)
        first, second = obj._sessions[3], obj._sessions[2]
        assert (first.cache.seq_length, second.cache.seq_length) == (2, 1)
        assert first.cache.layers[0].has_previous_state
        assert second.cache.layers[0].has_previous_state
        assert first.features.flatten().tolist() == [1, 2]
        assert second.features.flatten().tolist() == [3]
        first.cache.layers[0].recurrent_states.fill_(99)
        assert second.cache.layers[0].recurrent_states.item() == 3
        assert first.spare_cache.layers[0].recurrent_states.item() == 2


def test_duplicate_admission_state_cannot_acquire_two_owners():
    obj, existing, _, erased = decoder()
    state = SimpleNamespace(batch_idx=-1, max_length=100)
    request = SpecAdmission(state, [TextToken(token_id=1)], {})
    def unexpected(*args, **kwargs):
        pytest.fail("duplicate state must be rejected before allocation")
    obj._prepare_admission = unexpected
    results = obj.admit_many([request, request])
    assert all(isinstance(result, ValueError) for result in results)
    assert state.batch_idx == -1 and not erased
    assert obj._sessions[1].state is existing


def test_unsupported_runtime_rejects_draft_config_before_loading():
    from kestrel.config import RuntimeConfig

    with pytest.raises(ValueError, match="requires a Qwen"):
        RuntimeConfig(model="moondream3-preview", draft_model_path="unused",
                      model_path="unused", max_batch_size=1)
