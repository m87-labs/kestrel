from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.cache import Qwen35InferenceCache
from kestrel.models.qwen35.spec_decoder import Qwen35DFlashDecoder
from test_prefix_finalize_commit import _packed_prefixes, _paired_outputs


def test_inactive_banks_reuse_tensors_without_mutating_committed_sources(monkeypatch):
    sources, branches, _ = _packed_prefixes(count=2, layers=2)
    banks = tuple(source.fork_recurrent_state() for source in sources)
    pointers = tuple(layer.recurrent_states.data_ptr() for bank in banks for layer in bank.layers)
    records = branches[0]._prefix_records
    lengths = torch.tensor([3, 9], dtype=torch.int32)
    outputs = _paired_outputs(records, lengths,
        tuple(torch.full_like(record.initial_state, 42) for record in records.values()))

    @contextmanager
    def finalize(*args):
        yield outputs

    def no_allocation(*args, **kwargs):
        raise AssertionError("commit allocated a destination")

    monkeypatch.setattr(torch, "empty_like", no_allocation)
    result = Qwen35InferenceCache.commit_recurrent_prefixes(
        branches, (3, 9), finalizer=finalize, _destinations=banks)
    assert result == banks
    assert pointers == tuple(layer.recurrent_states.data_ptr() for bank in banks for layer in bank.layers)
    for row, source in enumerate(sources):
        for index, layer in enumerate(source.layers):
            assert torch.all(layer.recurrent_states == row + 10 * index)
            assert torch.all(result[row].layers[index].recurrent_states == 42)


def test_active_cache_cannot_be_used_as_staging_bank():
    sources, branches, _ = _packed_prefixes()
    with pytest.raises(ValueError, match="inactive cache banks"):
        Qwen35InferenceCache.commit_recurrent_prefixes(branches, (1, 2), _destinations=sources)


def test_repeated_bank_swaps_keep_storage_bounded():
    current, _, _ = _packed_prefixes(count=2, layers=2)
    spare = tuple(source.fork_recurrent_state() for source in current)
    owners = {id(cache) for cache in (*current, *spare)}
    pointers = {layer.recurrent_states.data_ptr() for cache in (*current, *spare) for layer in cache.layers}
    for value in range(3):
        _, branches = Qwen35InferenceCache.fork_packed_recurrent_state(current)
        _, fixtures, _ = _packed_prefixes(count=2, layers=2)
        records = fixtures[0]._prefix_records
        for branch in branches:
            branch._prefix_records = records
            branch.advance_to(branch._prefix_start + 16)
        outputs = _paired_outputs(records, torch.tensor([2, 3], dtype=torch.int32),
            tuple(torch.full_like(record.initial_state, value) for record in records.values()))

        @contextmanager
        def finalize(*args):
            yield outputs

        result = Qwen35InferenceCache.commit_recurrent_prefixes(
            branches, (2, 3), finalizer=finalize, _destinations=spare)
        spare, current = current, result
        assert {id(cache) for cache in (*current, *spare)} == owners
        assert {layer.recurrent_states.data_ptr() for cache in (*current, *spare)
                for layer in cache.layers} == pointers
        assert [cache.seq_length for cache in current] == [10 + 2 * (value + 1), 11 + 3 * (value + 1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_finalization_overlaps_draft_but_target_waits():
    decoder = Qwen35DFlashDecoder.__new__(Qwen35DFlashDecoder)
    decoder.runtime = SimpleNamespace(device=torch.device("cuda:0"))
    decoder._commit_stream = torch.cuda.Stream()
    decoder._commit_inputs_ready = torch.cuda.Event()
    decoder._commit_ready = torch.cuda.Event()
    decoder._commit_pending = False
    compute = torch.cuda.Stream()
    with torch.cuda.stream(compute):
        state = torch.zeros(1, device="cuda")
        draft = torch.empty_like(state)
        state.fill_(0)
        draft.fill_(2)
        torch.cuda._sleep(1)
        compute.synchronize()
        with decoder._finalization_stream():
            torch.cuda._sleep(200_000_000)
            state.fill_(7)
        # Independent draft work must not inherit the finalization dependency.
        draft.fill_(2)
        compute.synchronize()
        assert not decoder._commit_ready.query()
        decoder._wait_for_commit()
        observed = state.clone()
    compute.synchronize()
    assert draft.item() == 2
    assert observed.item() == 7


def test_partial_copy_failure_does_not_publish_inactive_bank(monkeypatch):
    sources, branches, _ = _packed_prefixes(count=2, layers=2)
    banks = tuple(source.fork_recurrent_state() for source in sources)
    records = branches[0]._prefix_records
    lengths = torch.tensor([3, 9], dtype=torch.int32)
    outputs = _paired_outputs(records, lengths,
        tuple(torch.full_like(record.initial_state, 42) for record in records.values()))
    exited = []

    @contextmanager
    def finalize(*args):
        try:
            yield outputs
        finally:
            exited.append(True)

    original = torch._foreach_copy_

    def fail(destinations, values):
        original(destinations, values)
        raise RuntimeError("partial copy")

    monkeypatch.setattr(torch, "_foreach_copy_", fail)
    with pytest.raises(RuntimeError, match="partial copy"):
        Qwen35InferenceCache.commit_recurrent_prefixes(
            branches, (3, 9), finalizer=finalize, _destinations=banks)
    assert exited == [True]
    for row, (source, branch, bank) in enumerate(zip(sources, branches, banks)):
        assert source.seq_length == bank.seq_length == 10 + row
        assert branch._prefix_source is source
        for index, layer in enumerate(source.layers):
            assert torch.all(layer.recurrent_states == row + 10 * index)
