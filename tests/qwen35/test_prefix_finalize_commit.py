from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.cache import Qwen35InferenceCache


def _packed_prefixes(count=2, layers=1):
    config = SimpleNamespace(layer_types=("linear_attention",) * layers)
    sources = []
    for row in range(count):
        source = Qwen35InferenceCache(config=config, paged_kv=(None,) * layers)
        source.seq_length = 10 + row
        for index, layer in enumerate(source.layers):
            layer.conv_states = torch.full((1, 2, 4), row + 10 * index, dtype=torch.bfloat16)
            layer.recurrent_states = torch.full((1, 2, 3, 3), row + 10 * index, dtype=torch.bfloat16)
            layer.has_previous_state = True
        sources.append(source)
    _, branches = Qwen35InferenceCache.fork_packed_recurrent_state(sources)
    records = {index: SimpleNamespace(
        prefix_context=object(), qkv=torch.zeros(1, count * 16, 2),
        initial_state=torch.zeros(count, 2, 3, 3, dtype=torch.bfloat16),
        conv_input=(torch.arange(count * 38).reshape(1, 2, count * 19)
                    + index * 1000).to(torch.bfloat16),
        module=SimpleNamespace(conv_kernel_size=4)) for index in range(layers)}
    for branch in branches:
        branch._prefix_records = records
        branch.advance_to(branch._prefix_start + 16)
    return sources, branches, records[0]


@pytest.mark.parametrize("lengths", [(1, 16), (16, 16), (7, 3)])
def test_packed_prefix_commit_keeps_sources_and_separate_rows(lengths):
    sources, branches, record = _packed_prefixes()
    snapshots = [(source.layers[0].conv_states.clone(),
                  source.layers[0].recurrent_states.clone()) for source in sources]

    @contextmanager
    def finalize(records, accepted):
        assert records[0].prefix_context is record.prefix_context
        assert accepted.tolist() == list(lengths)
        yield (accepted[:, None, None, None].expand(2, 2, 3, 3).to(torch.bfloat16),)

    committed = Qwen35InferenceCache.commit_recurrent_prefixes(
        branches, lengths, finalizer=finalize)
    for row, (source, branch, result, length, before) in enumerate(
            zip(sources, branches, committed, lengths, snapshots)):
        assert result is not source and result is not branch
        assert result.seq_length == source.seq_length + length
        assert torch.equal(source.layers[0].conv_states, before[0])
        assert torch.equal(source.layers[0].recurrent_states, before[1])
        assert torch.all(result.layers[0].recurrent_states == length)
        start = row * 19 + length - 1
        assert torch.equal(result.layers[0].conv_states, record.conv_input[..., start:start + 4])
        assert branch._prefix_source is None and not branch._prefix_records
    committed[0].layers[0].recurrent_states.zero_()
    assert torch.all(committed[1].layers[0].recurrent_states == lengths[1])


def test_failed_packed_finalizer_keeps_sources_and_records_retryable():
    sources, branches, record = _packed_prefixes()
    records = branches[0]._prefix_records

    @contextmanager
    def fail(*args):
        raise RuntimeError("finalizer failure")
        yield

    with pytest.raises(RuntimeError, match="finalizer failure"):
        Qwen35InferenceCache.commit_recurrent_prefixes(branches, (3, 9), finalizer=fail)
    for row, (source, branch) in enumerate(zip(sources, branches)):
        assert source.seq_length == 10 + row
        assert torch.all(source.layers[0].recurrent_states == row)
        assert branch._prefix_source is source
        assert branch._prefix_records is records


def test_packed_prefix_rejects_reordered_owners_before_finalizing():
    _, branches, _ = _packed_prefixes()

    def never(*args):
        raise AssertionError("finalizer must not run")

    with pytest.raises(RuntimeError, match="row ownership"):
        Qwen35InferenceCache.commit_recurrent_prefixes(branches[::-1], (3, 9), finalizer=never)


@pytest.mark.parametrize("failure", ("second_layer", "recurrent_copy", "conv_copy"))
def test_c8_multilayer_partial_failure_never_mutates_committed_or_verified_states(monkeypatch, failure):
    sources, branches, _ = _packed_prefixes(count=8, layers=3)
    records = branches[0]._prefix_records
    counts = (1, 16, 3, 15, 7, 16, 2, 9)
    snapshots = [(owner, owner.seq_length,
                  [(layer.conv_states.clone(), layer.recurrent_states.clone())
                   for layer in owner.layers]) for owner in (*sources, *branches)]
    written = []

    @contextmanager
    def finalize(records, accepted):
        templates = tuple(record.initial_state for record in records.values())
        outputs = tuple(torch.empty_like(template) for template in templates)
        for index, output in enumerate(outputs):
            output.fill_(index + 100)
            written.append(index)
            if failure == "second_layer" and index == 1:
                raise RuntimeError("second layer failed after writing")
        yield outputs

    original_copy = torch._foreach_copy_
    copy_calls = []

    def copy_many(destinations, values):
        copy_calls.append(len(destinations))
        fail_call = {"recurrent_copy": 1, "conv_copy": 2}.get(failure)
        if len(copy_calls) == fail_call:
            destinations[0].copy_(values[0])
            raise RuntimeError("copyback failed after writing")
        return original_copy(destinations, values)

    monkeypatch.setattr(torch, "_foreach_copy_", copy_many)
    with pytest.raises(RuntimeError, match="failed after writing"):
        Qwen35InferenceCache.commit_recurrent_prefixes(branches, counts, finalizer=finalize)
    assert len(written) >= 2
    for owner, length, layers in snapshots:
        assert owner.seq_length == length
        for layer, (conv, recurrent) in zip(owner.layers, layers):
            assert torch.equal(layer.conv_states, conv)
            assert torch.equal(layer.recurrent_states, recurrent)
    for row, (source, branch) in enumerate(zip(sources, branches)):
        assert branch._prefix_source is source
        assert branch._prefix_start == source.seq_length
        assert branch._prefix_row == row
        assert branch._prefix_records is records


def test_c8_multilayer_mixed_full_partial_commit_has_independent_destinations():
    sources, branches, _ = _packed_prefixes(count=8, layers=3)
    records = branches[0]._prefix_records
    counts = (1, 16, 3, 15, 7, 16, 2, 9)

    @contextmanager
    def finalize(records, accepted):
        templates = tuple(record.initial_state for record in records.values())
        yield tuple((accepted[:, None, None, None] + index * 32)
                    .expand_as(template).to(template.dtype)
                    for index, template in enumerate(templates))

    results = Qwen35InferenceCache.commit_recurrent_prefixes(branches, counts, finalizer=finalize)
    for row, (source, result, count) in enumerate(zip(sources, results, counts)):
        assert result.seq_length == source.seq_length + count
        for index, layer in enumerate(result.layers):
            assert torch.all(layer.recurrent_states == count + index * 32)
            start = row * 19 + count - 1
            assert torch.equal(layer.conv_states, records[index].conv_input[..., start:start + 4])
            assert torch.all(source.layers[index].recurrent_states == row + index * 10)
    results[0].layers[0].recurrent_states.zero_()
    assert torch.all(results[0].layers[1].recurrent_states == counts[0] + 32)
    assert torch.all(results[1].layers[0].recurrent_states == counts[1])


def test_commit_overwrites_poisoned_destinations_without_initialization_copy(monkeypatch):
    sources, branches, record = _packed_prefixes()
    paged = object()
    for owner in (*sources, *branches):
        owner.layers += (paged,)
    empty_like = torch.empty_like
    copy_many = torch._foreach_copy_
    allocations, copies = [], []

    def poisoned_empty(template):
        result = empty_like(template).fill_(float("nan"))
        allocations.append(result)
        return result

    def counted_copy(destinations, values):
        assert all(torch.isnan(tensor).all() for tensor in destinations)
        if not copies:
            assert all(tensor.is_contiguous() for tensor in values)
        else:
            assert all(not tensor.is_contiguous() for tensor in values)
        copies.append(len(destinations))
        return copy_many(destinations, values)

    @contextmanager
    def finalize(records, accepted):
        assert allocations == []
        assert copies == []
        yield (torch.full_like(record.initial_state, 42),)

    monkeypatch.setattr(torch, "empty_like", poisoned_empty)
    monkeypatch.setattr(torch, "_foreach_copy_", counted_copy)
    results = Qwen35InferenceCache.commit_recurrent_prefixes(
        branches, (3, 16), finalizer=finalize)
    assert copies == [2, 2]
    assert len(allocations) == 4
    for row, result in enumerate(results):
        assert result.layers[-1] is paged
        assert result._prefix_source is None and not result._prefix_records
        assert result._prefix_start == result._prefix_row == 0
        layer = result.layers[0]
        assert layer.has_previous_state
        assert torch.all(layer.recurrent_states == 42)
        assert torch.isfinite(layer.conv_states).all()
        for field in ("conv_states", "recurrent_states"):
            assert getattr(layer, field).data_ptr() != getattr(sources[row].layers[0], field).data_ptr()


def test_commit_rejects_uncovered_recurrent_layer_before_allocating(monkeypatch):
    sources, branches, _ = _packed_prefixes()
    sources[1].layers += (sources[1].layers[0],)

    def never(*args, **kwargs):
        raise AssertionError("must validate before allocation or finalization")

    monkeypatch.setattr(torch, "empty_like", never)
    with pytest.raises(ValueError, match="recurrent layer layout"):
        Qwen35InferenceCache.commit_recurrent_prefixes(branches, (3, 16), finalizer=never)


def test_destination_allocation_failure_releases_lease_without_publishing(monkeypatch):
    sources, branches, record = _packed_prefixes()
    records = branches[0]._prefix_records
    states = [(source.layers[0].conv_states.clone(), source.layers[0].recurrent_states.clone())
              for source in sources]
    events = []

    @contextmanager
    def finalize(records, accepted):
        events.append("submitted")
        try:
            yield (torch.full_like(record.initial_state, 42),)
        finally:
            events.append("released")

    def fail(template):
        assert events == ["submitted"]
        raise RuntimeError("destination allocation failed")

    monkeypatch.setattr(torch, "empty_like", fail)
    with pytest.raises(RuntimeError, match="destination allocation failed"):
        Qwen35InferenceCache.commit_recurrent_prefixes(branches, (3, 16), finalizer=finalize)
    assert events == ["submitted", "released"]
    for source, branch, (conv, recurrent) in zip(sources, branches, states):
        assert branch._prefix_source is source
        assert branch._prefix_records is records
        assert source.seq_length == branch._prefix_start
        assert torch.equal(source.layers[0].conv_states, conv)
        assert torch.equal(source.layers[0].recurrent_states, recurrent)
