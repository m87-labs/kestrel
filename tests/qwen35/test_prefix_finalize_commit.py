from contextlib import contextmanager
from asyncio import CancelledError
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.cache import Qwen35InferenceCache, _finalize_recurrent_prefixes


def _paired_outputs(records, lengths, states):
    outputs = []
    for record, state in zip(records.values(), states, strict=True):
        width = record.module.conv_kernel_size
        history = torch.cat([
            record.conv_input[..., row * (16 + width - 1) + length - 1:
                              row * (16 + width - 1) + length - 1 + width]
            for row, length in enumerate(lengths.tolist())], dim=0).contiguous()
        for row in range(lengths.numel()):
            outputs.extend((state[row:row + 1], history[row:row + 1]))
    return tuple(outputs)


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
        yield _paired_outputs(records, accepted,
            (accepted[:, None, None, None].expand(2, 2, 3, 3).to(torch.bfloat16),))

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


def test_c8_cached_row_outputs_remove_768_commit_slice_constructions(monkeypatch):
    sources, branches, _ = _packed_prefixes(count=8, layers=48)
    records = branches[0]._prefix_records
    lengths = torch.tensor([1, 16, 3, 15, 7, 16, 2, 9], dtype=torch.int32)
    constructed = []
    getitem = torch.Tensor.__getitem__

    def count_slice(tensor, index):
        result = getitem(tensor, index)
        if (tensor.ndim in (3, 4) and tensor.shape[0] == 8
                and isinstance(index, slice) and index.step is None
                and isinstance(index.start, int) and isinstance(index.stop, int)
                and index.stop - index.start == 1):
            constructed.append(1)
        return result

    def finalize(contexts, accepted, *, out_states):
        for state in out_states:
            state.fill_(42)

    monkeypatch.setattr(torch.Tensor, "__getitem__", count_slice)
    monkeypatch.setattr("kestrel_kernels.get_runtime", lambda: SimpleNamespace(
        gated_delta=SimpleNamespace(finalize_packed_gated_delta_prefix=finalize)))
    cached = _finalize_recurrent_prefixes(records, lengths)
    assert len(cached) == len(constructed) == 768
    identities = tuple(map(id, cached))
    constructed.clear()

    @contextmanager
    def lease(bound_records, accepted):
        assert bound_records is records
        yield cached

    results = Qwen35InferenceCache.commit_recurrent_prefixes(
        branches, tuple(lengths.tolist()), finalizer=lease)
    assert constructed == []
    assert tuple(map(id, cached)) == identities
    for index in range(48):
        for row in range(8):
            state, history = cached[2 * (index * 8 + row):2 * (index * 8 + row + 1)]
            layer = results[row].layers[index]
            assert torch.equal(layer.recurrent_states, state)
            assert torch.equal(layer.conv_states, history)
            assert not torch._C._overlaps(layer.recurrent_states, state)
            assert not torch._C._overlaps(layer.conv_states, history)
            assert torch.all(sources[row].layers[index].recurrent_states == row + index * 10)


def test_c8_commit_reads_record_geometry_once_per_layer():
    _, branches, _ = _packed_prefixes(count=8, layers=48)
    records = branches[0]._prefix_records
    lengths = torch.ones(8, dtype=torch.int32)
    outputs = _paired_outputs(records, lengths,
        tuple(torch.full_like(record.initial_state, 42) for record in records.values()))
    accesses = []

    class ShapeProbe:
        def __init__(self, tensor):
            self.tensor = tensor

        @property
        def shape(self):
            accesses.append(self)
            return self.tensor.shape

        @property
        def device(self):
            return self.tensor.device

    for record in records.values():
        for field in ("qkv", "initial_state", "conv_input"):
            setattr(record, field, ShapeProbe(getattr(record, field)))

    @contextmanager
    def finalize(bound_records, accepted):
        assert bound_records is records
        assert len(accesses) == 48 * 3
        assert len({id(probe) for probe in accesses}) == 48 * 3
        yield outputs

    committed = Qwen35InferenceCache.commit_recurrent_prefixes(
        branches, (1,) * 8, finalizer=finalize)
    assert len(accesses) == 144
    assert all(result.seq_length == 11 + row for row, result in enumerate(committed))


@pytest.mark.parametrize("field", ("conv_states", "recurrent_states"))
def test_late_owner_geometry_failure_precedes_allocation_and_finalization(monkeypatch, field):
    sources, branches, _ = _packed_prefixes(count=8, layers=48)
    records = branches[0]._prefix_records
    layer = sources[-1].layers[-1]
    tensor = getattr(layer, field)
    setattr(layer, field, tensor.expand(2, *tensor.shape[1:]))

    def never(*args, **kwargs):
        raise AssertionError("all source geometry must be validated before writes")

    monkeypatch.setattr(torch, "empty_like", never)
    monkeypatch.setattr(torch, "tensor", never)
    with pytest.raises(ValueError, match="state geometry"):
        Qwen35InferenceCache.commit_recurrent_prefixes(branches, (1,) * 8, finalizer=never)
    for source, branch in zip(sources, branches):
        assert branch._prefix_source is source
        assert branch._prefix_records is records
        assert source.seq_length == branch._prefix_start


@pytest.mark.parametrize("failure", ("second_layer", "recurrent_copy", "conv_copy", "cancel"))
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
        yield _paired_outputs(records, accepted, outputs)

    original_copy = torch._foreach_copy_
    copy_calls = []

    def copy_many(destinations, values):
        copy_calls.append(len(destinations))
        fail_call = {"recurrent_copy": 1, "conv_copy": 2, "cancel": 2}.get(failure)
        if len(copy_calls) == fail_call:
            destinations[0].copy_(values[0])
            if failure == "cancel":
                raise CancelledError("copyback cancelled after writing")
            raise RuntimeError("copyback failed after writing")
        return original_copy(destinations, values)

    monkeypatch.setattr(torch, "_foreach_copy_", copy_many)
    error = CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(error, match="after writing"):
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
        states = tuple((accepted[:, None, None, None] + index * 32)
                    .expand_as(template).to(template.dtype)
                    for index, template in enumerate(templates))
        yield _paired_outputs(records, accepted, states)

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
        assert all(tensor.is_contiguous() for tensor in values)
        copies.append(len(destinations))
        return copy_many(destinations, values)

    @contextmanager
    def finalize(records, accepted):
        assert allocations == []
        assert copies == []
        yield _paired_outputs(records, accepted, (torch.full_like(record.initial_state, 42),))

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
            yield _paired_outputs(records, accepted, (torch.full_like(record.initial_state, 42),))
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


@pytest.mark.parametrize("count", (1, 8))
@pytest.mark.parametrize("token_major", (False, True))
@pytest.mark.parametrize("width", (1, 4))
@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda",
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))))
def test_history_gather_matches_slices_with_poisoned_rejected_storage(monkeypatch, count, token_major, width, device):
    lengths = torch.tensor(([1, 16, 3, 15, 7, 16, 2, 9])[:count], dtype=torch.int32, device=device)
    span, channels = 16 + width - 1, 5
    shape = (1, count * span, channels) if token_major else (1, channels, count * span)
    history = torch.full(shape, float("nan"), dtype=torch.bfloat16, device=device)
    if token_major:
        history = history.transpose(1, 2)
    expected = torch.arange(count * channels * width, device=device).reshape(count, channels, width).bfloat16()
    expected[..., 0] = -0.0
    for row, length in enumerate(lengths.tolist()):
        start = row * span + length - 1
        history[..., start:start + width].copy_(expected[row:row + 1])
    initial = torch.ones((count, 2, 3, 3), dtype=torch.bfloat16, device=device)
    record = SimpleNamespace(conv_input=history, initial_state=initial,
        prefix_context=object(), module=SimpleNamespace(conv_kernel_size=width))
    before = history.contiguous().view(torch.int16).clone()

    def finalize(contexts, accepted, *, out_states):
        assert accepted is lengths
        out_states[0].fill_(23)

    monkeypatch.setattr("kestrel_kernels.get_runtime", lambda: SimpleNamespace(
        gated_delta=SimpleNamespace(finalize_packed_gated_delta_prefix=finalize)))
    outputs = _finalize_recurrent_prefixes({0: record}, lengths)
    state, gathered = torch.cat(outputs[::2]), torch.cat(outputs[1::2])
    assert gathered.is_contiguous()
    assert torch.equal(gathered.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(history.contiguous().view(torch.int16), before)
    assert torch.all(initial == 1) and torch.all(state == 23)
    assert not torch._C._overlaps(gathered, history)


def test_all_history_geometry_checked_before_state_writes(monkeypatch):
    _, branches, _ = _packed_prefixes(count=8, layers=2)
    records = branches[0]._prefix_records
    records[1].conv_input = records[1].conv_input[..., :-1]

    def never(*args, **kwargs):
        raise AssertionError("must validate every history before state finalization")

    monkeypatch.setattr("kestrel_kernels.get_runtime", never)
    with pytest.raises(ValueError, match="history geometry"):
        _finalize_recurrent_prefixes(records, torch.ones(8, dtype=torch.int32))


@pytest.mark.parametrize("fail_gather", (False, True))
def test_eager_history_finalization_transaction(monkeypatch, fail_gather):
    sources, branches, _ = _packed_prefixes(count=8, layers=2)
    records = branches[0]._prefix_records
    lengths = (1, 16, 3, 15, 7, 16, 2, 9)
    calls = []

    def finalize(contexts, accepted, *, out_states):
        for index, output in enumerate(out_states):
            output.fill_(31 + index)
        calls.append("states_written")

    monkeypatch.setattr("kestrel_kernels.get_runtime", lambda: SimpleNamespace(
        gated_delta=SimpleNamespace(finalize_packed_gated_delta_prefix=finalize)))
    if fail_gather:
        def fail(*args, **kwargs):
            assert calls == ["states_written"]
            raise RuntimeError("history gather failed after states")
        monkeypatch.setattr(torch, "gather", fail)
        with pytest.raises(RuntimeError, match="history gather failed"):
            Qwen35InferenceCache.commit_recurrent_prefixes(branches, lengths)
        for source, branch in zip(sources, branches):
            assert branch._prefix_source is source and branch._prefix_records is records
            assert source.seq_length == branch._prefix_start
    else:
        results = Qwen35InferenceCache.commit_recurrent_prefixes(branches, lengths)
        for row, (result, length) in enumerate(zip(results, lengths)):
            for index, layer in enumerate(result.layers):
                assert torch.all(layer.recurrent_states == 31 + index)
                start = row * 19 + length - 1
                assert torch.equal(layer.conv_states, records[index].conv_input[..., start:start + 4])
    for row, source in enumerate(sources):
        for index, layer in enumerate(source.layers):
            assert torch.all(layer.recurrent_states == row + index * 10)
            assert torch.all(layer.conv_states == row + index * 10)
