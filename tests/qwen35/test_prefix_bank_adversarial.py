"""Independent commit-boundary checks for caller-owned output storage."""
from contextlib import contextmanager

import pytest
import torch
import random

from kestrel.models.qwen35.cache import Qwen35InferenceCache
from kestrel.models.qwen35.prefix_output_bank import validate_disjoint_outputs
from tests.qwen35.test_prefix_finalize_commit import _packed_prefixes, _paired_outputs


@contextmanager
def _write_bank(records, accepted, *, out_states):
    for layer, state in enumerate(out_states):
        state.fill_(100 + layer)
    yield _paired_outputs(records, accepted, out_states)


@pytest.mark.parametrize('fault', ['detached', 'reversed_rows', 'wrong_layer'])
def test_finalizer_cannot_substitute_unbound_outputs(fault):
    sources, branches, _ = _packed_prefixes(count=4, layers=2)
    destinations = [source.fork_recurrent_state() for source in sources]
    old_states = [tuple(layer.recurrent_states for layer in dest.layers)
                  for dest in destinations]

    @contextmanager
    def corrupt(records, accepted, *, out_states):
        for state in out_states:
            state.fill_(91)
        outputs = list(_paired_outputs(records, accepted, out_states))
        if fault == 'detached':
            outputs[0] = outputs[0].clone()
        elif fault == 'reversed_rows':
            outputs[0], outputs[2] = outputs[2], outputs[0]
        else:
            outputs[0], outputs[8] = outputs[8], outputs[0]
        yield tuple(outputs)

    with pytest.raises(ValueError, match='bound.*bank'):
        Qwen35InferenceCache.commit_recurrent_prefixes(
            branches, (1, 3, 7, 16), finalizer=corrupt, _destinations=destinations)
    for row, (source, branch, dest) in enumerate(zip(sources, branches, destinations)):
        assert branch._prefix_source is source and branch._prefix_records
        assert dest.seq_length == source.seq_length
        for index, layer in enumerate(dest.layers):
            assert layer.recurrent_states is old_states[row][index]
            assert torch.all(source.layers[index].recurrent_states == row + index * 10)


def test_cross_layer_active_source_alias_rejected_before_finalizer():
    sources, branches, _ = _packed_prefixes(count=4, layers=2)
    destinations = [source.fork_recurrent_state() for source in sources]
    Qwen35InferenceCache.commit_recurrent_prefixes(
        branches, (1, 3, 7, 16), finalizer=_write_bank, _destinations=destinations)
    # Reuse the bank, but make an active source in a DIFFERENT layer alias it.
    # Same-layer-only checks would miss this and permit input corruption.
    sources[3].layers[1].recurrent_states = destinations[0].layers[0].recurrent_states
    _, fresh = Qwen35InferenceCache.fork_packed_recurrent_state(sources)
    _, template_branches, _ = _packed_prefixes(count=4, layers=2)
    records = template_branches[0]._prefix_records
    for branch in fresh:
        branch._prefix_records = records
        branch.advance_to(branch._prefix_start + 16)
    before = sources[3].layers[1].recurrent_states.clone()

    def never(*args, **kwargs):
        raise AssertionError('overlap must fail before finalizer submission')

    with pytest.raises(ValueError, match='overlaps active source'):
        Qwen35InferenceCache.commit_recurrent_prefixes(
            fresh, (1, 3, 7, 16), finalizer=never, _destinations=destinations)
    assert torch.equal(sources[3].layers[1].recurrent_states, before)
    assert all(branch._prefix_records is records for branch in fresh)


def test_multiple_source_intervals_match_independent_dense_address_oracle():
    # Unsorted and nested source ranges exercise the prefix-maximum endpoint
    # needed by the interval search, rather than only adjacent/single ranges.
    storage = torch.empty(128)
    rng = random.Random(813)
    for _ in range(100):
        ranges = [sorted(rng.sample(range(129), 2)) for _ in range(12)]
        low, high = sorted(rng.sample(range(129), 2))
        sources = [storage[a:b] for a, b in ranges]
        expected = any(set(range(low, high)) & set(range(a, b)) for a, b in ranges)
        if expected:
            with pytest.raises(ValueError, match='overlaps'):
                validate_disjoint_outputs((storage[low:high],), sources)
        else:
            validate_disjoint_outputs((storage[low:high],), sources)


@pytest.mark.parametrize('view', ['transpose', 'broadcast', 'offset', 'holes'])
def test_noncontiguous_sources_keep_conservative_extent(view):
    storage = torch.empty(64)
    source = {
        'transpose': lambda: storage[:16].reshape(4, 4).T,
        'broadcast': lambda: storage[3:4].expand(8),
        'offset': lambda: storage[7:23].reshape(4, 4),
        'holes': lambda: storage[2:18:2],
    }[view]()
    offsets = [source.storage_offset() + i * source.stride(0)
               + (j * source.stride(1) if source.ndim == 2 else 0)
               for i in range(source.shape[0])
               for j in range(source.shape[1] if source.ndim == 2 else 1)]
    for position in range(64):
        if min(offsets) <= position <= max(offsets):
            with pytest.raises(ValueError, match='overlaps'):
                validate_disjoint_outputs((storage[position:position + 1],), (source,))
        else:
            validate_disjoint_outputs((storage[position:position + 1],), (source,))
