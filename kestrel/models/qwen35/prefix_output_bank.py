"""Explicit packed output ownership for private recurrent-cache commit banks.

These tensors are finalizer destinations, never borrowed graph output scratch.
The descriptor counts owned payload, not allocator overhead or latency.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from bisect import bisect_left
import weakref

import torch


@dataclass(frozen=True)
class PrefixOutputGeometry:
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    device: torch.device
    payload_bytes: int

    @classmethod
    def from_templates(cls, templates):
        templates = tuple(templates)
        if not templates or any(t.ndim != 4 or not t.is_contiguous() for t in templates):
            raise ValueError('prefix output templates require contiguous rank-four states')
        count, device = templates[0].shape[0], templates[0].device
        if count < 1 or any(t.shape[0] != count or t.device != device for t in templates):
            raise ValueError('prefix output templates must share a nonempty cohort and device')
        return cls(tuple(tuple(t.shape) for t in templates), tuple(t.dtype for t in templates),
                   device, sum(t.numel() * t.element_size() for t in templates))


def _storage_interval(tensor):
    """Conservative byte interval including holes; no GPU reads or pointer math on device."""
    count = tensor.numel()
    if count == 0:
        return None
    start = tensor.data_ptr()
    if tensor.is_contiguous():
        extent = count
    else:
        strides = tensor.stride()
        if any(stride < 0 for stride in strides):
            raise ValueError('negative-stride source state is unsupported')
        extent = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, strides))
    return tensor.device, start, start + extent * tensor.element_size()


def validate_disjoint_outputs(outputs, sources):
    """All-layer/device overlap audit in O(N log N), conservative for strided holes."""
    grouped = {}
    for source in sources:
        interval = _storage_interval(source)
        if interval is not None:
            device, start, end = interval
            grouped.setdefault(device, []).append((start, end))
    indexed = {}
    for device, intervals in grouped.items():
        intervals.sort()
        starts, ends, maximum = [], [], 0
        for start, end in intervals:
            starts.append(start)
            maximum = max(maximum, end)
            ends.append(maximum)
        indexed[device] = starts, ends
    for output in outputs:
        device, start, end = _storage_interval(output)
        starts, ends = indexed.get(device, ((), ()))
        before_end = bisect_left(starts, end)
        if before_end and ends[before_end - 1] > start:
            raise ValueError('prefix output bank overlaps active source state')


class PrefixOutputBank:
    """One fixed row assignment; cache references keep storage alive after eviction."""
    def __init__(self, destinations, indices, templates, *, allocate=True):
        self.geometry = PrefixOutputGeometry.from_templates(templates)
        self.indices = tuple(indices)
        self.owners = tuple(weakref.ref(cache) for cache in destinations)
        if len(self.owners) != self.geometry.shapes[0][0] or len(self.indices) != len(self.geometry.shapes):
            raise ValueError('prefix bank owner/layer count does not match geometry')
        if len({id(cache) for cache in destinations}) != len(self.owners):
            raise ValueError('prefix bank owners must be distinct')
        self.states = None
        self.failed = False
        self._writer_stream = None
        self._lock = threading.Lock()
        self._active = False
        if allocate:
            self.materialize()

    def materialize(self):
        if self.states is not None:
            return
        self.states = tuple(torch.empty(shape, dtype=dtype, device=self.geometry.device)
                            for shape, dtype in zip(self.geometry.shapes, self.geometry.dtypes))
        self.row_states = tuple(tuple(state[row:row + 1] for row in range(len(self.owners)))
                                for state in self.states)
        self.row_descriptors = tuple(tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t.data_ptr())
                                          for t in rows) for rows in self.row_states)

    def is_current(self):
        return not self.failed and all((cache := reference()) is not None
                                       and getattr(cache, '_prefix_output_bank', None) is self
                                       for reference in self.owners)

    def owns_any(self, caches):
        return any(reference() is cache for reference in self.owners for cache in caches)

    def matches(self, destinations, indices, templates):
        if self.failed or self.states is None:
            return False
        if (tuple(indices) != self.indices or self.geometry != PrefixOutputGeometry.from_templates(templates)
                or len(destinations) != len(self.owners)):
            return False
        for row, (reference, cache) in enumerate(zip(self.owners, destinations)):
            if reference() is not cache or getattr(cache, '_prefix_output_bank', None) is not self:
                return False
            for layer, index in enumerate(self.indices):
                actual = cache.layers[index].recurrent_states
                if actual is not self.row_states[layer][row]:
                    return False
                if (tuple(actual.shape), tuple(actual.stride()), actual.dtype, actual.device, actual.data_ptr()) != self.row_descriptors[layer][row]:
                    return False
        return True

    @contextmanager
    def lease(self, source_tensors):
        with self._lock:
            if self.failed:
                raise RuntimeError('prefix output bank failed and must be retired')
            if self.states is None:
                raise RuntimeError('prefix output bank allocation was not prepared')
            if self._active:
                raise RuntimeError('prefix output bank already has an exclusive writer')
            stream = (torch.cuda.current_stream(self.geometry.device).cuda_stream
                      if self.geometry.device.type == 'cuda' else None)
            if self._writer_stream is not None and stream != self._writer_stream:
                raise RuntimeError('prefix output bank cannot switch writer streams')
            validate_disjoint_outputs(self.states, source_tensors)
            self._writer_stream = stream
            self._active = True
        try:
            yield self.states
        except BaseException:
            self.failed = True
            raise
        finally:
            with self._lock:
                self._active = False

    def validate_returned_states(self, returned):
        expected = tuple(t for rows in self.row_states for t in rows)
        if len(returned) != len(expected):
            raise ValueError('finalizer output count differs from bound bank')
        for actual, target in zip(returned, expected):
            if (actual.shape != target.shape or actual.stride() != target.stride()
                    or actual.dtype != target.dtype or actual.device != target.device
                    or actual.data_ptr() != target.data_ptr()):
                raise ValueError('finalizer did not return the bound output bank views')

    def publish(self, destinations):
        if not self._active:
            raise RuntimeError('prefix bank publication requires its exclusive lease')
        if len(destinations) != len(self.owners) or any(ref() is not cache for ref, cache in zip(self.owners, destinations)):
            raise ValueError('prefix bank row owners changed')
        for row, cache in enumerate(destinations):
            for layer, index in enumerate(self.indices):
                cache.layers[index].recurrent_states = self.row_states[layer][row]
            cache._prefix_output_bank = self
