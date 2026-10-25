"""Bounded graph-owner retention, not a total CUDA allocator memory budget."""
from collections import OrderedDict
from contextlib import contextmanager
import threading


class DraftOwnerCache:
    def __init__(self, *, max_entries, known_tensor_bytes):
        if any(type(v) is not int or v <= 0 for v in (max_entries, known_tensor_bytes)):
            raise ValueError('draft owner bounds must be positive integers')
        self.max_entries = max_entries
        self.known_tensor_bytes = known_tensor_bytes
        self._entries = OrderedDict()
        self._pins = {}
        self._lock = threading.RLock()
        self.closed = False

    @property
    def reserved_bytes(self):
        return sum(size for _, size in self._entries.values())

    def _retire(self, key):
        owner, _ = self._entries[key]
        if self._pins.get(id(owner), 0) or owner.active:
            raise RuntimeError('cannot evict a leased draft owner')
        del self._entries[key]
        owner.shutdown()  # synchronizes graph work before releasing its storage

    @contextmanager
    def lease(self, key, reservation, create, *, compatible=None):
        with self._lock:
            if self.closed:
                raise RuntimeError('draft owner cache is closed')
            if type(reservation) is not int or not 0 < reservation <= self.known_tensor_bytes:
                raise RuntimeError('draft owner exceeds known-owned-tensor-byte budget')
            if key not in self._entries and compatible is not None:
                candidates = [(size, candidate) for candidate, (o, size) in self._entries.items()
                              if not o.closed and not o.failed and compatible(candidate)]
                if candidates:
                    key = min(candidates, key=lambda item: item[0])[1]
            if key in self._entries and (self._entries[key][0].failed or self._entries[key][0].closed):
                self._retire(key)
            if key not in self._entries:
                while (len(self._entries) >= self.max_entries
                       or self.reserved_bytes + reservation > self.known_tensor_bytes):
                    victim = next((k for k, (o, _) in self._entries.items()
                                   if not o.active and not self._pins.get(id(o), 0)), None)
                    if victim is None:
                        raise RuntimeError('draft owner budget exhausted by live leases')
                    self._retire(victim)
                # Factory must clean up partial construction on failure. Never
                # install a partial owner or silently fall back after an OOM.
                owner = create()
                if owner.known_owned_tensor_bytes() > reservation:
                    owner.shutdown()
                    raise RuntimeError('draft owner exceeded tensor reservation during construction')
                self._entries[key] = owner, reservation
            owner, _ = self._entries[key]
            self._entries.move_to_end(key)
            self._pins[id(owner)] = self._pins.get(id(owner), 0) + 1
        try:
            yield owner
        finally:
            with self._lock:
                self._pins[id(owner)] -= 1
                if not self._pins[id(owner)]:
                    del self._pins[id(owner)]
                if owner.failed and not self._pins.get(id(owner), 0) and not owner.active:
                    if key in self._entries:
                        self._retire(key)

    def validate_retained_bytes(self, owner):
        with self._lock:
            entry = next(((o, size) for o, size in self._entries.values() if o is owner), None)
            if entry is None or owner.known_owned_tensor_bytes() > entry[1]:
                owner.failed = True
                raise RuntimeError('draft graph exceeded known-owned-tensor reservation')

    @contextmanager
    def pin(self, owner):
        with self._lock:
            if self.closed or owner.closed or owner.failed or not any(o is owner for o, _ in self._entries.values()):
                raise RuntimeError('draft owner is no longer retained')
            self._pins[id(owner)] = self._pins.get(id(owner), 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._pins[id(owner)] -= 1
                if not self._pins[id(owner)]:
                    del self._pins[id(owner)]
                if owner.failed and not self._pins.get(id(owner), 0) and not owner.active:
                    key = next((k for k, (o, _) in self._entries.items() if o is owner), None)
                    if key is not None:
                        self._retire(key)

    def detach_cache_owner(self, cache):
        with self._lock:
            for owner, _ in self._entries.values():
                if any(c is cache for c in owner.caches):
                    if owner.active or self._pins.get(id(owner), 0):
                        raise RuntimeError('cannot detach a leased draft cache owner')
                    owner.release_bindings()

    def shutdown(self):
        with self._lock:
            if self.closed:
                return
            if self._pins or any(o.active for o, _ in self._entries.values()):
                raise RuntimeError('cannot close draft owners with live leases')
            self.closed = True
            error = None
            for key in tuple(self._entries):
                try:
                    self._retire(key)
                except BaseException as exc:
                    error = error or exc
            if error is not None:
                raise error
