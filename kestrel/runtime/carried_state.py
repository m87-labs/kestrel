"""Physical carried-state reconciliation across execution paths."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class StatePhysicalForm:
    representation: str
    storage_axis_order: tuple[str, ...] = ()
    storage_dtype: str | None = None

    def __post_init__(self) -> None:
        if not self.representation:
            raise ValueError("state representation must be non-empty")
        if len(self.storage_axis_order) != len(set(self.storage_axis_order)):
            raise ValueError("state storage axes must be unique")
        if self.storage_axis_order and not self.storage_dtype:
            raise ValueError("state storage axes require a dtype")


@dataclass(frozen=True)
class StateRepresentationRequirement:
    buffer: str
    representation: str
    storage_axis_order: tuple[str, ...] = ()
    storage_dtype: str | None = None

    @property
    def physical_form(self) -> StatePhysicalForm:
        return StatePhysicalForm(
            self.representation, self.storage_axis_order, self.storage_dtype)


Transition = Callable[[StatePhysicalForm, StatePhysicalForm, tuple[int, ...]], None]


class CarriedStateCoordinator:
    """Convert only rows whose authoritative physical form changes."""

    def __init__(
        self,
        *,
        buffers: Iterable[str],
        rows: Iterable[int],
        transitions: Mapping[str, Transition],
    ) -> None:
        self._buffers = tuple(dict.fromkeys(map(str, buffers)))
        self._transitions = dict(transitions)
        unknown = set(self._transitions) - set(self._buffers)
        if unknown:
            raise ValueError(f"state transitions name unknown buffers {sorted(unknown)}")
        self._authoritative = {
            (buffer, int(row)): None
            for buffer in self._buffers
            for row in rows
        }

    def mark_coherent(self, rows: Iterable[int]) -> None:
        for row in map(int, rows):
            for buffer in self._buffers:
                self._authoritative[(buffer, row)] = None

    def prepare(
        self,
        requirements: Iterable[StateRepresentationRequirement],
        rows: Iterable[int],
    ) -> None:
        rows = tuple(dict.fromkeys(map(int, rows)))
        requirements = tuple(requirements)
        buffers = [item.buffer for item in requirements]
        if len(buffers) != len(set(buffers)):
            raise ValueError("an execution path repeats a state buffer")
        unknown = set(buffers) - set(self._buffers)
        if unknown:
            raise ValueError(f"execution path names unknown buffers {sorted(unknown)}")
        for requirement in requirements:
            target = requirement.physical_form
            grouped = defaultdict(list)
            for row in rows:
                source = self._authoritative[(requirement.buffer, row)]
                if source is not None and source != target:
                    grouped[source].append(row)
            for source, selected in grouped.items():
                try:
                    transition = self._transitions[requirement.buffer]
                except KeyError as exc:
                    raise RuntimeError(
                        f"state buffer {requirement.buffer!r} cannot transition") from exc
                transition(source, target, tuple(selected))
            for row in rows:
                self._authoritative[(requirement.buffer, row)] = target

    def physical_form(self, buffer: str, row: int) -> StatePhysicalForm | None:
        return self._authoritative[(str(buffer), int(row))]

    def representation(self, buffer: str, row: int) -> str | None:
        form = self.physical_form(buffer, row)
        return None if form is None else form.representation


__all__ = [
    "CarriedStateCoordinator",
    "StatePhysicalForm",
    "StateRepresentationRequirement",
]
