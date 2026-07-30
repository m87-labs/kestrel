"""Dataflow helpers for bounded sibling projections."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from kestrel.runtime.compilation import _scalar_buffer_key


class BoundedProjection(Protocol):
    input_min: torch.Tensor
    input_max: torch.Tensor

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def forward_bounded_input(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class PackedProjectionBindingStats:
    groups: int
    source_parameter_bytes: int
    packed_parameter_bytes: int
    source_bound_bytes: int
    packed_bound_bytes: int


class PackedBoundedProjections(nn.Module):
    """One packed linear with independently bounded output views."""

    def __init__(
        self,
        in_features: int,
        out_features: Sequence[int],
        *,
        source_names: Sequence[str],
        source_weight_leaf: str = "linear.weight",
        use_bounds: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = tuple(int(size) for size in out_features)
        self.source_names = tuple(source_names)
        self.source_weight_leaf = source_weight_leaf
        self.use_bounds = bool(use_bounds)
        if not self.out_features or any(size <= 0 for size in self.out_features):
            raise ValueError("packed projection output sizes must be positive")
        if len(self.source_names) != len(self.out_features):
            raise ValueError(
                "packed projection source names and output sizes must align"
            )
        if not self.source_weight_leaf:
            raise ValueError("packed projection source weight leaf cannot be empty")

        total_out_features = sum(self.out_features)
        self.linear = nn.Linear(
            self.in_features,
            total_out_features,
            bias=False,
        )
        if self.use_bounds:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer(
                "output_min",
                torch.full((total_out_features,), -float("inf")),
            )
            self.register_buffer(
                "output_max",
                torch.full((total_out_features,), float("inf")),
            )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.use_bounds:
            hidden_states = torch.clamp(
                hidden_states,
                self.input_min,
                self.input_max,
            )
        packed = self.linear(hidden_states)
        if self.use_bounds:
            packed = torch.clamp(packed, self.output_min, self.output_max)
        return packed.split(self.out_features, dim=-1)


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def bind_declared_packed_bounded_projections(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> PackedProjectionBindingStats:
    """Pack declared sibling weights after proving compatible transforms."""

    groups = 0
    source_parameter_bytes = 0
    packed_parameter_bytes = 0
    source_bound_bytes = 0
    packed_bound_bytes = 0
    for module_name, child in module.named_modules():
        if not isinstance(child, PackedBoundedProjections):
            continue
        parent_name, separator, _ = module_name.rpartition(".")
        source_prefix = f"{parent_name}." if separator else ""
        target_prefix = f"{module_name}." if module_name else ""
        target_weight_key = target_prefix + "linear.weight"
        if target_weight_key in state_dict:
            continue

        source_weight_keys = [
            f"{source_prefix}{name}.{child.source_weight_leaf}"
            for name in child.source_names
        ]
        missing = [key for key in source_weight_keys if key not in state_dict]
        if missing:
            raise KeyError(
                f"packed projection {module_name!r} is missing source weights: "
                + ", ".join(repr(key) for key in missing)
            )
        weights = [state_dict[key] for key in source_weight_keys]
        first_weight = weights[0]
        for key, weight, output_size in zip(
            source_weight_keys,
            weights,
            child.out_features,
        ):
            expected_shape = (output_size, child.in_features)
            if (
                weight.layout is not torch.strided
                or not weight.is_contiguous()
                or tuple(weight.shape) != expected_shape
                or weight.dtype != first_weight.dtype
                or weight.device != first_weight.device
            ):
                raise ValueError(
                    f"cannot pack projection weight {key!r}: expected contiguous "
                    f"{expected_shape} {first_weight.dtype} on {first_weight.device}, "
                    f"got {tuple(weight.shape)} {weight.dtype} {weight.device} "
                    f"layout={weight.layout} contiguous={weight.is_contiguous()}"
                )

        packed_bounds: dict[str, torch.Tensor] = {}
        source_bound_keys: list[str] = []
        bound_values: dict[str, list[torch.Tensor]] = {}
        if child.use_bounds:
            for bound_name in (
                "input_min",
                "input_max",
                "output_min",
                "output_max",
            ):
                keys = [
                    f"{source_prefix}{name}.{bound_name}"
                    for name in child.source_names
                ]
                missing = [key for key in keys if key not in state_dict]
                if missing:
                    raise KeyError(
                        f"packed projection {module_name!r} is missing bounds: "
                        + ", ".join(repr(key) for key in missing)
                    )
                values = [state_dict[key] for key in keys]
                if any(
                    value.ndim != 0
                    or value.dtype != values[0].dtype
                    or value.device != values[0].device
                    for value in values
                ):
                    raise ValueError(
                        f"packed projection {module_name!r} requires scalar "
                        f"{bound_name} values with one dtype and device"
                    )
                bound_values[bound_name] = values
                source_bound_keys.extend(keys)

            for bound_name in ("input_min", "input_max"):
                values = bound_values[bound_name]
                first_key = _scalar_buffer_key(values[0])
                if any(
                    _scalar_buffer_key(value) != first_key
                    for value in values[1:]
                ):
                    raise ValueError(
                        f"cannot pack projection {module_name!r} with different "
                        f"{bound_name} values"
                    )
                packed_bounds[bound_name] = values[0]
            for bound_name in ("output_min", "output_max"):
                packed_bounds[bound_name] = torch.cat(
                    [
                        value.expand(output_size)
                        for value, output_size in zip(
                            bound_values[bound_name],
                            child.out_features,
                        )
                    ]
                )

        packed_weight = torch.cat(weights, dim=0)
        state_dict[target_weight_key] = packed_weight
        for bound_name, value in packed_bounds.items():
            state_dict[target_prefix + bound_name] = value
        for key in source_weight_keys:
            state_dict.pop(key)
        for key in source_bound_keys:
            state_dict.pop(key)

        groups += 1
        source_parameter_bytes += sum(_tensor_bytes(weight) for weight in weights)
        packed_parameter_bytes += _tensor_bytes(packed_weight)
        source_bound_bytes += sum(
            _tensor_bytes(value)
            for values in bound_values.values()
            for value in values
        )
        packed_bound_bytes += sum(
            _tensor_bytes(value)
            for value in packed_bounds.values()
        )

    return PackedProjectionBindingStats(
        groups=groups,
        source_parameter_bytes=source_parameter_bytes,
        packed_parameter_bytes=packed_parameter_bytes,
        source_bound_bytes=source_bound_bytes,
        packed_bound_bytes=packed_bound_bytes,
    )


def apply_sibling_bounded_projections(
    hidden_states: torch.Tensor,
    projections: Sequence[BoundedProjection],
) -> tuple[torch.Tensor, ...]:
    """Apply sibling projections with one clamp when their input bounds alias.

    Immutable scalar canonicalization makes bit-identical bounds share their
    Tensor object. Identity therefore proves that the input transform is the
    same without a device comparison or model-specific policy. Unprepared or
    heterogeneous projection groups retain their independent exact paths.
    """

    if not projections:
        return ()
    first = projections[0]
    lower = getattr(first, "input_min", None)
    upper = getattr(first, "input_max", None)
    shares_bounds = (
        lower is not None
        and upper is not None
        and all(
            getattr(projection, "input_min", None) is lower
            and getattr(projection, "input_max", None) is upper
            for projection in projections[1:]
        )
    )
    if not shares_bounds:
        return tuple(projection(hidden_states) for projection in projections)

    bounded = torch.clamp(hidden_states, lower, upper)
    return tuple(
        projection.forward_bounded_input(bounded)
        for projection in projections
    )
