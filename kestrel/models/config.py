"""Strict helpers for model architecture descriptors."""

from dataclasses import MISSING, fields
from typing import Any, Mapping


def required_config(
    data: Mapping[str, Any],
    name: str,
    scope: str,
) -> Any:
    if name not in data:
        raise ValueError(f"{scope} config is missing required field {name!r}")
    return data[name]


def required_config_kwargs(
    cls: type,
    data: Mapping[str, Any],
    *,
    scope: str,
    transformed: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    required = tuple(
        field.name
        for field in fields(cls)
        if field.name not in transformed and field.default is MISSING
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"{scope} config is missing required fields: {missing}")
    return {name: data[name] for name in required}


__all__ = ["required_config", "required_config_kwargs"]
