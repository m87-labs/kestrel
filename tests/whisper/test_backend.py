from __future__ import annotations

from types import SimpleNamespace

import pytest

from kestrel.models.whisper import register_backend
from kestrel.models.whisper import runtime_abi


def test_backend_registration_is_explicit_and_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(runtime_abi, "_BACKEND_PROVIDER", None)
    backend = SimpleNamespace(
        create_prefill=lambda _bindings: None,
        create_decode=lambda _bindings: None,
        native_provenance=lambda _bindings, _session: {},
    )

    def provider():
        return backend

    register_backend(provider)
    register_backend(provider)

    assert runtime_abi.create_backend() is backend


def test_backend_registration_rejects_implicit_or_conflicting_providers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runtime_abi, "_BACKEND_PROVIDER", None)
    with pytest.raises(RuntimeError, match="No optimized Whisper backend"):
        runtime_abi.create_backend()
    with pytest.raises(TypeError, match="must be callable"):
        register_backend(None)  # type: ignore[arg-type]

    def first():
        return object()

    register_backend(first)
    with pytest.raises(RuntimeError, match="different Whisper backend"):
        register_backend(lambda: object())


def test_backend_provider_must_return_the_complete_contract(monkeypatch) -> None:
    monkeypatch.setattr(runtime_abi, "_BACKEND_PROVIDER", lambda: object())

    with pytest.raises(TypeError, match="required contract"):
        runtime_abi.create_backend()
