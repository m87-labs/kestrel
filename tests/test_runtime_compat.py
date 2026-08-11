from __future__ import annotations

from types import SimpleNamespace

import pytest

from kestrel.runtime._compat import resolve_runtime_contract
from kestrel.runtime.sampling import SamplingHooks


def test_v1_runtime_uses_text_template_eos_and_default_hooks() -> None:
    runtime = SimpleNamespace(prompt_template=SimpleNamespace(eos_id=7))

    contract = resolve_runtime_contract(runtime)

    assert contract.eos_token_ids == frozenset({7})
    assert contract.sampling_hooks == SamplingHooks()


def test_v2_runtime_owns_plural_eos_ids() -> None:
    hooks = SamplingHooks()
    runtime = SimpleNamespace(
        runtime_api_version=2,
        sampling_hooks=hooks,
        eos_token_ids=(7, 9),
    )

    contract = resolve_runtime_contract(runtime)

    assert contract.sampling_hooks is hooks
    assert contract.eos_token_ids == frozenset({7, 9})


def test_v2_runtime_requires_nonempty_eos_ids() -> None:
    runtime = SimpleNamespace(
        runtime_api_version=2,
        sampling_hooks=SamplingHooks(),
        eos_token_ids=(),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        resolve_runtime_contract(runtime)


def test_unknown_runtime_api_version_is_rejected() -> None:
    runtime = SimpleNamespace(runtime_api_version=3)

    with pytest.raises(ValueError, match="Unsupported"):
        resolve_runtime_contract(runtime)
