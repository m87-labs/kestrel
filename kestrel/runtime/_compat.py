"""Central compatibility boundary for externally injected AR runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kestrel.runtime.sampling import SamplingHooks


CURRENT_RUNTIME_API_VERSION = 2


@dataclass(frozen=True)
class RuntimeContract:
    """Capabilities used by engine/scheduler code after one version check."""

    sampling_hooks: SamplingHooks
    eos_token_ids: frozenset[int]


def resolve_runtime_contract(runtime: Any) -> RuntimeContract:
    """Resolve the current contract or the pre-extension injected-runtime shim.

    Runtime API v1 predates runtime-level EOS ids and explicit sampling hooks.
    It remains accepted for text-only injected runtimes. All in-tree runtimes
    advertise v2 and must implement the uniform surface directly; missing v2
    members are errors rather than per-call fallbacks.
    """

    version = int(getattr(runtime, "runtime_api_version", 1))
    if version == 1:
        return RuntimeContract(
            sampling_hooks=SamplingHooks(),
            eos_token_ids=frozenset({int(runtime.prompt_template.eos_id)}),
        )
    if version != CURRENT_RUNTIME_API_VERSION:
        raise ValueError(f"Unsupported autoregressive runtime API version {version}")

    hooks = runtime.sampling_hooks
    if not isinstance(hooks, SamplingHooks):
        raise TypeError("runtime.sampling_hooks must be SamplingHooks")
    eos_token_ids = frozenset(map(int, runtime.eos_token_ids))
    if not eos_token_ids:
        raise ValueError("runtime.eos_token_ids must not be empty")
    return RuntimeContract(
        sampling_hooks=hooks,
        eos_token_ids=eos_token_ids,
    )


__all__ = ["CURRENT_RUNTIME_API_VERSION", "RuntimeContract", "resolve_runtime_contract"]
