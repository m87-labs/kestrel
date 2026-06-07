"""Multi-model construction: Engine.create(models=[...]) builds a runtime
per configured model from its registered spec.

The engine hosts a default autoregressive model plus any co-hosted models
(e.g. single-pass). These tests pin the construction wiring on CPU with stub
factories — no GPU, no real weights:
  - the ``models=`` list is tracked (default first, deduped) and exposed
    pre-start via ``model()`` / ``_configured_models``;
  - ``_build_configured_runtimes`` builds every model from its spec,
    including a *tokenizer-free* single-pass spec (ModelSpec generalization);
  - ``max_lora_rank`` is forwarded only to the default AR model.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.models.registry import ModelSpec, register, _REGISTRY
from kestrel.runtime import ExecutionShape

from tests.scheduler._fake_runtime import FakeRuntime


class _SPStub:
    """Minimal single-pass runtime built from a spec factory."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = torch.device("cpu")
        self.execution_shape = ExecutionShape.SINGLE_PASS

    def tasks(self) -> tuple[str, ...]:
        return ("segment",)

    def forward(self, task: str, inputs: Any) -> Any:  # pragma: no cover - unused here
        return {"task": task, "inputs": inputs}

    def shutdown(self) -> None:
        pass


# model_path set so RuntimeConfig.__post_init__ skips weight resolution for
# these synthetic model names (it only downloads when model_path is None).
def _cfg(model: str) -> RuntimeConfig:
    return RuntimeConfig(model=model, device="cpu", model_path="/unused")


def test_models_none_is_single_default() -> None:
    """Omitting models= keeps today's single-model behavior."""
    eng = InferenceEngine(_cfg("a"))
    assert eng._model_ids == ["a"]
    assert eng._configured_models() == {"a"}


def test_models_arg_tracks_cohosted_ids() -> None:
    """models= is recorded default-first and deduplicated; co-hosted ids are
    addressable via model() before startup, and unknown ids are rejected."""
    eng = InferenceEngine(_cfg("a"), models=["b", "a", "c"])
    assert eng._model_ids == ["a", "b", "c"]
    assert eng._configured_models() == {"a", "b", "c"}
    assert eng.model("b").model_id == "b"  # handle for a co-hosted model pre-start
    with pytest.raises(ValueError, match="Unknown model 'z'"):
        eng.model("z")


def test_build_configured_runtimes_builds_all_from_specs() -> None:
    """Every configured model is built from its spec — including a
    tokenizer-free single-pass spec — with a per-model config, and
    max_lora_rank goes only to the default AR model."""
    built_kwargs: dict[str, dict[str, Any]] = {}

    def ar_factory(cfg: RuntimeConfig, **kwargs: Any) -> FakeRuntime:
        built_kwargs[cfg.model] = dict(kwargs)
        return FakeRuntime(model_name=cfg.model)

    def sp_factory(cfg: RuntimeConfig, **kwargs: Any) -> _SPStub:
        built_kwargs[cfg.model] = dict(kwargs)
        return _SPStub(cfg.model)

    # Tokenizer-free specs: no tokenizer_id / checkpoint_format / filename /
    # repo_id — proving the ModelSpec generalization (those are now optional).
    register(ModelSpec(name="mm-ar", runtime=ar_factory))
    register(ModelSpec(name="mm-sp", runtime=sp_factory))
    try:
        eng = object.__new__(InferenceEngine)
        eng._runtime_cfg = RuntimeConfig(model="mm-ar", device="cpu")
        eng._default_model = "mm-ar"
        eng._model_ids = ["mm-ar", "mm-sp"]
        eng._runtimes = {}
        eng._compute_stream = None

        eng._build_configured_runtimes(13)

        assert set(eng._runtimes) == {"mm-ar", "mm-sp"}
        assert eng._runtimes["mm-ar"].execution_shape is ExecutionShape.AUTOREGRESSIVE
        assert eng._runtimes["mm-sp"].execution_shape is ExecutionShape.SINGLE_PASS
        # Each runtime is built with a per-model config carrying its own id.
        assert eng._runtimes["mm-sp"].model_name == "mm-sp"
        # max_lora_rank → default AR only; compute_stream is supplied to
        # every runtime factory so both execution shapes share the engine
        # stream. CPU tests use None for that stream.
        assert built_kwargs["mm-ar"] == {"max_lora_rank": 13, "compute_stream": None}
        assert built_kwargs["mm-sp"] == {"compute_stream": None}
    finally:
        _REGISTRY.pop("mm-ar", None)
        _REGISTRY.pop("mm-sp", None)


def test_build_configured_runtimes_skips_already_built() -> None:
    """An already-present runtime (e.g. an injected default) is not rebuilt."""
    calls: list[str] = []

    def sp_factory(cfg: RuntimeConfig, **kwargs: Any) -> _SPStub:
        calls.append(cfg.model)
        return _SPStub(cfg.model)

    register(ModelSpec(name="mm-sp2", runtime=sp_factory))
    try:
        eng = object.__new__(InferenceEngine)
        eng._runtime_cfg = _cfg("preexisting")  # injected default; not registered
        eng._default_model = "preexisting"
        eng._model_ids = ["preexisting", "mm-sp2"]
        injected = _SPStub("preexisting")
        eng._runtimes = {"preexisting": injected}
        eng._compute_stream = None

        eng._build_configured_runtimes(None)

        assert eng._runtimes["preexisting"] is injected  # not rebuilt
        assert calls == ["mm-sp2"]  # only the missing one was built
    finally:
        _REGISTRY.pop("mm-sp2", None)
