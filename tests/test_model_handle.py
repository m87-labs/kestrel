"""ModelHandle: per-model view, capability gating, verb forwarding.

ModelHandle is one generic class for every model; capability
availability is a runtime check (tasks()/supports), and unsupported
calls raise rather than silently doing the wrong thing. These tests pin
that contract with stub runtimes — no GPU.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from kestrel.engine import InferenceEngine, ModelHandle
from kestrel.runtime import ExecutionShape
from kestrel.skills import SkillRegistry, QuerySkill, CaptionSkill, SegmentSkill


class _StubSinglePass:
    def __init__(self, model_name: str, tasks: tuple[str, ...]) -> None:
        self.model_name = model_name
        self.execution_shape = ExecutionShape.SINGLE_PASS
        self._tasks = tasks

    def tasks(self) -> tuple[str, ...]:
        return self._tasks


def _engine() -> InferenceEngine:
    """Minimal engine with an AR default + a single-pass model registered."""
    eng = object.__new__(InferenceEngine)
    eng._default_model = "ar-model"
    eng._runtimes = {
        "ar-model": SimpleNamespace(
            model_name="ar-model", execution_shape=ExecutionShape.AUTOREGRESSIVE
        ),
        "sp-model": _StubSinglePass("sp-model", ("segment_masks",)),
    }
    eng._skills = SkillRegistry([QuerySkill(), CaptionSkill(), SegmentSkill()])
    return eng


def test_model_returns_handle_bound_to_id() -> None:
    eng = _engine()
    h = eng.model("sp-model")
    assert isinstance(h, ModelHandle)
    assert h.model_id == "sp-model"


def test_model_defaults_to_default_model() -> None:
    eng = _engine()
    assert eng.model().model_id == "ar-model"


def test_model_rejects_unknown_id() -> None:
    eng = _engine()
    with pytest.raises(ValueError, match="Unknown model 'nope'.*configured"):
        eng.model("nope")


def test_ar_handle_advertises_skill_vocabulary() -> None:
    eng = _engine()
    h = eng.model("ar-model")
    assert h.tasks == ("query", "caption", "segment")
    assert h.supports("query") is True
    assert h.supports("segment_masks") is False


def test_single_pass_handle_advertises_runtime_tasks() -> None:
    eng = _engine()
    h = eng.model("sp-model")
    assert h.tasks == ("segment_masks",)
    assert h.supports("segment_masks") is True
    assert h.supports("query") is False


def test_unsupported_capability_raises_clearly() -> None:
    eng = _engine()
    sp = eng.model("sp-model")
    # query is an AR skill; the single-pass model doesn't serve it.
    with pytest.raises(ValueError, match="does not support 'query'"):
        asyncio.run(sp.query(None, "hi?"))


def test_default_ar_verb_forwards_to_engine() -> None:
    eng = _engine()
    captured: dict[str, Any] = {}

    async def fake_query(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "RESULT"

    eng.query = fake_query  # type: ignore[method-assign]
    h = eng.model("ar-model")  # the default AR model
    out = asyncio.run(h.query(None, "what is this?", reasoning=True))
    assert out == "RESULT"
    assert captured["question"] == "what is this?"
    assert captured["reasoning"] is True


def test_ar_verb_on_non_default_model_fails_loud() -> None:
    """The AR verbs don't route by model id yet, so a non-default AR
    handle must refuse rather than silently hit the default model."""
    eng = _engine()
    # Register a second AR model; the verbs can't route to it.
    eng._runtimes["ar-other"] = SimpleNamespace(
        model_name="ar-other", execution_shape=ExecutionShape.AUTOREGRESSIVE
    )

    async def fake_query(**kwargs: Any) -> str:  # pragma: no cover - must not run
        raise AssertionError("should not forward for a non-default AR model")

    eng.query = fake_query  # type: ignore[method-assign]
    h = eng.model("ar-other")
    with pytest.raises(NotImplementedError, match="not yet routable"):
        asyncio.run(h.query(None, "hi?"))


def test_handle_verb_defaults_match_engine() -> None:
    """A handle verb is a thin forwarder; its keyword defaults must match
    InferenceEngine's, or the same call drifts behavior through the handle
    (e.g. query's reasoning default). Introspect both, compare shared
    keyword params — guards all verbs, not just the one Codex caught."""
    import inspect

    for verb in ("query", "caption", "detect", "point", "segment"):
        eng_params = inspect.signature(getattr(InferenceEngine, verb)).parameters
        h_params = inspect.signature(getattr(ModelHandle, verb)).parameters
        for name, h_p in h_params.items():
            if name == "self" or h_p.default is inspect.Parameter.empty:
                continue
            eng_p = eng_params.get(name)
            assert eng_p is not None, f"{verb}: handle has param {name!r} engine lacks"
            assert h_p.default == eng_p.default, (
                f"{verb}: default for {name!r} drifted — "
                f"handle={h_p.default!r} engine={eng_p.default!r}"
            )


def test_run_on_ar_model_rejected_with_clear_message() -> None:
    """run() is the single-pass escape hatch; on an AR model it should say
    so, not die downstream in engine.run()."""
    eng = _engine()
    h = eng.model("ar-model")
    with pytest.raises(ValueError, match="run\\(\\) is for single-pass"):
        asyncio.run(h.run("query", {}))


def test_run_gates_on_task_then_forwards() -> None:
    eng = _engine()
    captured: dict[str, Any] = {}

    async def fake_run(model: str, task: str, inputs: Any) -> str:
        captured.update(model=model, task=task, inputs=inputs)
        return "MASKS"

    eng.run = fake_run  # type: ignore[method-assign]
    h = eng.model("sp-model")

    out = asyncio.run(h.run("segment_masks", {"points": [[1, 2]]}))
    assert out == "MASKS"
    assert captured == {
        "model": "sp-model",
        "task": "segment_masks",
        "inputs": {"points": [[1, 2]]},
    }

    # An unknown task is rejected before forwarding.
    with pytest.raises(ValueError, match="does not support 'bogus'"):
        asyncio.run(h.run("bogus", {}))
