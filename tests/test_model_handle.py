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
from kestrel.skills import SkillRegistry
from kestrel.models.moondream.skills import QuerySkill, CaptionSkill, SegmentSkill


class _StubSinglePass:
    def __init__(self, model_name: str, tasks: tuple[str, ...]) -> None:
        self.model_name = model_name
        self.execution_shape = ExecutionShape.SINGLE_PASS
        self._tasks = tasks

    def tasks(self) -> tuple[str, ...]:
        return self._tasks


def _engine() -> InferenceEngine:
    """Minimal engine with an AR default + a single-pass model registered."""
    ar_skills = SkillRegistry([QuerySkill(), CaptionSkill(), SegmentSkill()])
    eng = object.__new__(InferenceEngine)
    eng._default_model = "ar-model"
    eng._runtimes = {
        "ar-model": SimpleNamespace(
            model_name="ar-model",
            execution_shape=ExecutionShape.AUTOREGRESSIVE,
            skills=lambda: ar_skills,
            tasks=lambda: ar_skills.names(),
        ),
        "sp-model": _StubSinglePass("sp-model", ("segment_masks",)),
    }
    eng._skills_override = None  # AR runtime owns its skills
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


def test_default_handle_tasks_honor_skill_override() -> None:
    """The default model reports the registry that actually executes its
    verbs. With a ``_skills_override`` set, ``tasks``/``supports`` must
    reflect the override — not the built runtime's own skills — so the
    handle never hides or rejects a skill the engine would run."""
    eng = _engine()
    # Override the default model's skills; the built runtime still advertises
    # its own ("query", "caption", "segment") via its skills= lambda.
    eng._skills_override = SkillRegistry([QuerySkill()])
    h = eng.model("ar-model")
    assert h.tasks == ("query",)
    assert h.supports("query") is True
    assert h.supports("caption") is False


def test_unsupported_capability_raises_clearly() -> None:
    eng = _engine()
    sp = eng.model("sp-model")
    # An AR verb on a non-default model fails loud (not routable).
    with pytest.raises(NotImplementedError, match="not yet routable"):
        asyncio.run(sp.query(None, "hi?"))
    # A task the model doesn't serve, via the generic escape hatch, names it.
    with pytest.raises(ValueError, match="does not support 'query'"):
        asyncio.run(sp.run("query", {}))


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


def test_text_only_query_through_handle() -> None:
    """A text-only query (no image) must work through the handle, matching
    engine.query's optional image — regression for image being required."""
    eng = _engine()
    captured: dict[str, Any] = {}

    async def fake_query(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "RESULT"

    eng.query = fake_query  # type: ignore[method-assign]
    h = eng.model("ar-model")
    out = asyncio.run(h.query(question="just text?"))  # no image
    assert out == "RESULT"
    assert captured["image"] is None
    assert captured["question"] == "just text?"


def test_ar_verb_on_non_default_model_fails_loud() -> None:
    """The AR verbs don't route by model id yet, so a non-default AR
    handle must refuse rather than silently hit the default model."""
    eng = _engine()
    # Register a second AR model; the verbs can't route to it.
    ar_skills = SkillRegistry([QuerySkill(), CaptionSkill(), SegmentSkill()])
    eng._runtimes["ar-other"] = SimpleNamespace(
        model_name="ar-other",
        execution_shape=ExecutionShape.AUTOREGRESSIVE,
        skills=lambda: ar_skills,
        tasks=lambda: ar_skills.names(),
    )

    async def fake_query(**kwargs: Any) -> str:  # pragma: no cover - must not run
        raise AssertionError("should not forward for a non-default AR model")

    eng.query = fake_query  # type: ignore[method-assign]
    h = eng.model("ar-other")
    with pytest.raises(NotImplementedError, match="not yet routable"):
        asyncio.run(h.query(None, "hi?"))


def test_handle_verb_signatures_match_engine() -> None:
    """A handle verb is a thin forwarder; its signature must match
    InferenceEngine's, or binding a model silently breaks or changes a
    valid call. The ways it can drift, each seen in review:
      - default drift changes outcomes (query reasoning),
      - an engine-optional param made required breaks calls (text-only
        query), and
      - reordering / making positional params keyword-only breaks
        positional calls (query(img, q, reasoning, refs); detect settings).
    So compare the full ordered parameter list — (name, kind, default) —
    for every param the handle declares beyond self. Guards all verbs.
    """
    import inspect

    for verb in ("query", "caption", "detect", "point", "segment"):
        eng = inspect.signature(getattr(InferenceEngine, verb)).parameters
        hnd = inspect.signature(getattr(ModelHandle, verb)).parameters
        eng_list = [(n, p.kind, p.default) for n, p in eng.items() if n != "self"]
        hnd_list = [(n, p.kind, p.default) for n, p in hnd.items() if n != "self"]
        # The handle forwards exactly the engine verb's parameters, so the
        # ordered (name, kind, default) lists must be identical.
        assert hnd_list == eng_list, (
            f"{verb}: handle signature drifted from engine.\n"
            f"  handle: {hnd_list}\n  engine: {eng_list}"
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
