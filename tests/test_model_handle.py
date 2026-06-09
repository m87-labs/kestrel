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


class _StubStreaming:
    def __init__(self, model_name: str, tasks: tuple[str, ...]) -> None:
        self.model_name = model_name
        self.execution_shape = ExecutionShape.STREAMING
        self._tasks = tasks

    def tasks(self) -> tuple[str, ...]:
        return self._tasks


def _engine() -> InferenceEngine:
    """Minimal engine with an AR default + a single-pass model registered."""
    ar_skills = SkillRegistry([QuerySkill(), CaptionSkill(), SegmentSkill()])
    eng = object.__new__(InferenceEngine)
    eng._default_model = "ar-model"
    eng._model_ids = ["ar-model", "sp-model"]
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
    eng._shutdown = False
    # Runtimes are injected (already built), so the engine is effectively
    # started: _ensure_started() (now awaited by the handle verbs) is a no-op.
    eng._initialized = True
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
    # A capability verb on a single-pass model dispatches to its run() lane,
    # which names the task the model doesn't serve.
    with pytest.raises(ValueError, match="does not support 'query'"):
        asyncio.run(sp.query(question="hi?"))
    # Same task, via the generic escape hatch.
    with pytest.raises(ValueError, match="does not support 'query'"):
        asyncio.run(sp.run("query", {}))


def _capture_run_skill(captured: dict[str, Any]):
    """Return a stub ``_run_skill`` that records how the handle called it."""

    async def run_skill(task: str, *, image: Any, prompt: Any,
                        settings: Any, stream: Any) -> str:
        captured.update(
            task=task, image=image, prompt=prompt, settings=settings, stream=stream
        )
        return "RESULT"

    return run_skill


def test_default_ar_verb_routes_through_skill() -> None:
    """A default-AR capability passes its raw prompt to the model's skill via
    the engine's skill path — the handle adds no model knowledge. ``image`` is
    just another prompt key; the handle lifts it out for the engine's image
    path rather than privileging it in the signature."""
    eng = _engine()
    captured: dict[str, Any] = {}
    eng._run_skill = _capture_run_skill(captured)  # type: ignore[method-assign]
    h = eng.model("ar-model")  # the default AR model
    img = object()
    out = asyncio.run(h.query(image=img, question="what is this?", reasoning=True))
    assert out == "RESULT"
    assert captured["task"] == "query"
    assert captured["image"] is img  # lifted out of the prompt
    assert captured["prompt"] == {"question": "what is this?", "reasoning": True}


def test_text_only_query_through_handle() -> None:
    """A text-only query (no image) must work through the handle — image is
    optional and defaults to None."""
    eng = _engine()
    captured: dict[str, Any] = {}
    eng._run_skill = _capture_run_skill(captured)  # type: ignore[method-assign]
    h = eng.model("ar-model")
    out = asyncio.run(h.query(question="just text?"))  # no image
    assert out == "RESULT"
    assert captured["image"] is None
    assert captured["prompt"] == {"question": "just text?"}


def test_capability_lifts_settings_and_mirrors_stream() -> None:
    """settings (sampling) is lifted out of the model prompt as its own arg.
    stream is passed to the engine to select streaming delivery AND left in
    the prompt — the skill reads it to enable incremental token deltas
    (popping it would yield a stream object that never emits)."""
    eng = _engine()
    captured: dict[str, Any] = {}
    eng._run_skill = _capture_run_skill(captured)  # type: ignore[method-assign]
    h = eng.model("ar-model")
    asyncio.run(
        h.caption(length="short", stream=True, settings={"temperature": 0.5})
    )
    assert captured["task"] == "caption"
    assert captured["settings"] == {"temperature": 0.5}  # lifted out
    assert captured["stream"] is True  # selects streaming delivery
    # stream stays in the prompt so the skill builds a streaming request:
    assert captured["prompt"] == {"length": "short", "stream": True}


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

    async def fake_run_skill(*a: Any, **k: Any) -> str:  # pragma: no cover
        raise AssertionError("should not route a non-default AR model")

    eng._run_skill = fake_run_skill  # type: ignore[method-assign]
    h = eng.model("ar-other")
    with pytest.raises(NotImplementedError, match="not yet routable"):
        asyncio.run(h.query(question="hi?"))


def test_capability_verbs_are_uniform_kwargs() -> None:
    """Every capability verb is uniform — ``verb(**prompt)``. Inputs are
    model-defined (including the media payload — no privileged ``image``
    param), so the handle declares no per-task or per-modality params. This
    is the contract that replaced the old handle/engine signature drift
    guard: there is nothing to drift because the handle no longer mirrors
    the engine's typed verbs.
    """
    import inspect

    for verb in ("query", "caption", "detect", "point", "segment"):
        params = list(inspect.signature(getattr(ModelHandle, verb)).parameters.values())
        assert [p.name for p in params] == ["self", "prompt"], verb
        assert params[1].kind is inspect.Parameter.VAR_KEYWORD, (
            f"{verb}: the only parameter must be **prompt"
        )


def test_capability_dispatches_to_run_for_single_pass() -> None:
    """A single-pass model interprets the whole prompt in its forward pass:
    the handle forwards it verbatim (media payload included) to run()."""
    eng = _engine()
    eng._runtimes["sp-seg"] = _StubSinglePass("sp-seg", ("segment",))
    captured: dict[str, Any] = {}

    async def fake_run(model: str, task: str, inputs: Any) -> str:
        captured.update(model=model, task=task, inputs=inputs)
        return "MASKS"

    eng.run = fake_run  # type: ignore[method-assign]
    h = eng.model("sp-seg")
    img = object()
    out = asyncio.run(h.segment(image=img, points=[[1, 2]], labels=[1]))
    assert out == "MASKS"
    assert captured["model"] == "sp-seg"
    assert captured["task"] == "segment"
    assert captured["inputs"] == {"image": img, "points": [[1, 2]], "labels": [1]}


def _prestart_engine_building(captured: dict[str, Any]) -> InferenceEngine:
    """A not-yet-started engine whose (stubbed) startup builds a co-hosted
    single-pass runtime — mirrors eager ``models=[...]`` construction."""
    eng = object.__new__(InferenceEngine)
    eng._default_model = "ar"
    eng._model_ids = ["ar", "sp"]
    eng._runtimes = {}  # nothing built yet (pre-start)
    eng._initialized = False

    async def fake_ensure_started() -> None:
        eng._runtimes["sp"] = _StubSinglePass("sp", ("segment",))
        eng._initialized = True

    async def fake_run(model: str, task: str, inputs: Any) -> str:
        captured.update(model=model, task=task, inputs=inputs)
        return "OK"

    eng._ensure_started = fake_ensure_started  # type: ignore[method-assign]
    eng.run = fake_run  # type: ignore[method-assign]
    return eng


def test_handle_run_starts_engine_before_task_check() -> None:
    """A co-hosted model's run() must start the engine before checking tasks:
    the runtime (and thus its tasks) only exists post-build, so a pre-start
    call must trigger startup, not raise "unknown model". Regression for the
    eager-construction handle contract (Codex #81)."""
    captured: dict[str, Any] = {}
    eng = _prestart_engine_building(captured)
    out = asyncio.run(eng.model("sp").run("segment", {"points": [[1, 2]]}))
    assert out == "OK"
    assert captured == {
        "model": "sp",
        "task": "segment",
        "inputs": {"points": [[1, 2]]},
    }


def test_handle_capability_starts_engine_for_cohosted_single_pass() -> None:
    """The same pre-start guarantee for a capability verb: it must start the
    engine so shape dispatch sees the built single-pass runtime rather than
    misrouting to the autoregressive path."""
    captured: dict[str, Any] = {}
    eng = _prestart_engine_building(captured)
    out = asyncio.run(eng.model("sp").segment(points=[[3, 4]]))
    assert out == "OK"
    assert captured == {
        "model": "sp",
        "task": "segment",
        "inputs": {"points": [[3, 4]]},
    }


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


def test_stream_forwards_to_engine_for_streaming_model() -> None:
    eng = _engine()
    eng._runtimes["tracker"] = _StubStreaming("tracker", ("point",))
    captured: dict[str, Any] = {}

    async def fake_stream(model: str, task: str, inputs: Any) -> str:
        captured.update(model=model, task=task, inputs=inputs)
        return "STREAM"

    eng.stream = fake_stream  # type: ignore[method-assign]
    out = asyncio.run(
        eng.model("tracker").stream("point", points=[[0.5, 0.5]], labels=[1])
    )

    assert out == "STREAM"
    assert captured == {
        "model": "tracker",
        "task": "point",
        "inputs": {"points": [[0.5, 0.5]], "labels": [1]},
    }


def test_stream_rejects_non_streaming_model() -> None:
    eng = _engine()
    with pytest.raises(ValueError, match="stream\\(\\) is for streaming models"):
        asyncio.run(eng.model("sp-model").stream("segment_masks", points=[[1, 2]]))


def test_engine_stream_validates_shape_and_task() -> None:
    async def go() -> None:
        eng = _engine()
        eng._runtimes["tracker"] = _StubStreaming("tracker", ("point",))

        with pytest.raises(ValueError, match="not a streaming model"):
            await eng.stream("sp-model", "segment_masks", {})

        with pytest.raises(ValueError, match="does not support 'detect'"):
            await eng.stream("tracker", "detect", {})

        with pytest.raises(NotImplementedError, match="streaming executor"):
            await eng.stream("tracker", "point", {"points": [[0.5, 0.5]]})

    asyncio.run(go())
