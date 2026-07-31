"""Qwen 3.5 query state and termination behavior."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from kestrel.skills import QueryRequest, QuerySkillState
from kestrel.runtime.tokens import TextToken
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.types import (
    GenerationRequest,
    RequestLifecycle,
    RequestPhase,
)
from kestrel.models.qwen35.prompt_template import (
    END_OF_TEXT_ID,
    IM_END_ID,
    Qwen35PromptTemplate,
)
from kestrel.models.qwen35.skills import build_skill_registry


_ANSWER_ID = 42
_NEWLINE_ID = 198


class _Tokenizer:
    def decode(self, token_ids) -> str:
        pieces = {_ANSWER_ID: "answer", _NEWLINE_ID: "\n"}
        return "".join(pieces.get(token_id, "") for token_id in token_ids)


def _runtime():
    return SimpleNamespace(
        model_name="Qwen/Qwen3.5-4B",
        prompt_template=Qwen35PromptTemplate(),
        spec=object(),
        tokenizer=_Tokenizer(),
    )


def _request(*, stream_callback=None, return_logprobs: bool = False):
    skill = build_skill_registry().resolve("query")
    context = QueryRequest(
        question="question",
        image=None,
        reasoning=False,
        stream=stream_callback is not None,
    )
    request = GenerationRequest(
        request_id=1,
        prompt="question",
        prompt_tokens=[],
        max_new_tokens=16,
        skill=skill,
        request_context=context,
        stream_callback=stream_callback,
        return_logprobs=return_logprobs,
    )
    state = skill.create_state(_runtime(), request, context)
    request.skill_state = state
    return request, state


def _lifecycle(*, stream_callback=None, return_logprobs: bool = False):
    request, state = _request(
        stream_callback=stream_callback,
        return_logprobs=return_logprobs,
    )
    lifecycle = RequestLifecycle(
        request=request,
        skill_state=state,
        sequence_state=SimpleNamespace(
            advance=lambda: None,
            max_length=128,
            prompt_length=0,
            reused_page_count=0,
        ),
        phase=RequestPhase.RUNNING,
    )
    request.lifecycle = lifecycle
    return lifecycle


def _scheduler(runtime):
    scheduler = GenerationScheduler.__new__(GenerationScheduler)
    scheduler.runtime = runtime
    scheduler._completed = deque()
    return scheduler


def test_query_skill_uses_template_stop_id() -> None:
    _, state = _request()
    assert isinstance(state, QuerySkillState)
    assert state.mask_is_stateful is True
    assert tuple(state.stop_token_ids(_runtime())) == (IM_END_ID,)


@pytest.mark.parametrize("stop_id", [IM_END_ID, END_OF_TEXT_ID])
def test_non_spec_query_stops_without_streaming_terminator(stop_id: int) -> None:
    streamed = []
    runtime = _runtime()
    lifecycle = _lifecycle(stream_callback=streamed.append)
    scheduler = _scheduler(runtime)

    for token_id in [_ANSWER_ID, stop_id, _NEWLINE_ID]:
        lifecycle.stage_token(runtime, TextToken(token_id=token_id))
        if scheduler._mark_finished_if_needed(lifecycle):
            break

    result = scheduler._completed.pop()
    assert result.finish_reason == "stop"
    assert result.output["answer"] == "answer"
    assert [token.token_id for token in result.tokens] == [_ANSWER_ID, stop_id]
    assert "".join(update.text for update in streamed) == "answer"


def test_spec_query_truncates_committed_run_at_im_end() -> None:
    runtime = _runtime()
    lifecycle = _lifecycle(return_logprobs=True)
    lifecycle.inflight_refs = 2
    scheduler = _scheduler(runtime)
    scheduler.running = [lifecycle]
    token_ids = [_ANSWER_ID, IM_END_ID, _NEWLINE_ID, END_OF_TEXT_ID]
    logprobs = [-0.1, -0.2, -0.3, -0.4]
    scheduler._materialize_spec_tokens = lambda *args: (
        [[TextToken(token_id=token_id) for token_id in token_ids]],
        [logprobs],
    )
    result = SimpleNamespace(
        tokens=[token_ids],
        logprobs=[logprobs],
        side_values=None,
    )

    scheduler._commit_spec(([lifecycle], result))

    completed = scheduler._completed.pop()
    assert completed.finish_reason == "stop"
    assert completed.output["answer"] == "answer"
    assert [token.token_id for token in completed.tokens] == [
        _ANSWER_ID,
        IM_END_ID,
    ]
    assert completed.logprobs == [-0.1, -0.2]
    assert len(completed.logprobs) == len(completed.tokens)
    assert lifecycle not in scheduler.running
