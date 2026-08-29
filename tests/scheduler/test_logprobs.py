from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from kestrel.models.moondream.region import SpatialDecodeTables
from kestrel.models.moondream.runtime import TextToken
from kestrel.runtime.sampling import SamplingHooks
from kestrel.scheduler.pipeline import (
    DecodeLaunch,
    DecodePendingCommit,
    LaunchHandle,
    PendingCommit,
)
from kestrel.scheduler.scheduler import GenerationScheduler, _MaskPlan
from kestrel.scheduler.spatial import compute_spatial_values
from kestrel.scheduler.types import GeneratedPrefix, GenerationRequest, RequestLifecycle
from kestrel.skills import DecodeStep, SkillFinalizeResult, SkillRegistry, SkillState
from tests.scheduler._fake_runtime import FakeRuntime


@dataclass(frozen=True)
class _SkillSpecStub:
    name: str = "stub"


class _SkillStateStub(SkillState):
    def __init__(self, request: GenerationRequest) -> None:
        super().__init__(_SkillSpecStub(), request)  # type: ignore[arg-type]

    def consume_step(self, runtime: object, step: DecodeStep) -> None:
        self.append_token(step.token)

    def finalize(self, runtime: object, *, reason: str) -> SkillFinalizeResult:
        return SkillFinalizeResult(text="", tokens=list(self.tokens), output={})


class _FailingConsumeSkillState(_SkillStateStub):
    def consume_step(self, runtime: object, step: DecodeStep) -> None:
        raise RuntimeError("consume failed")


class _FailingFinalizeSkillState(_SkillStateStub):
    def finalize(self, runtime: object, *, reason: str) -> SkillFinalizeResult:
        raise RuntimeError("finalize failed")


class _ReasoningStreamSkillState(_SkillStateStub):
    def __init__(self, request: GenerationRequest) -> None:
        super().__init__(request)
        self._reasoning_pending = True

    def pop_reasoning_stream_delta(self, runtime: object) -> str | None:
        if not self._reasoning_pending:
            return None
        self._reasoning_pending = False
        return "think"


class _AudioStreamState(_SkillStateStub):
    def pop_stream_output(self, runtime: object) -> dict[str, object] | None:
        return {
            "audio": [0.25, -0.25],
            "sample_rate": 24_000,
        }


def _make_lifecycle(*, return_logprobs: bool | None) -> RequestLifecycle:
    request = GenerationRequest(
        request_id=7,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=4,
        skill=_SkillSpecStub(),  # type: ignore[arg-type]
        request_context=object(),
        return_logprobs=return_logprobs,
    )
    state = _SkillStateStub(request)
    lifecycle = RequestLifecycle(request=request, skill_state=state)
    request.lifecycle = lifecycle
    return lifecycle


def _make_lifecycle_with_state(
    state_cls: type[_SkillStateStub],
    *,
    return_logprobs: bool | None,
) -> RequestLifecycle:
    seq = _make_lifecycle(return_logprobs=return_logprobs)
    state = state_cls(seq.request)
    seq.skill_state = state
    seq.request.skill_state = state
    return seq


def _scheduler(batch: int = 1) -> GenerationScheduler:
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = SimpleNamespace()
    scheduler._hooks = SamplingHooks()
    scheduler._greedy_tail_hooks_eligible = True
    scheduler._sampling_rng = torch.Generator()
    scheduler._sampling_temps = torch.empty((batch,), dtype=torch.float32)
    scheduler._sampling_top_ps = torch.empty((batch,), dtype=torch.float32)
    scheduler._sampling_temps_by_batch = torch.full(
        (batch,), 0.7, dtype=torch.float32
    )
    scheduler._sampling_top_ps_by_batch = torch.ones((batch,), dtype=torch.float32)
    return scheduler


def _sequence(
    *,
    temperature: float,
    return_logprobs: bool | None,
    suppress_next_token_ids: tuple[int, ...] | None = None,
    token_count: int = 0,
    generated_prefix_length: int = 0,
    allowed_token_ids: list[int] | None = None,
    suppressed_token_ids: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        finalized=False,
        skill_state=SimpleNamespace(
            token_count=token_count,
            allowed_token_ids=lambda runtime: allowed_token_ids,
            suppressed_token_ids=lambda runtime: suppressed_token_ids,
        ),
        request=SimpleNamespace(
            temperature=temperature,
            return_logprobs=return_logprobs,
            generated_prefix_length=generated_prefix_length,
            suppress_next_token_ids=suppress_next_token_ids,
        ),
    )


def _spatial_tables(dim: int) -> SpatialDecodeTables:
    values = torch.arange(dim, dtype=torch.float32)
    return SpatialDecodeTables(
        coord_value_lut=values,
        size_value_lut=values,
        coord_logits_dim=dim,
    )


def _patch_spatial_logits(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch: int,
    dim: int,
) -> None:
    def fake_spatial_decode_logits(
        hidden: torch.Tensor,
        tables: SpatialDecodeTables,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (batch, dim)
        return torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)

    monkeypatch.setattr(
        "kestrel.scheduler.spatial.spatial_decode_logits",
        fake_spatial_decode_logits,
    )


def test_scheduler_result_omits_logprobs_by_default() -> None:
    seq = _make_lifecycle(return_logprobs=None)
    seq.stage_token(SimpleNamespace(), TextToken(10))

    result = GenerationScheduler._build_result(_scheduler(), seq)

    assert result.tokens == [TextToken(10)]
    assert result.logprobs is None


def test_scheduler_result_returns_requested_token_logprobs() -> None:
    seq = _make_lifecycle(return_logprobs=True)
    seq.stage_token(SimpleNamespace(), TextToken(10), logprob=-1.25)
    seq.stage_token(SimpleNamespace(), TextToken(11), logprob=-0.5)

    result = GenerationScheduler._build_result(_scheduler(), seq)

    assert result.tokens == [TextToken(10), TextToken(11)]
    assert result.logprobs == [-1.25, -0.5]


def test_decode_step_exposes_selected_logprob_before_skill_consumption() -> None:
    seen: list[float | None] = []

    class _CapturingState(_SkillStateStub):
        def consume_step(self, runtime: object, step: DecodeStep) -> None:
            seen.append(step.logprob)
            super().consume_step(runtime, step)

    seq = _make_lifecycle_with_state(_CapturingState, return_logprobs=True)
    seq.stage_token(SimpleNamespace(), TextToken(10), logprob=-0.75)

    assert seen == [-0.75]
    assert seq.logprobs == [-0.75]


def test_stage_token_emits_reasoning_without_answer_text() -> None:
    updates = []
    seq = _make_lifecycle_with_state(
        _ReasoningStreamSkillState,
        return_logprobs=None,
    )
    seq.request.stream_callback = updates.append

    seq.stage_token(SimpleNamespace(), TextToken(10))
    seq.stage_token(SimpleNamespace(), TextToken(11))

    assert len(updates) == 2
    assert updates[0].request_id == 7
    assert updates[0].token == TextToken(10)
    assert updates[0].token_index == 0
    assert updates[0].text == ""
    assert updates[0].reasoning == "think"
    assert updates[1].token == TextToken(11)
    assert updates[1].token_index == 1
    assert updates[1].text == ""
    assert updates[1].reasoning is None


def test_stage_token_emits_update_without_text_or_reasoning_delta() -> None:
    updates = []
    seq = _make_lifecycle(return_logprobs=None)
    seq.request.stream_callback = updates.append

    seq.stage_token(SimpleNamespace(), TextToken(10))

    assert len(updates) == 1
    assert updates[0].request_id == 7
    assert updates[0].token == TextToken(10)
    assert updates[0].token_index == 0
    assert updates[0].text == ""
    assert updates[0].reasoning is None


def test_stream_update_preserves_capability_defined_output() -> None:
    updates = []
    seq = _make_lifecycle_with_state(_AudioStreamState, return_logprobs=None)
    seq.request.stream_callback = updates.append

    seq.stage_token(SimpleNamespace(), TextToken(10))

    assert len(updates) == 1
    assert updates[0].text == ""
    assert updates[0].output == {
        "audio": [0.25, -0.25],
        "sample_rate": 24_000,
    }


def test_scheduler_result_keeps_generated_prefix_logprobs_aligned() -> None:
    request = GenerationRequest(
        request_id=7,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=4,
        skill=_SkillSpecStub(),  # type: ignore[arg-type]
        request_context=object(),
        return_logprobs=True,
        generated_prefix=GeneratedPrefix(
            tokens=(TextToken(10), TextToken(11)),
            logprobs=(-0.1, -0.2),
        ),
    )
    state = _SkillStateStub(request)
    state.consume_step(SimpleNamespace(), DecodeStep(TextToken(10), 0))
    state.consume_step(SimpleNamespace(), DecodeStep(TextToken(11), 1))
    lifecycle = RequestLifecycle(request=request, skill_state=state)
    request.lifecycle = lifecycle

    lifecycle.stage_token(SimpleNamespace(), TextToken(12), logprob=-0.3)
    result = GenerationScheduler._build_result(_scheduler(), lifecycle)

    assert result.tokens == [TextToken(10), TextToken(11), TextToken(12)]
    assert result.logprobs == [-0.1, -0.2, -0.3]
    assert result.metrics.decode_tokens == 1


def test_generation_request_tracks_generated_prefix_prefill_shape() -> None:
    request = GenerationRequest(
        request_id=7,
        prompt="prompt",
        prompt_tokens=[TextToken(1)],
        max_new_tokens=4,
        skill=_SkillSpecStub(),  # type: ignore[arg-type]
        request_context=object(),
        generated_prefix=GeneratedPrefix(tokens=(TextToken(10), TextToken(11))),
    )

    assert request.prompt_length == 1
    assert request.generated_prefix_length == 2
    assert request.remaining_new_tokens == 2
    assert request.prefill_tokens == [TextToken(1), TextToken(10), TextToken(11)]
    assert request.target_length == 5


def test_generation_request_validates_generated_prefix() -> None:
    with pytest.raises(ValueError, match="shorter"):
        GenerationRequest(
            request_id=7,
            prompt="prompt",
            prompt_tokens=[TextToken(1)],
            max_new_tokens=2,
            skill=_SkillSpecStub(),  # type: ignore[arg-type]
            request_context=object(),
            generated_prefix=GeneratedPrefix(tokens=(TextToken(10), TextToken(11))),
        )
    with pytest.raises(ValueError, match="same length"):
        GenerationRequest(
            request_id=7,
            prompt="prompt",
            prompt_tokens=[TextToken(1)],
            max_new_tokens=4,
            skill=_SkillSpecStub(),  # type: ignore[arg-type]
            request_context=object(),
            generated_prefix=GeneratedPrefix(tokens=(TextToken(10),), logprobs=()),
        )
    with pytest.raises(ValueError, match="return_logprobs"):
        GenerationRequest(
            request_id=7,
            prompt="prompt",
            prompt_tokens=[TextToken(1)],
            max_new_tokens=4,
            skill=_SkillSpecStub(),  # type: ignore[arg-type]
            request_context=object(),
            return_logprobs=True,
            generated_prefix=GeneratedPrefix(tokens=(TextToken(10),)),
        )


def test_scheduler_result_rejects_misaligned_logprobs() -> None:
    seq = _make_lifecycle(return_logprobs=True)
    seq.stage_token(SimpleNamespace(), TextToken(10), logprob=-1.25)
    seq.logprobs.append(-0.5)

    result = GenerationScheduler._build_result(_scheduler(), seq)

    assert result.tokens == []
    assert result.logprobs is None
    assert result.finish_reason == "error"
    assert result.output == {"error": "Internal logprobs/token alignment mismatch"}


def test_requested_logprobs_require_sampling_result() -> None:
    seq = _make_lifecycle(return_logprobs=True)

    with pytest.raises(RuntimeError, match="Missing token logprob"):
        seq.stage_token(SimpleNamespace(), TextToken(10))


def test_logprob_is_not_appended_when_token_consume_fails() -> None:
    seq = _make_lifecycle_with_state(
        _FailingConsumeSkillState,
        return_logprobs=True,
    )

    with pytest.raises(RuntimeError, match="consume failed"):
        seq.stage_token(SimpleNamespace(), TextToken(10), logprob=-1.25)

    assert seq.logprobs == []


def test_logprob_alignment_check_preserves_finalize_error() -> None:
    seq = _make_lifecycle_with_state(
        _FailingFinalizeSkillState,
        return_logprobs=True,
    )
    seq.stage_token(SimpleNamespace(), TextToken(10), logprob=-1.25)

    result = GenerationScheduler._build_result(_scheduler(), seq)

    assert result.finish_reason == "error"
    assert result.output == {"error": "finalize failed"}
    assert result.logprobs is None


def test_sample_batch_omits_logprob_keyword_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        sampled = torch.tensor([3], dtype=torch.long)
        if out is not None:
            out.copy_(sampled)
            return out
        return sampled

    monkeypatch.setattr(
        "kestrel.scheduler.scheduler.sample_step_from_logits",
        fake_sample_step_from_logits,
    )

    sampled, _, _, logprobs = GenerationScheduler._sample_batch(
        _scheduler(),
        torch.zeros((1, 8), dtype=torch.float32),
        [_sequence(temperature=0.7, return_logprobs=None)],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
        batch_idx=torch.tensor([0], dtype=torch.long),
        logprobs_out=torch.empty((1,), dtype=torch.float32),
    )

    assert sampled.tolist() == [3]
    assert logprobs is None


def test_sample_batch_requests_packed_sampling_when_runtime_requires_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    scheduler._hooks = SamplingHooks(require_packed_sampling=True)
    sample = Mock(return_value=torch.tensor([3]))
    monkeypatch.setattr(
        "kestrel.scheduler.scheduler.sample_step_from_logits",
        sample,
    )

    GenerationScheduler._sample_batch(
        scheduler,
        torch.zeros((1, 8), dtype=torch.float32),
        [_sequence(temperature=0.7, return_logprobs=None)],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
        batch_idx=torch.tensor([0], dtype=torch.long),
    )

    assert sample.call_args.kwargs["require_packed"] is True


def test_scheduler_rejects_custom_greedy_with_speculative_decode() -> None:
    runtime = FakeRuntime()
    runtime.spec = object()  # type: ignore[assignment]
    runtime.sampling_hooks = SamplingHooks(
        sample_greedy=lambda logits, out, **_kwargs: out
    )

    with pytest.raises(ValueError, match="require non-speculative"):
        GenerationScheduler(
            runtime,
            compute_stream=None,
            skill_registry=SkillRegistry([]),
        )


def test_scheduler_rejects_custom_logits_processing_with_speculative_decode() -> None:
    runtime = FakeRuntime()
    runtime.spec = object()  # type: ignore[assignment]
    runtime.sampling_hooks = SamplingHooks(
        process_logits=lambda logits, **_kwargs: logits.zero_()
    )

    with pytest.raises(ValueError, match="require non-speculative"):
        GenerationScheduler(
            runtime,
            compute_stream=None,
            skill_registry=SkillRegistry([]),
        )


def test_scheduler_rejects_custom_token_scoring_with_speculative_decode() -> None:
    runtime = FakeRuntime()
    runtime.spec = object()  # type: ignore[assignment]
    runtime.sampling_hooks = SamplingHooks(
        score_sampled_tokens=lambda _logits, **_kwargs: None
    )

    with pytest.raises(ValueError, match="require non-speculative"):
        GenerationScheduler(
            runtime,
            compute_stream=None,
            skill_registry=SkillRegistry([]),
        )


def test_scheduler_rejects_custom_sampling_params_with_speculative_decode() -> None:
    runtime = FakeRuntime()
    runtime.spec = object()  # type: ignore[assignment]
    runtime.sampling_hooks = SamplingHooks(
        adjust_sampling_params=lambda _temperatures, _top_ps, **_kwargs: None
    )

    with pytest.raises(ValueError, match="require non-speculative"):
        GenerationScheduler(
            runtime,
            compute_stream=None,
            skill_registry=SkillRegistry([]),
        )


@pytest.mark.parametrize(
    ("temperature", "return_logprobs"),
    ((0.0, None), (0.7, True)),
)
def test_sample_batch_processes_logits_before_generic_sampling(
    monkeypatch: pytest.MonkeyPatch,
    temperature: float,
    return_logprobs: bool | None,
) -> None:
    calls: list[tuple[list[int], int]] = []
    scheduler = _scheduler()

    def process_logits(
        logits: torch.Tensor,
        *,
        sequences: list[object],
        batch_idx: torch.Tensor,
    ) -> None:
        calls.append((batch_idx.tolist(), len(sequences)))
        assert torch.isneginf(logits[0, 0])
        logits[0, 1] = -float("inf")

    scheduler._hooks = SamplingHooks(process_logits=process_logits)
    if temperature > 0.0:
        scheduler._sampling_temps_by_batch.fill_(temperature)

    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        logprobs_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert torch.isneginf(logits[0, 0])
        assert torch.isneginf(logits[0, 1])
        sampled = torch.tensor([2], dtype=torch.long)
        if logprobs_out is not None:
            logprobs_out.fill_(-0.25)
        if out is not None:
            out.copy_(sampled)
            return out
        return sampled

    monkeypatch.setattr(
        "kestrel.scheduler.scheduler.sample_step_from_logits",
        fake_sample_step_from_logits,
    )
    sampled, _, _, logprobs = GenerationScheduler._sample_batch(
        scheduler,
        torch.tensor([[5.0, 4.0, 3.0]], dtype=torch.float32),
        [
            _sequence(
                temperature=temperature,
                return_logprobs=return_logprobs,
                suppressed_token_ids=[0],
            )
        ],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
        batch_idx=torch.tensor([0], dtype=torch.long),
        logprobs_out=torch.empty((1,), dtype=torch.float32),
    )

    assert calls == [([0], 1)]
    assert sampled.tolist() == [2]
    assert (logprobs is not None) is (return_logprobs is True)


def test_sample_batch_applies_model_owned_selected_token_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    observed: list[list[int]] = []

    def score_sampled_tokens(
        logits: torch.Tensor,
        *,
        sampled_ids: torch.Tensor,
        token_logprobs: torch.Tensor,
        sequences: list[object],
        batch_idx: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
    ) -> None:
        assert logits.shape == (1, 3)
        assert len(sequences) == 1
        torch.testing.assert_close(temperatures, torch.tensor([0.7]))
        assert top_ps.tolist() == [1.0]
        observed.append((sampled_ids.tolist(), batch_idx.tolist()))
        token_logprobs.fill_(-2.5)

    scheduler._hooks = SamplingHooks(
        score_sampled_tokens=score_sampled_tokens
    )

    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        logprobs_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del logits, temperatures, top_p, generator
        assert logprobs_out is not None
        logprobs_out.fill_(0.0)
        sampled = torch.tensor([2], dtype=torch.long)
        if out is not None:
            out.copy_(sampled)
            return out
        return sampled

    monkeypatch.setattr(
        "kestrel.scheduler.scheduler.sample_step_from_logits",
        fake_sample_step_from_logits,
    )
    sampled, _, _, logprobs = GenerationScheduler._sample_batch(
        scheduler,
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        [_sequence(temperature=0.7, return_logprobs=True)],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
        batch_idx=torch.tensor([0], dtype=torch.long),
        logprobs_out=torch.empty((1,), dtype=torch.float32),
    )

    assert sampled.tolist() == [2]
    assert observed == [([2], [0])]
    assert logprobs is not None
    torch.testing.assert_close(logprobs, torch.tensor([-2.5]))


def test_sample_batch_fuses_greedy_model_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    scheduler._hooks = SamplingHooks(
        score_sampled_tokens=lambda *_args, **_kwargs: pytest.fail(
            "greedy model scores should already be complete"
        )
    )

    def fake_greedy(
        logits: torch.Tensor,
        *,
        out: torch.Tensor,
        logprobs_out: torch.Tensor,
        require_packed: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert logits.shape == (1, 3)
        assert require_packed is False
        out.fill_(2)
        logprobs_out.fill_(-0.25)
        return out, logprobs_out

    monkeypatch.setattr(
        "kestrel.scheduler.scheduler.greedy_logprobs_from_logits",
        fake_greedy,
    )
    sampled, temps, top_ps, logprobs = GenerationScheduler._sample_batch(
        scheduler,
        torch.tensor([[1.0, 2.0, 3.0]]),
        [_sequence(temperature=0.0, return_logprobs=True)],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
        batch_idx=torch.tensor([0], dtype=torch.long),
        logprobs_out=torch.empty((1,), dtype=torch.float32),
    )

    assert sampled.tolist() == [2]
    assert temps is None and top_ps is None
    assert logprobs is not None and logprobs.tolist() == [-0.25]


def test_sample_batch_applies_model_owned_sampling_params_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(batch=2)
    observed: list[tuple[list[int], list[int]]] = []

    def adjust_sampling_params(
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        *,
        sequences: list[object],
        batch_idx: torch.Tensor,
    ) -> None:
        assert len(sequences) == 2
        torch.testing.assert_close(temperatures, torch.tensor([0.8, 0.6]))
        torch.testing.assert_close(top_ps, torch.tensor([0.7, 0.5]))
        observed.append(batch_idx.tolist())
        temperatures[0] = 0.0
        top_ps[0] = 1.0

    scheduler._hooks = SamplingHooks(
        adjust_sampling_params=adjust_sampling_params
    )
    scheduler._sampling_temps_by_batch[:2].copy_(torch.tensor([0.8, 0.6]))
    scheduler._sampling_top_ps_by_batch[:2].copy_(torch.tensor([0.7, 0.5]))

    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        logprobs_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del logits, generator, logprobs_out
        torch.testing.assert_close(temperatures, torch.tensor([0.0, 0.6]))
        torch.testing.assert_close(top_p, torch.tensor([1.0, 0.5]))
        assert out is not None
        out.copy_(torch.tensor([2, 1]))
        return out

    monkeypatch.setattr(
        "kestrel.scheduler.scheduler.sample_step_from_logits",
        fake_sample_step_from_logits,
    )
    sampled, temps, top_ps, _ = GenerationScheduler._sample_batch(
        scheduler,
        torch.ones((2, 3), dtype=torch.float32),
        [
            _sequence(temperature=0.8, return_logprobs=False),
            _sequence(temperature=0.6, return_logprobs=False),
        ],  # type: ignore[list-item]
        torch.empty((2,), dtype=torch.long),
        batch_idx=torch.tensor([0, 1], dtype=torch.long),
    )

    assert sampled.tolist() == [2, 1]
    assert observed == [[0, 1]]
    assert temps is not None and top_ps is not None
    torch.testing.assert_close(temps, torch.tensor([0.0, 0.6]))
    torch.testing.assert_close(top_ps, torch.tensor([1.0, 0.5]))


def test_sample_batch_calls_custom_greedy_after_static_and_one_shot_masks() -> None:
    calls: list[tuple[list[int], int]] = []
    scheduler = _scheduler()

    def sample_greedy(
        logits: torch.Tensor,
        out: torch.Tensor,
        *,
        sequences: list[object],
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((batch_idx.tolist(), len(sequences)))
        assert torch.isneginf(logits[0, 0])
        assert torch.isneginf(logits[0, 1])
        torch.argmax(logits, dim=-1, out=out)
        return out

    scheduler._hooks = SamplingHooks(sample_greedy=sample_greedy)
    sampled, _, _, _ = GenerationScheduler._sample_batch(
        scheduler,
        torch.tensor([[5.0, 4.0, 3.0]], dtype=torch.float32),
        [
            _sequence(
                temperature=0.0,
                return_logprobs=None,
                suppressed_token_ids=[0],
                suppress_next_token_ids=(1,),
            )
        ],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
        batch_idx=torch.tensor([0], dtype=torch.long),
    )

    assert calls == [([0], 1)]
    assert sampled.tolist() == [2]


@pytest.mark.parametrize(
    ("temperature", "return_logprobs", "message"),
    [
        (0.7, None, "requires greedy requests"),
        (0.0, True, "does not support token logprobs"),
    ],
)
def test_sample_batch_rejects_unsupported_custom_greedy_modes(
    temperature: float,
    return_logprobs: bool | None,
    message: str,
) -> None:
    scheduler = _scheduler()
    scheduler._hooks = SamplingHooks(
        sample_greedy=lambda _logits, out, **_kwargs: out
    )

    with pytest.raises(ValueError, match=message):
        GenerationScheduler._sample_batch(
            scheduler,
            torch.zeros((1, 8), dtype=torch.float32),
            [
                _sequence(
                    temperature=temperature,
                    return_logprobs=return_logprobs,
                )
            ],  # type: ignore[list-item]
            torch.empty((1,), dtype=torch.long),
            batch_idx=torch.tensor([0], dtype=torch.long),
            logprobs_out=torch.empty((1,), dtype=torch.float32),
        )


def test_sample_batch_uses_sampler_for_greedy_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        logprobs_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        calls.append((temperatures.clone(), top_p.clone(), logprobs_out is not None))
        sampled = torch.tensor([4], dtype=torch.long)
        if logprobs_out is not None:
            logprobs_out.copy_(torch.tensor([-1.5], dtype=torch.float32))
        if out is not None:
            out.copy_(sampled)
            return out
        return sampled

    monkeypatch.setattr(
        "kestrel.scheduler.scheduler.sample_step_from_logits",
        fake_sample_step_from_logits,
    )

    sampled, temps, top_ps, logprobs = GenerationScheduler._sample_batch(
        _scheduler(),
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
        [_sequence(temperature=0.0, return_logprobs=True)],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
        logprobs_out=torch.empty((1,), dtype=torch.float32),
    )

    assert sampled.tolist() == [4]
    assert logprobs is not None
    torch.testing.assert_close(logprobs, torch.tensor([-1.5], dtype=torch.float32))
    assert temps is not None and top_ps is not None
    torch.testing.assert_close(temps, torch.zeros((1,), dtype=torch.float32))
    torch.testing.assert_close(top_ps, torch.ones((1,), dtype=torch.float32))
    assert len(calls) == 1
    call_temps, call_top_ps, call_logprobs = calls[0]
    torch.testing.assert_close(call_temps, torch.zeros((1,), dtype=torch.float32))
    torch.testing.assert_close(call_top_ps, torch.ones((1,), dtype=torch.float32))
    assert call_logprobs is True


def test_sample_batch_suppresses_next_token_only() -> None:
    logits = torch.tensor([[5.0, 4.0, 0.0]], dtype=torch.float32)

    sampled, _, _, _ = GenerationScheduler._sample_batch(
        _scheduler(),
        logits.clone(),
        [
            _sequence(
                temperature=0.0,
                return_logprobs=None,
                suppress_next_token_ids=(0,),
                token_count=0,
            )
        ],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
    )
    assert sampled.tolist() == [1]

    sampled, _, _, _ = GenerationScheduler._sample_batch(
        _scheduler(),
        logits.clone(),
        [
            _sequence(
                temperature=0.0,
                return_logprobs=None,
                suppress_next_token_ids=(0,),
                token_count=1,
            )
        ],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
    )
    assert sampled.tolist() == [0]


def test_sample_batch_suppresses_first_token_after_generated_prefix() -> None:
    logits = torch.tensor([[5.0, 4.0, 0.0]], dtype=torch.float32)

    sampled, _, _, _ = GenerationScheduler._sample_batch(
        _scheduler(),
        logits.clone(),
        [
            _sequence(
                temperature=0.0,
                return_logprobs=None,
                suppress_next_token_ids=(0,),
                token_count=2,
                generated_prefix_length=2,
            )
        ],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
    )
    assert sampled.tolist() == [1]

    sampled, _, _, _ = GenerationScheduler._sample_batch(
        _scheduler(),
        logits.clone(),
        [
            _sequence(
                temperature=0.0,
                return_logprobs=None,
                suppress_next_token_ids=(0,),
                token_count=3,
                generated_prefix_length=2,
            )
        ],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
    )
    assert sampled.tolist() == [0]


def test_sample_batch_suppression_preserves_baseline_logprob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        logprobs_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert torch.isneginf(logits[0, 0])
        sampled = torch.tensor([1], dtype=torch.long)
        if logprobs_out is not None:
            logprobs_out.copy_(torch.tensor([-999.0], dtype=torch.float32))
        if out is not None:
            out.copy_(sampled)
            return out
        return sampled

    monkeypatch.setattr(
        "kestrel.scheduler.scheduler.sample_step_from_logits",
        fake_sample_step_from_logits,
    )

    scheduler = _scheduler()
    scheduler._sampling_temps_by_batch.fill_(1.0)
    logits = torch.tensor([[10.0, 0.0, -10.0]], dtype=torch.float32)
    sampled, _, _, logprobs = GenerationScheduler._sample_batch(
        scheduler,
        logits,
        [
            _sequence(
                temperature=1.0,
                return_logprobs=True,
                suppress_next_token_ids=(0,),
            )
        ],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
        batch_idx=torch.tensor([0], dtype=torch.long),
        logprobs_out=torch.empty((1,), dtype=torch.float32),
    )

    expected = torch.log_softmax(
        torch.tensor([[10.0, 0.0, -10.0]], dtype=torch.float32), dim=-1
    )[0, 1]
    assert sampled.tolist() == [1]
    assert logprobs is not None
    torch.testing.assert_close(logprobs, expected.view(1))


def test_sample_batch_suppression_logprob_overwrite_is_row_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        logprobs_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert torch.isneginf(logits[0, 0])
        assert logits[1, 0] == 1.0
        sampled = torch.tensor([1, 2], dtype=torch.long)
        if logprobs_out is not None:
            logprobs_out.copy_(torch.tensor([-999.0, -2.0], dtype=torch.float32))
        if out is not None:
            out.copy_(sampled)
            return out
        return sampled

    monkeypatch.setattr(
        "kestrel.scheduler.scheduler.sample_step_from_logits",
        fake_sample_step_from_logits,
    )

    scheduler = _scheduler(batch=2)
    scheduler._sampling_temps_by_batch.fill_(1.0)
    logits = torch.tensor(
        [
            [10.0, 0.0, -10.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    sampled, _, _, logprobs = GenerationScheduler._sample_batch(
        scheduler,
        logits,
        [
            _sequence(
                temperature=1.0,
                return_logprobs=True,
                suppress_next_token_ids=(0,),
            ),
            _sequence(temperature=1.0, return_logprobs=True),
        ],  # type: ignore[list-item]
        torch.empty((2,), dtype=torch.long),
        batch_idx=torch.tensor([0, 1], dtype=torch.long),
        logprobs_out=torch.empty((2,), dtype=torch.float32),
    )

    expected_row0 = torch.log_softmax(
        torch.tensor([[10.0, 0.0, -10.0]], dtype=torch.float32), dim=-1
    )[0, 1]
    assert sampled.tolist() == [1, 2]
    assert logprobs is not None
    torch.testing.assert_close(
        logprobs,
        torch.tensor([float(expected_row0), -2.0], dtype=torch.float32),
    )


def test_selected_logprobs_follow_sampling_epsilon_boundary() -> None:
    logits = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, -1e-6, -2e-6],
        ],
        dtype=torch.float32,
    )
    sampled_ids = torch.tensor([0, 1, 1], dtype=torch.long)
    temperatures = torch.tensor([0.0, 5e-7, 2e-6], dtype=torch.float32)

    logprobs = GenerationScheduler._selected_logprobs_from_logits(
        logits,
        sampled_ids,
        temperatures,
    )

    expected_temp_scaled = torch.log_softmax(logits[2] / temperatures[2], dim=-1)[1]
    assert logprobs[0].item() == 0.0
    assert torch.isneginf(logprobs[1]).item()
    torch.testing.assert_close(logprobs[2], expected_temp_scaled)


def test_sample_batch_suppression_composes_with_allowed_tokens() -> None:
    sampled, _, _, _ = GenerationScheduler._sample_batch(
        _scheduler(),
        torch.tensor([[5.0, 4.0, 3.0]], dtype=torch.float32),
        [
            _sequence(
                temperature=0.0,
                return_logprobs=None,
                suppress_next_token_ids=(0,),
                allowed_token_ids=[0, 1],
            )
        ],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
    )

    assert sampled.tolist() == [1]


def test_sample_batch_plan_waits_through_event_abstraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EventStub:
        waits = 0

        def wait(self) -> None:
            self.waits += 1

    def fail_current_stream() -> None:
        raise AssertionError("sampling should not query a CUDA stream")

    monkeypatch.setattr(torch.cuda, "current_stream", fail_current_stream)

    event = EventStub()
    sampled, _, _, _ = GenerationScheduler._sample_batch(
        _scheduler(),
        torch.tensor([[5.0, 4.0, 3.0]], dtype=torch.float32),
        [_sequence(temperature=0.0, return_logprobs=None)],  # type: ignore[list-item]
        torch.empty((1,), dtype=torch.long),
        plan=_MaskPlan(
            disallow=torch.tensor([[True, False, False]]),
            event=event,
            suppress_rows=[],
            all_greedy=True,
            any_return_logprobs=False,
        ),
    )

    assert event.waits == 1
    assert sampled.tolist() == [1]


@pytest.mark.parametrize(
    (
        "all_greedy",
        "with_post_sample",
        "expected_pending_copies",
        "expected_commit_records",
        "expected_transfer_fence",
    ),
    [
        (True, False, 0, 0, True),
        (True, True, 0, 1, False),
        (False, False, 1, 1, False),
    ],
)
def test_finalize_only_reuses_transfer_fence_for_plain_fused_greedy(
    monkeypatch: pytest.MonkeyPatch,
    all_greedy: bool,
    with_post_sample: bool,
    expected_pending_copies: int,
    expected_commit_records: int,
    expected_transfer_fence: bool,
) -> None:
    class EventStub:
        def __init__(self) -> None:
            self.records = 0

        def record(self) -> None:
            self.records += 1

    class PendingStub:
        def __init__(self) -> None:
            self.shape = (8,)
            self.index_copies = 0

        def index_copy_(self, *_args) -> None:
            self.index_copies += 1

    batch_indices = torch.tensor([1, 4], dtype=torch.long)
    sampled_ids = torch.empty((2,), dtype=torch.long)
    step_done = EventStub()
    commit_done = EventStub()
    transfer = object()
    slot = SimpleNamespace(
        compute_stream=object(),
        logits=torch.tensor([[1.0, 4.0], [5.0, 2.0]]),
        hidden_last=torch.empty((2, 1)),
        sampled_ids=sampled_ids,
        sampled_logprobs=torch.empty((2,), dtype=torch.float32),
        meta=SimpleNamespace(batch_idx=SimpleNamespace(gpu=batch_indices)),
        step_done_event=step_done,
        commit_done_event=commit_done,
        render=SimpleNamespace(transfer=lambda *_args, **_kwargs: transfer),
    )
    scheduler = object.__new__(GenerationScheduler)
    other_slot = SimpleNamespace(compute_stream=object())
    assert other_slot.compute_stream is not slot.compute_stream
    scheduler.runtime = SimpleNamespace(decode_slots=(other_slot, slot))
    runtime_step = object()
    post_sample_calls = []

    def post_sample(*args, **kwargs):
        post_sample_calls.append((args, kwargs))
        return runtime_step

    scheduler._hooks = SamplingHooks(
        post_sample=post_sample if with_post_sample else None
    )
    scheduler._greedy_tail_hooks_eligible = True
    pending = PendingStub()
    scheduler._pending_token_ids = pending
    scheduler._greedy_tail_workspaces = (object(), object())
    generic_calls = []

    def sample_batch(*args, **kwargs):
        generic_calls.append((args, kwargs))
        out = args[2]
        out.copy_(torch.tensor([1, 0]))
        return out, None, None, None

    scheduler._sample_batch = sample_batch
    calls = []

    def fused(
        logits,
        batch_idx,
        pending,
        workspace,
        *,
        out,
        batch_indices_in_bounds=False,
    ):
        calls.append(
            (
                logits,
                batch_idx,
                pending,
                workspace,
                out,
                batch_indices_in_bounds,
            )
        )
        out.copy_(torch.tensor([1, 0]))
        return out

    monkeypatch.setattr("kestrel.scheduler.scheduler.greedy_tail_from_logits", fused)
    sequences = [
        SimpleNamespace(packed_pending_ready=False),
        SimpleNamespace(packed_pending_ready=False),
    ]
    handle = LaunchHandle(
        kind="decode",
        sequences=sequences,
        payload=DecodeLaunch(slot_id=1),
    )
    plan = _MaskPlan(
        disallow=None,
        event=None,
        suppress_rows=[],
        all_greedy=all_greedy,
        any_return_logprobs=False,
    )

    result = scheduler._finalize_sampling_on_stream(handle, plan)

    assert len(calls) == int(all_greedy)
    assert len(generic_calls) == int(not all_greedy)
    if all_greedy:
        assert calls[0][1].data_ptr() == batch_indices.data_ptr()
        assert torch.equal(calls[0][1], batch_indices)
        assert calls[0][2] is scheduler._pending_token_ids
        assert calls[0][3] is scheduler._greedy_tail_workspaces[1]
        assert calls[0][4] is sampled_ids
        assert calls[0][5] is True
        assert torch.all(calls[0][1] >= 0)
        assert torch.all(calls[0][1] < scheduler._pending_token_ids.shape[0])
    assert sampled_ids.tolist() == [1, 0]
    assert step_done.records == 1
    assert pending.index_copies == expected_pending_copies
    assert commit_done.records == expected_commit_records
    assert len(post_sample_calls) == int(with_post_sample)
    assert all(sequence.packed_pending_ready for sequence in sequences)
    assert result.transfer is transfer
    assert (
        result.payload.pending_write_covered_by_transfer is expected_transfer_fence
    )
    assert result.payload.runtime_step is (
        runtime_step if with_post_sample else None
    )


@pytest.mark.parametrize("transfer_fences_pending", [False, True])
def test_decode_commit_uses_the_required_pending_write_fence(
    transfer_fences_pending: bool,
) -> None:
    class CommitEventStub:
        def __init__(self) -> None:
            self.synchronizes = 0

        def synchronize(self) -> None:
            self.synchronizes += 1

    commit_done = CommitEventStub()
    slot = SimpleNamespace(
        commit_done_event=commit_done,
        meta=SimpleNamespace(batch_idx=SimpleNamespace(cpu=torch.empty((0,)))),
    )
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = SimpleNamespace(decode_slots=(slot,))
    scheduler._hooks = SamplingHooks()
    step = PendingCommit(
        kind="decode",
        sequences=[],
        transfer=SimpleNamespace(
            wait=lambda: (torch.empty((0,), dtype=torch.long), None)
        ),
        payload=DecodePendingCommit(
            slot_id=0,
            pending_write_covered_by_transfer=transfer_fences_pending,
        ),
    )

    scheduler.commit_step(step)

    assert commit_done.synchronizes == int(not transfer_fences_pending)


@pytest.mark.parametrize(
    ("plan", "hooks"),
    [
        (None, SamplingHooks()),
        (
            _MaskPlan(None, None, [], False, False),
            SamplingHooks(),
        ),
        (
            _MaskPlan(None, None, [], True, True),
            SamplingHooks(),
        ),
        (
            _MaskPlan(torch.zeros((1, 1), dtype=torch.bool), None, [], True, False),
            SamplingHooks(),
        ),
        (
            _MaskPlan(None, None, [(0, (1,))], True, False),
            SamplingHooks(),
        ),
        (
            _MaskPlan(None, None, [], True, False),
            SamplingHooks(process_logits=lambda *_args, **_kwargs: None),
        ),
        (
            _MaskPlan(None, None, [], True, False),
            SamplingHooks(adjust_sampling_params=lambda *_args, **_kwargs: None),
        ),
        (
            _MaskPlan(None, None, [], True, False),
            SamplingHooks(sample_greedy=lambda *_args, **_kwargs: None),
        ),
        (
            _MaskPlan(None, None, [], True, False),
            SamplingHooks(score_sampled_tokens=lambda *_args, **_kwargs: None),
        ),
    ],
)
def test_greedy_tail_eligibility_rejects_modified_sampling(plan, hooks) -> None:
    scheduler = object.__new__(GenerationScheduler)
    scheduler._hooks = hooks
    scheduler._greedy_tail_hooks_eligible = bool(
        hooks.process_logits is None
        and hooks.adjust_sampling_params is None
        and hooks.sample_greedy is None
        and hooks.score_sampled_tokens is None
    )

    assert not scheduler._can_publish_greedy_tail(plan)


@pytest.mark.parametrize("temperature", [0.7, 0.0])
def test_spatial_logprobs_are_added_for_coord_and_size_tokens(
    monkeypatch: pytest.MonkeyPatch,
    temperature: float,
) -> None:
    batch = 3
    dim = 3
    coord_id = 90
    size_id = 91
    calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        logprobs_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        calls.append((temperatures.clone(), top_p.clone(), logprobs_out is not None))
        if logits.shape[0] == batch:
            if logprobs_out is not None:
                logprobs_out.copy_(torch.tensor([-0.1, -0.2, -0.3]))
            return torch.tensor([0, 1, 2], dtype=torch.long)
        if logprobs_out is not None:
            logprobs_out.copy_(
                torch.tensor([-0.4, -0.5, -0.6, -0.7, -0.8, -0.9])
            )
        return torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.long)

    _patch_spatial_logits(monkeypatch, batch=batch, dim=dim)
    monkeypatch.setattr(
        "kestrel.scheduler.spatial.sample_step_from_logits",
        fake_sample_step_from_logits,
    )

    sample_kwargs = {}
    if temperature > 0.0:
        sample_kwargs = {
            "temperatures": torch.full((batch,), temperature),
            "top_ps": torch.ones((batch,), dtype=torch.float32),
        }
    token_logprobs = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
    compute_spatial_values(
        torch.tensor([coord_id, size_id, 123], dtype=torch.long),
        torch.zeros((batch, 2), dtype=torch.float32),
        [SimpleNamespace(temperature=temperature) for _ in range(batch)],  # type: ignore[list-item]
        _spatial_tables(dim),
        **sample_kwargs,
        token_logprobs=token_logprobs,
        coord_id=coord_id,
        size_id=size_id,
        out_coord=torch.empty((batch, 1), dtype=torch.float32),
        out_size=torch.empty((batch, 2), dtype=torch.float32),
    )

    torch.testing.assert_close(
        token_logprobs,
        torch.tensor([-1.1, -3.3, -3.0], dtype=torch.float32),
    )
    assert [call[2] for call in calls] == [True, True]
    expected_temp = torch.full((batch,), temperature, dtype=torch.float32)
    torch.testing.assert_close(calls[0][0], expected_temp)
    torch.testing.assert_close(calls[1][0], expected_temp.repeat(2))
    torch.testing.assert_close(calls[0][1], torch.ones((batch,), dtype=torch.float32))
    torch.testing.assert_close(calls[1][1], torch.ones((batch * 2,), dtype=torch.float32))


def test_spatial_decode_omits_logprob_keyword_without_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = 1
    dim = 2
    _patch_spatial_logits(monkeypatch, batch=batch, dim=dim)

    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        assert "logprobs_out" not in kwargs
        return torch.zeros((logits.shape[0],), dtype=torch.long)

    monkeypatch.setattr(
        "kestrel.scheduler.spatial.sample_step_from_logits",
        fake_sample_step_from_logits,
    )

    coord, size = compute_spatial_values(
        torch.tensor([123], dtype=torch.long),
        torch.zeros((batch, 2), dtype=torch.float32),
        [SimpleNamespace(temperature=0.7)],  # type: ignore[list-item]
        _spatial_tables(dim),
        temperatures=torch.full((batch,), 0.7),
        top_ps=torch.ones((batch,), dtype=torch.float32),
        out_coord=torch.empty((batch, 1), dtype=torch.float32),
        out_size=torch.empty((batch, 2), dtype=torch.float32),
    )

    assert coord.tolist() == [[0.0]]
    assert size.tolist() == [[0.0, 0.0]]
