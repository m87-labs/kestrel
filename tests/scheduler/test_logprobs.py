from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from kestrel.moondream.region import SpatialDecodeTables
from kestrel.moondream.runtime import TextToken
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.spatial import compute_spatial_values
from kestrel.scheduler.types import GeneratedPrefix, GenerationRequest, RequestLifecycle
from kestrel.skills import DecodeStep, SkillFinalizeResult, SkillState


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
