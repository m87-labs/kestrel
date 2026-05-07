from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from kestrel.moondream.region import SpatialDecodeTables
from kestrel.moondream.runtime import TextToken
from kestrel.scheduler.scheduler import GenerationScheduler
from kestrel.scheduler.spatial import compute_spatial_values
from kestrel.scheduler.types import GenerationRequest, RequestLifecycle
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
        return SkillFinalizeResult(
            text="",
            tokens=list(self.tokens),
            output={},
        )


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


def test_scheduler_result_omits_logprobs_by_default() -> None:
    seq = _make_lifecycle(return_logprobs=None)
    seq.stage_token(SimpleNamespace(), TextToken(10))

    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = SimpleNamespace()
    result = GenerationScheduler._build_result(scheduler, seq)

    assert result.tokens == [TextToken(10)]
    assert result.logprobs is None


def test_scheduler_result_returns_requested_token_logprobs() -> None:
    seq = _make_lifecycle(return_logprobs=True)
    seq.stage_token(SimpleNamespace(), TextToken(10), logprob=-1.25)
    seq.stage_token(SimpleNamespace(), TextToken(11), logprob=-0.5)

    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = SimpleNamespace()
    result = GenerationScheduler._build_result(scheduler, seq)

    assert result.tokens == [TextToken(10), TextToken(11)]
    assert result.logprobs == [-1.25, -0.5]


def test_requested_logprobs_require_sampling_result() -> None:
    seq = _make_lifecycle(return_logprobs=True)

    with pytest.raises(RuntimeError, match="Missing token logprob"):
        seq.stage_token(SimpleNamespace(), TextToken(10))


def test_sample_batch_omits_logprob_keyword_without_opt_in(monkeypatch) -> None:
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
    scheduler = object.__new__(GenerationScheduler)
    scheduler.runtime = SimpleNamespace()
    scheduler._sampling_rng = torch.Generator()
    scheduler._sampling_temps_by_batch = torch.tensor([0.7], dtype=torch.float32)
    scheduler._sampling_top_ps_by_batch = torch.tensor([1.0], dtype=torch.float32)
    scheduler._sampling_temps = torch.empty((1,), dtype=torch.float32)
    scheduler._sampling_top_ps = torch.empty((1,), dtype=torch.float32)
    seqs = [
        SimpleNamespace(
            finalized=False,
            skill_state=SimpleNamespace(
                allowed_token_ids=lambda runtime: None,
                suppressed_token_ids=lambda runtime: None,
            ),
            request=SimpleNamespace(temperature=0.7, return_logprobs=None),
        )
    ]

    sampled, _, _, logprobs = GenerationScheduler._sample_batch(
        scheduler,
        torch.zeros((1, 8), dtype=torch.float32),
        seqs,  # type: ignore[arg-type]
        torch.empty((1,), dtype=torch.long),
        batch_idx=torch.tensor([0], dtype=torch.long),
        logprobs_out=torch.empty((1,), dtype=torch.float32),
    )

    assert sampled.tolist() == [3]
    assert logprobs is None


def test_spatial_logprobs_are_added_for_coord_and_size_tokens(monkeypatch) -> None:
    batch = 3
    coord_id = 90
    size_id = 91
    token_ids = torch.tensor([coord_id, size_id, 123], dtype=torch.long)
    token_logprobs = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
    hidden_last = torch.zeros((batch, 2), dtype=torch.float32)
    requests = [SimpleNamespace(temperature=0.7) for _ in range(batch)]
    spatial_tables = SpatialDecodeTables(
        coord_value_lut=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
        size_value_lut=torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32),
        coord_logits_dim=3,
    )

    def fake_spatial_decode_logits(
        hidden: torch.Tensor,
        tables: SpatialDecodeTables,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((batch, 3), dtype=torch.float32),
            torch.zeros((batch, 3), dtype=torch.float32),
            torch.zeros((batch, 3), dtype=torch.float32),
        )

    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        logprobs_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape[0] == batch:
            if logprobs_out is not None:
                logprobs_out.copy_(torch.tensor([-0.1, -0.2, -0.3]))
            return torch.tensor([0, 1, 2], dtype=torch.long)
        if logprobs_out is not None:
            logprobs_out.copy_(
                torch.tensor([-0.4, -0.5, -0.6, -0.7, -0.8, -0.9])
            )
        return torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.long)

    monkeypatch.setattr(
        "kestrel.scheduler.spatial.spatial_decode_logits",
        fake_spatial_decode_logits,
    )
    monkeypatch.setattr(
        "kestrel.scheduler.spatial.sample_step_from_logits",
        fake_sample_step_from_logits,
    )

    compute_spatial_values(
        token_ids,
        hidden_last,
        requests,  # type: ignore[arg-type]
        spatial_tables,
        temperatures=torch.full((batch,), 0.7),
        top_ps=torch.ones((batch,), dtype=torch.float32),
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


def test_spatial_decode_omits_logprob_keyword_without_buffer(monkeypatch) -> None:
    batch = 1
    spatial_tables = SpatialDecodeTables(
        coord_value_lut=torch.tensor([0.0, 1.0], dtype=torch.float32),
        size_value_lut=torch.tensor([0.5, 1.0], dtype=torch.float32),
        coord_logits_dim=2,
    )

    def fake_spatial_decode_logits(
        hidden: torch.Tensor,
        tables: SpatialDecodeTables,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((batch, 2), dtype=torch.float32),
            torch.zeros((batch, 2), dtype=torch.float32),
            torch.zeros((batch, 2), dtype=torch.float32),
        )

    def fake_sample_step_from_logits(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.zeros((logits.shape[0],), dtype=torch.long)

    monkeypatch.setattr(
        "kestrel.scheduler.spatial.spatial_decode_logits",
        fake_spatial_decode_logits,
    )
    monkeypatch.setattr(
        "kestrel.scheduler.spatial.sample_step_from_logits",
        fake_sample_step_from_logits,
    )

    coord, size = compute_spatial_values(
        torch.tensor([123], dtype=torch.long),
        torch.zeros((batch, 2), dtype=torch.float32),
        [SimpleNamespace(temperature=0.7)],  # type: ignore[list-item]
        spatial_tables,
        temperatures=torch.full((batch,), 0.7),
        top_ps=torch.ones((batch,), dtype=torch.float32),
        out_coord=torch.empty((batch, 1), dtype=torch.float32),
        out_size=torch.empty((batch, 2), dtype=torch.float32),
    )

    assert coord.tolist() == [[0.0]]
    assert size.tolist() == [[0.5, 0.5]]
