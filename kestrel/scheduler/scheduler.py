"""Flexible batching scheduler for Moondream text inference."""

from __future__ import annotations

from typing import List, Optional

import time

import pyvips
import torch
from torch import Tensor

from kestrel.moondream.runtime import MoondreamRuntime, TextToken, Token
from kestrel.moondream.image_crops import OverlapCropOutput
from kestrel.skills import QuerySkill, SkillRegistry, SkillSpec, SkillState

from .queues import RequestQueue, RunningQueue
from .types import (
    GenerationRequest,
    RequestMetrics,
    ScheduledSequence,
    SchedulerResult,
    StreamCallback,
)
from .sampling import sample_tokens


class GenerationScheduler:
    """Batched prefill+decode driver that mirrors flex-nano-vllm semantics."""

    def __init__(
        self,
        runtime: MoondreamRuntime,
        *,
        default_temperature: float = 0.2,
        default_top_p: float = 0.9,
        skill_registry: Optional[SkillRegistry] = None,
    ) -> None:
        self.runtime = runtime
        self.waiting: RequestQueue[GenerationRequest] = RequestQueue()
        self.running: RunningQueue[ScheduledSequence] = RunningQueue()
        self.completed: list[ScheduledSequence] = []
        self._next_request_id = 0
        self._default_temperature = max(float(default_temperature), 0.0)
        self._default_top_p = float(default_top_p)
        if not (0.0 < self._default_top_p <= 1.0):
            raise ValueError("default_top_p must be in the range (0, 1]")
        self._skills = skill_registry or SkillRegistry([QuerySkill()])

    # ------------------------------------------------------------------
    # Submission

    def enqueue_request(
        self,
        request: GenerationRequest,
        skill_state: SkillState,
    ) -> None:
        """Insert a fully constructed request/skill state into the waiting queue."""

        if request.skill_state is not None and request.skill_state is not skill_state:
            raise ValueError("GenerationRequest already has an associated SkillState")
        request.skill_state = skill_state
        self.waiting.push(request)

    # ------------------------------------------------------------------
    # Execution

    def run(self) -> List[SchedulerResult]:
        """Process queued requests until completion and return their outputs."""

        while len(self.waiting) or len(self.running):
            progressed = False
            progressed |= self._try_prefill()
            progressed |= self._decode_step()

            if not progressed:
                stalled = self.waiting.peek()
                if stalled is not None and not self.runtime.can_reserve(
                    stalled.target_length
                ):
                    raise RuntimeError(
                        "Scheduler stalled: insufficient KV cache capacity for request "
                        f"{stalled.request_id} (needs {stalled.target_length} tokens)."
                    )
                # No work left that we can process right now.
                break

        return [self._build_result(seq) for seq in self.completed]

    # ------------------------------------------------------------------
    # Internal helpers

    def _issue_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _try_prefill(self) -> bool:
        progress = False
        while len(self.waiting) and len(self.running) < self.runtime.max_batch_size:
            request = self.waiting.peek()
            if request is None:
                break
            if not self.runtime.can_reserve(request.target_length):
                break

            request = self.waiting.pop()
            tokens = request.prompt_tokens.view(1, -1).to(
                device=self.runtime.device, dtype=torch.long
            )
            prefill_start = time.perf_counter()
            state, logits = self.runtime.start_sequence(
                prompt_tokens=tokens,
                image=request.image,
                image_crops=request.image_crops,
                max_new_tokens=request.max_new_tokens,
                adapter=request.adapter,
            )
            skill_state = request.skill_state
            if skill_state is None:
                skill_state = request.skill.create_state(
                    self.runtime, request, request.request_context
                )
            request.skill_state = skill_state
            seq = ScheduledSequence(
                request=request,
                state=state,
                skill_state=skill_state,
            )
            seq.skill_state.on_prefill(self.runtime)
            seq.started_at = request.submitted_at
            seq.prefill_started_at = prefill_start

            if request.max_new_tokens <= 0:
                seq.first_token_time = prefill_start
                self._finalize_sequence(seq, "length")
                progress = True
                continue

            first_logits = logits.squeeze(0)
            sampled = self._sample_batch(first_logits.unsqueeze(0), [seq.request])
            token_value = int(sampled[0])
            seq.stage_token(self.runtime, token_value)

            if self._mark_finished_if_needed(seq):
                progress = True
                continue

            self.running.push(seq)
            progress = True
        return progress

    def _decode_step(self) -> bool:
        if not len(self.running):
            return False

        active: list[ScheduledSequence] = []
        idle: list[ScheduledSequence] = []
        for seq in self.running.take_all():
            if seq.state.at_capacity():
                self._finalize_sequence(seq, "length")
                continue
            if seq.needs_decode():
                active.append(seq)
            else:
                idle.append(seq)

        if not active:
            self.running.extend(idle)
            return False

        pending_tokens: list[Token] = []
        all_text = True
        for seq in active:
            token = seq.pending_token
            if token is None:
                raise RuntimeError(
                    "ScheduledSequence has no pending token during decode"
                )
            pending_tokens.append(token)
            if not isinstance(token, TextToken):
                all_text = False

        if all_text:
            token_input: Tensor | Sequence[Token] = torch.tensor(
                [token.token_id for token in pending_tokens], dtype=torch.long
            )
        else:
            token_input = pending_tokens

        logits = self.runtime.decode_batch([seq.state for seq in active], token_input)

        sampled_tokens = self._sample_batch(logits, [seq.request for seq in active])
        sampled_cpu = sampled_tokens.cpu().tolist()

        for seq, row, token_value in zip(active, logits, sampled_cpu):
            seq.state.advance()
            seq.stage_token(self.runtime, int(token_value))
            if not self._mark_finished_if_needed(seq):
                idle.append(seq)

        self.running.extend(idle)
        return True

    def _sample_batch(
        self, logits: Tensor, requests: List[GenerationRequest]
    ) -> Tensor:
        batch = len(requests)
        if batch == 0:
            return torch.empty(0, dtype=torch.long, device=logits.device)

        allowed_tokens: list[Optional[Sequence[int]]] = []
        restrict = False
        for req in requests:
            state = req.skill_state
            if state is None:
                allowed_tokens.append(None)
                continue
            allowed = state.allowed_token_ids(self.runtime)
            allowed_tokens.append(allowed)
            if allowed:
                restrict = True

        if restrict:
            logits = logits.clone()
            for i, allowed in enumerate(allowed_tokens):
                if not allowed:
                    continue
                idx = torch.tensor(allowed, device=logits.device, dtype=torch.long)
                row = logits[i]
                pruned = torch.full_like(row, float("-inf"))
                pruned[idx] = row[idx]
                logits[i] = pruned

        if all(req.temperature <= 0.0 for req in requests):
            return torch.argmax(logits, dim=-1)

        temps = torch.empty(batch, dtype=torch.float32)
        top_ps = torch.empty(batch, dtype=torch.float32)
        for i, req in enumerate(requests):
            temps[i] = req.temperature
            top_ps[i] = req.top_p
        temps = temps.to(device=logits.device)
        top_ps = top_ps.to(device=logits.device)
        return sample_tokens(logits, temps, top_ps)

    def _mark_finished_if_needed(self, seq: ScheduledSequence) -> bool:
        last_token = seq.last_token
        eos_id = self.runtime.config.tokenizer.eos_id
        eos_hit = isinstance(last_token, TextToken) and last_token.token_id == eos_id
        max_new_hit = seq.skill_state.token_count >= seq.request.max_new_tokens
        max_len_hit = seq.total_length >= seq.state.max_length

        if not (eos_hit or max_new_hit or max_len_hit):
            return False

        if eos_hit:
            reason = "stop"
        else:
            reason = "length"

        self._finalize_sequence(seq, reason)
        return True

    def _finalize_sequence(self, seq: ScheduledSequence, reason: str) -> None:
        if seq.finished:
            return
        seq.finished = True
        seq.finish_reason = reason
        seq.completed_at = time.perf_counter()
        if seq.first_token_time is None:
            seq.first_token_time = seq.completed_at
        if seq.state.batch_idx in self.runtime.active_sequences:
            self.runtime.release_sequence(seq.state)
        self.completed.append(seq)

    def _resolve_temperature(self, temperature: Optional[float]) -> float:
        if temperature is None:
            return self._default_temperature
        return max(float(temperature), 0.0)

    def _resolve_top_p(self, top_p: Optional[float]) -> float:
        value = self._default_top_p if top_p is None else float(top_p)
        if value <= 0.0 or value > 1.0:
            raise ValueError("top_p must be in the range (0, 1]")
        return value

    def _build_result(self, seq: ScheduledSequence) -> SchedulerResult:
        finish_reason = seq.finish_reason or "unknown"
        finalize = seq.skill_state.finalize(
            self.runtime, reason=finish_reason
        )
        prompt_tokens = seq.state.prompt_length
        decode_tokens = len(finalize.tokens)
        queued_at = seq.request.submitted_at
        prefill_started_at = seq.prefill_started_at or queued_at
        completed_at = seq.completed_at or time.perf_counter()
        first_token_time = seq.first_token_time or completed_at
        prefill_time_ms = max((first_token_time - prefill_started_at) * 1000.0, 0.0)
        ttft_ms = max((first_token_time - queued_at) * 1000.0, 0.0)
        decode_time_ms = max((completed_at - first_token_time) * 1000.0, 0.0)
        metrics = RequestMetrics(
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
            prefill_time_ms=prefill_time_ms,
            ttft_ms=ttft_ms,
            decode_time_ms=decode_time_ms,
        )
        return SchedulerResult(
            request_id=seq.request.request_id,
            tokens=finalize.tokens,
            finish_reason=finish_reason,
            metrics=metrics,
            output=finalize.output,
        )
