"""Flexible batching scheduler for Moondream text inference."""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torch import Tensor

from kestrel.models import MoondreamTextRuntime

from .queues import RequestQueue, RunningQueue
from .types import GenerationRequest, ScheduledSequence, SchedulerResult


class GenerationScheduler:
    """Batched prefill+decode driver that mirrors flex-nano-vllm semantics."""

    def __init__(self, runtime: MoondreamTextRuntime, *, greedy: bool = True) -> None:
        self.runtime = runtime
        self.waiting: RequestQueue[GenerationRequest] = RequestQueue()
        self.running: RunningQueue[ScheduledSequence] = RunningQueue()
        self.completed: list[ScheduledSequence] = []
        self.greedy = greedy
        self._next_request_id = 0

    # ------------------------------------------------------------------
    # Submission

    def submit(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        prompt_tokens: Optional[Tensor] = None,
        request_id: Optional[int] = None,
    ) -> int:
        """Queue a new prompt for generation."""

        if prompt_tokens is None:
            prompt_tokens = self.runtime.build_prompt_tokens(prompt)
        request = GenerationRequest(
            request_id=request_id if request_id is not None else self._issue_request_id(),
            prompt=prompt,
            prompt_tokens=prompt_tokens.detach().clone().to("cpu"),
            max_new_tokens=max_new_tokens,
        )
        self.waiting.push(request)
        return request.request_id

    def submit_many(
        self,
        prompts: Iterable[str],
        *,
        max_new_tokens: int,
    ) -> List[int]:
        ids: List[int] = []
        for prompt in prompts:
            ids.append(self.submit(prompt, max_new_tokens=max_new_tokens))
        return ids

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
                if stalled is not None and not self.runtime.can_reserve(stalled.target_length):
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
        while (
            len(self.waiting)
            and len(self.running) < self.runtime.max_batch_size
        ):
            request = self.waiting.peek()
            if request is None:
                break
            if not self.runtime.can_reserve(request.target_length):
                break

            request = self.waiting.pop()
            tokens = request.prompt_tokens.view(1, -1).to(
                device=self.runtime.device, dtype=torch.long
            )
            state, logits = self.runtime.start_sequence(
                prompt_tokens=tokens, max_new_tokens=request.max_new_tokens
            )
            seq = ScheduledSequence(request=request, state=state)

            if request.max_new_tokens <= 0:
                self._finalize_sequence(seq, "max_new_tokens")
                progress = True
                continue

            first_logits = logits.squeeze(0)
            next_token = self._select_next_token(first_logits)
            seq.stage_token(next_token, first_logits)

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

        token_ids = torch.tensor(
            [seq.pending_token for seq in active],
            dtype=torch.long,
        )
        logits = self.runtime.decode_batch([seq.state for seq in active], token_ids)

        for seq, row in zip(active, logits):
            seq.state.advance()
            next_token = self._select_next_token(row)
            seq.stage_token(next_token, row)
            if not self._mark_finished_if_needed(seq):
                idle.append(seq)

        self.running.extend(idle)
        return True

    def _select_next_token(self, logits: Tensor) -> int:
        if not self.greedy:
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())
        return int(torch.argmax(logits, dim=-1).item())

    def _mark_finished_if_needed(self, seq: ScheduledSequence) -> bool:
        last_token = seq.last_token
        eos_id = self.runtime.config.tokenizer.eos_id
        eos_hit = last_token == eos_id
        max_new_hit = len(seq.generated_tokens) >= seq.request.max_new_tokens
        max_len_hit = seq.total_length >= seq.state.max_length

        if not (eos_hit or max_new_hit or max_len_hit):
            return False

        if eos_hit:
            reason = "eos"
        elif max_new_hit:
            reason = "max_new_tokens"
        else:
            reason = "length"

        self._finalize_sequence(seq, reason)
        return True

    def _finalize_sequence(self, seq: ScheduledSequence, reason: str) -> None:
        if seq.finished:
            return
        seq.finished = True
        seq.finish_reason = reason
        if seq.state.batch_idx in self.runtime.active_sequences:
            self.runtime.release_sequence(seq.state)
        self.completed.append(seq)

    def _build_result(self, seq: ScheduledSequence) -> SchedulerResult:
        text = self.runtime.tokenizer.decode(seq.generated_tokens) if seq.generated_tokens else ""
        return SchedulerResult(
            request_id=seq.request.request_id,
            prompt=seq.request.prompt,
            tokens=list(seq.generated_tokens),
            text=text,
            finish_reason=seq.finish_reason or "unknown",
        )
