"""Single-sequence greedy DFlash with native target sequence verification."""

from dataclasses import dataclass
from typing import Any

import torch

from kestrel.runtime.spec import DraftResult, SpecStepResult
from kestrel.runtime.tokens import TextToken
from kestrel_kernels import get_runtime
from .cache import Qwen35InferenceCache
from .dflash import DFlashContextCache, load_dflash_drafter


@dataclass
class _Session:
    state: Any
    cache: Qwen35InferenceCache
    draft_cache: DFlashContextCache
    features: torch.Tensor
    bonus: int


class Qwen35DFlashDecoder:
    """Own one scheduler row; the scheduler owns output staging and lengths."""

    def __init__(self, runtime, draft_path):
        if runtime.max_batch_size != 1 or runtime.device.type != "cuda" or runtime.dtype != torch.bfloat16:
            raise ValueError("DFlash requires one CUDA BF16 sequence")
        self.runtime = runtime
        self.text = runtime.model.model.language_model
        self.draft = load_dflash_drafter(draft_path, device=runtime.device)
        config = self.draft.config
        target = self.text.config
        if (config.hidden_size != target.hidden_size
                or not config.target_layer_ids
                or any(i < 0 or i >= target.num_hidden_layers for i in config.target_layer_ids)
                or not 0 <= config.mask_token_id < target.vocab_size
                or config.block_size < 2):
            raise ValueError("DFlash checkpoint does not match target dimensions, taps, or vocabulary")
        self.num_speculative_tokens = config.block_size - 1
        self.num_lookahead_tokens = config.block_size
        self._session = None

    @property
    def free_slots(self):
        return int(self._session is None and bool(self.runtime.page_table.free_batch_idx))

    @staticmethod
    def _unconstrained(allowed, suppressed):
        if allowed is not None or suppressed:
            raise ValueError("Qwen DFlash currently requires unconstrained greedy text generation")

    def admit(self, state, prompt_tokens, *, image=None, image_crops=None,
              allowed_token_ids=None, suppressed_token_ids=None,
              suppress_next_token_ids=None, temperature=0.0, top_p=1.0):
        self._unconstrained(allowed_token_ids, suppressed_token_ids)
        if (image is not None or image_crops is not None or suppress_next_token_ids
                or temperature != 0.0 or not 0.0 < top_p <= 1.0
                or getattr(state, "return_logprobs", False) or getattr(state, "lora_slot", 0)):
            raise ValueError("Qwen DFlash supports greedy text without images, adapters, masks, or logprobs")
        if not prompt_tokens or any(not isinstance(token, TextToken) for token in prompt_tokens):
            raise ValueError("Qwen DFlash requires a nonempty text-token prompt")
        capacity = int(state.max_length) + self.num_lookahead_tokens
        if capacity > self.runtime.max_seq_length:
            raise ValueError("request plus speculative lookahead exceeds model context")
        if not self.free_slots:
            raise RuntimeError("Qwen DFlash sequence slot is occupied")
        tokens = [int(token.token_id) for token in prompt_tokens]
        if any(token < 0 or token >= self.runtime.vocab_size for token in tokens):
            raise ValueError("prompt token is outside the target vocabulary")
        pages = self.runtime.page_table
        slot = pages.allocate()
        state.batch_idx = slot
        try:
            pages.reserve(slot, capacity)
            pages.commit_block_table([slot])
            cache = Qwen35InferenceCache(config=self.text.config, paged_kv=self.runtime._paged_kv)
            self.runtime._linear_state_pool.bind_prefill_state(cache)
            expected, features, cache = self._target(tokens, cache, slot, capture=False)
            self._session = _Session(state, cache, DFlashContextCache(capacity), features, expected[-1])
            return expected[-1], None
        except Exception:
            pages.erase(slot)
            state.batch_idx = -1
            raise

    def _target(self, tokens, committed, slot, *, capture):
        cache = committed.fork_recurrent_state(capture_prefix=capture)
        device = self.runtime.device
        start, length = committed.seq_length, len(tokens)
        ids = torch.tensor([tokens], device=device, dtype=torch.long)
        positions = torch.arange(start, start+length, device=device)[None]
        page_row = self.runtime.page_table.page_table[slot:slot+1]
        page_size = self.runtime.page_size
        slots = page_row[0, positions // page_size].long() * page_size + positions % page_size
        cu, topology = get_runtime().gated_delta.bind_packed_prefill_topology(
            sequence_lengths=(length,), device=device)
        output = self.text(
            input_ids=ids, past_key_values=cache, position_ids=positions,
            cache_position_ids=positions, slot_mapping=slots, page_table=page_row,
            paged_kv_seqlens_k=torch.tensor([start+length], device=device, dtype=torch.int32),
            cu_seq_lens_q=cu, sequence_lengths=(length,), topology_token=topology,
            seq_idx=torch.zeros((1, length), device=device, dtype=torch.int32),
            gdn_state_indices=torch.tensor([slot], device=device, dtype=torch.long),
            gdn_state_indices_allocator_owned=True, capture_layers=self.draft.config.target_layer_ids)
        cache.advance_to(start+length)
        features = torch.cat(output.layer_hidden_states, dim=-1)
        expected = self.runtime.model.lm_head(output.last_hidden_state).argmax(-1)[0].tolist()
        return expected, features, cache

    def propose(self, ctx):
        config = self.draft.config
        start = ctx.cache.seq_length
        noise = torch.full((1, config.block_size), config.mask_token_id,
                           device=self.runtime.device, dtype=torch.long)
        noise[0, 0] = ctx.bonus
        positions = torch.arange(ctx.draft_cache.length, start+config.block_size,
                                 device=self.runtime.device)[None]
        hidden = self.draft(self.text.embed_tokens(noise), ctx.features, positions,
                            context_cache=ctx.draft_cache)
        ids = self.runtime.model.lm_head(hidden[:, 1:]).argmax(-1).to(torch.int32)
        return DraftResult(token_ids=ids)

    def commit_accept(self, ctx):
        verified, features, expected, count = ctx
        session = self._session
        cache = verified.commit_recurrent_prefix(count)
        session.cache = cache
        session.features = features[:, :count]
        session.bonus = expected[count-1]

    def step(self, states, *, allowed_token_ids=None, suppressed_token_ids=None, commit_caps=None):
        session = self._session
        if len(states) != 1 or session is None or states[0] is not session.state:
            raise ValueError("Qwen DFlash requires its single admitted sequence")
        for values in (allowed_token_ids, suppressed_token_ids, commit_caps):
            if values is not None and len(values) != 1:
                raise ValueError("per-sequence options must match the admitted sequence")
        self._unconstrained(None if allowed_token_ids is None else allowed_token_ids[0],
                            None if suppressed_token_ids is None else suppressed_token_ids[0])
        cap = None if commit_caps is None else commit_caps[0]
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ValueError("commit cap must be a positive integer")
        candidate = [session.bonus, *self.propose(session).token_ids[0].tolist()]
        expected, features, verified = self._target(
            candidate, session.cache, session.state.batch_idx, capture=True)
        accepted = 0
        for proposed, wanted in zip(candidate[1:], expected):
            if proposed != wanted:
                break
            accepted += 1
        count = min(accepted+1, cap if cap is not None else accepted+1)
        self.commit_accept((verified, features, expected, count))
        # The input's first token was emitted by admit/the preceding step.
        # Outputs are accepted drafts followed by the replacement/bonus token.
        return SpecStepResult(tokens=[expected[:count]], accept_counts=[count-1])

    def retire(self, state):
        session = self._session
        if session is None:
            return
        if session.state is not state:
            raise ValueError("cannot retire a different speculative sequence")
        self.runtime.page_table.erase(state.batch_idx)
        # Scheduler cleanup still uses batch_idx to remove active_sequences.
        self._session = None
