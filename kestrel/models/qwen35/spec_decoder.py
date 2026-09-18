"""Independent greedy DFlash sessions with native sequence verification."""

from copy import copy
from dataclasses import dataclass, replace
from typing import Any

import torch

from kestrel.runtime.spec import DraftResult, SpecStepResult
from kestrel.runtime.tokens import TextToken
from kestrel_kernels import get_runtime
from .cache import Qwen35InferenceCache
from .dflash import DFlashContextCache, load_dflash_drafter
from .gdn_state import LinearAttentionState


@dataclass
class _Session:
    state: Any
    cache: Qwen35InferenceCache
    draft_cache: DFlashContextCache
    features: torch.Tensor
    bonus: int
    failed: bool = False


class Qwen35DFlashDecoder:
    """Own independent scheduler rows and pack their target verification."""

    def __init__(self, runtime, draft_path):
        if runtime.device.type != "cuda" or runtime.dtype != torch.bfloat16:
            raise ValueError("DFlash requires CUDA BF16 sequences")
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
        self._sessions = {}

    @property
    def free_slots(self):
        return min(self.runtime.max_batch_size - len(self._sessions),
                   len(self.runtime.page_table.free_batch_idx))

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
        if any(session.state is state for session in self._sessions.values()):
            raise ValueError("speculative sequence is already admitted")
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
            # KV retains the global allocator slot; this private recurrent
            # cache owns only its row, which the first target fork clones.
            for layer in cache.layers:
                if isinstance(layer, LinearAttentionState):
                    layer.recurrent_states = layer.recurrent_states[slot:slot + 1]
            expected, features, cache = self._target(tokens, cache, slot, capture=False)
            self._sessions[slot] = _Session(state, cache, DFlashContextCache(capacity), features, expected[-1])
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
            gdn_state_indices=torch.zeros(1, device=device, dtype=torch.long),
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

    def _target_many(self, candidates, sessions):
        """Verify independent sequences together, retaining separate commit owners."""
        device = self.runtime.device
        lengths = tuple(len(tokens) for tokens in candidates)
        branches = [session.cache.fork_recurrent_state(capture_prefix=True) for session in sessions]
        packed = copy(branches[0])
        packed._prefix_records = {}
        layers = []
        for index, layer in enumerate(packed.layers):
            if isinstance(layer, LinearAttentionState):
                layer = copy(layer)
                layer.conv_states = torch.cat([branch.layers[index].conv_states for branch in branches])
                layer.recurrent_states = torch.cat([branch.layers[index].recurrent_states for branch in branches])
            layers.append(layer)
        packed.layers = tuple(layers)
        ids = torch.tensor([sum(candidates, [])], device=device, dtype=torch.long)
        positions = torch.cat([
            torch.arange(session.cache.seq_length, session.cache.seq_length + length, device=device)
            for session, length in zip(sessions, lengths)])[None]
        slot_ids = torch.tensor([session.state.batch_idx for session in sessions], device=device, dtype=torch.long)
        local_state_indices = torch.zeros_like(slot_ids)
        page_table = self.runtime.page_table.page_table.index_select(0, slot_ids)
        page_size = self.runtime.page_size
        position_rows = positions[0].split(lengths)
        slot_mapping = torch.cat([
            page_table[row, position // page_size].long() * page_size + position % page_size
            for row, position in enumerate(position_rows)])[None]
        cu, topology = get_runtime().gated_delta.bind_packed_prefill_topology(
            sequence_lengths=lengths, device=device)
        output = self.text(
            input_ids=ids, past_key_values=packed, position_ids=positions,
            cache_position_ids=positions, slot_mapping=slot_mapping, page_table=page_table,
            paged_kv_seqlens_k=torch.tensor([
                session.cache.seq_length + length for session, length in zip(sessions, lengths)
            ], device=device, dtype=torch.int32),
            cu_seq_lens_q=cu, sequence_lengths=lengths, topology_token=topology,
            seq_idx=torch.cat([torch.full((length,), row, device=device, dtype=torch.int32)
                               for row, length in enumerate(lengths)])[None],
            gdn_state_indices=torch.arange(len(sessions), device=device, dtype=torch.long),
            gdn_state_indices_allocator_owned=True, capture_layers=self.draft.config.target_layer_ids)
        features = torch.cat(output.layer_hidden_states, dim=-1).split(lengths, dim=1)
        expected = self.runtime.model.lm_head(output.last_hidden_state).argmax(-1)[0].split(lengths)
        for index, record in packed._prefix_records.items():
            records = record.split_sequences(lengths)
            for row, branch in enumerate(branches):
                branch.layers[index].conv_states.copy_(packed.layers[index].conv_states[row:row + 1])
                branch.layers[index].recurrent_states.copy_(
                    packed.layers[index].recurrent_states[row:row + 1])
                branch._prefix_records[index] = replace(records[row], state_indices=local_state_indices[row:row + 1])
        for branch, session, length in zip(branches, sessions, lengths):
            branch.advance_to(session.cache.seq_length + length)
        return [(tokens.tolist(), feature, branch)
                for tokens, feature, branch in zip(expected, features, branches)]

    def commit_accept(self, ctx):
        session, verified, features, expected, count = ctx
        cache = verified.commit_recurrent_prefix(count)
        session.cache = cache
        session.features = features[:, :count]
        session.bonus = expected[count-1]

    def step(self, states, *, allowed_token_ids=None, suppressed_token_ids=None, commit_caps=None):
        if not states:
            raise ValueError("Qwen DFlash requires admitted sequences")
        for values in (allowed_token_ids, suppressed_token_ids, commit_caps):
            if values is not None and len(values) != len(states):
                raise ValueError("per-sequence options must match the admitted sequences")
        sessions = []
        seen = set()
        for row, state in enumerate(states):
            slot = state.batch_idx
            session = self._sessions.get(slot)
            if session is None or session.state is not state or slot in seen:
                raise ValueError("Qwen DFlash requires distinct admitted sequences")
            if session.failed:
                raise RuntimeError("failed speculative sequence must be retired")
            seen.add(slot)
            self._unconstrained(None if allowed_token_ids is None else allowed_token_ids[row],
                                None if suppressed_token_ids is None else suppressed_token_ids[row])
            cap = None if commit_caps is None else commit_caps[row]
            if cap is not None and (type(cap) is not int or cap < 1):
                raise ValueError("commit cap must be a positive integer")
            sessions.append((session, cap))
        pending = []
        try:
            candidates = [[session.bonus, *self.propose(session).token_ids[0].tolist()]
                          for session, _ in sessions]
            if len(sessions) == 1:
                session = sessions[0][0]
                results = [self._target(candidates[0], session.cache, session.state.batch_idx, capture=True)]
            else:
                results = self._target_many(candidates, [session for session, _ in sessions])
            for (session, cap), candidate, (expected, features, verified) in zip(sessions, candidates, results):
                accepted = 0
                for proposed, wanted in zip(candidate[1:], expected):
                    if proposed != wanted:
                        break
                    accepted += 1
                count = min(accepted+1, cap if cap is not None else accepted+1)
                pending.append((session, verified, features, expected, count))
            for ctx in pending:
                self.commit_accept(ctx)
        except Exception:
            # Draft caches and shared KV suffixes may already have advanced.
            # Keep slots owned until scheduler retirement; retry is not safe.
            for session, _ in sessions:
                session.failed = True
            raise
        # Each input's first token was emitted by admit/the preceding step.
        return SpecStepResult(tokens=[expected[:count] for _, _, _, expected, count in pending],
                              accept_counts=[count-1 for _, _, _, _, count in pending])

    def retire(self, state):
        session = self._sessions.get(state.batch_idx)
        if session is None:
            return
        if session.state is not state:
            raise ValueError("cannot retire a different speculative sequence")
        self.runtime.page_table.erase(state.batch_idx)
        # Scheduler cleanup still uses batch_idx to remove active_sequences.
        del self._sessions[state.batch_idx]
