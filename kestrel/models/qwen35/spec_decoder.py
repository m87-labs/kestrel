"""Independent greedy DFlash sessions with automatic target verification."""

from dataclasses import dataclass, replace
from contextlib import ExitStack, contextmanager, nullcontext
from collections import Counter, OrderedDict
from typing import Any
import time

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
    spare_cache: Qwen35InferenceCache | None = None


class Qwen35DFlashDecoder:
    """Own independent scheduler rows and pack their target verification."""

    _native_projections = None

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
                or (config.selector_rank and config.vocab_size != target.vocab_size)
                or config.block_size < 2):
            raise ValueError("DFlash checkpoint does not match target dimensions, taps, or vocabulary")
        self._native_verification_width = getattr(runtime._cfg, "verification_width", None)
        if self._native_verification_width is not None and (
                type(self._native_verification_width) is not int
                or not 1 <= self._native_verification_width <= config.block_size
                or runtime._cfg.verification_path != "native"):
            raise ValueError("native verification width must fit the draft block")
        self.num_speculative_tokens = config.block_size - 1
        self.num_lookahead_tokens = config.block_size
        self._sessions = {}
        self._verification_layouts = OrderedDict()
        self._query_metadata_workspaces = OrderedDict()
        self._target_graph = None
        self._replay_graph = None
        self._draft_graph = None
        from .draft_owner_cache import DraftOwnerCache
        self._draft_owners = DraftOwnerCache(
            max_entries=runtime._cfg.draft_graph_cache_entries,
            known_tensor_bytes=runtime._cfg.draft_graph_known_tensor_bytes)
        self._draft_graph_enabled = runtime._cfg.enable_cuda_graphs
        self._graph_failed = False
        self._closed = False
        self._commit_stream = torch.cuda.Stream(device=runtime.device)
        self._commit_inputs_ready = torch.cuda.Event()
        self._commit_ready = torch.cuda.Event()
        self._commit_pending = False
        self._generated_verification = None
        self._width_policy = None
        self._draft_confidence = None
        self._batch_width_policies = {}
        self._batch_span_verifiers = {}
        self._batch_draft_confidence = {}
        self._draft_timing_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self._candidate_buffers = {}
        from .generated_verification import Qwen35GeneratedVerification

        verification_path = runtime._cfg.verification_path
        if verification_path == "auto":
            verification_path = runtime.decode_path
        if (verification_path == "generated" or
                (verification_path == "auto" and
                 Qwen35GeneratedVerification.supports(runtime, self.draft))):
            self._generated_verification = Qwen35GeneratedVerification(runtime, self.draft)
            self._generated_verifiers = {1: self._generated_verification}
            for sequences in range(2, runtime.max_batch_size + 1):
                self._generated_verifiers[sequences] = Qwen35GeneratedVerification(
                    runtime, self.draft, sequences=sequences,
                    weights=self._generated_verification.weights,
                    **({"verification_rows": {2: 8, 3: 2, 4: 4}.get(sequences, 2)}
                       if self._generated_verification.segmented else {}))
        if (self._generated_verification is not None
                and self._generated_verification.segmented and self._draft_graph_enabled
                and self.draft.candidate_selector is None):
            from .verification_policy import VerificationWidthPolicy
            self._width_policy = VerificationWidthPolicy()
            self._span_verifiers = {config.block_size: self._generated_verification}
            for width in self._width_policy.widths:
                if width not in self._span_verifiers:
                    self._span_verifiers[width] = Qwen35GeneratedVerification(
                        runtime, self.draft, weights=self._generated_verification.weights,
                        verification_rows=width)
            self._draft_confidence = torch.empty((1, config.block_size-1),
                                                dtype=torch.float32, device=runtime.device)
            for sequences, widths in ((2, (2, 4, 8)), (4, (2, 4))):
                if sequences > runtime.max_batch_size:
                    continue
                self._batch_width_policies[sequences] = VerificationWidthPolicy(widths)
                verifiers = {widths[-1]: self._generated_verifiers[sequences]}
                for width in widths[:-1]:
                    verifiers[width] = Qwen35GeneratedVerification(
                        runtime, self.draft, sequences=sequences,
                        weights=self._generated_verification.weights, verification_rows=width)
                self._batch_span_verifiers[sequences] = verifiers
                self._batch_draft_confidence[sequences] = torch.empty(
                    (sequences, widths[-1] - 1), dtype=torch.float32, device=runtime.device)
        if runtime._cfg.verification_path == "native":
            from .verification_linear import NativeVerificationProjections
            self._native_projections = NativeVerificationProjections(
                (*self.text.modules(), runtime.model.lm_head),
                max_verification_tokens=config.block_size * runtime.max_batch_size,
                attention_splits=runtime._cfg.verification_attention_splits,
                attention_implementation=runtime._cfg.verification_attention_implementation,
                projection_path=runtime._cfg.verification_projection_path)
        try:
            if runtime._cfg.enable_cuda_graphs and self._generated_verification is None:
                from .spec_target_graph import Qwen35TargetGraph
                from .spec_replay_graph import Qwen35ReplayGraph
                self._target_graph = Qwen35TargetGraph(runtime, self.text,
                    config.target_layer_ids, config.block_size, finalize_stream=self._commit_stream)
                self._replay_graph = Qwen35ReplayGraph(runtime, config.block_size,
                    target.num_hidden_layers)
        except BaseException:
            self.shutdown()
            raise

    def _verify(self, leases, *, state_sources=None, last_token_only=False,
                verification=False, **kwargs):
        if leases is not None and self._target_graph is not None and not verification:
            raise ValueError('target graph requires an explicit verification scope')
        scope = (self._native_projections.activate()
                 if verification and self._native_projections is not None else nullcontext())
        with scope:
            if leases is None or self._target_graph is None:
                if state_sources is not None:
                    raise RuntimeError("unmaterialized recurrent rows require a verification graph")
                output = self.text(**kwargs)
                hidden = output.last_hidden_state
                if last_token_only:
                    hidden = hidden[:, -1:]
                return output, self.runtime.model.lm_head(hidden).argmax(-1)
            return leases.enter_context(self._target_graph.launch(state_sources=state_sources, **kwargs))

    def shutdown(self):
        self._closed = True
        try:
            self._commit_stream.synchronize()
            if self._target_graph is not None:
                self._target_graph.shutdown()
            self._query_metadata_workspaces.clear()
        finally:
            try:
                if self._replay_graph is not None:
                    self._replay_graph.shutdown()
            finally:
                try:
                    self._draft_owners.shutdown()
                    self._draft_graph = None
                finally:
                    if self._native_projections is not None:
                        self._native_projections.close()

    @property
    def free_slots(self):
        return min(self.runtime.max_batch_size - len(self._sessions),
                   len(self.runtime.page_table.free_batch_idx))

    @staticmethod
    def _unconstrained(allowed, suppressed):
        if allowed is not None or suppressed:
            raise ValueError("Qwen DFlash currently requires unconstrained greedy text generation")

    def _prepare_admission(self, state, prompt_tokens, *, image=None, image_crops=None,
              allowed_token_ids=None, suppressed_token_ids=None,
              suppress_next_token_ids=None, temperature=0.0, top_p=1.0):
        if self._closed or self._graph_failed:
            raise RuntimeError('speculative decoder is shut down or its verification graph failed')
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
        if not self._sessions and self._width_policy is not None:
            self._width_policy.reset_request()
            for policy in self._batch_width_policies.values():
                policy.reset_request()
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
            return slot, capacity, tokens, cache
        except Exception:
            pages.erase(slot)
            state.batch_idx = -1
            raise

    def admit(self, state, prompt_tokens, **options):
        slot, capacity, tokens, cache = self._prepare_admission(state, prompt_tokens, **options)
        try:
            expected, features, cache = self._target(tokens, cache, slot, capture=False)
            self._sessions[slot] = _Session(state, cache, DFlashContextCache(capacity), features,
                expected[-1], spare_cache=cache.fork_recurrent_state())
            return expected[-1], None
        except Exception:
            self.runtime.page_table.erase(slot)
            state.batch_idx = -1
            raise

    def admit_many(self, requests):
        """Prefill independent new sessions together; report errors per request."""
        if len(requests) == 1:
            from kestrel.runtime.spec import admit_independently
            return admit_independently(self, requests)
        prepared, results = [], [None] * len(requests)
        counts = Counter(id(request.state) for request in requests)
        for index, request in enumerate(requests):
            try:
                if counts[id(request.state)] != 1:
                    raise ValueError("duplicate speculative admission state")
                if len(prepared) >= self.runtime.max_batch_size - len(self._sessions):
                    raise RuntimeError("Qwen DFlash sequence slot is occupied")
                values = self._prepare_admission(request.state, request.prompt_tokens, **request.options)
                prepared.append((index, request.state, *values))
            except Exception as error:
                results[index] = error
        if not prepared:
            return results
        try:
            outputs = self._target_prefill_many(
                [item[4] for item in prepared], [item[5] for item in prepared],
                [item[2] for item in prepared])
            sessions = [
                _Session(state, cache, DFlashContextCache(capacity), features, token,
                         spare_cache=cache.fork_recurrent_state())
                for (_, state, _, capacity, _, _), (token, features, cache)
                in zip(prepared, outputs, strict=True)]
        except Exception as error:
            for index, state, slot, *_ in prepared:
                self.runtime.page_table.erase(slot)
                state.batch_idx = -1
                results[index] = error
            return results
        for (index, _, slot, *_), session in zip(prepared, sessions, strict=True):
            self._sessions[slot] = session
            results[index] = session.bonus, None
        return results

    def _target_prefill_many(self, prompts, caches, slots):
        self._wait_for_commit()
        lengths = tuple(map(len, prompts))
        device = self.runtime.device
        packed = Qwen35InferenceCache(config=self.text.config, paged_kv=self.runtime._paged_kv)
        for layer, source in zip(packed.layers, caches[0].layers, strict=True):
            if isinstance(layer, LinearAttentionState):
                state = source.recurrent_states
                layer.recurrent_states = torch.empty(
                    (len(prompts), *state.shape[1:]), device=state.device, dtype=state.dtype)
        ids = torch.tensor([[token for prompt in prompts for token in prompt]],
                           device=device, dtype=torch.long)
        positions = torch.cat([torch.arange(length, device=device) for length in lengths])[None]
        slot_ids = torch.tensor(slots, device=device, dtype=torch.long)
        page_table = self.runtime.page_table.page_table.index_select(0, slot_ids)
        page_size = self.runtime.page_size
        slot_mapping = torch.cat([
            page_table[row, position // page_size].long() * page_size + position % page_size
            for row, position in enumerate(positions[0].split(lengths))])[None]
        cu, topology = get_runtime().gated_delta.bind_packed_prefill_topology(
            sequence_lengths=lengths, device=device)
        output = self.text(
            input_ids=ids, past_key_values=packed, position_ids=positions,
            cache_position_ids=positions, slot_mapping=slot_mapping, page_table=page_table,
            paged_kv_seqlens_k=torch.tensor(lengths, device=device, dtype=torch.int32),
            cu_seq_lens_q=cu, sequence_lengths=lengths, topology_token=topology,
            seq_idx=torch.cat([torch.full((length,), row, device=device, dtype=torch.int32)
                               for row, length in enumerate(lengths)])[None],
            gdn_state_indices=torch.arange(len(prompts), device=device, dtype=torch.long),
            gdn_state_indices_allocator_owned=True, capture_layers=self.draft.config.target_layer_ids)
        features = torch.cat(output.layer_hidden_states, dim=-1).split(lengths, dim=1)
        ends = cu[1:].long() - 1
        last = output.last_hidden_state.index_select(1, ends)
        tokens = self.runtime.model.lm_head(last).argmax(-1)[0].tolist()
        for row, (cache, length) in enumerate(zip(caches, lengths, strict=True)):
            for owned, layer in zip(cache.layers, packed.layers, strict=True):
                if isinstance(layer, LinearAttentionState):
                    owned.conv_states = layer.conv_states[row:row + 1]
                    owned.recurrent_states = layer.recurrent_states[row:row + 1]
                    owned.has_previous_state = layer.has_previous_state
            cache.advance_to(length)
        return list(zip(tokens, features, caches, strict=True))

    def _target(self, tokens, committed, slot, *, capture, leases=None):
        self._wait_for_commit()
        if capture and self._generated_verification is not None:
            if self._width_policy is not None:
                if not self._width_policy.calibrated:
                    if len(tokens) != self.draft.config.block_size:
                        raise ValueError("verification calibration requires the full draft proposal")
                    samples = {width: [] for width in self._width_policy.widths}
                    for repeat in range(5):
                        order = self._width_policy.widths[::1 if repeat % 2 == 0 else -1]
                        for width in order:
                            samples[width].append(self._span_verifiers[width].measure_replay(
                                tokens[:width].unsqueeze(0), [committed], [slot]))
                    self._width_policy.set_target_samples(samples)
                self._generated_verification = self._span_verifiers[len(tokens)]
            else:
                self._generated_verification = self._generated_verifiers[1]
            return self._generated_verification.target(tokens, committed, slot)
        graph_staging = capture and leases is not None and self._target_graph is not None
        if graph_staging:
            cache, _ = Qwen35InferenceCache.fork_packed_recurrent_state(
                [committed], materialize=False)
        else:
            cache = committed.fork_recurrent_state(capture_prefix=capture)
        device = self.runtime.device
        start, length = committed.seq_length, len(tokens)
        ids = torch.tensor([tokens], device=device, dtype=torch.long)
        from .query_metadata import build_query_metadata
        metadata = build_query_metadata(self, (length,), (slot,), (start,)) if capture else None
        if metadata is None:
            positions = torch.arange(start, start+length, device=device)[None]
            page_row = self.runtime.page_table.page_table[slot:slot+1]
            page_size = self.runtime.page_size
            slots = page_row[0, positions // page_size].long() * page_size + positions % page_size
            used = torch.tensor([start+length], device=device, dtype=torch.int32)
        else:
            positions, page_row, slots, used = metadata.positions, metadata.pages, metadata.slots, metadata.used
        cu, topology = get_runtime().gated_delta.bind_packed_prefill_topology(
            sequence_lengths=(length,), device=device)
        output, predictions = self._verify(leases, last_token_only=not capture, verification=capture,
            state_sources=[committed] if graph_staging else None,
            input_ids=ids, past_key_values=cache, position_ids=positions,
            cache_position_ids=positions, slot_mapping=slots, page_table=page_row,
            paged_kv_seqlens_k=used,
            cu_seq_lens_q=cu, sequence_lengths=(length,), topology_token=topology,
            seq_idx=(metadata.seq_idx if metadata is not None else
                     torch.zeros((1, length), device=device, dtype=torch.int32)),
            gdn_state_indices=torch.zeros(1, device=device, dtype=torch.long),
            gdn_state_indices_allocator_owned=True, capture_layers=self.draft.config.target_layer_ids)
        cache.advance_to(start+length)
        features = torch.cat(output.layer_hidden_states, dim=-1)
        expected = predictions[0].tolist()
        return expected, features, cache

    @contextmanager
    def _draft_tokens(self, noise, features, positions, caches, anchors=None):
        confidence = (getattr(self, "_draft_confidence", None) if len(caches) == 1 else
                      getattr(self, "_batch_draft_confidence", {}).get(len(caches)))
        proposal_count = (self.draft.config.block_size - 1 if confidence is None
                          else confidence.shape[-1])
        if (confidence is None and self._generated_verification is not None
                and self.draft.candidate_selector is None):
            proposal_count = self._generated_verifiers[len(caches)].block - 1
        eligible = (self._draft_graph_enabled
                    and all(cache.layers for cache in caches)
                    and all(value.shape[1] <= self.draft.config.block_size for value in features))
        if not eligible:
            if len(caches) == 1:
                hidden = self.draft(noise[0], features[0], positions[0], context_cache=caches[0])
            else:
                hidden = torch.cat(self.draft.forward_many(
                    noise, features, positions, context_caches=caches), dim=0)
            hidden = hidden[:, :proposal_count + 1]
            if confidence is None:
                yield self.draft.select_tokens(hidden, self.runtime.model.lm_head, anchors)
            else:
                yield self.draft.select_tokens(hidden, self.runtime.model.lm_head, anchors,
                                                confidence=confidence)
            return
        from .draft_workspace import DFlashDraftGraphSession
        parameter = next(self.draft.parameters())
        confidence_key = None if confidence is None else (
            id(confidence), tuple(confidence.shape), tuple(confidence.stride()),
            confidence.dtype, confidence.device)
        key = (len(caches), proposal_count, max(cache.capacity for cache in caches),
               parameter.device, parameter.dtype, self.draft.config.block_size, confidence_key)
        reservation = DFlashDraftGraphSession.known_tensor_reservation(
            self.draft, caches, proposal_count, projected=self.runtime.model.lm_head is not None)
        previous_owner = self._draft_graph
        self._draft_graph = None
        with self._draft_owners.lease(key, reservation, lambda: DFlashDraftGraphSession(
                self.draft, caches, lm_head=self.runtime.model.lm_head,
                confidence=confidence, proposal_count=proposal_count),
                compatible=lambda candidate: (candidate[:2] == key[:2]
                    and candidate[3:] == key[3:] and candidate[2] >= key[2])) as owner:
            if (owner is not previous_owner
                    or any(left is not right for left, right in zip(owner.caches, caches))
                    or owner.lengths != tuple(cache.length for cache in caches)):
                try:
                    owner.rebind(caches)
                except BaseException:
                    owner.failed = True
                    raise
            self._draft_graph = owner
            try:
                with owner.launch(noise, features, positions, anchors) as tokens:
                    self._draft_owners.validate_retained_bytes(owner)
                    yield tokens
            finally:
                if owner.failed or owner.closed:
                    self._draft_graph = None

    @contextmanager
    def _pin_draft_owner(self, owner):
        try:
            with self._draft_owners.pin(owner):
                yield
        finally:
            if owner.failed or owner.closed:
                self._draft_graph = None

    def propose(self, ctx):
        if self._width_policy is None:
            result = self._propose_fixed(ctx)
            width = getattr(self, '_native_verification_width', None)
            return (result if width is None else
                    DraftResult(token_ids=result.token_ids[:, :width-1]))
        warmed = (self._draft_graph is not None and len(self._draft_graph.caches) == 1
                  and bool(ctx.draft_cache.layers)
                  and ctx.draft_cache.capacity <= self._draft_graph.workspaces[0].capacity
                  and self._draft_graph.graph.has_staged_entry)
        self._last_draft_warm = warmed
        begin, end = self._draft_timing_events
        begin.record()
        result = self._propose_fixed(ctx)
        end.record()
        probabilities = self._draft_confidence[0].tolist()
        # This blocking D2H copy follows the timing event on the current stream.
        if warmed:
            self._width_policy.record_draft(begin.elapsed_time(end))
        width = self._width_policy.choose(probabilities)
        return DraftResult(token_ids=result.token_ids[:, :width-1])

    def _propose_fixed(self, ctx):
        config = self.draft.config
        start = ctx.cache.seq_length
        session = self._draft_graph
        if (session is not None and not session.closed and not session.failed and len(session.caches) == 1
                and session.caches[0] is ctx.draft_cache
                and session.lengths == (ctx.draft_cache.length,)
                and ctx.features.shape[1] <= session.context_rows
                and session.graph.has_staged_entry):
            with self._pin_draft_owner(session), session.launch_tokens(ctx.features, self.text.embed_tokens.weight,
                                       bonus=ctx.bonus, target_position=start) as ids:
                pass
            return DraftResult(token_ids=ids)
        noise = torch.full((1, config.block_size), config.mask_token_id,
                           device=self.runtime.device, dtype=torch.long)
        noise[0, 0] = ctx.bonus
        positions = torch.arange(ctx.draft_cache.length, start+config.block_size,
                                 device=self.runtime.device)[None]
        consumer_stream = torch.cuda.current_stream(self.runtime.device)
        with self._draft_tokens([self.text.embed_tokens(noise)], [ctx.features],
                                [positions], [ctx.draft_cache], noise[:, 0]) as ids:
            producer_stream = torch.cuda.current_stream(self.runtime.device)
        if producer_stream != consumer_stream:
            ids.record_stream(consumer_stream)
        return DraftResult(token_ids=ids)

    def _propose_many(self, sessions):
        config = self.draft.config
        count = len(sessions)
        policy = getattr(self, "_batch_width_policies", {}).get(count)
        graph = self._draft_graph
        warmed = (graph is not None and not graph.closed and not graph.failed and len(graph.caches) == count
                  and graph.graph.has_staged_entry
                  and all(left is right.draft_cache for left, right in zip(graph.caches, sessions))
                  and graph.lengths == tuple(session.draft_cache.length for session in sessions)
                  and all(session.features.shape[1] <= config.block_size for session in sessions))
        self._last_draft_warm = warmed
        if policy is not None:
            begin, end = self._draft_timing_events
            begin.record()
        if warmed:
            with self._pin_draft_owner(graph), graph.launch_tokens_many(
                    [session.features for session in sessions], self.text.embed_tokens.weight,
                    bonuses=[session.bonus for session in sessions],
                    target_positions=[session.cache.seq_length for session in sessions]) as tokens:
                if policy is not None:
                    end.record()
                ids = tokens.tolist()
        else:
            noise = torch.full((count, config.block_size), config.mask_token_id,
                               device=self.runtime.device, dtype=torch.long)
            noise[:, 0] = torch.tensor([session.bonus for session in sessions],
                                       device=self.runtime.device)
            positions = [torch.arange(session.draft_cache.length,
                                      session.cache.seq_length + config.block_size,
                                      device=self.runtime.device)[None]
                         for session in sessions]
            with self._draft_tokens(
                    self.text.embed_tokens(noise).split(1), [session.features for session in sessions],
                    positions, [session.draft_cache for session in sessions], noise[:, 0]) as tokens:
                if policy is not None:
                    end.record()
                ids = tokens.tolist()
        if policy is not None:
            confidence = self._batch_draft_confidence[count].flatten().tolist()
            if warmed:
                policy.record_draft(begin.elapsed_time(end))
            width = policy.choose(confidence, sequences=count)
        else:
            width = (self._generated_verifiers[count].block
                     if self._generated_verification is not None else config.block_size)
        return [[session.bonus, *row[:width - 1]] for session, row in zip(sessions, ids)]

    def _target_many(self, candidates, sessions, *, leases=None):
        if self._generated_verification is not None:
            self._wait_for_commit()
            count = len(sessions)
            committed = [session.cache for session in sessions]
            slots = [session.state.batch_idx for session in sessions]
            policy = getattr(self, "_batch_width_policies", {}).get(count)
            if policy is not None:
                verifiers = self._batch_span_verifiers[count]
                if not policy.calibrated:
                    if any(len(row) != policy.widths[-1] for row in candidates):
                        raise ValueError("batched calibration requires complete maximum-span proposals")
                    tokens = torch.tensor(candidates, device=self.runtime.device, dtype=torch.long)
                    samples = {width: [] for width in policy.widths}
                    for repeat in range(5):
                        for width in policy.widths[::1 if repeat % 2 == 0 else -1]:
                            samples[width].append(verifiers[width].measure_replay(
                                tokens[:, :width], committed, slots))
                    policy.set_target_samples(samples)
                self._generated_verification = verifiers[len(candidates[0])]
            else:
                self._generated_verification = self._generated_verifiers[count]
            return self._generated_verification.target_many(candidates, committed, slots)
        """Verify independent sequences together, retaining separate commit owners."""
        self._wait_for_commit()
        device = self.runtime.device
        lengths = tuple(len(tokens) for tokens in candidates)
        cacheable = (bool(lengths) and len(set(lengths)) == 1
                     and 1 <= lengths[0] <= self.num_lookahead_tokens
                     and len(lengths) <= self.runtime.max_batch_size)
        sources = [session.cache for session in sessions]
        graph_staging = leases is not None and self._target_graph is not None
        packed, branches = Qwen35InferenceCache.fork_packed_recurrent_state(
            sources, materialize=not graph_staging)
        ids = torch.tensor([sum(candidates, [])], device=device, dtype=torch.long)
        from .query_metadata import build_query_metadata
        metadata = build_query_metadata(self, lengths,
            [session.state.batch_idx for session in sessions],
            [session.cache.seq_length for session in sessions])
        positions, page_table, slot_mapping, used = (metadata.positions, metadata.pages,
                                                    metadata.slots, metadata.used)
        layout = self._verification_layouts.get(lengths) if cacheable else None
        if layout is None:
            cu, topology = get_runtime().gated_delta.bind_packed_prefill_topology(
                sequence_lengths=lengths, device=device)
            state_indices = torch.arange(len(sessions), device=device, dtype=torch.long)
            local_state_indices = torch.zeros_like(state_indices)
            layout = cu, topology, state_indices, local_state_indices
            if cacheable:
                if len(self._verification_layouts) >= self.runtime.max_batch_size:
                    self._verification_layouts.pop(next(iter(self._verification_layouts)))
                self._verification_layouts[lengths] = layout
        if cacheable:
            self._verification_layouts.pop(lengths)
            self._verification_layouts[lengths] = layout
        cu, topology, state_indices, local_state_indices = layout
        output, predictions = self._verify(leases, verification=True, state_sources=sources if graph_staging else None,
            input_ids=ids, past_key_values=packed, position_ids=positions,
            cache_position_ids=positions, slot_mapping=slot_mapping, page_table=page_table,
            paged_kv_seqlens_k=used,
            cu_seq_lens_q=cu, sequence_lengths=lengths, topology_token=topology,
            seq_idx=metadata.seq_idx,
            gdn_state_indices=state_indices,
            gdn_state_indices_allocator_owned=True, capture_layers=self.draft.config.target_layer_ids)
        features = torch.cat(output.layer_hidden_states, dim=-1).split(lengths, dim=1)
        if predictions.shape != (1, sum(lengths)):
            raise ValueError("packed verification predictions do not match sequence lengths")
        if packed._borrowed_recurrent_state:
            for branch in branches:
                branch._borrowed_recurrent_state = True
            for index, layer in enumerate(packed.layers):
                if isinstance(layer, LinearAttentionState):
                    for row, branch in enumerate(branches):
                        branch.layers[index].conv_states = layer.conv_states[row:row + 1]
                        branch.layers[index].recurrent_states = layer.recurrent_states[row:row + 1]
        if packed._prefix_records and all(
                record.prefix_context is not None for record in packed._prefix_records.values()):
            for branch in branches:
                branch._prefix_records = packed._prefix_records
        else:
            for index, record in packed._prefix_records.items():
                records = record.split_sequences(lengths)
                for row, branch in enumerate(branches):
                    branch._prefix_records[index] = replace(records[row], state_indices=local_state_indices[row:row + 1])
        for branch, session, length in zip(branches, sessions, lengths):
            branch.advance_to(session.cache.seq_length + length)
        # Keep the host transfer after branch setup so that work can overlap
        # verification. Transfer once instead of one copy/sync per sequence.
        flat_expected = predictions[0].tolist()
        expected = []
        offset = 0
        for length in lengths:
            expected.append(flat_expected[offset:offset + length])
            offset += length
        return [(tokens, feature, branch)
                for tokens, feature, branch in zip(expected, features, branches)]

    def commit_accept(self, ctx):
        if self._generated_verification is not None:
            # Tried separate-stream replay: C1 stayed near 396 tok/s;
            # keep the single-stream owner without additional event traffic.
            self._generated_verification.commit(ctx)
            return
        session, verified, features, expected, count = ctx
        cache = verified.commit_recurrent_prefix(count, replay_graph=self._replay_graph)
        session.cache = cache
        session.features = features[:, :count]
        session.bonus = expected[count-1]

    def step(self, states, *, allowed_token_ids=None, suppressed_token_ids=None, commit_caps=None):
        if self._closed or self._graph_failed:
            raise RuntimeError('speculative decoder is shut down or its verification graph failed')
        policy = (self._width_policy if len(states) == 1 else
                  getattr(self, "_batch_width_policies", {}).get(len(states)))
        measured = policy is not None and policy.calibrated
        self._last_draft_warm = False
        started = time.perf_counter()
        with ExitStack() as leases:
            result = self._step(states, allowed_token_ids=allowed_token_ids,
                suppressed_token_ids=suppressed_token_ids, commit_caps=commit_caps, leases=leases)
        if measured and self._last_draft_warm:
            policy.record_cycle(self._generated_verification.block,
                                           (time.perf_counter()-started)*1000)
        return result

    def _step(self, states, *, allowed_token_ids, suppressed_token_ids, commit_caps, leases):
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
            if len(sessions) == 1:
                session = sessions[0][0]
                proposed = self.propose(session).token_ids[0]
                if self._generated_verification is not None:
                    # Keep the draft on device until verification completes;
                    # acceptance needs host tokens, but verification does not.
                    from kestrel_kernels.verification_pointwise import stage_candidate
                    width = proposed.numel() + 1
                    candidate = self._candidate_buffers.get(width)
                    if candidate is None:
                        candidate = torch.empty(width, dtype=proposed.dtype, device=proposed.device)
                        self._candidate_buffers[width] = candidate
                    stage_candidate(proposed, candidate, session.bonus)
                    results = [self._target(candidate, session.cache,
                        session.state.batch_idx, capture=True, leases=leases)]
                    candidates = [candidate.tolist()]
                else:
                    candidates = [[session.bonus, *proposed.tolist()]]
                    results = [self._target(candidates[0], session.cache,
                        session.state.batch_idx, capture=True, leases=leases)]
            else:
                candidates = self._propose_many([session for session, _ in sessions])
                width = getattr(self, '_native_verification_width', None)
                if width is not None:
                    candidates = [candidate[:width] for candidate in candidates]
                results = self._target_many(candidates, [session for session, _ in sessions], leases=leases)
            for (session, cap), candidate, (expected, features, verified) in zip(sessions, candidates, results):
                accepted = 0
                for proposed, wanted in zip(candidate[1:], expected):
                    if proposed != wanted:
                        break
                    accepted += 1
                count = min(accepted+1, cap if cap is not None else accepted+1)
                pending.append((session, verified, features, expected, count))
            if all(verified._prefix_records and all(
                    record.prefix_context is not None for record in verified._prefix_records.values())
                    for _, verified, _, _, _ in pending):
                with self._finalization_stream():
                    committed = Qwen35InferenceCache.commit_recurrent_prefixes(
                        [ctx[1] for ctx in pending], [ctx[4] for ctx in pending],
                        finalizer=(None if self._target_graph is None
                                   else self._target_graph.finalize_prefixes),
                        _destinations=[session.spare_cache for session, _ in sessions])
                for (session, _, features, expected, count), cache in zip(pending, committed):
                    session.spare_cache = session.cache
                    session.cache = cache
                    session.features = features[:, :count]
                    session.bonus = expected[count - 1]
            else:
                for ctx in pending:
                    self.commit_accept(ctx)
        except Exception:
            if self._target_graph is not None:
                self._graph_failed = True
            # Draft caches and shared KV suffixes may already have advanced.
            # Keep slots owned until scheduler retirement; retry is not safe.
            for session, _ in sessions:
                session.failed = True
            raise
        # Each input's first token was emitted by admit/the preceding step.
        return SpecStepResult(tokens=[expected[:count] for _, _, _, expected, count in pending],
                              accept_counts=[count-1 for _, _, _, _, count in pending])

    def _wait_for_commit(self):
        if self._commit_pending:
            torch.cuda.current_stream(self.runtime.device).wait_event(self._commit_ready)

    @contextmanager
    def _finalization_stream(self):
        if self._target_graph is None:
            # Eager prefix tensors are released after commit. Keep their reads
            # on the allocation stream; graph-owned buffers outlive async work.
            yield
            return
        stream = self._commit_stream
        self._commit_inputs_ready.record(torch.cuda.current_stream(self.runtime.device))
        stream.wait_event(self._commit_inputs_ready)
        with torch.cuda.stream(stream):
            try:
                yield
            finally:
                # Even a failed partial copy may still be using slot storage.
                self._commit_ready.record(stream)
                self._commit_pending = True

    def retire(self, state):
        session = self._sessions.get(state.batch_idx)
        if session is None:
            return
        if session.state is not state:
            raise ValueError("cannot retire a different speculative sequence")
        if self._commit_pending:
            self._commit_ready.synchronize()
        self._draft_owners.detach_cache_owner(session.draft_cache)
        self.runtime.page_table.erase(state.batch_idx)
        # Scheduler cleanup still uses batch_idx to remove active_sequences.
        del self._sessions[state.batch_idx]
