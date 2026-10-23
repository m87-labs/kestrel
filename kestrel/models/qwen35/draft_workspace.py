"""Fixed-address draft KV storage; committed lengths remain session-owned."""

import torch
from contextlib import contextmanager

from kestrel_kernels import get_runtime


class DraftLayerKVWorkspace:
    """One layer's independent slots, including private padding-write storage.

    Sources contain a fixed context extent followed by a fixed query block.
    Only live context and query rows enter the attention prefix. Padding writes
    have distinct addresses beyond the logical capacity, never a shared sink.
    This object does not advance committed lengths or own a graph/output lease.
    """

    def __init__(self, *, slots, capacity, context_rows, query_rows,
                 heads, head_dim, device, dtype=torch.bfloat16):
        dimensions = (slots, capacity, context_rows, query_rows, heads, head_dim)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("draft KV workspace dimensions must be positive integers")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("draft KV workspace requires BF16 or FP16")
        self.slots = slots
        self.capacity = capacity
        self.context_rows = context_rows
        self.query_rows = query_rows
        self.source_rows = context_rows + query_rows
        self.storage_rows = capacity + context_rows
        shape = (slots, self.storage_rows, heads, head_dim)
        # The attention tile may physically load a masked tail. Initialize it
        # once, so unused V lanes cannot contain allocator NaNs (0 * NaN).
        self.keys = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros_like(self.keys)
        self.scale = torch.ones((), device=device, dtype=torch.float32)

    def append_inputs(self, committed_lengths, context_lengths):
        """Validate all slots before building fixed-shape graph tensor inputs."""
        if len(committed_lengths) != self.slots or len(context_lengths) != self.slots:
            raise ValueError("draft KV lengths must describe every slot")
        mappings, used = [], []
        for slot, (start, length) in enumerate(zip(committed_lengths, context_lengths)):
            if (type(start) is not int or type(length) is not int or start < 0
                    or not 0 <= length <= self.context_rows
                    or start + length + self.query_rows > self.capacity):
                raise ValueError("draft KV append exceeds its logical capacity")
            base = slot * self.storage_rows
            mappings.extend(base + start + row if row < length
                            else base + self.capacity + row
                            for row in range(self.context_rows))
            mappings.extend(base + start + length + row for row in range(self.query_rows))
            used.append(start + length + self.query_rows)
        return (torch.tensor(mappings, device=self.keys.device, dtype=torch.int64),
                torch.tensor(used, device=self.keys.device, dtype=torch.int32))

    def write(self, keys, values, slot_mapping):
        """Submit an indexed append without changing host session state."""
        expected = (self.slots * self.source_rows, *self.keys.shape[2:])
        for source in (keys, values):
            if (source.shape != expected or source.dtype != self.keys.dtype
                    or source.device != self.keys.device or source.stride(-1) != 1
                    or source.stride(-2) != source.shape[-1]):
                raise ValueError("draft KV source does not match its workspace")
        if (slot_mapping.shape != (expected[0],) or slot_mapping.dtype != torch.int64
                or slot_mapping.device != self.keys.device or not slot_mapping.is_contiguous()):
            raise ValueError("draft KV mapping must be contiguous on-device int64")
        get_runtime().cache.reshape_and_cache_flash(
            keys, values, self.keys, self.values, slot_mapping,
            "auto", self.scale, self.scale)
        return self.keys[:, :self.capacity], self.values[:, :self.capacity]


class DFlashDraftGraphSession:
    """One fixed ordered set of draft-cache owners; never shared across slots.

    Rebind compatible owners between leases; retire on geometry changes. The
    authoritative caches receive only new context; transient query KV stays here.
    A failed launch/consumer poisons the session rather than attempting rollback.
    """

    def __init__(self, model, caches, *, lm_head=None, confidence=None, proposal_count=None):
        from kestrel.runtime.single_pass_graph import FixedShapeSinglePassGraph

        self.model = model
        self.lm_head = lm_head
        self.confidence = confidence
        self.caches = tuple(caches)
        if not self.caches or len({id(cache) for cache in self.caches}) != len(self.caches):
            raise ValueError("draft graph requires distinct cache owners")
        self.lengths = tuple(cache.length for cache in self.caches)
        self.failed = False
        self.closed = False
        self.active = False
        config = model.config
        if proposal_count is not None and (type(proposal_count) is not int
                or not 1 <= proposal_count < config.block_size):
            raise ValueError("proposal count must fit the draft block")
        self.proposal_count = config.block_size - 1 if proposal_count is None else proposal_count
        if confidence is not None and confidence.shape[-1] != self.proposal_count:
            raise ValueError("draft confidence must match the projected proposal count")
        self.context_rows = config.block_size
        self.query_rows = config.block_size
        parameter = next(model.parameters())
        if parameter.device.type != "cuda":
            raise ValueError("draft graph requires CUDA")
        capacity = max(cache.capacity for cache in self.caches)
        self.workspaces = []
        for cache in self.caches:
            if not 0 <= cache.length <= cache.capacity:
                raise ValueError("draft graph cache has invalid committed length")
            if len(cache.layers) != len(model.layers):
                raise ValueError("draft graph requires initialized per-layer context caches")
            for layer in cache.layers:
                expected = (1, cache.capacity, config.num_key_value_heads, config.head_dim)
                for tensor in (layer.keys, layer.values):
                    if (tensor is None or tensor.shape != expected or tensor.device != parameter.device
                            or tensor.dtype != parameter.dtype):
                        raise ValueError("draft graph cache does not match model geometry")
        for index in range(len(model.layers)):
            workspace = DraftLayerKVWorkspace(
                slots=len(self.caches), capacity=capacity, context_rows=self.context_rows,
                query_rows=self.query_rows, heads=config.num_key_value_heads,
                head_dim=config.head_dim, device=parameter.device, dtype=parameter.dtype)
            for slot, cache in enumerate(self.caches):
                workspace.keys[slot, :cache.length].copy_(cache.layers[index].keys[0, :cache.length])
                workspace.values[slot, :cache.length].copy_(cache.layers[index].values[0, :cache.length])
            self.workspaces.append(workspace)
        self.stream = torch.cuda.Stream(device=parameter.device)
        self.targets = parameter.new_empty((len(self.caches), self.context_rows,
                                            len(config.target_layer_ids) * config.hidden_size))
        self.positions = torch.empty((len(self.caches), self.context_rows + self.query_rows),
                                     device=parameter.device, dtype=torch.int64)
        self.graph = FixedShapeSinglePassGraph(
            enabled=True, device=parameter.device, stream=self.stream,
            run_forward=self._forward, max_entries=1)
        self._context_copy_plan = self._bind_context_copy_plan()
        self._staging_launch_plan = None
        self._staging_implementation = None

    def _bind_context_copy_plan(self):
        """Retain independent cache destinations while this owner cohort lives."""
        self._context_copy_fallback_reason = None
        from kestrel_kernels.draft_context_copy import DraftContextCopyPlan
        sources, destinations = [], []
        for index, workspace in enumerate(self.workspaces):
            sources.extend((workspace.keys, workspace.values))
            destinations.extend((
                tuple(cache.layers[index].keys for cache in self.caches),
                tuple(cache.layers[index].values for cache in self.caches)))
        tensors = (*sources, *(value for row in destinations for value in row))
        if any(not value.is_contiguous() for value in tensors):
            self._context_copy_fallback_reason = "noncontiguous_cache_layout"
            return None
        ranges = sorted((value.data_ptr(), value.data_ptr() + value.numel() * value.element_size())
                        for value in tensors)
        if any(end > following for (_, end), (following, _) in zip(ranges, ranges[1:])):
            raise ValueError("draft context copy requires disjoint owner storage")
        return DraftContextCopyPlan(sources, destinations,
                                    context_rows=self.context_rows, stream=self.stream)

    @property
    def dispatch_report(self):
        """Actual bound paths for installed-build qualification; no timing claim."""
        from dataclasses import asdict
        copy_plan = self._context_copy_plan
        return {
            "owners": len(self.caches),
            "context_copy": {
                "implementation": "packed" if copy_plan is not None else "foreach",
                "fallback_reason": self._context_copy_fallback_reason,
                "launch_plan": asdict(copy_plan.launch_plan) if copy_plan is not None else None,
            },
            "staging_implementation": self._staging_implementation,
            "staging_launch_plan": (asdict(self._staging_launch_plan)
                                    if self._staging_launch_plan is not None else None),
        }

    def rebind(self, caches):
        """Replace retired owners whose logical capacities fit retained storage."""
        if self.failed or self.closed or self.active:
            raise RuntimeError("cannot rebind an active, failed, or retired draft graph")
        caches = tuple(caches)
        if (len(caches) != len(self.caches) or len({id(cache) for cache in caches}) != len(caches)
                or any(cache.capacity > self.workspaces[0].capacity for cache in caches)):
            raise ValueError("draft graph rebind exceeds retained slot capacity")
        for cache in caches:
            if not 0 <= cache.length <= cache.capacity or len(cache.layers) != len(self.workspaces):
                raise ValueError("draft graph rebind cache geometry mismatch")
            for layer, workspace in zip(cache.layers, self.workspaces):
                expected = (1, cache.capacity, *workspace.keys.shape[2:])
                for tensor in (layer.keys, layer.values):
                    if (tensor is None or tensor.shape != expected or tensor.device != workspace.keys.device
                            or tensor.dtype != workspace.keys.dtype):
                        raise ValueError("draft graph rebind cache geometry mismatch")
        self.stream.wait_stream(torch.cuda.current_stream(self.stream.device))
        with torch.cuda.stream(self.stream):
            for index, workspace in enumerate(self.workspaces):
                workspace.keys.zero_()
                workspace.values.zero_()
                for slot, cache in enumerate(caches):
                    layer = cache.layers[index]
                    workspace.keys[slot, :cache.length].copy_(layer.keys[0, :cache.length])
                    workspace.values[slot, :cache.length].copy_(layer.values[0, :cache.length])
                    layer.keys.record_stream(self.stream)
                    layer.values.record_stream(self.stream)
        self.caches = caches
        self.lengths = tuple(cache.length for cache in caches)
        self._context_copy_plan = self._bind_context_copy_plan()
        self._staging_launch_plan = None
        self._staging_implementation = None

    def _forward(self, hidden, targets, positions, mapping, used, anchors):
        context = self.model.hidden_norm(self.model.fc(targets))
        cos, sin = self.model.rotary_emb(hidden, positions[..., None])
        query_rotary = (cos[:, -hidden.shape[1]:], sin[:, -hidden.shape[1]:])
        key_rotary = (cos, sin)
        if all(layer.attention_conv is None and layer.mlp_conv is None for layer in self.model.layers):
            residual = None
            for layer, workspace in zip(self.model.layers, self.workspaces):
                norm = layer.input_layernorm
                normalized = (norm(hidden) if residual is None else
                              get_runtime().dense.add_rmsnorm(hidden, residual, norm.weight, norm.eps))
                output = layer.self_attn.forward_stable(
                    normalized, context, query_rotary, key_rotary, workspace, mapping, used)
                norm = layer.post_attention_layernorm
                normalized = get_runtime().dense.add_rmsnorm(output, hidden, norm.weight, norm.eps)
                residual = output
                hidden = layer.mlp(normalized)
            norm = self.model.norm
            hidden = get_runtime().dense.add_rmsnorm(hidden, residual, norm.weight, norm.eps)
        else:
            for layer, workspace in zip(self.model.layers, self.workspaces):
                hidden = layer.run(hidden, lambda value: layer.self_attn.forward_stable(
                    value, context, query_rotary, key_rotary, workspace, mapping, used))
            hidden = self.model.norm(hidden)
        if self.lm_head is None:
            return (hidden,)
        # Keep the complete draft transformer; project only consumed proposals.
        hidden = hidden[:, :self.proposal_count + 1]
        if self.confidence is not None:
            return (self.model.select_tokens(hidden, self.lm_head, anchors,
                                             confidence=self.confidence),)
        return (self.model.select_tokens(hidden, self.lm_head, anchors),)

    @contextmanager
    def launch(self, noise_embeddings, target_hiddens, position_ids, anchors=None):
        if self.failed or self.closed or self.active:
            raise RuntimeError("draft graph session is failed or retired")
        count = len(self.caches)
        if not (len(noise_embeddings) == len(target_hiddens) == len(position_ids) == count):
            raise ValueError("draft graph inputs must match its cache owners")
        if tuple(cache.length for cache in self.caches) != self.lengths:
            raise ValueError("draft graph cache was advanced outside its owner session")
        lengths = tuple(target.shape[1] for target in target_hiddens)
        config = self.model.config
        parameter = next(self.model.parameters())
        consumer_stream = torch.cuda.current_stream(parameter.device)
        if anchors is None:
            if self.lm_head is not None and self.model.candidate_selector is not None:
                raise ValueError("DFlash2 requires anchor tokens")
            anchors = torch.zeros(count, device=parameter.device, dtype=torch.long)
        if anchors.shape != (count,) or anchors.device != parameter.device or anchors.dtype != torch.long:
            raise ValueError("draft anchors must match cache owners and device")
        for noise, target, positions, cache, length in zip(
                noise_embeddings, target_hiddens, position_ids, self.caches, lengths):
            if (noise.shape != (1, self.query_rows, config.hidden_size)
                    or target.shape != (1, length, len(config.target_layer_ids) * config.hidden_size)
                    or positions.shape != (1, length + self.query_rows)
                    or not 0 <= length <= self.context_rows
                    or cache.length + length + self.query_rows > cache.capacity
                    or any(t.device != parameter.device for t in (noise, target, positions))
                    or noise.dtype != parameter.dtype or target.dtype != parameter.dtype
                    or positions.dtype != torch.int64):
                raise ValueError("draft graph input geometry or capacity mismatch")
        # Stage on the retained stream so the next call cannot overwrite these
        # buffers before the preceding graph's input copies have consumed them.
        self.stream.wait_stream(consumer_stream)
        with torch.cuda.stream(self.stream):
            targets, positions = self.targets, self.positions
            targets.zero_()
            positions.zero_()
            target_dst, target_src, position_dst, position_src = [], [], [], []
            for slot, (target, position, length) in enumerate(zip(target_hiddens, position_ids, lengths)):
                target_dst.append(targets[slot, :length])
                target_src.append(target[0])
                position_dst.extend((positions[slot, :length], positions[slot, self.context_rows:]))
                position_src.extend((position[0, :length], position[0, length:]))
            torch._foreach_copy_(target_dst, target_src)
            torch._foreach_copy_(position_dst, position_src)
            mapping, used = self.workspaces[0].append_inputs(self.lengths, lengths)
            inputs = (torch.cat(noise_embeddings, dim=0), targets, positions, mapping, used, anchors)
            for value in (*noise_embeddings, *target_hiddens, *position_ids):
                value.record_stream(self.stream)
        # These staging tensors are allocated on the caller stream but copied
        # on our retained stream; keep their storage live through that copy.
        for value in inputs:
            value.record_stream(self.stream)
        self._staging_launch_plan = None
        self._staging_implementation = None
        self.active = True
        try:
            with self.graph.launch(*inputs) as (output,):
                yield output
                self._commit_context(lengths)
            # Cache ownership returns to the caller along with host lengths.
            # A later eager draft must observe these post-consumer KV copies.
            if consumer_stream != self.stream:
                consumer_stream.wait_stream(self.stream)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False

    def _commit_context(self, lengths):
        if self._context_copy_plan is not None:
            self._context_copy_plan.run(self.lengths, lengths)
        else:
            destinations, sources = [], []
            for index, workspace in enumerate(self.workspaces):
                for slot, (cache, start, length) in enumerate(zip(self.caches, self.lengths, lengths)):
                    if length:
                        destinations.extend((cache.layers[index].keys[0, start:start + length],
                                             cache.layers[index].values[0, start:start + length]))
                        sources.extend((workspace.keys[slot, start:start + length],
                                        workspace.values[slot, start:start + length]))
                        cache.layers[index].keys.record_stream(self.stream)
                        cache.layers[index].values.record_stream(self.stream)
            if destinations:
                torch._foreach_copy_(destinations, sources)
        self.lengths = tuple(start + length for start, length in zip(self.lengths, lengths))
        for cache, length in zip(self.caches, self.lengths):
            cache.length = length

    @contextmanager
    def launch_tokens(self, features, embedding, *, bonus, target_position):
        """Fuse C1 staging into the captured inputs under the normal graph lease."""
        from kestrel_kernels.draft_staging import stage_draft_inputs

        if self.failed or self.closed or self.active:
            raise RuntimeError("draft graph session is failed or retired")
        if len(self.caches) != 1 or self.lm_head is None:
            raise ValueError("direct draft token staging requires one token-producing owner")
        cache = self.caches[0]
        length = features.shape[1]
        if (self.lengths != (cache.length,) or cache.length + length != target_position
                or cache.length + length + self.query_rows > cache.capacity):
            raise ValueError("draft staging cache ownership or capacity mismatch")
        consumer = torch.cuda.current_stream(features.device)
        def stage(inputs):
            stage_draft_inputs(features, embedding, inputs, start=cache.length,
                               bonus=bonus, mask_token=self.model.config.mask_token_id,
                               capacity=self.workspaces[0].capacity)
            self._staging_implementation = "single"
            features.record_stream(self.stream)
            embedding.record_stream(self.stream)
        self._staging_launch_plan = None
        self._staging_implementation = None
        self.active = True
        try:
            with self.graph.launch_staged(stage) as (output,):
                yield output
                self._commit_context((length,))
            if consumer != self.stream:
                consumer.wait_stream(self.stream)
                output.record_stream(consumer)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False

    @contextmanager
    def launch_tokens_many(self, features, embedding, *, bonuses, target_positions):
        """Stage independent owners directly into the retained batched graph inputs."""
        from kestrel_kernels.draft_staging import stage_draft_inputs_many
        if self.failed or self.closed or self.active:
            raise RuntimeError("draft graph session is failed or retired")
        count = len(self.caches)
        if (self.lm_head is None or len(features) != count or len(bonuses) != count
                or len(target_positions) != count):
            raise ValueError("direct draft staging requires complete token-producing owners")
        lengths = tuple(feature.shape[1] for feature in features)
        if (self.lengths != tuple(cache.length for cache in self.caches)
                or any(cache.length + length != position
                       or cache.length + length + self.query_rows > cache.capacity
                       for cache, length, position in zip(self.caches, lengths, target_positions))):
            raise ValueError("draft staging cache ownership or capacity mismatch")
        consumer = torch.cuda.current_stream(embedding.device)
        def stage(inputs):
            self._staging_launch_plan = stage_draft_inputs_many(features, embedding, inputs, starts=self.lengths,
                bonuses=bonuses, mask_token=self.model.config.mask_token_id,
                capacity=self.workspaces[0].capacity)
            self._staging_implementation = "packed"
            for feature in features:
                feature.record_stream(self.stream)
            embedding.record_stream(self.stream)
        self._staging_launch_plan = None
        self._staging_implementation = None
        self.active = True
        try:
            with self.graph.launch_staged(stage) as (output,):
                yield output
                self._commit_context(lengths)
            if consumer != self.stream:
                consumer.wait_stream(self.stream)
                output.record_stream(consumer)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False

    def shutdown(self):
        if self.active:
            raise RuntimeError("cannot retire an active draft graph lease")
        self.closed = True
        self.graph.shutdown()
        self._context_copy_plan = None
        # A caller can still hold this retired session while constructing its
        # replacement. Release synchronized storage now to avoid overlapping
        # two full draft KV workspaces during a batch geometry change.
        self.workspaces.clear()
        self.caches = ()
        self.targets = None
        self.positions = None
