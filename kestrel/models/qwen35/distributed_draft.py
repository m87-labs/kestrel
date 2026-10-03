"""Retained tensor-parallel DFlash graphs on an explicit CUDA device group."""

from contextlib import contextmanager
from dataclasses import replace

import torch

from kestrel_kernels.graph_team import CudaGraphTeam
from kestrel_kernels.peer_graph import BF16PeerCollective, PeerCopies, PeerPrefixCopies

from .dflash import DFlashDraftModel
from .draft_workspace import DraftLayerKVWorkspace


# A failed partial warmup/capture cannot safely release peer-visible storage.
_failed_captures = []


class DistributedDFlashDraftSession:
    """One greedy DFlash sequence, with captured transfers and native launches.

    Device order defines rank order, not CUDA ordinal order. The first device
    owns canonical caches, input staging and output consumption. This session
    requires ordinary BF16 transformer drafts and a BF16 vocabulary head;
    unsupported draft architectures or missing AOT shapes fail explicitly.
    """

    def __init__(self, model, caches, *, lm_head, devices):
        devices, caches = tuple(devices), tuple(caches)
        config = model.config
        world = len(devices)
        if (world not in (2, 4, 8) or len(set(devices)) != world
                or any(type(device) is not int or device < 0 for device in devices)
                or len(caches) != 1 or config.conv_kernel_size or config.selector_rank
                or config.num_attention_heads % world or config.num_key_value_heads % world
                or config.intermediate_size % world
                or not isinstance(lm_head, torch.nn.Linear) or lm_head.bias is not None
                or lm_head.weight.dtype != torch.bfloat16 or lm_head.weight.ndim != 2
                or lm_head.weight.shape[0] % world or lm_head.weight.shape[1] != config.hidden_size):
            raise ValueError("distributed DFlash requires one divisible BF16 transformer draft")
        self._devices, self.config = devices, config
        self._primary = torch.cuda.current_stream(devices[0])
        if (any(value.dtype != torch.bfloat16 or value.device != self._primary.device
                for value in model.parameters()) or lm_head.weight.device != self._primary.device):
            raise ValueError("draft weights must be BF16 on the primary device")
        self.failed = self.closed = self.active = False
        self._models, self._weights = [], []
        self._streams = tuple(torch.cuda.Stream(device=device) for device in devices)
        ready = torch.cuda.Event()
        ready.record(self._primary)
        local = replace(config, num_attention_heads=config.num_attention_heads // world,
                        num_key_value_heads=config.num_key_value_heads // world,
                        intermediate_size=config.intermediate_size // world)
        state, vocabulary = model.state_dict(), lm_head.weight.shape[0] // world
        for rank, (device, stream) in enumerate(zip(devices, self._streams, strict=True)):
            with torch.cuda.device(device), torch.cuda.stream(stream):
                stream.wait_event(ready)
                with torch.device("meta"):
                    shard = DFlashDraftModel(local)
                sliced = {}
                for name, value in state.items():
                    axis = (0 if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                               "gate_proj.weight", "up_proj.weight")) else
                            1 if name.endswith(("o_proj.weight", "down_proj.weight")) else None)
                    if axis is not None:
                        value = value.chunk(world, dim=axis)[rank]
                    sliced[name] = value.to(device=device, copy=True).contiguous()
                shard.load_state_dict(sliced, strict=True, assign=True)
                shard.rotary_emb.inv_freq = model.rotary_emb.inv_freq.to(device=device, copy=True)
                self._models.append(shard.requires_grad_(False).eval())
                self._weights.append(lm_head.weight[rank * vocabulary:(rank + 1) * vocabulary]
                                     .to(device=device, copy=True).contiguous())
        for stream in self._streams:
            stream.synchronize()
        self._team = None
        self._bind(caches)

    @property
    def devices(self):
        return self._devices

    @property
    def stream(self):
        return self._primary

    def _validate_caches(self, caches):
        if len(caches) != 1 or not 0 <= caches[0].length <= caches[0].capacity:
            raise ValueError("distributed draft requires one valid cache owner")
        cache, config = caches[0], self.config
        if len(cache.layers) != config.num_hidden_layers:
            raise ValueError("distributed draft requires initialized context")
        expected = (1, cache.capacity, config.num_key_value_heads, config.head_dim)
        if any(value is None or value.shape != expected or value.dtype != torch.bfloat16
               or value.device != self._primary.device
               for layer in cache.layers for value in (layer.keys, layer.values)):
            raise ValueError("distributed draft cache geometry does not match its model")

    def _bind(self, caches):
        self._validate_caches(caches)
        self.caches, self.lengths = tuple(caches), (caches[0].length,)
        config, world, cache = self.config, len(self.devices), caches[0]
        self._collective = BF16PeerCollective(self.devices, (1, config.block_size, config.hidden_size),
                                             collectives=2 * config.num_hidden_layers)
        self._workspaces, self._packed, self._inputs = [], [], []
        self._staging, self._commits, self._gathers = [], [], []
        self._local_scores, self._local_ids = [], []
        self._graphs, self._team = None, None
        ready = torch.cuda.Event()
        ready.record(self._primary)
        for rank, (device, stream) in enumerate(zip(self.devices, self._streams, strict=True)):
            with torch.cuda.device(device), torch.cuda.stream(stream):
                stream.wait_event(ready)
                workspaces = []
                for layer in cache.layers:
                    workspace = DraftLayerKVWorkspace(
                        slots=1, capacity=cache.capacity, context_rows=config.block_size,
                        query_rows=config.block_size, heads=config.num_key_value_heads // world,
                        head_dim=config.head_dim, device=torch.device("cuda", device))
                    for name in ("keys", "values"):
                        getattr(workspace, name)[:, :cache.length].copy_(
                            getattr(layer, name)[:, :cache.length].chunk(world, dim=2)[rank])
                    workspaces.append(workspace)
                self._workspaces.append(workspaces)
                specs = (
                    ((1, config.block_size, config.hidden_size), torch.bfloat16),
                    ((1, config.block_size, len(config.target_layer_ids) * config.hidden_size), torch.bfloat16),
                    ((1, 2 * config.block_size), torch.int64),
                    ((2 * config.block_size,), torch.int64), ((1,), torch.int32), ((3,), torch.int32))
                sizes = [torch.Size(shape).numel() * dtype.itemsize for shape, dtype in specs]
                aligned = [(size + 255) // 256 * 256 for size in sizes]
                packed = torch.zeros(sum(aligned), device=device, dtype=torch.uint8)
                views, offset = [], 0
                for (shape, dtype), size, stride in zip(specs, sizes, aligned, strict=True):
                    views.append(packed[offset:offset + size].view(dtype).view(shape))
                    offset += stride
                self._packed.append(packed)
                self._inputs.append(tuple(views))
                self._local_scores.append(torch.empty(config.block_size - 1, device=device, dtype=torch.bfloat16))
                self._local_ids.append(torch.empty(config.block_size - 1, device=device, dtype=torch.int64))
                sources, destinations = [], []
                local_width = config.num_key_value_heads // world * config.head_dim
                for workspace, layer in zip(workspaces, cache.layers, strict=True):
                    for name in ("keys", "values"):
                        sources.append(getattr(workspace, name).view(workspace.storage_rows, local_width))
                        destinations.append(getattr(layer, name).view(cache.capacity, -1)
                                            [:, rank * local_width:(rank + 1) * local_width])
                self._commits.append(PeerPrefixCopies(device, sources, destinations, views[-1]))
        with torch.cuda.device(self.devices[0]), torch.cuda.stream(self._primary):
            self._scores = torch.empty((world, config.block_size - 1), device=self.devices[0], dtype=torch.bfloat16)
            self._ids = torch.empty((world, config.block_size - 1), device=self.devices[0], dtype=torch.int64)
        for rank, (device, stream) in enumerate(zip(self.devices, self._streams, strict=True)):
            with torch.cuda.device(device), torch.cuda.stream(stream):
                self._staging.append(None if rank == 0 else
                                     PeerCopies(device, (self._packed[0],), (self._packed[rank],)))
                self._gathers.append(PeerCopies(device,
                    (self._local_scores[rank], self._local_ids[rank]),
                    (self._scores[rank], self._ids[rank])))
        for stream in self._streams:
            stream.synchronize()

    def _map(self, function):
        values = []
        for rank, (device, stream) in enumerate(zip(self.devices, self._streams, strict=True)):
            with torch.cuda.device(device), torch.cuda.stream(stream):
                values.append(function(rank))
        return values

    def _reduce(self, values, index):
        outputs = self._map(lambda rank: torch.empty_like(values[rank]))
        self._map(lambda rank: self._collective.launch_rank(rank, values[rank], outputs[rank], index))
        return outputs

    @torch.no_grad()
    def _forward(self):
        self._map(lambda rank: self._staging[rank].launch() if rank else None)
        context = self._map(lambda rank: self._models[rank].hidden_norm(
            self._models[rank].fc(self._inputs[rank][1])))
        rotary = self._map(lambda rank: self._models[rank].rotary_emb(
            self._inputs[rank][0], self._inputs[rank][2][..., None]))
        hidden = [inputs[0] for inputs in self._inputs]
        for index in range(self.config.num_hidden_layers):
            normalized = self._map(lambda rank: self._models[rank].layers[index].input_layernorm(hidden[rank]))
            partials = self._map(lambda rank: self._models[rank].layers[index].self_attn.forward_stable(
                normalized[rank], context[rank],
                tuple(value[:, -self.config.block_size:] for value in rotary[rank]), rotary[rank],
                self._workspaces[rank][index], self._inputs[rank][3], self._inputs[rank][4]))
            summed = self._reduce(partials, 2 * index)
            hidden = self._map(lambda rank: hidden[rank] + summed[rank])
            partials = self._map(lambda rank: self._models[rank].layers[index].mlp(
                self._models[rank].layers[index].post_attention_layernorm(hidden[rank])))
            summed = self._reduce(partials, 2 * index + 1)
            hidden = self._map(lambda rank: hidden[rank] + summed[rank])
        hidden = self._map(lambda rank: self._models[rank].norm(hidden[rank]))
        logits = self._map(lambda rank: torch.nn.functional.linear(hidden[rank][:, 1:], self._weights[rank]))
        self._map(lambda rank: torch.max(logits[rank].reshape(self.config.block_size - 1, -1), dim=-1,
                                         out=(self._local_scores[rank], self._local_ids[rank])))
        self._map(lambda rank: self._local_ids[rank].add_(rank * self._weights[rank].shape[0]))
        self._map(lambda rank: self._commits[rank].launch())
        self._map(lambda rank: self._gathers[rank].launch())

    def _capture(self):
        try:
            for _ in range(3):
                self._forward()
                for stream in self._streams:
                    stream.synchronize()
            graphs = [torch.cuda.CUDAGraph() for _ in self.devices]
            for device, stream, graph in zip(self.devices, self._streams, graphs, strict=True):
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    graph.capture_begin(capture_error_mode="relaxed")
            self._forward()
            for device, stream, graph in zip(self.devices, self._streams, graphs, strict=True):
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    graph.capture_end()
            self._graphs = graphs
            self._team = CudaGraphTeam(self.devices, graphs, self._streams, primary=self._primary,
                owners=(self._models, self._weights, self._workspaces, self._packed,
                        self._staging, self._commits, self._gathers, self._scores, self._ids,
                        self._collective, self.caches))
        except BaseException:
            self.failed = True
            _failed_captures.append(self)
            raise

    @contextmanager
    def launch(self, noise_embeddings, target_hiddens, position_ids, anchors=None):
        if self.closed or self.failed or self.active:
            raise RuntimeError("distributed draft session unavailable")
        if torch.cuda.current_stream(self.devices[0]) != self._primary:
            raise ValueError("distributed draft requires its owning primary stream")
        if len(noise_embeddings) != 1 or len(target_hiddens) != 1 or len(position_ids) != 1:
            raise ValueError("distributed draft requires one sequence")
        noise, target, positions = noise_embeddings[0], target_hiddens[0], position_ids[0]
        length, start = target.shape[1], self.lengths[0]
        inputs = self._inputs[0]
        if (self.caches[0].length != start or not 0 <= length <= self.config.block_size
                or start + length + self.config.block_size > self.caches[0].capacity
                or noise.shape != inputs[0].shape or target.shape != (1, length, inputs[1].shape[-1])
                or positions.shape != (1, length + self.config.block_size)
                or noise.dtype != torch.bfloat16 or target.dtype != torch.bfloat16
                or positions.dtype != torch.int64
                or any(value.device != self._primary.device for value in (noise, target, positions))):
            raise ValueError("distributed draft inputs or cache length differ")
        self.active = True
        try:
            mapping, used = self._workspaces[0][0].append_inputs((start,), (length,))
            inputs[0].copy_(noise)
            inputs[1].zero_()
            inputs[1][:, :length].copy_(target)
            inputs[2].zero_()
            inputs[2][:, :length].copy_(positions[:, :length])
            inputs[2][:, self.config.block_size:].copy_(positions[:, length:])
            inputs[3].copy_(mapping)
            inputs[4].copy_(used)
            inputs[5].copy_(torch.tensor((start, start, length), device=self._primary.device, dtype=torch.int32))
            if self._team is None:
                ready = torch.cuda.Event()
                ready.record(self._primary)
                for stream in self._streams:
                    stream.wait_event(ready)
                self._capture()
            self._team.replay()
            winners = self._scores.argmax(0)
            yield self._ids.gather(0, winners[None]).to(torch.int32)
            self.lengths = (start + length,)
            self.caches[0].length = self.lengths[0]
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False

    def rebind(self, caches):
        caches = tuple(caches)
        if self.failed or self.closed or self.active:
            raise RuntimeError("cannot rebind an active, failed, or closed draft")
        self._validate_caches(caches)
        if self._team is not None:
            self._team.close()
        self._collective.close()
        self._bind(caches)

    def shutdown(self):
        if self.active:
            raise RuntimeError("cannot retire an active distributed draft")
        if self.closed:
            return
        if self._team is not None:
            self._team.close()
        elif self.failed:
            raise RuntimeError("failed partial capture retains its owners until process exit")
        self._collective.close()
        self.closed = True
