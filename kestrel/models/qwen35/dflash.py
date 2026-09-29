"""Inference-only DFlash draft layers with checkpoint-defined attention."""

from dataclasses import dataclass, field
import json
from itertools import accumulate
from pathlib import Path

import torch
from torch import nn

from kestrel.ops.attention import dense_attention
from kestrel.ops.rotary import (
    MultidimensionalRotaryEmbedding, default_inv_freq,
)
from kestrel_kernels import get_runtime
from kestrel_kernels.dynamic_conv import causal_dynamic_conv1d
from kestrel_kernels.candidate_chain import greedy_candidate_chain


@dataclass
class _LayerContextBuffer:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def append(self, keys, values, start, capacity):
        end = start + keys.shape[2]
        if end > capacity:
            raise ValueError("DFlash context capacity exceeded")
        shape = (keys.shape[0], capacity, keys.shape[1], keys.shape[3])
        if self.keys is None:
            self.keys, self.values = keys.new_empty(shape), values.new_empty(shape)
        if (self.values is None or self.keys.shape != shape or self.values.shape != shape
                or self.keys.dtype != keys.dtype or self.values.dtype != values.dtype
                or self.keys.device != keys.device or self.values.device != values.device):
            raise ValueError("DFlash context buffer does not match this session")
        self.keys[:, start:end].copy_(keys.transpose(1, 2))
        self.values[:, start:end].copy_(values.transpose(1, 2))
        return self.keys[:, :end].transpose(1, 2), self.values[:, :end].transpose(1, 2)


@dataclass
class DFlashContextCache:
    """One contiguous sequence's verified context and transient draft capacity."""

    capacity: int
    length: int = field(default=0, init=False)
    layers: list[_LayerContextBuffer] = field(default_factory=list, init=False)

    def __post_init__(self):
        if type(self.capacity) is not int or self.capacity <= 0:
            raise ValueError("DFlash context capacity must be a positive integer")


@dataclass(frozen=True)
class DFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    layer_types: tuple[str, ...]
    sliding_window: int | None
    causal_override: bool | None = None
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    vocab_size: int = 0

    @classmethod
    def from_dict(cls, data):
        draft = data["dflash_config"]
        architecture = data.get("architectures", ["DFlashDraftModel"])
        if architecture not in (["DFlashDraftModel"], ["DFlash2DraftModel"]):
            raise ValueError("unsupported DFlash draft architecture")
        is_v2 = architecture == ["DFlash2DraftModel"]
        extra = tuple(int(draft[name]) for name in (
            "conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"
        )) if is_v2 else (0, 0, 0, 0)
        if is_v2 and (min(extra) <= 0 or int(data["hidden_size"]) % extra[1]
                      or extra[3] > int(data["vocab_size"])):
            raise ValueError("invalid DFlash 2 convolution or selector dimensions")
        layers = int(data["num_hidden_layers"])
        if layers <= 0:
            raise ValueError("DFlash requires positive layer count")
        if draft.get("use_swa", False):
            raise ValueError("DFlash use_swa override is unsupported; specify layer_types")
        kinds = tuple(data.get("layer_types") or ["full_attention"] * layers)
        if len(kinds) != layers or set(kinds) - {"full_attention", "sliding_attention"}:
            raise ValueError("unsupported DFlash layer_types")
        window = draft.get("swa_window_size", data.get("sliding_window"))
        if "sliding_attention" in kinds and (window is None or int(window) <= 0):
            raise ValueError("DFlash sliding layers require a positive window")
        rope = data.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("DFlash requires default RoPE")
        if data.get("hidden_act", "silu") != "silu" or data.get("attention_bias", False):
            raise ValueError("DFlash requires bias-free attention and SiLU")
        heads, kv_heads = int(data["num_attention_heads"]), int(data["num_key_value_heads"])
        if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
            raise ValueError("DFlash attention heads must divide into KV groups")
        causal = data.get("is_causal")
        if causal is None:
            causal = draft.get("causal")
        if causal is not None and not isinstance(causal, bool):
            raise ValueError("DFlash causality must be boolean")
        taps = tuple(int(i) for i in draft["target_layer_ids"])
        if not taps or len(set(taps)) != len(taps) or min(taps) < 0:
            raise ValueError("DFlash requires distinct target layer indices")
        if "num_target_layers" in data and max(taps) >= int(data["num_target_layers"]):
            raise ValueError("DFlash target layer index exceeds target depth")
        block_size = int(draft.get("block_size", data.get("block_size", 16)))
        if block_size <= 1:
            raise ValueError("DFlash block must include a seed and a draft token")
        return cls(
            int(data["hidden_size"]), int(data["intermediate_size"]), layers,
            heads, kv_heads, int(data["head_dim"]), float(data["rms_norm_eps"]),
            float(rope.get("rope_theta", data.get("rope_theta", 10000000))),
            block_size,
            int(draft["mask_token_id"]), taps, kinds,
            None if window is None else int(window), causal, *extra,
            int(data.get("vocab_size", 0)),
        )

    def attention(self, layer):
        sliding = self.layer_types[layer] == "sliding_attention"
        causal = sliding if self.causal_override is None else self.causal_override
        return causal, self.sliding_window if sliding else None


class _Norm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.eps = eps

    def forward(self, value):
        return get_runtime().dense.rmsnorm(value, self.weight, self.eps)


class _Attention(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.head_dim = config.head_dim
        self.causal, self.window = config.attention(index)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = _Norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _Norm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden, context, query_rotary, key_rotary, *, cache=None, past_context=0,
                cache_capacity=0):
        batch, rows, _ = hidden.shape
        joined = torch.cat((context, hidden), dim=1)
        q = self.q_norm(self.q_proj(hidden).reshape(batch, rows, -1, self.head_dim))
        k = self.k_norm(self.k_proj(joined).reshape(batch, joined.shape[1], -1, self.head_dim))
        v = self.v_proj(joined).reshape(batch, joined.shape[1], -1, self.head_dim).transpose(1, 2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        # Tried FP32-product rotary: C1 serving 0.679 -> 0.852s from worse
        # draft acceptance; retain the checkpoint's BF16 intermediate products.
        q, _ = get_runtime().rotary.text_mrope_apply(q, q[:, :0], *query_rotary)
        k, _ = get_runtime().rotary.text_mrope_apply(k, k[:, :0], *key_rotary)
        if cache is not None:
            k, v = cache.append(k, v, past_context, cache_capacity)
        output = dense_attention(
            q, k, v, scaling=self.head_dim ** -0.5, causal=self.causal,
            window_size_left=None if self.window is None else self.window - 1,
            window_size_right=None if self.window is None else (0 if self.causal else self.window - 1),
        )
        return self.o_proj(output.reshape(batch, rows, -1))

    def forward_packed(self, hidden, contexts, query_rotary, key_rotary, caches, layer_index,
                       query_lengths, cu_q, cu_k):
        query_pieces = hidden.split(query_lengths, dim=1)
        joined_lengths = tuple(context.shape[1] + query.shape[1]
                               for context, query in zip(contexts, query_pieces))
        joined = torch.cat([torch.cat((context, query), dim=1)
                            for context, query in zip(contexts, query_pieces)], dim=1)
        q = self.q_norm(self.q_proj(hidden).reshape(1, hidden.shape[1], -1, self.head_dim))
        k = self.k_norm(self.k_proj(joined).reshape(1, joined.shape[1], -1, self.head_dim))
        v = self.v_proj(joined).reshape(1, joined.shape[1], -1, self.head_dim).transpose(1, 2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        q, _ = get_runtime().rotary.text_mrope_apply(q, q[:, :0], *query_rotary)
        k, _ = get_runtime().rotary.text_mrope_apply(k, k[:, :0], *key_rotary)
        keys, values = [], []
        for key, value, cache in zip(k.split(joined_lengths, dim=2),
                                     v.split(joined_lengths, dim=2), caches):
            key, value = cache.layers[layer_index].append(key, value, cache.length, cache.capacity)
            keys.append(key)
            values.append(value)
        output = dense_attention(
            q, torch.cat(keys, dim=2), torch.cat(values, dim=2),
            scaling=self.head_dim ** -0.5, causal=self.causal,
            window_size_left=None if self.window is None else self.window - 1,
            window_size_right=None if self.window is None else (0 if self.causal else self.window - 1),
            cu_seqlens=cu_q, cu_seqlens_k=cu_k)
        return self.o_proj(output.reshape(1, hidden.shape[1], -1))

    def forward_stable(self, hidden, context, query_rotary, key_rotary, workspace,
                       slot_mapping, used_keys):
        """Fixed-shape layer primitive with session-owned committed KV lengths.

        Context padding is projected but its KV writes land in private guard
        storage. Query positions follow the live context, not its padded size.
        Callers retain the workspace and lease outputs until consumed.
        """
        batch, rows, _ = hidden.shape
        if (hidden.device.type != "cuda" or batch != workspace.slots
                or rows != workspace.query_rows
                or context.shape[:2] != (batch, workspace.context_rows)):
            raise ValueError("stable draft attention requires matching CUDA workspace inputs")
        joined = torch.cat((context, hidden), dim=1)
        q = self.q_norm(self.q_proj(hidden).reshape(batch, rows, -1, self.head_dim))
        k = self.k_norm(self.k_proj(joined).reshape(batch, joined.shape[1], -1, self.head_dim))
        v = self.v_proj(joined).reshape(batch, joined.shape[1], -1, self.head_dim)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        q, _ = get_runtime().rotary.text_mrope_apply(q, q[:, :0], *query_rotary)
        k, _ = get_runtime().rotary.text_mrope_apply(k, k[:, :0], *key_rotary)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        keys, values = workspace.write(k.reshape(-1, k.shape[2], self.head_dim),
                                       v.reshape(-1, v.shape[2], self.head_dim), slot_mapping)
        output, _ = get_runtime().attention.flash_attn_fwd(
            q, keys, values, seqused_k=used_keys,
            softmax_scale=self.head_dim ** -0.5, causal=self.causal,
            window_size_left=None if self.window is None else self.window - 1,
            window_size_right=None if self.window is None else (0 if self.causal else self.window - 1),
            require_native=True, pack_gqa=False)
        return self.o_proj(output.reshape(batch, rows, -1))


class _MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _DynamicConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.group_size = config.conv_group_size
        self.kernel_size = config.conv_kernel_size
        self.base_kernel = nn.Parameter(torch.empty(2, self.kernel_size, config.hidden_size))
        self.kernel_projection = nn.Linear(
            config.hidden_size, 2 * self.kernel_size * config.hidden_size // self.group_size,
            bias=False)

    def prepare(self, hidden):
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, hidden.shape[-1] // self.group_size)
        return causal_dynamic_conv1d(hidden, dynamic[..., 0, :, :], self.base_kernel[0],
                                     group_size=self.group_size), dynamic[..., 1, :, :]

    def finish(self, hidden, dynamic):
        return causal_dynamic_conv1d(hidden, dynamic, self.base_kernel[1], group_size=self.group_size)


class _CandidateSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.selector_top_k
        self.predecessor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.hidden_projection = nn.Linear(config.hidden_size, config.selector_rank, bias=False)

    def forward(self, hidden, logits, anchors):
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        projected = self.hidden_projection(hidden)
        return greedy_candidate_chain(projected, unary, candidates,
                                      self.predecessor_codebook.weight,
                                      self.successor_codebook.weight, anchors)


class _Layer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = _Attention(config, index)
        self.mlp = _MLP(config)
        self.input_layernorm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.attention_conv = _DynamicConv(config) if config.conv_kernel_size else None
        self.mlp_conv = _DynamicConv(config) if config.conv_kernel_size else None

    def run(self, hidden, attention, lengths=None):
        # Packed sequences must not share convolution history at row boundaries.
        lengths = lengths or (hidden.shape[1],)
        normalized = self.input_layernorm(hidden)
        kernels = None
        if self.attention_conv is not None:
            prepared = [self.attention_conv.prepare(part) for part in normalized.split(lengths, dim=1)]
            normalized = torch.cat([part for part, _ in prepared], dim=1)
            kernels = [kernel for _, kernel in prepared]
        output = attention(normalized)
        if kernels is not None:
            output = torch.cat([self.attention_conv.finish(part, kernel) for part, kernel in
                                zip(output.split(lengths, dim=1), kernels)], dim=1)
        hidden = hidden + output
        normalized = self.post_attention_layernorm(hidden)
        if self.mlp_conv is None:
            return hidden + self.mlp(normalized)
        prepared = [self.mlp_conv.prepare(part) for part in normalized.split(lengths, dim=1)]
        output = self.mlp(torch.cat([part for part, _ in prepared], dim=1))
        return hidden + torch.cat([self.mlp_conv.finish(part, kernel) for part, (_, kernel) in
                                   zip(output.split(lengths, dim=1), prepared)], dim=1)

    def forward(self, hidden, context, query_rotary, key_rotary, **cache_args):
        return self.run(hidden, lambda value: self.self_attn(
            value, context, query_rotary, key_rotary, **cache_args))


class DFlashDraftModel(nn.Module):
    """Consume concatenated target taps and an embedded candidate block."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(_Layer(config, i) for i in range(config.num_hidden_layers))
        self.norm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.fc = nn.Linear(len(config.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.candidate_selector = _CandidateSelector(config) if config.selector_rank else None
        self.rotary_emb = MultidimensionalRotaryEmbedding(
            config.head_dim, config.rope_theta, dimensions=1)

    def forward(self, noise_embedding, target_hidden, position_ids, *, context_cache=None):
        """Append new verified target taps, then evaluate a transient query block.

        With a cache, target_hidden and position_ids contain only newly added
        context followed by query positions. Query K/V never advance the cache.
        Positions must be contiguous absolute sequence positions, continuing
        the previously committed context; a cache cannot be shared by sessions.
        """
        if position_ids.shape != (noise_embedding.shape[0], target_hidden.shape[1] + noise_embedding.shape[1]):
            raise ValueError("DFlash positions must cover context plus query block")
        if context_cache is not None:
            if context_cache.length + position_ids.shape[1] > context_cache.capacity:
                raise ValueError("DFlash context capacity exceeded")
            if not context_cache.layers:
                context_cache.layers = [_LayerContextBuffer() for _ in self.layers]
            if len(context_cache.layers) != len(self.layers):
                raise ValueError("DFlash context cache has the wrong layer count")
        context = (self.hidden_norm(self.fc(target_hidden)) if target_hidden.shape[1]
                   else noise_embedding.new_empty((*target_hidden.shape[:2], self.config.hidden_size)))
        cos, sin = self.rotary_emb(noise_embedding, position_ids[..., None])
        rows = noise_embedding.shape[1]
        query_rotary, key_rotary = (cos[:, -rows:], sin[:, -rows:]), (cos, sin)
        hidden = noise_embedding
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden, context, query_rotary, key_rotary,
                cache=None if context_cache is None else context_cache.layers[index],
                past_context=0 if context_cache is None else context_cache.length,
                cache_capacity=0 if context_cache is None else context_cache.capacity)
        result = self.norm(hidden)
        if context_cache is not None:
            # Failed evaluation may overwrite only the uncommitted suffix.
            context_cache.length += target_hidden.shape[1]
        return result

    def forward_many(self, noise_embeddings, target_hiddens, position_ids, *, context_caches):
        """Evaluate independent draft blocks with separate committed KV owners."""
        count = len(noise_embeddings)
        if (not count or len(target_hiddens) != count or len(position_ids) != count
                or len(context_caches) != count or len({id(c) for c in context_caches}) != count):
            raise ValueError("packed DFlash requires distinct caches and matching sequence inputs")
        query_lengths = tuple(value.shape[1] for value in noise_embeddings)
        context_lengths = tuple(value.shape[1] for value in target_hiddens)
        key_lengths = []
        for noise, target, positions, cache in zip(
                noise_embeddings, target_hiddens, position_ids, context_caches):
            if (noise.shape[0] != 1 or target.shape[0] != 1 or noise.shape[1] < 1
                    or positions.shape != (1, target.shape[1] + noise.shape[1])):
                raise ValueError("packed DFlash inputs must each describe one nonempty query sequence")
            end = cache.length + positions.shape[1]
            if end > cache.capacity:
                raise ValueError("DFlash context capacity exceeded")
            if cache.layers and len(cache.layers) != len(self.layers):
                raise ValueError("DFlash context cache has the wrong layer count")
            key_lengths.append(end)
        for cache in context_caches:
            if not cache.layers:
                cache.layers = [_LayerContextBuffer() for _ in self.layers]
        noise = torch.cat(noise_embeddings, dim=1)
        targets = torch.cat(target_hiddens, dim=1)
        context = (self.hidden_norm(self.fc(targets)) if targets.shape[1]
                   else noise.new_empty((1, 0, self.config.hidden_size)))
        contexts = context.split(context_lengths, dim=1)
        positions = torch.cat(position_ids, dim=1)
        cos, sin = self.rotary_emb(noise, positions[..., None])
        joined_lengths = tuple(context + query for context, query in zip(context_lengths, query_lengths))
        q_cos = torch.cat([part[:, -length:] for part, length in
                           zip(cos.split(joined_lengths, dim=1), query_lengths)], dim=1)
        q_sin = torch.cat([part[:, -length:] for part, length in
                           zip(sin.split(joined_lengths, dim=1), query_lengths)], dim=1)
        query_rotary, key_rotary = (q_cos, q_sin), (cos, sin)
        cu_q = torch.tensor([0, *accumulate(query_lengths)], device=noise.device, dtype=torch.int32)
        cu_k = torch.tensor([0, *accumulate(key_lengths)], device=noise.device, dtype=torch.int32)
        hidden = noise
        for index, layer in enumerate(self.layers):
            hidden = layer.run(hidden, lambda value: layer.self_attn.forward_packed(
                value, contexts, query_rotary, key_rotary, context_caches,
                index, query_lengths, cu_q, cu_k), query_lengths)
        result = self.norm(hidden).split(query_lengths, dim=1)
        for cache, length in zip(context_caches, context_lengths):
            cache.length += length
        return result

    def select_tokens(self, hidden, lm_head, anchors):
        rows = hidden[:, 1:]
        logits = lm_head(rows.reshape(1, -1, rows.shape[-1])).reshape(
            rows.shape[0], rows.shape[1], -1)
        if self.candidate_selector is None:
            return logits.argmax(-1).to(torch.int32)
        return self.candidate_selector(rows, logits, anchors).to(torch.int32)


def load_dflash_drafter(path: str | Path, *, device: torch.device) -> DFlashDraftModel:
    from safetensors.torch import load_file

    path = Path(path)
    config = DFlashConfig.from_dict(json.loads((path / "config.json").read_text()))
    with torch.device("meta"):
        model = DFlashDraftModel(config)
    state = load_file(path / "model.safetensors", device=str(device))
    for name in ("predecessor_codebook", "successor_codebook"):
        key = f"candidate_selector.{name}"
        if key in state:
            state[key + ".weight"] = state.pop(key)
    if any(value.dtype != torch.bfloat16 for value in state.values()):
        raise ValueError("DFlash draft checkpoint must contain BF16 weights")
    model.load_state_dict(state, strict=True, assign=True)
    model.rotary_emb.inv_freq = default_inv_freq(config.head_dim, config.rope_theta, device=device)
    return model.requires_grad_(False).eval()
