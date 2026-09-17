"""Inference-only DFlash draft layers with checkpoint-defined attention."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
from torch import nn

from kestrel.ops.attention import dense_attention
from kestrel.ops.rotary import (
    MultidimensionalRotaryEmbedding, apply_rotary, default_inv_freq,
)
from kestrel_kernels import get_runtime


@dataclass
class _LayerContextBuffer:
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None


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

    @classmethod
    def from_dict(cls, data):
        draft = data["dflash_config"]
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
            None if window is None else int(window), causal,
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

    def forward(self, hidden, context, cos, sin, *, cache=None, past_context=0,
                cache_capacity=0):
        batch, rows, _ = hidden.shape
        joined = torch.cat((context, hidden), dim=1)
        q = self.q_norm(self.q_proj(hidden).reshape(batch, rows, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(joined).reshape(batch, joined.shape[1], -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(joined).reshape(batch, joined.shape[1], -1, self.head_dim).transpose(1, 2)
        q = apply_rotary(q, cos[:, -rows:], sin[:, -rows:])
        k = apply_rotary(k, cos, sin)
        if cache is not None:
            end = past_context + joined.shape[1]
            if end > cache_capacity:
                raise ValueError("DFlash context capacity exceeded")
            shape = (batch, cache_capacity, k.shape[1], self.head_dim)
            if cache.keys is None:
                keys, values = k.new_empty(shape), v.new_empty(shape)
                cache.keys, cache.values = keys, values
            if (cache.values is None
                    or cache.keys.shape != shape or cache.values.shape != shape
                    or cache.keys.dtype != k.dtype or cache.values.dtype != v.dtype
                    or cache.keys.device != k.device or cache.values.device != v.device):
                raise ValueError("DFlash context buffer does not match this session")
            cache.keys[:, past_context:end].copy_(k.transpose(1, 2))
            cache.values[:, past_context:end].copy_(v.transpose(1, 2))
            k = cache.keys[:, :end].transpose(1, 2)
            v = cache.values[:, :end].transpose(1, 2)
        output = dense_attention(
            q, k, v, scaling=self.head_dim ** -0.5, causal=self.causal,
            window_size_left=None if self.window is None else self.window - 1,
            window_size_right=None if self.window is None or not self.causal else 0,
        )
        return self.o_proj(output.reshape(batch, rows, -1))


class _MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _Layer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = _Attention(config, index)
        self.mlp = _MLP(config)
        self.input_layernorm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _Norm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, context, cos, sin, **cache_args):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), context, cos, sin,
                                        **cache_args)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class DFlashDraftModel(nn.Module):
    """Consume concatenated target taps and an embedded candidate block."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(_Layer(config, i) for i in range(config.num_hidden_layers))
        self.norm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.fc = nn.Linear(len(config.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = _Norm(config.hidden_size, config.rms_norm_eps)
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
        hidden = noise_embedding
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden, context, cos, sin,
                cache=None if context_cache is None else context_cache.layers[index],
                past_context=0 if context_cache is None else context_cache.length,
                cache_capacity=0 if context_cache is None else context_cache.capacity)
        result = self.norm(hidden)
        if context_cache is not None:
            # Failed evaluation may overwrite only the uncommitted suffix.
            context_cache.length += target_hidden.shape[1]
        return result


def load_dflash_drafter(path: str | Path, *, device: torch.device) -> DFlashDraftModel:
    from safetensors.torch import load_file

    path = Path(path)
    config = DFlashConfig.from_dict(json.loads((path / "config.json").read_text()))
    with torch.device("meta"):
        model = DFlashDraftModel(config)
    state = load_file(path / "model.safetensors", device=str(device))
    if any(value.dtype != torch.bfloat16 for value in state.values()):
        raise ValueError("DFlash draft checkpoint must contain BF16 weights")
    model.load_state_dict(state, strict=True, assign=True)
    model.rotary_emb.inv_freq = default_inv_freq(config.head_dim, config.rope_theta, device=device)
    return model.requires_grad_(False).eval()
