"""Gemma 4 model implementation."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from kestrel_kernels import get_runtime
from kestrel.kv_cache import PagedKVCache
from kestrel.ops import attention as attention_ops
from kestrel.ops import rotary as rotary_ops
from kestrel.runtime.bounded_projection import (
    BoundedLinear,
    PackedBoundedProjections,
    PackedLinear,
)
from torch import nn
from torch.nn import functional as F

from .config import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    attention_kv_heads,
)
from .paged_cache import kv_source_layers

_dense_runtime = get_runtime().dense
_rotary_runtime = get_runtime().rotary
_kestrel_gated_activation_into = _dense_runtime.gated_activation_into
_prepare_neox_rotary = _rotary_runtime.prepare_neox
_apply_neox_rotary = _rotary_runtime.apply_neox


def _rmsnorm_state(
    dim: int,
    eps: float,
    *,
    with_scale: bool = True,
) -> nn.ModuleDict:
    state = nn.ModuleDict()
    weight = torch.ones(dim, dtype=torch.float32)
    if with_scale:
        state.weight = nn.Parameter(weight)
    else:
        state.register_buffer("weight", weight, persistent=False)
    state.eps = eps
    return state


class Gemma4TextRotaryEmbedding(nn.Module):

    def __init__(
        self,
        config: Gemma4TextConfig,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.inv_freq: dict[str, torch.Tensor] = {}
        schedules = {
            "default": rotary_ops.default_inv_freq,
            "proportional": rotary_ops.proportional_inv_freq,
        }
        for layer_type in set(config.layer_types):
            params = config.rope[layer_type]
            head_dim = (
                config.global_head_dim
                if layer_type == "full_attention"
                and params.kind == "proportional"
                else config.head_dim
            )
            try:
                schedule = schedules[params.kind]
            except KeyError as exc:
                raise ValueError(f"unsupported RoPE schedule {params.kind!r}") from exc
            self.inv_freq[layer_type] = schedule(
                head_dim,
                params.theta,
                partial_rotary_factor=params.partial_rotary_factor,
                factor=params.factor,
                device=device,
            )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self.inv_freq[layer_type].to(x.device)
        self.inv_freq[layer_type] = inv
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = position_ids.float()[..., None] * inv.float()
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


class Gemma4TextAttention(nn.Module):

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        *,
        kv_source_layer_idx: int,
        publishes_kv: bool,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.kv_source_layer_idx = kv_source_layer_idx
        self.owns_kv = kv_source_layer_idx == layer_idx
        self.publishes_kv = publishes_kv
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = (
            config.global_head_dim if (not self.is_sliding and config.global_head_dim) else config.head_dim
        )
        use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        num_kv_heads = attention_kv_heads(config, is_sliding=self.is_sliding)

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.q_norm = _rmsnorm_state(self.head_dim, config.rms_norm_eps)

        if self.owns_kv:
            self.k_norm = _rmsnorm_state(self.head_dim, config.rms_norm_eps)
            self.v_norm = _rmsnorm_state(
                self.head_dim, config.rms_norm_eps, with_scale=False)
            self.k_proj = nn.Linear(
                config.hidden_size, num_kv_heads * self.head_dim, bias=False
            )
            self.v_proj = (
                nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=False)
                if not use_alternative_attention
                else None
            )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        transient_kv: list[tuple[torch.Tensor, torch.Tensor] | None],
        paged_kv_layer: PagedKVCache | None = None,
        cache_position_ids: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        paged_kv_seqlens_k: Optional[torch.Tensor] = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = _dense_runtime.rmsnorm(
            query_states, self.q_norm.weight, self.q_norm.eps)

        key_states: Optional[torch.Tensor] = None
        value_states: Optional[torch.Tensor] = None
        if not self.owns_kv:
            query_states, _ = _apply_neox_rotary(
                query_states, None, position_embeddings
            )
            query_states = query_states.transpose(1, 2)
            if page_table is None:
                source = transient_kv[self.kv_source_layer_idx]
                if source is None:
                    raise RuntimeError(
                        f"Gemma layer {self.layer_idx} has no transient K/V from "
                        f"producer {self.kv_source_layer_idx}"
                    )
                key_states, value_states = source
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = (
                self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
            )
            key_states = _dense_runtime.rmsnorm(
                key_states, self.k_norm.weight, self.k_norm.eps)
            query_states, key_states = _apply_neox_rotary(
                query_states, key_states, position_embeddings
            )
            query_states = query_states.transpose(1, 2)
            assert key_states is not None
            key_states = key_states.transpose(1, 2)

            value_states = _dense_runtime.rmsnorm(
                value_states, self.v_norm.weight, self.v_norm.eps)
            value_states = value_states.transpose(1, 2)

        if self.owns_kv and paged_kv_layer is not None:
            if cache_position_ids is None or slot_mapping is None:
                raise RuntimeError("Gemma paged K/V write requires positions and slots")
            assert key_states is not None and value_states is not None
            paged_kv_layer.update(
                input_pos=cache_position_ids,
                k_val=key_states.transpose(1, 2),
                v_val=value_states.transpose(1, 2),
                slot_mapping=slot_mapping,
            )
        if self.publishes_kv:
            assert key_states is not None and value_states is not None
            transient_kv[self.layer_idx] = (key_states, value_states)

        if page_table is not None:
            if paged_kv_layer is None or paged_kv_seqlens_k is None:
                raise RuntimeError("Gemma paged attention requires K/V storage metadata")
            attn_out = attention_ops.paged_attention(
                query_states,
                paged_kv_layer=paged_kv_layer,
                page_table=page_table,
                paged_kv_seqlens_k=paged_kv_seqlens_k,
                scaling=1.0,
                sliding_window=self.sliding_window,
            )
        else:
            assert key_states is not None and value_states is not None
            attn_out = attention_ops.dense_attention(
                query_states,
                key_states,
                value_states,
                scaling=1.0,
                causal=not self.is_sliding,
                window_size_left=(self.sliding_window - 1) if self.is_sliding else None,
                window_size_right=0 if self.is_sliding else None,
                cu_seqlens=cu_seqlens,
            )
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_out)


class Gemma4TextMLP(nn.Module):

    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared = layer_idx >= first_shared > 0
        use_double_wide = config.use_double_wide_mlp and is_kv_shared

        self.intermediate_size = config.intermediate_size * (2 if use_double_wide else 1)
        self.gate_up_proj = PackedLinear(
            config.hidden_size,
            (self.intermediate_size, self.intermediate_size),
            source_names=("gate_proj", "up_proj"),
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        hidden = gate_up.new_empty(*gate_up.shape[:-1], self.intermediate_size)
        _kestrel_gated_activation_into(
            hidden,
            gate_up,
            activation="gelu_tanh",
            layout="contiguous",
        )
        return self.down_proj(hidden)


class Gemma4TextDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        *,
        kv_source_layer_idx: int,
        publishes_kv: bool,
    ) -> None:
        super().__init__()
        self.self_attn = Gemma4TextAttention(
            config,
            layer_idx,
            kv_source_layer_idx=kv_source_layer_idx,
            publishes_kv=publishes_kv,
        )
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = _rmsnorm_state(
            config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _rmsnorm_state(
            config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = _rmsnorm_state(
            config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = _rmsnorm_state(
            config.hidden_size, config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = _rmsnorm_state(
                config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor],
        transient_kv: list[tuple[torch.Tensor, torch.Tensor] | None],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        paged_kv_layer: PagedKVCache | None = None,
        cache_position_ids: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        paged_kv_seqlens_k: Optional[torch.Tensor] = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = _dense_runtime.rmsnorm(
            hidden_states,
            self.input_layernorm.weight,
            self.input_layernorm.eps,
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            transient_kv=transient_kv,
            paged_kv_layer=paged_kv_layer,
            cache_position_ids=cache_position_ids,
            slot_mapping=slot_mapping,
            page_table=page_table,
            paged_kv_seqlens_k=paged_kv_seqlens_k,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = _dense_runtime.rmsnorm(
            hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = _dense_runtime.rmsnorm(
            hidden_states,
            self.pre_feedforward_layernorm.weight,
            self.pre_feedforward_layernorm.eps,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = _dense_runtime.rmsnorm(
            hidden_states,
            self.post_feedforward_layernorm.weight,
            self.post_feedforward_layernorm.eps,
        )
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = F.gelu(hidden_states, approximate="tanh")
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = _dense_runtime.rmsnorm(
                hidden_states,
                self.post_per_layer_input_norm.weight,
                self.post_per_layer_input_norm.eps,
            )
            hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4TextModel(nn.Module):

    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
        )
        self.embed_tokens.register_buffer(
            "embed_scale",
            torch.tensor(config.hidden_size**0.5),
            persistent=False,
        )
        sources = kv_source_layers(config)
        self.kv_source_layers = sources
        published = {
            source for layer, source in enumerate(sources) if layer != source
        }
        self.layers = nn.ModuleList(
            [
                Gemma4TextDecoderLayer(
                    config,
                    layer,
                    kv_source_layer_idx=sources[layer],
                    publishes_kv=layer in published,
                )
                for layer in range(config.num_hidden_layers)
            ]
        )
        self.norm = _rmsnorm_state(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.unique_layer_types = set(config.layer_types or [])

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                padding_idx=0,
            )
            self.embed_tokens_per_layer.register_buffer(
                "embed_scale",
                torch.tensor(config.hidden_size_per_layer_input**0.5),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.tensor(2.0**-0.5),
                persistent=False,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = _rmsnorm_state(
                config.hidden_size_per_layer_input, config.rms_norm_eps)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.embed_tokens.embed_scale

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        per = (
            self.embed_tokens_per_layer(input_ids)
            * self.embed_tokens_per_layer.embed_scale
        )
        return per.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        proj = proj.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        proj = _dense_runtime.rmsnorm(
            proj,
            self.per_layer_projection_norm.weight,
            self.per_layer_projection_norm.eps,
        )
        if per_layer_inputs is None:
            return proj
        return (proj + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Sequence[PagedKVCache | None] | None = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        cache_position_ids: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        paged_kv_seqlens_k: Optional[torch.Tensor] = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None and per_layer_inputs is not None:
            raise ValueError("per_layer_inputs requires inputs_embeds (not input_ids)")

        if input_ids is not None:
            inputs_embeds = self.embed(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                assert input_ids is not None
                per_layer_inputs = self.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        seq_len = inputs_embeds.shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=inputs_embeds.device
            ).unsqueeze(0)

        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            cos, sin = self.rotary_emb(inputs_embeds, position_ids, layer_type)
            position_embeddings[layer_type] = _prepare_neox_rotary(cos, sin)

        hidden_states = inputs_embeds
        transient_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None
        ] * len(self.layers)
        for i, layer in enumerate(self.layers):
            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            layer_type = self.config.layer_types[i]
            hidden_states = layer(
                hidden_states,
                per_layer_input=per_layer_input,
                transient_kv=transient_kv,
                position_embeddings=position_embeddings[layer_type],
                paged_kv_layer=(
                    kv_cache[self.kv_source_layers[i]]
                    if kv_cache is not None
                    else None
                ),
                cache_position_ids=cache_position_ids,
                slot_mapping=slot_mapping,
                page_table=page_table,
                paged_kv_seqlens_k=paged_kv_seqlens_k,
                cu_seqlens=cu_seqlens,
            )

        hidden_states = _dense_runtime.rmsnorm(
            hidden_states, self.norm.weight, self.norm.eps)
        return hidden_states

class Gemma4VisionPatchEmbedder(nn.Module):

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3 * config.patch_size**2, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, config.position_embedding_size, config.hidden_size)
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        positions = pixel_position_ids.clamp(min=0)
        position_embeddings = (
            self.position_embedding_table[0, positions[..., 0]]
            + self.position_embedding_table[1, positions[..., 1]]
        )
        position_embeddings = torch.where(
            padding_positions.unsqueeze(-1), 0.0, position_embeddings
        )
        return hidden_states + position_embeddings


class Gemma4VisionPooler(nn.Module):

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.root_hidden_size = config.hidden_size**0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: {k=}^2 times {length=} must be {input_seq_len}."
            )
        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are patches"
                f" ({hidden_states.shape[1]})."
            )
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, padding_positions


class Gemma4VisionMLP(nn.Module):

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = PackedBoundedProjections(
            config.hidden_size,
            (self.intermediate_size, self.intermediate_size),
            source_names=("gate_proj", "up_proj"),
            use_bounds=config.use_clipped_linears,
        )
        self.down_proj = BoundedLinear(
            self.intermediate_size,
            config.hidden_size,
            use_bounds=config.use_clipped_linears,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward_packed(x)
        hidden = gate_up.new_empty(*gate_up.shape[:-1], self.intermediate_size)
        _kestrel_gated_activation_into(
            hidden,
            gate_up,
            activation="gelu_tanh",
            layout="contiguous",
        )
        return self.down_proj(hidden)


class Gemma4VisionAttention(nn.Module):

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.qkv_proj = PackedBoundedProjections(
            config.hidden_size,
            (
                config.num_attention_heads * self.head_dim,
                config.num_key_value_heads * self.head_dim,
                config.num_key_value_heads * self.head_dim,
            ),
            source_names=("q_proj", "k_proj", "v_proj"),
            use_bounds=config.use_clipped_linears,
        )
        self.o_proj = BoundedLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            use_bounds=config.use_clipped_linears,
        )

        self.q_norm = _rmsnorm_state(self.head_dim, config.rms_norm_eps)
        self.k_norm = _rmsnorm_state(self.head_dim, config.rms_norm_eps)
        self.v_norm = _rmsnorm_state(
            self.head_dim, config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seqused_k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states, key_states, value_states = self.qkv_proj(hidden_states)
        query_states = query_states.view(hidden_shape)
        query_states = _dense_runtime.rmsnorm(
            query_states, self.q_norm.weight, self.q_norm.eps)
        query_states = rotary_ops.apply_multidimensional_rotary(
            query_states,
            cos,
            sin,
            position_ids,
        )

        key_states = key_states.view(hidden_shape)
        key_states = _dense_runtime.rmsnorm(
            key_states, self.k_norm.weight, self.k_norm.eps)
        key_states = rotary_ops.apply_multidimensional_rotary(
            key_states,
            cos,
            sin,
            position_ids,
        )

        value_states = value_states.view(hidden_shape)
        value_states = _dense_runtime.rmsnorm(
            value_states, self.v_norm.weight, self.v_norm.eps)

        attn_out = attention_ops.variable_length_attention(
            query_states,
            key_states,
            value_states,
            used_key_lengths=seqused_k,
            scaling=1.0,
        )
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_out)


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.self_attn = Gemma4VisionAttention(config)
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = _rmsnorm_state(
            config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _rmsnorm_state(
            config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = _rmsnorm_state(
            config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = _rmsnorm_state(
            config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seqused_k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = _dense_runtime.rmsnorm(
            hidden_states,
            self.input_layernorm.weight,
            self.input_layernorm.eps,
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seqused_k=seqused_k,
            position_ids=position_ids,
        )
        hidden_states = _dense_runtime.rmsnorm(
            hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = _dense_runtime.rmsnorm(
            hidden_states,
            self.pre_feedforward_layernorm.weight,
            self.pre_feedforward_layernorm.eps,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = _dense_runtime.rmsnorm(
            hidden_states,
            self.post_feedforward_layernorm.weight,
            self.post_feedforward_layernorm.eps,
        )
        hidden_states = residual + hidden_states
        return hidden_states


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        if config.rope.kind != "default":
            raise ValueError(
                "vision RoPE requires the default frequency schedule, "
                f"got {config.rope.kind!r}"
            )
        self.rotary_emb = rotary_ops.MultidimensionalRotaryEmbedding(
            config.head_dim,
            config.rope.theta,
            dimensions=2,
        )
        self.layers = nn.ModuleList(
            [Gemma4VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(inputs_embeds, pixel_position_ids)
        seqused_k = attention_mask.sum(dim=-1, dtype=torch.int32)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                seqused_k=seqused_k,
                position_ids=pixel_position_ids,
            )
        return hidden_states


class Gemma4VisionModel(nn.Module):

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)
        if config.standardize:
            self.register_buffer("std_bias", torch.empty(config.hidden_size))
            self.register_buffer("std_scale", torch.empty(config.hidden_size))

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        pooling_kernel_size = self.config.pooling_kernel_size
        # HF derives ``output_length`` from the input grid divided by
        # pooling_kernel_size**2 along the patch axis (axis -2 of
        # pixel_values *as the processor returns it* — see HF source).
        output_length = pixel_values.shape[-2] // (pooling_kernel_size * pooling_kernel_size)

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        encoded = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
        )

        hidden_states, pooler_mask = self.pooler(
            hidden_states=encoded,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        hidden_states = hidden_states[pooler_mask]

        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return hidden_states


class Gemma4VisionEmbedder(nn.Module):

    def __init__(
        self,
        vision_config: Gemma4VisionConfig,
        text_config: Gemma4TextConfig,
    ) -> None:
        super().__init__()
        self.embedding_projection = nn.Linear(
            vision_config.hidden_size, text_config.hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = _rmsnorm_state(
            vision_config.hidden_size,
            vision_config.rms_norm_eps,
            with_scale=False,
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        normed = _dense_runtime.rmsnorm(
            inputs_embeds,
            self.embedding_pre_projection_norm.weight,
            self.embedding_pre_projection_norm.eps,
        )
        return self.embedding_projection(normed)


class Gemma4Model(nn.Module):

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        text_cfg = config.text_config

        self.language_model = Gemma4TextModel(text_cfg)

        self.vision_tower = Gemma4VisionModel(config.vision_config)
        self.embed_vision = Gemma4VisionEmbedder(config.vision_config, text_cfg)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image patches and project into text-embedding space."""
        vision_hidden = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
        )
        return self.embed_vision(vision_hidden)

class Gemma4InferenceModel(nn.Module):

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        text_cfg = config.text_config
        self.model = Gemma4Model(config)
        self.lm_head = nn.Linear(text_cfg.hidden_size, text_cfg.vocab_size, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

__all__ = [
    "Gemma4InferenceModel",
    "Gemma4Model",
]
