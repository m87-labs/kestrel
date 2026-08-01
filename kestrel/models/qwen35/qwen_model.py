# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from kestrel.ops.rotary import default_inv_freq

from .inference_ops import (
    create_causal_mask,
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
    kestrel_vision_flash_attention_forward,
    sdpa_attention_forward,
    torch_compilable_check,
)
from .qwen_config import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from .cache import Qwen35InferenceCache

from kestrel_kernels import get_runtime
from kestrel_kernels import moe as _MOE_API
from kestrel_kernels.moe.errors import FP8_MOE_REQUIRES_COMPACT_CONFIG

_kestrel_runtime = get_runtime()
_flash_attn_fwd = _kestrel_runtime.attention.flash_attn_fwd
_kestrel_causal_conv1d_packed = _kestrel_runtime.gated_delta.causal_conv1d_packed
_kestrel_causal_conv1d_update_indexed = (
    _kestrel_runtime.gated_delta.causal_conv1d_update_indexed
)
_kestrel_packed_prefill_prepare = _kestrel_runtime.gated_delta.packed_prefill_prepare
_kestrel_packed_recurrent_decode_replay_indexed = (
    _kestrel_runtime.gated_delta.packed_recurrent_gated_delta_rule_decode_replay_indexed
)
_kestrel_packed_recurrent_decode_replay_indexed_gqa = (
    _kestrel_runtime.gated_delta.packed_recurrent_gated_delta_rule_decode_replay_indexed_gqa
)
_kestrel_packed_recurrent_prefill = (
    _kestrel_runtime.gated_delta.packed_recurrent_gated_delta_rule_prefill
)
_kestrel_gated_rmsnorm = _kestrel_runtime.gated_delta.gated_rmsnorm
_kestrel_rmsnorm = _kestrel_runtime.dense.rmsnorm
_kestrel_supports_packed_gdn = _kestrel_runtime.gated_delta.supports_packed_gdn
_kestrel_add_rmsnorm = _kestrel_runtime.dense.add_rmsnorm
_kestrel_gated_activation_into = _kestrel_runtime.dense.gated_activation_into
_kestrel_fused_mlp_gelu_bias_residual = _kestrel_runtime.dense.fused_mlp_gelu_bias_residual
_kestrel_text_mrope_apply = _kestrel_runtime.rotary.text_mrope_apply
_kestrel_spatial_rope_apply = _kestrel_runtime.rotary.spatial_rope_apply
_kestrel_moe_runtime = _kestrel_runtime.moe
_kestrel_moe_topk_fwd = _kestrel_moe_runtime.topk_fwd
_KESTREL_MOE_DECODE_MAX_TOKENS = 16
_KESTREL_MOE_MIN_PREFILL_BUCKET_TOKENS = 64
_KESTREL_MOE_FP8_WEIGHT_SCALE_LAYOUT = "block128_interleaved8"


def _rmsnorm_state(dim: int, eps: float) -> nn.ModuleDict:
    state = nn.ModuleDict()
    state.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
    state.eps = eps
    return state


@dataclass
class _TextModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Qwen35InferenceCache | None = None


def _module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.get_default_dtype()


def _kestrel_moe_capacity_for_tokens(tokens: int) -> tuple[int, str]:
    tokens = int(tokens)
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if tokens <= _KESTREL_MOE_DECODE_MAX_TOKENS:
        return tokens, "decode"
    return (
        max(_KESTREL_MOE_MIN_PREFILL_BUCKET_TOKENS, 1 << (tokens - 1).bit_length()),
        "prefill",
    )


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = default_inv_freq(dim, theta)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Qwen3_5TextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        inv_freq = default_inv_freq(
            config.head_dim,
            config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
            device=device,
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = config.mrope_section

    @torch.no_grad()
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3_5 has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gated_rmsnorm = _kestrel_gated_rmsnorm

    def forward(self, hidden_states, gate=None):
        return self.gated_rmsnorm(
            hidden_states,
            gate,
            self.weight,
            self.variance_epsilon,
        )


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens: torch.Tensor | None = None,
):
    # Single-sequence (numel==2: [0, T]) falls through to the core chunk math so the
    # host read below is skipped under CUDA-graph capture (graph-safety).
    if cu_seqlens is not None and int(cu_seqlens.numel()) > 2:
        offsets = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        outputs = []
        final_states = []
        for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            row_initial_state = None
            if initial_state is not None:
                row_initial_state = initial_state[row : row + 1]
            out, state = torch_chunk_gated_delta_rule(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                chunk_size=chunk_size,
                initial_state=row_initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            outputs.append(out)
            if state is not None:
                final_states.append(state)
        core_attn_out = torch.cat(outputs, dim=1)
        last_recurrent_state = (
            torch.cat(final_states, dim=0) if final_states else None
        )
        return core_attn_out, last_recurrent_state

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _packed_seq_idx_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32).contiguous()
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    sequence_ids = torch.arange(lengths.numel(), device=device, dtype=torch.int32)
    return torch.repeat_interleave(
        sequence_ids,
        lengths,
        output_size=total_tokens,
    )[None, :]


def _install_linear_conv_state(layer: Any, conv_states: torch.Tensor) -> None:
    layer.dtype = conv_states.dtype
    layer.device = conv_states.device
    layer.max_batch_size = conv_states.shape[0]
    layer.conv_kernel_size = conv_states.shape[-1]
    layer.conv_states = conv_states
    layer.is_conv_states_initialized = True


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = "silu"
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        ssm_dtype = torch.float32
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=ssm_dtype))

        A = torch.empty(self.num_v_heads, dtype=ssm_dtype).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_packed = _kestrel_causal_conv1d_packed
        self.causal_conv1d_update_indexed = _kestrel_causal_conv1d_update_indexed
        self.packed_prefill_prepare = _kestrel_packed_prefill_prepare
        self.packed_recurrent_decode_replay_indexed = (
            _kestrel_packed_recurrent_decode_replay_indexed
        )
        self.packed_recurrent_decode_replay_indexed_gqa = (
            _kestrel_packed_recurrent_decode_replay_indexed_gqa
        )
        self.packed_recurrent_prefill = _kestrel_packed_recurrent_prefill
        self.supports_packed_gdn = _kestrel_supports_packed_gdn

        self.in_proj = nn.Linear(
            self.hidden_size,
            self.conv_dim + self.value_dim + 2 * self.num_v_heads,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Qwen35InferenceCache | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        cu_seq_lens_q: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        gdn_state_indices: torch.Tensor | None = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        # We have cached `conv_state` / `recurrent_state` to continue from. The two cached modes
        # (single-token decode and chunk-tokens continuation) share the state read here; they only
        # diverge in how the conv input is assembled and which kernel consumes the states below,
        # which we gate locally on `seq_len`.
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        cu_seqlens_q = cu_seq_lens_q
        supports_native_packed_gdn = (
            self.supports_packed_gdn(
                hidden_states.device,
                hidden_states.dtype,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
            )
            and self.num_v_heads % self.num_k_heads == 0
            and self.head_v_dim == self.head_k_dim
        )
        packed_prefill = (
            cu_seqlens_q is not None
            and batch_size == 1
            and cache_params is not None
            and not use_precomputed_states
        )
        native_prefill = packed_prefill and supports_native_packed_gdn

        # getting projected states from cache if it exists
        if use_precomputed_states:
            layer_cache = cache_params.layers[self.layer_idx]
            conv_state = layer_cache.conv_states
            recurrent_state = layer_cache.recurrent_states
            state_indices = gdn_state_indices
            if state_indices is not None:
                state_indices = state_indices.to(
                    device=conv_state.device,
                    dtype=torch.long,
                ).view(-1)

        in_proj = self.in_proj(hidden_states)
        mixed_qkv, z, b, a = torch.split(
            in_proj,
            [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads],
            dim=-1,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        core_attn_out = None
        last_recurrent_state = None
        packed_recurrent_state = None

        if use_precomputed_states and seq_len == 1:
            # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
            if state_indices is None:
                raise RuntimeError("Qwen cached GDN decode requires gdn_state_indices")
            mixed_qkv_decode = self.causal_conv1d_update_indexed(
                mixed_qkv.squeeze(-1),
                conv_state,
                self.conv1d.weight.squeeze(1),
                state_indices,
                self.conv1d.bias,
                self.activation,
            )
            if supports_native_packed_gdn:
                # ReplaySSM is the only single-token GDN decode path. The ring
                # buffers are seeded at prefill (native packed prefill here, or
                # the state-pool capture for the captured path) and bound to the
                # cache before decode, so they are always present.
                #
                # The replay ring is indexed by VALUE head (num_v_heads) -- both
                # the symmetric and GQA kernels read it that way (the GQA kernel
                # applies the k->v fan-out to q/k internally). On a GQA layer
                # (num_v_heads != num_k_heads) the symmetric
                # ``packed_recurrent_decode_replay_indexed`` validates replay_k
                # as num_k_heads-shaped, so after the value-head ring allocation
                # it rejects the (correct) v-head ring and its torch fallback
                # raises ``bad replay_k shape``. Route GQA shapes to the GQA
                # decode kernel, which expects the v-head ring; keep the
                # symmetric kernel for symmetric (num_v_heads == num_k_heads)
                # layers.
                replay_checkpoint = layer_cache.replay_checkpoint_states
                replay_k = layer_cache.replay_k
                replay_u = layer_cache.replay_u
                replay_g = layer_cache.replay_g
                replay_lengths = layer_cache.replay_lengths
                assert (
                    replay_checkpoint is not None
                    and replay_k is not None
                    and replay_u is not None
                    and replay_g is not None
                    and replay_lengths is not None
                ), "ReplaySSM decode state was not seeded before GDN decode"
                decode_fn = (
                    self.packed_recurrent_decode_replay_indexed_gqa
                    if self.num_v_heads != self.num_k_heads
                    else self.packed_recurrent_decode_replay_indexed
                )
                core_attn_out, last_recurrent_state = decode_fn(
                    mixed_qkv_decode,
                    a,
                    b,
                    self.A_log,
                    self.dt_bias,
                    replay_checkpoint,
                    replay_k,
                    replay_u,
                    replay_g,
                    replay_lengths,
                    state_indices,
                )
            else:
                mixed_qkv = mixed_qkv_decode[:, :, None]
        else:
            # Multi-token forward (prefill, or chunked-tokens decode when the cache has prior state).
            packed_conv_state = None
            if use_precomputed_states:
                # Prior single-token decode may have run the ReplaySSM path, which
                # advances the replay ring buffer but not recurrent_states. Fold
                # the buffer back in so this chunk continuation starts from the
                # true current state instead of a stale recurrent_state.
                layer_cache.materialize_recurrent_from_replay(state_indices)
                recurrent_state = layer_cache.recurrent_states
                if state_indices is not None:
                    conv_state = conv_state.index_select(0, state_indices)
                    recurrent_state = recurrent_state.index_select(0, state_indices)
                elif conv_state.shape[0] != batch_size:
                    raise RuntimeError(
                        "Qwen cached chunk decode requires gdn_state_indices "
                        "when state batch differs from token batch"
                    )
                # Cached chunked-tokens decode: prepend the cached conv context so the causal conv
                # sees the correct left-context rather than zero-padding. Dropped from the output
                # at the end of this branch.
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                if native_prefill:
                    num_sequences = (
                        int(cu_seqlens_q.numel() - 1)
                        if packed_prefill
                        else batch_size
                    )
                    layer = cache_params.layers[self.layer_idx]
                    conv_shape = (
                        num_sequences,
                        self.conv_dim,
                        self.conv_kernel_size,
                    )
                    if (
                        not layer.is_conv_states_initialized
                        or layer.conv_states is None
                        or tuple(layer.conv_states.shape) != conv_shape
                    ):
                        _install_linear_conv_state(
                            layer,
                            torch.empty(
                                conv_shape,
                                device=mixed_qkv.device,
                                dtype=mixed_qkv.dtype,
                            ),
                        )
                    packed_conv_state = layer.conv_states
                    recurrent_shape = (
                        num_sequences,
                        self.num_v_heads,
                        self.head_k_dim,
                        self.head_v_dim,
                    )
                    if (
                        not layer.is_recurrent_states_initialized
                        or layer.recurrent_states is None
                        or tuple(layer.recurrent_states.shape) != recurrent_shape
                    ):
                        layer.recurrent_states = torch.empty(
                            recurrent_shape,
                            device=mixed_qkv.device,
                            dtype=torch.float32,
                        )
                        layer.is_recurrent_states_initialized = True
                    packed_recurrent_state = layer.recurrent_states
                    layer.has_previous_state = True
                elif not packed_prefill:
                    new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                    cache_params.layers[self.layer_idx].update_conv_state(
                        new_conv_state,
                        state_indices=state_indices if use_precomputed_states else None,
                    )
            if native_prefill:
                if seq_idx is None:
                    seq_idx = _packed_seq_idx_from_cu_seqlens(
                        cu_seqlens_q,
                        mixed_qkv.shape[-1],
                        mixed_qkv.device,
                    )
                recurrence_cu_seqlens = cu_seqlens_q
                # Tried fusing packed conv + q/k/v/g/beta prep in CuTe DSL:
                # Triton Bench on H100 was 0.0358 ms vs 0.0250 ms at seq=384,
                # and 0.0515 ms vs 0.0333 ms at seq=768,chunks=2; keeping the
                # separate kernels.
                mixed_qkv = self.causal_conv1d_packed(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    seq_idx=seq_idx,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    final_state=packed_conv_state,
                )
            elif packed_prefill:
                if seq_idx is None:
                    seq_idx = _packed_seq_idx_from_cu_seqlens(
                        cu_seqlens_q,
                        mixed_qkv.shape[-1],
                        mixed_qkv.device,
                    )
                recurrence_cu_seqlens = cu_seqlens_q
                if cache_params is not None:
                    num_sequences = int(cu_seqlens_q.numel() - 1)
                    layer = cache_params.layers[self.layer_idx]
                    conv_shape = (
                        num_sequences,
                        self.conv_dim,
                        self.conv_kernel_size,
                    )
                    if (
                        not layer.is_conv_states_initialized
                        or layer.conv_states is None
                        or tuple(layer.conv_states.shape) != conv_shape
                    ):
                        _install_linear_conv_state(
                            layer,
                            torch.empty(
                                conv_shape,
                                device=mixed_qkv.device,
                                dtype=mixed_qkv.dtype,
                            ),
                        )
                    packed_conv_state = layer.conv_states
                mixed_qkv = self.causal_conv1d_packed(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    seq_idx=seq_idx,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    final_state=packed_conv_state,
                    out=None,
                )
                if cache_params is not None and packed_conv_state is not None:
                    cache_params.layers[self.layer_idx].update_conv_state(
                        packed_conv_state,
                        state_indices=state_indices if use_precomputed_states else None,
                    )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        if core_attn_out is None:
            mixed_qkv = mixed_qkv.transpose(1, 2)
            prepped_qk = (
                native_prefill
                and supports_native_packed_gdn
                and not use_precomputed_states
            )
            if prepped_qk:
                query, key, value, g, beta = self.packed_prefill_prepare(
                    mixed_qkv,
                    a,
                    b,
                    self.A_log,
                    self.dt_bias,
                )
            else:
                query, key, value = torch.split(
                    mixed_qkv,
                    [
                        self.key_dim,
                        self.key_dim,
                        self.value_dim,
                    ],
                    dim=-1,
                )

                query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
                key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
                value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

                beta = b.sigmoid()
                # If the model is loaded in fp16, without the .float() here, A might be -inf
                g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
                if self.num_v_heads // self.num_k_heads > 1:
                    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
                    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

            if core_attn_out is None and prepped_qk:
                # Native serial prefill recurrence beats Kestrel-prep+FLA on
                # most Qwen packed shapes (for example seq=384,chunks=1:
                # 0.252 ms vs 0.433 ms on H100). It still loses at
                # seq=768,chunks=1 (0.489 ms vs 0.441 ms); closing that
                # remaining case needs a native chunk/WY-parallel recurrence.
                core_attn_out, last_recurrent_state = self.packed_recurrent_prefill(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    recurrence_cu_seqlens,
                    output_final_state=cache_params is not None,
                    final_state=packed_recurrent_state,
                )
            elif core_attn_out is None:
                initial_recurrent_state = (
                    recurrent_state if use_precomputed_states else None
                )
                if initial_recurrent_state is not None:
                    if (
                        state_indices is not None
                        and initial_recurrent_state.shape[0] != batch_size
                    ):
                        initial_recurrent_state = initial_recurrent_state.index_select(
                            0,
                            state_indices,
                        )
                    elif initial_recurrent_state.shape[0] != batch_size:
                        raise RuntimeError(
                            "Qwen cached recurrent decode requires gdn_state_indices "
                            "when state batch differs from token batch"
                        )
                core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=initial_recurrent_state,
                    output_final_state=cache_params is not None,
                    use_qk_l2norm_in_kernel=True,
                    # The chunked FLA kernel takes a single `cu_seqlens` arg; for packed self-attention this matches q-side lengths.
                    cu_seqlens=cu_seqlens_q,
                )

        # Update cache
        if (
            cache_params is not None
            and last_recurrent_state is not None
            and last_recurrent_state is not packed_recurrent_state
        ):
            cache_params.layers[self.layer_idx].update_recurrent_state(
                last_recurrent_state,
                state_indices=state_indices if use_precomputed_states else None,
            )
        elif cache_params is not None and native_prefill:
            # Native packed prefill writes recurrent_states in place, bypassing
            # update_recurrent_state -> _reset_replay_rows. Seed the ReplaySSM
            # ring buffer from the committed state here so single-token decode
            # takes the replay path in eager mode too (the captured path seeds
            # the same checkpoint via the state-pool capture).
            seed_layer = cache_params.layers[self.layer_idx]
            if (
                hasattr(seed_layer, "_reset_replay_rows")
                and getattr(seed_layer, "recurrent_states", None) is not None
            ):
                seed_layer._reset_replay_rows(seed_layer.recurrent_states, None)

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


# Adapted from GLM's rotary embedding helper.
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if unsqueeze_dim != 1:
        raise ValueError("Qwen text rotary expects [B, H, S, D] query/key tensors")
    return _kestrel_text_mrope_apply(q, k, cos, sin)


def paged_attention_forward(
    query: torch.Tensor,
    *,
    paged_kv_layer: Any,
    page_table: torch.Tensor | None,
    paged_kv_seqlens_k: torch.Tensor | None,
    paged_kv_seqlens_q: torch.Tensor | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
):
    if page_table is None or paged_kv_seqlens_k is None:
        raise RuntimeError("Qwen paged attention requires page_table and seqused_k")

    q_bshd = query.transpose(1, 2).contiguous()
    q_for_attention = q_bshd
    seqused_q = paged_kv_seqlens_q
    if cu_seq_lens_q is not None:
        # Packed query layout: cu_seq_lens_q describes the per-sequence block
        # boundaries over a flat [total_q, H, D] tensor. The single-sequence /
        # production-prefill callers already pass batch dim 1 ([1, total, H, D]);
        # the batched spec-verify caller passes [B, T, H, D] (equal-length blocks)
        # so the FlashAttention varlen path takes the shipped packed cubin (the
        # batched seqused_q path is JIT-only and unsafe under graph capture).
        # Flatten the batch into the packed token axis either way.
        q_for_attention = q_bshd.reshape(-1, q_bshd.shape[-2], q_bshd.shape[-1]).contiguous()
        seqused_q = None
    if q_for_attention.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"Qwen paged attention requires fp16/bf16 tensors, got {q_for_attention.dtype}"
        )

    k_cache = paged_kv_layer.k_cache.permute(0, 2, 1, 3)
    v_cache = paged_kv_layer.v_cache.permute(0, 2, 1, 3)
    out, _ = _flash_attn_fwd(
        q_for_attention,
        k_cache,
        v_cache,
        page_table=page_table,
        cu_seqlens_q=cu_seq_lens_q,
        seqused_q=seqused_q,
        seqused_k=paged_kv_seqlens_k,
        paged_kv_non_tma=True,
        causal=True,
        k_scale=getattr(paged_kv_layer, "k_scale", None),
        v_scale=getattr(paged_kv_layer, "v_scale", None),
    )
    if cu_seq_lens_q is not None:
        # Restore the [B, T, H, D] (or [1, total, H, D]) shape the caller expects.
        out = out.reshape(q_bshd.shape[0], q_bshd.shape[1], *out.shape[-2:])
    return out, None


class Qwen3_5Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_gate_size = config.num_attention_heads * self.head_dim * 2
        self.kv_size = config.num_key_value_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_gate_size + 2 * self.kv_size,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        # Unlike OLMo, these normalize only the head dimension.
        self.q_norm = _rmsnorm_state(self.head_dim, config.rms_norm_eps)
        self.k_norm = _rmsnorm_state(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Qwen35InferenceCache | None = None,
        *,
        cache_position_ids: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        paged_kv_seqlens_q: torch.Tensor | None = None,
        paged_kv_seqlens_k: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_gate, key_states, value_states = self.qkv_proj(hidden_states).split(
            [self.q_gate_size, self.kv_size, self.kv_size],
            dim=-1,
        )
        query_states, gate = torch.chunk(
            q_gate.reshape(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )

        query_states = _kestrel_rmsnorm(
            query_states.reshape(hidden_shape), self.q_norm.weight, self.q_norm.eps
        ).transpose(1, 2)
        key_states = _kestrel_rmsnorm(
            key_states.reshape(hidden_shape), self.k_norm.weight, self.k_norm.eps
        ).transpose(1, 2)
        value_states = value_states.reshape(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is None:
            attn_output, attn_weights = sdpa_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0,
                scaling=self.scaling,
            )
        else:
            paged_kv_layer = past_key_values.layers[self.layer_idx]
            if cache_position_ids is None or slot_mapping is None:
                raise RuntimeError(
                    "Qwen paged KV update requires cache_position_ids and slot_mapping"
                )
            paged_kv_layer.update(
                input_pos=cache_position_ids,
                k_val=key_states.transpose(1, 2),
                v_val=value_states.transpose(1, 2),
                slot_mapping=slot_mapping,
            )
            attn_output, attn_weights = paged_attention_forward(
                query_states,
                paged_kv_layer=paged_kv_layer,
                page_table=page_table,
                paged_kv_seqlens_q=paged_kv_seqlens_q,
                paged_kv_seqlens_k=paged_kv_seqlens_k,
                cu_seq_lens_q=cu_seq_lens_q,
            )

        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3_5MLP(nn.Module):
    def __init__(
        self,
        config: Qwen3_5Config,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        hidden = torch.empty(
            (*gate_up.shape[:-1], self.intermediate_size),
            device=gate_up.device,
            dtype=gate_up.dtype,
        )
        _kestrel_gated_activation_into(
            hidden,
            gate_up,
            activation="silu",
            layout="interleaved_i8",
        )
        return self.down_proj(hidden)


class Qwen3_5Experts(nn.Module):
    """Packed Qwen 3.5 MoE expert weights, matching HF safetensor keys."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        if config.num_experts is None or config.moe_intermediate_size is None:
            raise ValueError("MoE config must define num_experts and moe_intermediate_size")
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.expert_weight_format = config.expert_weight_format
        if self.expert_weight_format not in ("bf16", "fp8_e4m3"):
            raise ValueError(
                f"unsupported Qwen expert weight format: {self.expert_weight_format!r}"
            )
        if self.intermediate_dim % 8 != 0:
            raise ValueError(
                "Qwen expert interleaved gate/up layout requires "
                f"moe_intermediate_size divisible by 8, got {self.intermediate_dim}"
            )
        param_dtype = (
            torch.uint8
            if self.expert_weight_format == "fp8_e4m3"
            else torch.get_default_dtype()
        )
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.intermediate_dim,
                self.hidden_dim,
                dtype=param_dtype,
            ),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
                dtype=param_dtype,
            ),
            requires_grad=False,
        )
        if self.expert_weight_format == "fp8_e4m3":
            self.register_buffer(
                "gate_up_proj_scale",
                torch.empty(
                    self.num_experts,
                    2,
                    (self.intermediate_dim + 127) // 128,
                    (self.hidden_dim + 127) // 128,
                    dtype=torch.float32,
                ),
            )
            self.register_buffer(
                "down_proj_scale",
                torch.empty(
                    self.num_experts,
                    (self.hidden_dim + 127) // 128,
                    (self.intermediate_dim + 127) // 128,
                    dtype=torch.float32,
                ),
            )
        self.act_fn = F.silu

    def _expert_gate_up(self, expert_idx: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj[expert_idx]
        if gate_up.dtype != torch.uint8:
            return gate_up
        scale = self.gate_up_proj_scale[expert_idx]
        gate_scale = scale[0].repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1
        )[: self.intermediate_dim, : self.hidden_dim]
        up_scale = scale[1].repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1
        )[: self.intermediate_dim, : self.hidden_dim]
        expanded_scale = torch.stack(
            (
                gate_scale.reshape(self.intermediate_dim // 8, 8, self.hidden_dim),
                up_scale.reshape(self.intermediate_dim // 8, 8, self.hidden_dim),
            ),
            dim=1,
        ).reshape(
            2 * self.intermediate_dim,
            self.hidden_dim,
        )
        return (
            gate_up.view(torch.float8_e4m3fn).to(torch.float32)
            * expanded_scale
        ).to(torch.bfloat16)

    def _expert_down(self, expert_idx: torch.Tensor) -> torch.Tensor:
        down = self.down_proj[expert_idx]
        if down.dtype != torch.uint8:
            return down
        scale = self.down_proj_scale[expert_idx]
        expanded_scale = scale.repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1
        )
        return (
            down.view(torch.float8_e4m3fn).to(torch.float32)
            * expanded_scale[: self.hidden_dim, : self.intermediate_dim]
        ).to(torch.bfloat16)

    def _split_interleaved_gate_up(
        self, gate_up: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_up = gate_up.reshape(
            gate_up.shape[0],
            self.intermediate_dim // 8,
            2,
            8,
        )
        gate = gate_up[:, :, 0, :].reshape(gate_up.shape[0], self.intermediate_dim)
        up = gate_up[:, :, 1, :].reshape(gate_up.shape[0], self.intermediate_dim)
        return gate, up

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and top_k_weights.dtype == torch.bfloat16
            and self.top_k is not None
        ):
            try:
                return self._forward_kestrel(
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                )
            except ValueError as exc:
                if (
                    self.expert_weight_format != "fp8_e4m3"
                    or str(exc) != FP8_MOE_REQUIRES_COMPACT_CONFIG
                ):
                    raise

        final_hidden_states = torch.zeros_like(hidden_states)
        expert_ids_for_mask = (
            top_k_index if top_k_index.dtype == torch.long else top_k_index.to(torch.long)
        )
        expert_mask = F.one_hot(expert_ids_for_mask, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hits = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hits:
            expert_idx = expert_idx_tensor[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = F.linear(current_state, self._expert_gate_up(expert_idx))
            gate, up = self._split_interleaved_gate_up(gate_up)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states,
                self._expert_down(expert_idx),
            )
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states

    def _forward_kestrel(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        tokens = int(hidden_states.shape[0])
        capacity_tokens, capacity_mode = _kestrel_moe_capacity_for_tokens(tokens)
        if top_k_index.dtype != torch.int32:
            top_k_index = top_k_index.to(torch.int32)
        spec = _MOE_API.MoeSpec(
            num_experts=self.num_experts,
            top_k=int(self.top_k),
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            activation="swiglu",
            weight_format=self.expert_weight_format,
            dtype=hidden_states.dtype,
            backend="auto" if self.expert_weight_format == "fp8_e4m3" else "triton",
        )
        handle = _kestrel_moe_runtime.prepare(
            spec,
            _MOE_API.MoeCapacity(
                max_tokens=capacity_tokens,
                mode=capacity_mode,
            ),
            device=hidden_states.device,
        )
        pack_kwargs: dict[str, Any] = {}
        if self.expert_weight_format == "fp8_e4m3":
            pack_kwargs = {
                "up_scale": self.gate_up_proj_scale,
                "down_scale": self.down_proj_scale,
                "weight_scale_layout": _KESTREL_MOE_FP8_WEIGHT_SCALE_LAYOUT,
            }
        weights = _MOE_API.pack_weights(
            handle.spec,
            up=self.gate_up_proj,
            down=self.down_proj,
            **pack_kwargs,
        )
        return _kestrel_moe_runtime.forward(
            handle,
            x=hidden_states,
            topk_ids=top_k_index,
            topk_weights=top_k_weights,
            weights=weights,
        )


class Qwen3_5TopKRouter(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        if config.num_experts is None or config.num_experts_per_tok is None:
            raise ValueError("MoE config must define num_experts and num_experts_per_tok")
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        if (
            router_logits.is_cuda
            and router_logits.dtype == torch.bfloat16
            and router_logits.is_contiguous()
            and self.top_k == 8
            and self.num_experts in (64, 256)
        ):
            return _kestrel_moe_topk_fwd(
                router_logits,
                self.top_k,
                softmax=True,
            )
        router_top_logits, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_scores = F.softmax(router_top_logits, dtype=torch.float, dim=-1).to(
            router_logits.dtype
        )
        return router_scores, router_indices


class Qwen3_5SparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        if config.shared_expert_intermediate_size is None:
            raise ValueError("MoE config must define shared_expert_intermediate_size")
        self.gate = Qwen3_5TopKRouter(config)
        self.experts = Qwen3_5Experts(config)
        self.shared_expert = Qwen3_5MLP(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(
            hidden_states_reshaped,
            selected_experts,
            routing_weights,
        )
        shared_expert_output = torch.sigmoid(
            self.shared_expert_gate(hidden_states_reshaped)
        ) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)


def qwen_add_rms_norm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        residual.is_cuda
        and x.is_cuda
        and weight.is_cuda
        and residual.shape == x.shape
        and residual.dtype == torch.bfloat16
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.float32
        and abs(float(eps) - 1.0e-6) < 1.0e-12
    ):
        return residual, _kestrel_add_rmsnorm(residual, x, weight, eps)
    # Tried MPS _kestrel_add_rmsnorm: standalone [1, 2048] add-RMSNorm was
    # 1.65x faster, but Qwen 64-token median fell to 16.8 tok/s vs 17.4 with
    # in-place add + PyTorch RMSNorm; keep the end-to-end winner.
    residual.add_(x)
    return residual, F.rms_norm(residual.float(), weight.shape, weight, eps).to(
        residual.dtype
    )


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = (
            Qwen3_5SparseMoeBlock(config)
            if config.is_moe
            else Qwen3_5MLP(
                config,
                config.intermediate_size,
            )
        )
        self.input_layernorm = _rmsnorm_state(
            config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _rmsnorm_state(
            config.hidden_size, config.rms_norm_eps)

    def _forward_from_normalized(
        self,
        residual: torch.Tensor,
        normalized_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        output_layernorm: nn.ModuleDict | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Qwen35InferenceCache | None = None,
        *,
        cache_position_ids: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        paged_kv_seqlens_q: torch.Tensor | None = None,
        paged_kv_seqlens_k: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        gdn_state_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = normalized_hidden_states

        # Token Mixer
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                seq_idx=seq_idx,
                gdn_state_indices=gdn_state_indices,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                cache_position_ids=cache_position_ids,
                slot_mapping=slot_mapping,
                page_table=page_table,
                paged_kv_seqlens_q=paged_kv_seqlens_q,
                paged_kv_seqlens_k=paged_kv_seqlens_k,
                cu_seq_lens_q=cu_seq_lens_q,
            )

        # Fully Connected
        residual, hidden_states = qwen_add_rms_norm(
            residual,
            hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        hidden_states = self.mlp(hidden_states)
        if output_layernorm is not None:
            residual, normalized_output = qwen_add_rms_norm(
                residual,
                hidden_states,
                output_layernorm.weight,
                output_layernorm.eps,
            )
            return residual, normalized_output

        hidden_states = residual + hidden_states

        return hidden_states, None

class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, hidden_size, bias=True)
        # cuBLASLt fused-MLP GELU epilogue mode that matches ``config.hidden_act``.
        # The vision encoder uses ``gelu_pytorch_tanh`` (tanh approximation); plain
        # ``gelu`` maps to the exact (erf) GELU.
        if config.hidden_act == "gelu_pytorch_tanh":
            self._gelu_approximate = "tanh"
        elif config.hidden_act == "gelu":
            self._gelu_approximate = "none"
        else:
            self._gelu_approximate = None

        self.act_fn = lambda value: F.gelu(
            value,
            approximate=self._gelu_approximate or "none",
        )
    def forward(self, hidden_state, residual=None, hidden_workspace=None):
        """``linear_fc2(act_fn(linear_fc1(hidden_state)))``, plus ``residual``.

        When ``residual`` is given on CUDA bf16/fp16, fc1+GELU+fc2+bias+residual
        fuse into a single cuBLASLt call (one kernel instead of the eager
        fc1/GELU/fc2/add chain), eliminating the per-block GELU launch. Every
        other case falls back to eager. Called through ``__call__`` so module
        forward hooks (e.g. the profiler's ``vision.mlp``) still fire.
        """
        if (
            residual is not None
            and self._gelu_approximate is not None
            and hidden_state.is_cuda
            and hidden_state.ndim == 2
            and hidden_state.dtype in (torch.bfloat16, torch.float16)
            and self.linear_fc1.weight.dtype == hidden_state.dtype
            and self.linear_fc2.weight.dtype == hidden_state.dtype
        ):
            x = hidden_state.contiguous()
            residual = residual.contiguous()
            m = x.shape[0]
            out = torch.empty_like(residual)
            if hidden_workspace is not None and hidden_workspace.shape[0] >= m:
                hidden = hidden_workspace[:m]
            else:
                hidden = torch.empty(
                    (m, self.intermediate_size), device=x.device, dtype=x.dtype
                )
            _kestrel_fused_mlp_gelu_bias_residual(
                out,
                hidden,
                x,
                self.linear_fc1.weight,
                self.linear_fc1.bias,
                self.linear_fc2.weight,
                self.linear_fc2.bias,
                residual,
                approximate=self._gelu_approximate,
            )
            return out
        out = self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
        return out if residual is None else residual + out


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.proj = nn.Linear(
            config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size,
            config.hidden_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _kestrel_spatial_rope_apply(q, k, cos, sin, axis_blocks=1)


class Qwen3_5VisionAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        *,
        use_flash_attention: bool,
    ) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.use_flash_attention = bool(use_flash_attention)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if self.use_flash_attention:
            # Flash Attention: Use cu_seqlens for variable length attention
            attn_output, _ = kestrel_vision_flash_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                scaling=self.scaling,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                sdpa_attention_forward(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3_5VisionBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        *,
        use_flash_attention: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(
            config=config,
            use_flash_attention=use_flash_attention,
        )
        self.mlp = Qwen3_5VisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        mlp_workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.Tensor`):
            Cumulative sequence lengths used for packed variable-length attention in Flash Attention kernels.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.mlp(
            self.norm2(hidden_states), hidden_states, mlp_workspace
        )
        return hidden_states


class Qwen3_5VisionModel(nn.Module):
    config: Qwen3_5VisionConfig

    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        *,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.config = config

        self.patch_embed = Qwen3_5VisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen3_5VisionBlock(
                    config,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(config.depth)
            ]
        )
        self.merger = Qwen3_5VisionPatchMerger(config)
        # One shared fused-MLP fc1/gelu workspace for all blocks (the intermediate
        # is consumed within each block's fused call and the blocks run
        # sequentially, so a single buffer suffices instead of one per block).
        self._mlp_hidden_workspace: torch.Tensor | None = None

    def _mlp_workspace(self, num_tokens, dtype, device) -> torch.Tensor:
        ws = self._mlp_hidden_workspace
        if (
            ws is None
            or ws.shape[0] < num_tokens
            or ws.dtype != dtype
            or ws.device != device
        ):
            ws = torch.empty(
                num_tokens, self.config.intermediate_size, dtype=dtype, device=device
            )
            self._mlp_hidden_workspace = ws
        return ws[:num_tokens]

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        bilinear_indices: torch.Tensor | None = None,
        bilinear_weights: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.config.spatial_merge_size,
            bilinear_indices=bilinear_indices,
            bilinear_weights=bilinear_weights,
        )
        position_ids = get_vision_position_ids(
            grid_thw,
            self.config.spatial_merge_size,
            position_ids,
        )
        cu_seqlens = get_vision_cu_seqlens(grid_thw, cu_seqlens)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Only the fused (CUDA bf16/fp16) MLP path consumes the workspace; skip
        # the allocation entirely on eager/CPU/MPS fallbacks.
        if hidden_states.is_cuda and hidden_states.dtype in (
            torch.bfloat16,
            torch.float16,
        ):
            mlp_workspace = self._mlp_workspace(
                seq_len, hidden_states.dtype, hidden_states.device
            )
        else:
            mlp_workspace = None
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                mlp_workspace=mlp_workspace,
            )

        return self.merger(hidden_states)


class Qwen3_5TextModel(nn.Module):
    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = _rmsnorm_state(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Qwen35InferenceCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        *,
        cache_position_ids: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        paged_kv_seqlens_q: torch.Tensor | None = None,
        paged_kv_seqlens_k: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        gdn_state_indices: torch.Tensor | None = None,
    ) -> _TextModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        causal_mask = (
            None
            if past_key_values is not None and attention_mask is None
            else create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=text_position_ids,
            )
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        decoder_layers = self.layers[: self.config.num_hidden_layers]

        if decoder_layers:
            state = decoder_layers[0].input_layernorm
            normalized_hidden_states = _kestrel_rmsnorm(
                hidden_states, state.weight, state.eps)

        for i, decoder_layer in enumerate(decoder_layers):
            layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            output_layernorm = (
                decoder_layers[i + 1].input_layernorm
                if i + 1 < len(decoder_layers)
                else self.norm
            )

            hidden_states, normalized_hidden_states = decoder_layer._forward_from_normalized(
                hidden_states,
                normalized_hidden_states,
                position_embeddings=position_embeddings,
                output_layernorm=output_layernorm,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position_ids=cache_position_ids,
                slot_mapping=slot_mapping,
                page_table=page_table,
                paged_kv_seqlens_q=paged_kv_seqlens_q,
                paged_kv_seqlens_k=paged_kv_seqlens_k,
                cu_seq_lens_q=cu_seq_lens_q,
                seq_idx=seq_idx,
                gdn_state_indices=gdn_state_indices,
            )

        hidden_states = (
            normalized_hidden_states
            if decoder_layers
            else _kestrel_rmsnorm(
                hidden_states, self.norm.weight, self.norm.eps)
        )

        return _TextModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask


class Qwen3_5Model(nn.Module):
    config: Qwen3_5Config

    def __init__(
        self,
        config: Qwen3_5Config,
        *,
        use_vision_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VisionModel(
            config.vision_config,
            use_flash_attention=use_vision_flash_attention,
        )
        self.language_model = Qwen3_5TextModel(config.text_config)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        *,
        bilinear_indices: torch.Tensor | None = None,
        bilinear_weights: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required with pixel_values")
        pixel_values = pixel_values.type(_module_dtype(self.visual))
        image_embeds = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            bilinear_indices=bilinear_indices,
            bilinear_weights=bilinear_weights,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
        )
        split_sizes = (
            image_grid_thw.prod(-1)
            // self.config.vision_config.spatial_merge_size**2
        ).tolist()
        return torch.split(image_embeds, split_sizes)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen35InferenceCache | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        cache_position_ids: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        paged_kv_seqlens_q: torch.Tensor | None = None,
        paged_kv_seqlens_k: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        gdn_state_indices: torch.Tensor | None = None,
        vision_bilinear_indices: torch.Tensor | None = None,
        vision_bilinear_weights: torch.Tensor | None = None,
        vision_position_ids: torch.Tensor | None = None,
        vision_cu_seqlens: torch.Tensor | None = None,
    ) -> _TextModelOutput:
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(
                pixel_values,
                image_grid_thw,
                bilinear_indices=vision_bilinear_indices,
                bilinear_weights=vision_bilinear_weights,
                position_ids=vision_position_ids,
                cu_seqlens=vision_cu_seqlens,
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_token_mask = input_ids == self.config.image_token_id
            image_mask = image_token_mask.unsqueeze(-1).expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            torch_compilable_check(
                inputs_embeds[image_mask].numel() == image_embeds.numel(),
                "Image features and image tokens do not match, "
                f"tokens: {image_token_mask.sum()}, "
                f"features: {image_embeds.shape[0]}",
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position_ids=cache_position_ids,
            slot_mapping=slot_mapping,
            page_table=page_table,
            paged_kv_seqlens_q=paged_kv_seqlens_q,
            paged_kv_seqlens_k=paged_kv_seqlens_k,
            cu_seq_lens_q=cu_seq_lens_q,
            seq_idx=seq_idx,
            gdn_state_indices=gdn_state_indices,
        )

        return _TextModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
        )


class Qwen3_5ForConditionalGeneration(nn.Module):
    config: Qwen3_5Config

    def __init__(
        self,
        config: Qwen3_5Config,
        *,
        use_vision_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(
            config,
            use_vision_flash_attention=use_vision_flash_attention,
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

__all__ = [
    "Qwen3_5VisionModel",
    "Qwen3_5TextModel",
    "Qwen3_5Model",
    "Qwen3_5ForConditionalGeneration",
]
