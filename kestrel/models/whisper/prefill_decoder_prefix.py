"""Fixed-shape Whisper Turbo control-prefix prefill orchestration.

This module composes the canonical dense, layer-norm, FlashAttention, exact
GELU MLP, and paged-cache-write primitives.  It intentionally does not own a
model-specific CUDA kernel family.
"""

from dataclasses import dataclass

import torch

from kestrel_kernels import get_runtime

from .prefill_stem import HIDDEN_SIZE
from .prefill_encoder import (
    ATTENTION_HEADS,
    DECODER_LAYERS,
    ENCODER_FRAMES,
    FFN_DIM,
    HEAD_DIM,
    LAYER_NORM_EPS,
    _prepare_attention,
    _prepare_layer_norm,
    _prepare_linear,
    _prepared_tensor,
)
from .runtime_abi import WhisperSelfKVArenas
from .weights import (
    LayerNormWeights,
    LinearWeights,
    WhisperModelWeights,
)

_KERNELS = get_runtime()
_ATTENTION = _KERNELS.attention
_CACHE = _KERNELS.cache
_DENSE = _KERNELS.dense
_LINEAR = _KERNELS.linear
_VISION = _KERNELS.vision

CONTROL_PREFIX_CAPACITY = 4
MAX_TARGET_POSITIONS = 448
VOCAB_SIZE = 51866


@dataclass(frozen=True)
class _PackedDecoderLayer:
    self_attention_layer_norm: LayerNormWeights
    self_qkv_weight: torch.Tensor
    self_qkv_bias: torch.Tensor
    self_attention_output: LinearWeights
    cross_attention_layer_norm: LayerNormWeights
    cross_query: LinearWeights
    cross_attention_output: LinearWeights
    final_layer_norm: LayerNormWeights
    fc1: LinearWeights
    fc2: LinearWeights


@dataclass(frozen=True)
class PreparedWhisperDecoderWeights:
    """BF16 decoder weights with self-QKV fused once at model load."""

    token_embedding: torch.Tensor
    position_embedding: torch.Tensor
    layers: tuple[_PackedDecoderLayer, ...]
    final_layer_norm: LayerNormWeights
    device: torch.device


@dataclass
class WhisperDecoderPrefixWorkspace:
    """All mutable storage for one fixed CUDA-graph prefix batch bucket."""

    batch_size: int
    hidden: torch.Tensor
    post_self_attention: torch.Tensor
    post_cross_attention: torch.Tensor
    normalized: torch.Tensor
    self_qkv: torch.Tensor
    cross_query: torch.Tensor
    attention: torch.Tensor
    mlp_hidden: torch.Tensor
    final_hidden: torch.Tensor
    last_hidden: torch.Tensor
    last_indices: torch.Tensor
    kv_scale: torch.Tensor

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> "WhisperDecoderPrefixWorkspace":
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("Whisper decoder-prefix batch_size must be positive")
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("Whisper decoder-prefix workspace requires a CUDA device")
        dtype = torch.bfloat16
        shape = (batch_size, CONTROL_PREFIX_CAPACITY, HIDDEN_SIZE)
        return cls(
            batch_size=batch_size,
            hidden=torch.empty(shape, device=resolved_device, dtype=dtype),
            post_self_attention=torch.empty(shape, device=resolved_device, dtype=dtype),
            post_cross_attention=torch.empty(
                shape, device=resolved_device, dtype=dtype
            ),
            normalized=torch.empty(shape, device=resolved_device, dtype=dtype),
            self_qkv=torch.empty(
                (batch_size, CONTROL_PREFIX_CAPACITY, 3 * HIDDEN_SIZE),
                device=resolved_device,
                dtype=dtype,
            ),
            cross_query=torch.empty(shape, device=resolved_device, dtype=dtype),
            attention=torch.empty(
                (
                    batch_size,
                    CONTROL_PREFIX_CAPACITY,
                    ATTENTION_HEADS,
                    HEAD_DIM,
                ),
                device=resolved_device,
                dtype=dtype,
            ),
            mlp_hidden=torch.empty(
                (batch_size * CONTROL_PREFIX_CAPACITY, FFN_DIM),
                device=resolved_device,
                dtype=dtype,
            ),
            final_hidden=torch.empty(shape, device=resolved_device, dtype=dtype),
            last_hidden=torch.empty(
                (batch_size, HIDDEN_SIZE), device=resolved_device, dtype=dtype
            ),
            last_indices=torch.empty(
                (batch_size,), device=resolved_device, dtype=torch.int64
            ),
            kv_scale=torch.ones((), device=resolved_device, dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return self.hidden.device


@dataclass(frozen=True)
class WhisperDecoderPrefixOutput:
    logits: torch.Tensor
    last_hidden_state: torch.Tensor


def prepare_whisper_decoder_weights(
    weights: WhisperModelWeights,
) -> PreparedWhisperDecoderWeights:
    """Validate and prepack the pinned four-layer Turbo decoder."""

    decoder = weights.decoder
    token_embedding = decoder.token_embedding
    position_embedding = decoder.position_embedding
    decoder_layers = decoder.layers
    if len(decoder_layers) != DECODER_LAYERS:
        raise ValueError(
            f"Whisper Turbo requires {DECODER_LAYERS} decoder layers, "
            f"got {len(decoder_layers)}"
        )
    device = token_embedding.device
    packed_layers = []
    for index, layer in enumerate(decoder_layers):
        prefix = f"decoder_layers[{index}]"
        qkv_weight, qkv_bias, self_output = _prepare_attention(
            f"{prefix}.self_attention", layer.self_attention, device
        )
        packed_layers.append(
            _PackedDecoderLayer(
                self_attention_layer_norm=_prepare_layer_norm(
                    f"{prefix}.self_attention_layer_norm",
                    layer.self_attention_layer_norm,
                    device,
                ),
                self_qkv_weight=qkv_weight,
                self_qkv_bias=qkv_bias,
                self_attention_output=self_output,
                cross_attention_layer_norm=_prepare_layer_norm(
                    f"{prefix}.cross_attention_layer_norm",
                    layer.cross_attention_layer_norm,
                    device,
                ),
                cross_query=_prepare_linear(
                    f"{prefix}.cross_attention.query",
                    layer.cross_attention.query,
                    out_features=HIDDEN_SIZE,
                    in_features=HIDDEN_SIZE,
                    bias=True,
                    device=device,
                ),
                cross_attention_output=_prepare_linear(
                    f"{prefix}.cross_attention.output",
                    layer.cross_attention.output,
                    out_features=HIDDEN_SIZE,
                    in_features=HIDDEN_SIZE,
                    bias=True,
                    device=device,
                ),
                final_layer_norm=_prepare_layer_norm(
                    f"{prefix}.final_layer_norm", layer.final_layer_norm, device
                ),
                fc1=_prepare_linear(
                    f"{prefix}.fc1",
                    layer.fc1,
                    out_features=FFN_DIM,
                    in_features=HIDDEN_SIZE,
                    bias=True,
                    device=device,
                ),
                fc2=_prepare_linear(
                    f"{prefix}.fc2",
                    layer.fc2,
                    out_features=HIDDEN_SIZE,
                    in_features=FFN_DIM,
                    bias=True,
                    device=device,
                ),
            )
        )
    return PreparedWhisperDecoderWeights(
        token_embedding=_prepared_tensor(
            "token_embedding",
            token_embedding,
            (VOCAB_SIZE, HIDDEN_SIZE),
            device,
        ),
        position_embedding=_prepared_tensor(
            "position_embedding",
            position_embedding,
            (MAX_TARGET_POSITIONS, HIDDEN_SIZE),
            device,
        ),
        layers=tuple(packed_layers),
        final_layer_norm=_prepare_layer_norm(
            "final_layer_norm", decoder.final_layer_norm, device
        ),
        device=device,
    )


def _validate_prefix_inputs(
    control_token_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    slot_mapping: torch.Tensor,
    compact_cross_keys: torch.Tensor,
    compact_cross_values: torch.Tensor,
    logits_out: torch.Tensor,
    weights: PreparedWhisperDecoderWeights,
    workspace: WhisperDecoderPrefixWorkspace,
    self_kv: WhisperSelfKVArenas,
) -> None:
    batch = workspace.batch_size
    device = weights.device
    if workspace.device != device or self_kv.keys[0].device != device:
        raise ValueError("Whisper prefix weights, workspace, and self-KV must colocate")
    if (
        tuple(control_token_ids.shape) != (batch, CONTROL_PREFIX_CAPACITY)
        or control_token_ids.dtype != torch.int64
        or control_token_ids.device != device
        or not control_token_ids.is_contiguous()
    ):
        raise ValueError("control_token_ids must be contiguous CUDA INT64 [batch, 4]")
    if (
        tuple(prefix_lengths.shape) != (batch,)
        or prefix_lengths.dtype != torch.int32
        or prefix_lengths.device != device
        or not prefix_lengths.is_contiguous()
    ):
        raise ValueError(
            "prefix_lengths must be contiguous CUDA INT32 [batch] with values 1, 3, or 4"
        )
    if (
        tuple(slot_mapping.shape) != (batch, CONTROL_PREFIX_CAPACITY)
        or slot_mapping.dtype != torch.int64
        or slot_mapping.device != device
        or not slot_mapping.is_contiguous()
    ):
        raise ValueError(
            "slot_mapping must be contiguous CUDA INT64 [batch, 4]; padded positions map to reserved page 0"
        )
    cross_shape = (
        DECODER_LAYERS,
        batch,
        ENCODER_FRAMES,
        ATTENTION_HEADS,
        HEAD_DIM,
    )
    for name, cross in (
        ("compact_cross_keys", compact_cross_keys),
        ("compact_cross_values", compact_cross_values),
    ):
        if (
            tuple(cross.shape) != cross_shape
            or cross.device != device
            or cross.dtype != torch.bfloat16
            or not cross.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA BF16 with shape {cross_shape}"
            )
    if (
        tuple(logits_out.shape) != (batch, VOCAB_SIZE)
        or logits_out.device != device
        or logits_out.dtype != torch.bfloat16
        or not logits_out.is_contiguous()
    ):
        raise ValueError(
            f"logits_out must be contiguous CUDA BF16 [batch, {VOCAB_SIZE}]"
        )


def _write_self_kv(
    qkv: torch.Tensor,
    slot_mapping: torch.Tensor,
    self_kv: WhisperSelfKVArenas,
    workspace: WhisperDecoderPrefixWorkspace,
    layer_index: int,
) -> None:
    qkv_heads = qkv.view(
        workspace.batch_size,
        CONTROL_PREFIX_CAPACITY,
        3,
        ATTENTION_HEADS,
        HEAD_DIM,
    )
    keys = qkv_heads[:, :, 1].reshape(
        workspace.batch_size * CONTROL_PREFIX_CAPACITY,
        ATTENTION_HEADS,
        HEAD_DIM,
    )
    values = qkv_heads[:, :, 2].reshape_as(keys)
    key_pool = self_kv.keys[layer_index]
    value_pool = self_kv.values[layer_index]
    _CACHE.reshape_and_cache_flash(
        keys,
        values,
        key_pool.unsqueeze(2).permute(0, 2, 1, 3),
        value_pool.unsqueeze(2).permute(0, 2, 1, 3),
        slot_mapping.view(-1),
        "auto",
        workspace.kv_scale,
        workspace.kv_scale,
    )


def _run_decoder_layer(
    hidden_states: torch.Tensor,
    layer: _PackedDecoderLayer,
    layer_index: int,
    compact_cross_keys: torch.Tensor,
    compact_cross_values: torch.Tensor,
    slot_mapping: torch.Tensor,
    workspace: WhisperDecoderPrefixWorkspace,
    self_kv: WhisperSelfKVArenas,
    *,
    require_packed: bool = False,
) -> torch.Tensor:
    batch = workspace.batch_size
    rows = batch * CONTROL_PREFIX_CAPACITY
    _DENSE.layernorm_bias_into(
        workspace.normalized,
        hidden_states,
        layer.self_attention_layer_norm.weight,
        layer.self_attention_layer_norm.bias,
        LAYER_NORM_EPS,
    )
    _LINEAR.linear(
        workspace.normalized,
        layer.self_qkv_weight,
        layer.self_qkv_bias,
        out=workspace.self_qkv,
    )
    _write_self_kv(workspace.self_qkv, slot_mapping, self_kv, workspace, layer_index)
    qkv = workspace.self_qkv.view(
        batch,
        CONTROL_PREFIX_CAPACITY,
        3,
        ATTENTION_HEADS,
        HEAD_DIM,
    )
    attention, _ = _ATTENTION.flash_attn_fwd(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        softmax_scale=HEAD_DIM**-0.5,
        causal=True,
        out=workspace.attention,
        require_native=True,
        require_packed=require_packed,
    )
    if attention.data_ptr() != workspace.attention.data_ptr():
        raise RuntimeError("Whisper self-attention did not honor stable output")
    assert layer.self_attention_output.bias is not None
    _VISION.fused_linear_bias_residual_into(
        x=workspace.attention.view(batch, CONTROL_PREFIX_CAPACITY, HIDDEN_SIZE),
        w=layer.self_attention_output.weight,
        b=layer.self_attention_output.bias,
        residual=hidden_states,
        out=workspace.post_self_attention,
    )

    _DENSE.layernorm_bias_into(
        workspace.normalized,
        workspace.post_self_attention,
        layer.cross_attention_layer_norm.weight,
        layer.cross_attention_layer_norm.bias,
        LAYER_NORM_EPS,
    )
    assert layer.cross_query.bias is not None
    _LINEAR.linear(
        workspace.normalized,
        layer.cross_query.weight,
        layer.cross_query.bias,
        out=workspace.cross_query,
    )
    attention, _ = _ATTENTION.flash_attn_fwd(
        workspace.cross_query.view(
            batch, CONTROL_PREFIX_CAPACITY, ATTENTION_HEADS, HEAD_DIM
        ),
        compact_cross_keys[layer_index],
        compact_cross_values[layer_index],
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        out=workspace.attention,
        require_native=True,
        require_packed=require_packed,
    )
    if attention.data_ptr() != workspace.attention.data_ptr():
        raise RuntimeError("Whisper cross-attention did not honor stable output")
    assert layer.cross_attention_output.bias is not None
    _VISION.fused_linear_bias_residual_into(
        x=workspace.attention.view(batch, CONTROL_PREFIX_CAPACITY, HIDDEN_SIZE),
        w=layer.cross_attention_output.weight,
        b=layer.cross_attention_output.bias,
        residual=workspace.post_self_attention,
        out=workspace.post_cross_attention,
    )

    _DENSE.layernorm_bias_into(
        workspace.normalized,
        workspace.post_cross_attention,
        layer.final_layer_norm.weight,
        layer.final_layer_norm.bias,
        LAYER_NORM_EPS,
    )
    assert layer.fc1.bias is not None and layer.fc2.bias is not None
    _DENSE.fused_mlp_gelu_bias_residual(
        workspace.hidden.view(rows, HIDDEN_SIZE),
        workspace.mlp_hidden,
        workspace.normalized.view(rows, HIDDEN_SIZE),
        layer.fc1.weight,
        layer.fc1.bias,
        layer.fc2.weight,
        layer.fc2.bias,
        workspace.post_cross_attention.view(rows, HIDDEN_SIZE),
        approximate="none",
    )
    return workspace.hidden


@torch.inference_mode()
def whisper_decoder_prefix(
    control_token_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    slot_mapping: torch.Tensor,
    compact_cross_keys: torch.Tensor,
    compact_cross_values: torch.Tensor,
    weights: PreparedWhisperDecoderWeights,
    workspace: WhisperDecoderPrefixWorkspace,
    self_kv: WhisperSelfKVArenas,
    *,
    logits_out: torch.Tensor,
    require_packed: bool = False,
) -> WhisperDecoderPrefixOutput:
    """Prefill mixed 1/3/4-token control prefixes and seed decode logits.

    ``control_token_ids`` always has four columns.  Entries beyond each row's
    ``prefix_lengths`` are semantically ignored: causal attention prevents
    them from affecting valid positions, their self-K/V writes must target
    reserved physical page zero, and only the last valid hidden row is sent to
    the tied output projection.  This fixed geometry keeps one graph for mixed
    automatic-language and explicit-language requests.
    """

    # Tried one native paged-decode-style step per control token: at B1/B4/B8
    # it cost 0.99-1.06x this batched graph for length 1, while this graph was
    # 3.30-3.60x faster for length 4 on H100/B200.  Keeping one Bx4 graph so
    # mixed automatic/explicit rows share a capture without losing L4 latency.
    _validate_prefix_inputs(
        control_token_ids,
        prefix_lengths,
        slot_mapping,
        compact_cross_keys,
        compact_cross_values,
        logits_out,
        weights,
        workspace,
        self_kv,
    )
    torch.index_select(
        weights.token_embedding,
        0,
        control_token_ids.view(-1),
        out=workspace.hidden.view(-1, HIDDEN_SIZE),
    )
    workspace.hidden.add_(weights.position_embedding[:CONTROL_PREFIX_CAPACITY])
    hidden_states = workspace.hidden
    for index, layer in enumerate(weights.layers):
        hidden_states = _run_decoder_layer(
            hidden_states,
            layer,
            index,
            compact_cross_keys,
            compact_cross_values,
            slot_mapping,
            workspace,
            self_kv,
            require_packed=require_packed,
        )
    _DENSE.layernorm_bias_into(
        workspace.final_hidden,
        hidden_states,
        weights.final_layer_norm.weight,
        weights.final_layer_norm.bias,
        LAYER_NORM_EPS,
    )
    torch.sub(prefix_lengths, 1, out=workspace.last_indices)
    last_indices = workspace.last_indices.view(-1, 1, 1).expand(-1, 1, HIDDEN_SIZE)
    torch.gather(
        workspace.final_hidden,
        1,
        last_indices,
        out=workspace.last_hidden.view(workspace.batch_size, 1, HIDDEN_SIZE),
    )
    _LINEAR.linear(
        workspace.last_hidden,
        weights.token_embedding,
        None,
        out=logits_out,
    )
    return WhisperDecoderPrefixOutput(
        logits=logits_out,
        last_hidden_state=workspace.last_hidden,
    )


__all__ = [
    "CONTROL_PREFIX_CAPACITY",
    "MAX_TARGET_POSITIONS",
    "VOCAB_SIZE",
    "PreparedWhisperDecoderWeights",
    "WhisperDecoderPrefixOutput",
    "WhisperDecoderPrefixWorkspace",
    "prepare_whisper_decoder_weights",
    "whisper_decoder_prefix",
]
