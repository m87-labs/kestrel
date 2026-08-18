"""Fixed-geometry Whisper large-v3-turbo encoder and cross-K/V prefill."""

from dataclasses import dataclass

import torch

from kestrel_kernels import get_runtime

from .prefill_stem import (
    ENCODER_FRAMES,
    HIDDEN_SIZE,
    WhisperAudioStemWeights,
    _validate_prepared_weights,
    prepare_whisper_audio_stem_weights,
    whisper_audio_stem,
)
from .runtime_abi import WhisperCrossArenas
from .weights import (
    AttentionWeights,
    LayerNormWeights,
    LinearWeights,
    WhisperModelWeights,
)

_KERNELS = get_runtime()
_ATTENTION = _KERNELS.attention
_DENSE = _KERNELS.dense
_LINEAR = _KERNELS.linear
_VISION = _KERNELS.vision

ENCODER_LAYERS = 32
DECODER_LAYERS = 4
ATTENTION_HEADS = 20
HEAD_DIM = 64
FFN_DIM = 5120
LAYER_NORM_EPS = 1e-5


@dataclass(frozen=True)
class _PackedEncoderLayer:
    self_attention_layer_norm: LayerNormWeights
    qkv_weight: torch.Tensor
    qkv_bias: torch.Tensor
    attention_output: LinearWeights
    final_layer_norm: LayerNormWeights
    fc1: LinearWeights
    fc2: LinearWeights


@dataclass(frozen=True)
class _PackedCrossProjection:
    key: LinearWeights
    value: LinearWeights


@dataclass(frozen=True)
class PreparedWhisperEncoderWeights:
    """BF16 inference weights, including load-time fused Q|K|V projections."""

    stem: WhisperAudioStemWeights
    layers: tuple[_PackedEncoderLayer, ...]
    final_layer_norm: LayerNormWeights
    cross_projections: tuple[_PackedCrossProjection, ...]
    device: torch.device


@dataclass
class WhisperEncoderWorkspace:
    """All mutable buffers for one fixed CUDA-graph batch bucket."""

    batch_size: int
    hidden: torch.Tensor
    post_attention: torch.Tensor
    normalized: torch.Tensor
    qkv: torch.Tensor
    attention: torch.Tensor
    mlp_hidden: torch.Tensor
    encoder_output: torch.Tensor
    compact_cross_keys: torch.Tensor
    compact_cross_values: torch.Tensor

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> "WhisperEncoderWorkspace":
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("Whisper encoder workspace batch_size must be positive")
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("Whisper encoder workspace requires a CUDA device")
        dtype = torch.bfloat16
        hidden_shape = (batch_size, ENCODER_FRAMES, HIDDEN_SIZE)
        cross_shape = (
            DECODER_LAYERS,
            batch_size,
            ENCODER_FRAMES,
            ATTENTION_HEADS,
            HEAD_DIM,
        )
        return cls(
            batch_size=batch_size,
            hidden=torch.empty(hidden_shape, device=resolved_device, dtype=dtype),
            post_attention=torch.empty(
                hidden_shape, device=resolved_device, dtype=dtype
            ),
            normalized=torch.empty(hidden_shape, device=resolved_device, dtype=dtype),
            qkv=torch.empty(
                (batch_size, ENCODER_FRAMES, 3 * HIDDEN_SIZE),
                device=resolved_device,
                dtype=dtype,
            ),
            attention=torch.empty(
                (
                    batch_size,
                    ENCODER_FRAMES,
                    ATTENTION_HEADS,
                    HEAD_DIM,
                ),
                device=resolved_device,
                dtype=dtype,
            ),
            mlp_hidden=torch.empty(
                (batch_size * ENCODER_FRAMES, FFN_DIM),
                device=resolved_device,
                dtype=dtype,
            ),
            encoder_output=torch.empty(
                hidden_shape, device=resolved_device, dtype=dtype
            ),
            compact_cross_keys=torch.empty(
                cross_shape, device=resolved_device, dtype=dtype
            ),
            compact_cross_values=torch.empty(
                cross_shape, device=resolved_device, dtype=dtype
            ),
        )

    @property
    def device(self) -> torch.device:
        return self.hidden.device


def _prepared_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"{name} must be bfloat16 or float32, got {tensor.dtype}")
    return tensor.detach().to(dtype=torch.bfloat16).contiguous()


def _prepare_linear(
    name: str,
    weights: LinearWeights,
    *,
    out_features: int,
    in_features: int,
    bias: bool,
    device: torch.device,
) -> LinearWeights:
    weight = _prepared_tensor(
        f"{name}.weight", weights.weight, (out_features, in_features), device
    )
    if bias:
        if weights.bias is None:
            raise ValueError(f"{name}.bias is required")
        prepared_bias = _prepared_tensor(
            f"{name}.bias", weights.bias, (out_features,), device
        )
    else:
        if weights.bias is not None:
            raise ValueError(f"{name}.bias must be None")
        prepared_bias = None
    return LinearWeights(weight, prepared_bias)


def _prepare_layer_norm(
    name: str,
    weights: LayerNormWeights,
    device: torch.device,
) -> LayerNormWeights:
    return LayerNormWeights(
        _prepared_tensor(f"{name}.weight", weights.weight, (HIDDEN_SIZE,), device),
        _prepared_tensor(f"{name}.bias", weights.bias, (HIDDEN_SIZE,), device),
    )


def _prepare_attention(
    name: str,
    weights: AttentionWeights,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, LinearWeights]:
    query = _prepare_linear(
        f"{name}.query",
        weights.query,
        out_features=HIDDEN_SIZE,
        in_features=HIDDEN_SIZE,
        bias=True,
        device=device,
    )
    key = _prepare_linear(
        f"{name}.key",
        weights.key,
        out_features=HIDDEN_SIZE,
        in_features=HIDDEN_SIZE,
        bias=False,
        device=device,
    )
    value = _prepare_linear(
        f"{name}.value",
        weights.value,
        out_features=HIDDEN_SIZE,
        in_features=HIDDEN_SIZE,
        bias=True,
        device=device,
    )
    output = _prepare_linear(
        f"{name}.output",
        weights.output,
        out_features=HIDDEN_SIZE,
        in_features=HIDDEN_SIZE,
        bias=True,
        device=device,
    )
    assert query.bias is not None and value.bias is not None
    qkv_weight = torch.cat((query.weight, key.weight, value.weight), dim=0).contiguous()
    qkv_bias = torch.cat(
        (query.bias, torch.zeros_like(query.bias), value.bias), dim=0
    ).contiguous()
    return qkv_weight, qkv_bias, output


def prepare_whisper_encoder_weights(
    weights: WhisperModelWeights,
) -> PreparedWhisperEncoderWeights:
    """Validate and prepack the pinned encoder and decoder cross projections."""

    encoder = weights.encoder
    decoder = weights.decoder
    stem = prepare_whisper_audio_stem_weights(
        encoder.conv1.weight,
        encoder.conv1.bias,
        encoder.conv2.weight,
        encoder.conv2.bias,
        encoder.position_embedding,
    )
    encoder_layers = encoder.layers
    cross_projections = tuple(layer.cross_attention for layer in decoder.layers)
    if len(encoder_layers) != ENCODER_LAYERS:
        raise ValueError(
            f"Whisper Turbo requires {ENCODER_LAYERS} encoder layers, "
            f"got {len(encoder_layers)}"
        )
    if len(cross_projections) != DECODER_LAYERS:
        raise ValueError(
            f"Whisper Turbo requires {DECODER_LAYERS} cross projections, "
            f"got {len(cross_projections)}"
        )
    _validate_prepared_weights(stem)
    device = stem.conv1_weight.device
    packed_layers = []
    for index, layer in enumerate(encoder_layers):
        prefix = f"encoder_layers[{index}]"
        self_ln = _prepare_layer_norm(
            f"{prefix}.self_attention_layer_norm",
            layer.self_attention_layer_norm,
            device,
        )
        qkv_weight, qkv_bias, attention_output = _prepare_attention(
            f"{prefix}.self_attention", layer.self_attention, device
        )
        final_ln = _prepare_layer_norm(
            f"{prefix}.final_layer_norm", layer.final_layer_norm, device
        )
        fc1 = _prepare_linear(
            f"{prefix}.fc1",
            layer.fc1,
            out_features=FFN_DIM,
            in_features=HIDDEN_SIZE,
            bias=True,
            device=device,
        )
        fc2 = _prepare_linear(
            f"{prefix}.fc2",
            layer.fc2,
            out_features=HIDDEN_SIZE,
            in_features=FFN_DIM,
            bias=True,
            device=device,
        )
        packed_layers.append(
            _PackedEncoderLayer(
                self_attention_layer_norm=self_ln,
                qkv_weight=qkv_weight,
                qkv_bias=qkv_bias,
                attention_output=attention_output,
                final_layer_norm=final_ln,
                fc1=fc1,
                fc2=fc2,
            )
        )

    packed_cross = []
    for index, attention in enumerate(cross_projections):
        prefix = f"cross_projections[{index}]"
        packed_cross.append(
            _PackedCrossProjection(
                key=_prepare_linear(
                    f"{prefix}.key",
                    attention.key,
                    out_features=HIDDEN_SIZE,
                    in_features=HIDDEN_SIZE,
                    bias=False,
                    device=device,
                ),
                value=_prepare_linear(
                    f"{prefix}.value",
                    attention.value,
                    out_features=HIDDEN_SIZE,
                    in_features=HIDDEN_SIZE,
                    bias=True,
                    device=device,
                ),
            )
        )

    return PreparedWhisperEncoderWeights(
        stem=stem,
        layers=tuple(packed_layers),
        final_layer_norm=_prepare_layer_norm(
            "final_layer_norm", encoder.final_layer_norm, device
        ),
        cross_projections=tuple(packed_cross),
        device=device,
    )


def _validate_workspace(
    workspace: WhisperEncoderWorkspace,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if workspace.batch_size != batch_size:
        raise ValueError(
            f"workspace batch_size is {workspace.batch_size}, input batch is {batch_size}"
        )
    if workspace.device != device:
        raise ValueError(f"workspace is on {workspace.device}, weights are on {device}")


def _run_encoder_layer(
    hidden_states: torch.Tensor,
    layer: _PackedEncoderLayer,
    workspace: WhisperEncoderWorkspace,
    *,
    require_packed: bool = False,
) -> torch.Tensor:
    batch = workspace.batch_size
    rows = batch * ENCODER_FRAMES
    _DENSE.layernorm_bias_into(
        workspace.normalized,
        hidden_states,
        layer.self_attention_layer_norm.weight,
        layer.self_attention_layer_norm.bias,
        LAYER_NORM_EPS,
    )
    _LINEAR.linear(
        workspace.normalized,
        layer.qkv_weight,
        layer.qkv_bias,
        out=workspace.qkv,
    )
    qkv = workspace.qkv.view(batch, ENCODER_FRAMES, 3, ATTENTION_HEADS, HEAD_DIM)
    attention, _ = _ATTENTION.flash_attn_fwd(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        out=workspace.attention,
        require_native=True,
        require_packed=require_packed,
    )
    if attention.data_ptr() != workspace.attention.data_ptr():
        raise RuntimeError("Whisper attention did not honor its stable output buffer")
    assert layer.attention_output.bias is not None
    _VISION.fused_linear_bias_residual_into(
        x=workspace.attention.view(batch, ENCODER_FRAMES, HIDDEN_SIZE),
        w=layer.attention_output.weight,
        b=layer.attention_output.bias,
        residual=hidden_states,
        out=workspace.post_attention,
    )
    _DENSE.layernorm_bias_into(
        workspace.normalized,
        workspace.post_attention,
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
        workspace.post_attention.view(rows, HIDDEN_SIZE),
        approximate="none",
    )
    return workspace.hidden


@torch.inference_mode()
def whisper_encoder(
    input_features: torch.Tensor,
    weights: PreparedWhisperEncoderWeights,
    workspace: WhisperEncoderWorkspace,
    *,
    require_packed: bool = False,
) -> torch.Tensor:
    """Run all 32 encoder layers into the workspace's stable output tensor."""

    if not input_features.is_cuda:
        raise ValueError("Whisper encoder input_features must be CUDA")
    if input_features.device != weights.device:
        raise ValueError("Whisper encoder input_features must match weight placement")
    _validate_workspace(
        workspace,
        batch_size=int(input_features.shape[0]),
        device=weights.device,
    )
    hidden_states = whisper_audio_stem(
        input_features,
        weights.stem,
        out=workspace.hidden,
        require_native=True,
        require_packed=require_packed,
    )
    for layer in weights.layers:
        hidden_states = _run_encoder_layer(
            hidden_states,
            layer,
            workspace,
            require_packed=require_packed,
        )
    _DENSE.layernorm_bias_into(
        workspace.encoder_output,
        hidden_states,
        weights.final_layer_norm.weight,
        weights.final_layer_norm.bias,
        LAYER_NORM_EPS,
    )
    return workspace.encoder_output


@torch.inference_mode()
def whisper_cross_kv(
    encoder_hidden_states: torch.Tensor,
    weights: PreparedWhisperEncoderWeights,
    workspace: WhisperEncoderWorkspace,
    cross_kv: WhisperCrossArenas,
    batch_idx: torch.Tensor,
) -> WhisperCrossArenas:
    """Project compact rows, then scatter into persistent global arenas."""

    expected = (workspace.batch_size, ENCODER_FRAMES, HIDDEN_SIZE)
    if tuple(encoder_hidden_states.shape) != expected:
        raise ValueError(
            f"encoder_hidden_states must have shape {expected}, "
            f"got {tuple(encoder_hidden_states.shape)}"
        )
    if (
        encoder_hidden_states.device != weights.device
        or encoder_hidden_states.dtype != torch.bfloat16
        or not encoder_hidden_states.is_contiguous()
    ):
        raise ValueError(
            "encoder_hidden_states must be contiguous BF16 with the weights"
        )
    _validate_workspace(
        workspace,
        batch_size=int(encoder_hidden_states.shape[0]),
        device=weights.device,
    )
    if cross_kv.keys.device != weights.device:
        raise ValueError("global cross K/V must be on the weight device")
    if (
        tuple(batch_idx.shape) != (workspace.batch_size,)
        or batch_idx.dtype != torch.int64
        or batch_idx.device != weights.device
        or not batch_idx.is_contiguous()
    ):
        raise ValueError(
            "batch_idx must be contiguous CUDA INT64 with one global row per input"
        )
    rows = workspace.batch_size * ENCODER_FRAMES
    encoder_rows = encoder_hidden_states.view(rows, HIDDEN_SIZE)
    # Tried fusing K|V per layer (4 GEMMs + 8 copies) and across all layers
    # (1 GEMM + 2 transpose-copies): with the same global scatter they took
    # 1.16-1.40x and 1.13-1.42x the latency of these 8 direct GEMMs at
    # B1/B4/B8 on H100/B200. Keeping direct projections, which also avoid
    # additional 2D/8D scratch arenas.
    for index, projection in enumerate(weights.cross_projections):
        _LINEAR.linear(
            encoder_rows,
            projection.key.weight,
            None,
            out=workspace.compact_cross_keys[index].view(rows, HIDDEN_SIZE),
        )
        assert projection.value.bias is not None
        _LINEAR.linear(
            encoder_rows,
            projection.value.weight,
            projection.value.bias,
            out=workspace.compact_cross_values[index].view(rows, HIDDEN_SIZE),
        )
    # Tried scattering each layer separately (8 launches): it ranged from
    # 7.2% faster to 8.7% slower cross-only, but improved the full pipeline by
    # at most 0.21%. Keeping two whole-arena launches and the simpler graph.
    cross_kv.keys.index_copy_(1, batch_idx, workspace.compact_cross_keys)
    cross_kv.values.index_copy_(1, batch_idx, workspace.compact_cross_values)
    return cross_kv


__all__ = [
    "ATTENTION_HEADS",
    "DECODER_LAYERS",
    "ENCODER_LAYERS",
    "FFN_DIM",
    "HEAD_DIM",
    "PreparedWhisperEncoderWeights",
    "WhisperEncoderWorkspace",
    "prepare_whisper_encoder_weights",
    "whisper_cross_kv",
    "whisper_encoder",
]
