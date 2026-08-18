"""Typed, inference-only ownership and strict loading for Whisper weights."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, cast

import torch
from torch import Tensor

from .config import WhisperTurboConfig


class WhisperCheckpointError(ValueError):
    """The checkpoint is not the exact tensor set required by the config."""


@dataclass(frozen=True, slots=True)
class LinearWeights:
    weight: Tensor
    bias: Tensor | None


@dataclass(frozen=True, slots=True)
class LayerNormWeights:
    weight: Tensor
    bias: Tensor


@dataclass(frozen=True, slots=True)
class Conv1dWeights:
    weight: Tensor
    bias: Tensor
    stride: int
    padding: int


@dataclass(frozen=True, slots=True)
class AttentionWeights:
    query: LinearWeights
    key: LinearWeights
    value: LinearWeights
    output: LinearWeights


@dataclass(frozen=True, slots=True)
class EncoderLayerWeights:
    self_attention_layer_norm: LayerNormWeights
    self_attention: AttentionWeights
    final_layer_norm: LayerNormWeights
    fc1: LinearWeights
    fc2: LinearWeights


@dataclass(frozen=True, slots=True)
class DecoderLayerWeights:
    self_attention_layer_norm: LayerNormWeights
    self_attention: AttentionWeights
    cross_attention_layer_norm: LayerNormWeights
    cross_attention: AttentionWeights
    final_layer_norm: LayerNormWeights
    fc1: LinearWeights
    fc2: LinearWeights


@dataclass(frozen=True, slots=True)
class WhisperEncoderWeights:
    conv1: Conv1dWeights
    conv2: Conv1dWeights
    position_embedding: Tensor
    layers: tuple[EncoderLayerWeights, ...]
    final_layer_norm: LayerNormWeights


@dataclass(frozen=True, slots=True)
class WhisperDecoderWeights:
    token_embedding: Tensor
    position_embedding: Tensor
    layers: tuple[DecoderLayerWeights, ...]
    final_layer_norm: LayerNormWeights

    @property
    def output_projection(self) -> Tensor:
        """Whisper ties ``proj_out.weight`` to the decoder token embedding."""

        return self.token_embedding


_WeightTree = TypeVar("_WeightTree", bound="WhisperModelWeights")


@dataclass(frozen=True, slots=True)
class WhisperModelWeights:
    encoder: WhisperEncoderWeights
    decoder: WhisperDecoderWeights

    def to(
        self: _WeightTree,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> _WeightTree:
        """Return an inference weight tree moved without creating parameters."""

        return cast(_WeightTree, _map_weight_tree(self, device=device, dtype=dtype))


def _map_weight_tree(
    value: Any,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> Any:
    if isinstance(value, Tensor):
        return value.detach().to(device=device, dtype=dtype).contiguous()
    if isinstance(value, tuple):
        return tuple(
            _map_weight_tree(item, device=device, dtype=dtype) for item in value
        )
    if is_dataclass(value):
        return type(value)(
            **{
                field.name: _map_weight_tree(
                    getattr(value, field.name), device=device, dtype=dtype
                )
                for field in fields(value)
            }
        )
    return value


def _add_linear(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    out_features: int,
    in_features: int,
    *,
    bias: bool,
) -> None:
    shapes[f"{prefix}.weight"] = (out_features, in_features)
    if bias:
        shapes[f"{prefix}.bias"] = (out_features,)


def _add_layer_norm(
    shapes: dict[str, tuple[int, ...]], prefix: str, width: int
) -> None:
    shapes[f"{prefix}.weight"] = (width,)
    shapes[f"{prefix}.bias"] = (width,)


def _add_attention(shapes: dict[str, tuple[int, ...]], prefix: str, width: int) -> None:
    _add_linear(shapes, f"{prefix}.q_proj", width, width, bias=True)
    _add_linear(shapes, f"{prefix}.k_proj", width, width, bias=False)
    _add_linear(shapes, f"{prefix}.v_proj", width, width, bias=True)
    _add_linear(shapes, f"{prefix}.out_proj", width, width, bias=True)


def expected_whisper_checkpoint_shapes(
    config: WhisperTurboConfig,
) -> Mapping[str, tuple[int, ...]]:
    """Return the complete official HF key/shape contract for ``config``."""

    d_model = config.d_model
    shapes: dict[str, tuple[int, ...]] = {
        "model.encoder.conv1.weight": (d_model, config.num_mel_bins, 3),
        "model.encoder.conv1.bias": (d_model,),
        "model.encoder.conv2.weight": (d_model, d_model, 3),
        "model.encoder.conv2.bias": (d_model,),
        "model.encoder.embed_positions.weight": (
            config.max_source_positions,
            d_model,
        ),
        "model.decoder.embed_tokens.weight": (config.vocab_size, d_model),
        "model.decoder.embed_positions.weight": (
            config.max_target_positions,
            d_model,
        ),
    }
    _add_layer_norm(shapes, "model.encoder.layer_norm", d_model)
    _add_layer_norm(shapes, "model.decoder.layer_norm", d_model)

    for index in range(config.encoder_layers):
        prefix = f"model.encoder.layers.{index}"
        _add_attention(shapes, f"{prefix}.self_attn", d_model)
        _add_layer_norm(shapes, f"{prefix}.self_attn_layer_norm", d_model)
        _add_linear(
            shapes,
            f"{prefix}.fc1",
            config.encoder_ffn_dim,
            d_model,
            bias=True,
        )
        _add_linear(
            shapes,
            f"{prefix}.fc2",
            d_model,
            config.encoder_ffn_dim,
            bias=True,
        )
        _add_layer_norm(shapes, f"{prefix}.final_layer_norm", d_model)

    for index in range(config.decoder_layers):
        prefix = f"model.decoder.layers.{index}"
        _add_attention(shapes, f"{prefix}.self_attn", d_model)
        _add_layer_norm(shapes, f"{prefix}.self_attn_layer_norm", d_model)
        _add_attention(shapes, f"{prefix}.encoder_attn", d_model)
        _add_layer_norm(shapes, f"{prefix}.encoder_attn_layer_norm", d_model)
        _add_linear(
            shapes,
            f"{prefix}.fc1",
            config.decoder_ffn_dim,
            d_model,
            bias=True,
        )
        _add_linear(
            shapes,
            f"{prefix}.fc2",
            d_model,
            config.decoder_ffn_dim,
            bias=True,
        )
        _add_layer_norm(shapes, f"{prefix}.final_layer_norm", d_model)
    return shapes


def _name_linear(named: dict[str, Tensor], prefix: str, linear: LinearWeights) -> None:
    named[f"{prefix}.weight"] = linear.weight
    if linear.bias is not None:
        named[f"{prefix}.bias"] = linear.bias


def _name_layer_norm(
    named: dict[str, Tensor], prefix: str, layer_norm: LayerNormWeights
) -> None:
    named[f"{prefix}.weight"] = layer_norm.weight
    named[f"{prefix}.bias"] = layer_norm.bias


def _name_attention(
    named: dict[str, Tensor], prefix: str, attention: AttentionWeights
) -> None:
    _name_linear(named, f"{prefix}.q_proj", attention.query)
    _name_linear(named, f"{prefix}.k_proj", attention.key)
    _name_linear(named, f"{prefix}.v_proj", attention.value)
    _name_linear(named, f"{prefix}.out_proj", attention.output)


def named_whisper_tensors(weights: WhisperModelWeights) -> Mapping[str, Tensor]:
    """Invert the typed tree into the official checkpoint ownership names."""

    encoder = weights.encoder
    decoder = weights.decoder
    named: dict[str, Tensor] = {
        "model.encoder.conv1.weight": encoder.conv1.weight,
        "model.encoder.conv1.bias": encoder.conv1.bias,
        "model.encoder.conv2.weight": encoder.conv2.weight,
        "model.encoder.conv2.bias": encoder.conv2.bias,
        "model.encoder.embed_positions.weight": encoder.position_embedding,
        "model.decoder.embed_tokens.weight": decoder.token_embedding,
        "model.decoder.embed_positions.weight": decoder.position_embedding,
    }
    _name_layer_norm(named, "model.encoder.layer_norm", encoder.final_layer_norm)
    _name_layer_norm(named, "model.decoder.layer_norm", decoder.final_layer_norm)
    for index, layer in enumerate(encoder.layers):
        prefix = f"model.encoder.layers.{index}"
        _name_attention(named, f"{prefix}.self_attn", layer.self_attention)
        _name_layer_norm(
            named,
            f"{prefix}.self_attn_layer_norm",
            layer.self_attention_layer_norm,
        )
        _name_linear(named, f"{prefix}.fc1", layer.fc1)
        _name_linear(named, f"{prefix}.fc2", layer.fc2)
        _name_layer_norm(named, f"{prefix}.final_layer_norm", layer.final_layer_norm)
    for index, layer in enumerate(decoder.layers):
        prefix = f"model.decoder.layers.{index}"
        _name_attention(named, f"{prefix}.self_attn", layer.self_attention)
        _name_layer_norm(
            named,
            f"{prefix}.self_attn_layer_norm",
            layer.self_attention_layer_norm,
        )
        _name_attention(named, f"{prefix}.encoder_attn", layer.cross_attention)
        _name_layer_norm(
            named,
            f"{prefix}.encoder_attn_layer_norm",
            layer.cross_attention_layer_norm,
        )
        _name_linear(named, f"{prefix}.fc1", layer.fc1)
        _name_linear(named, f"{prefix}.fc2", layer.fc2)
        _name_layer_norm(named, f"{prefix}.final_layer_norm", layer.final_layer_norm)
    return named


def validate_whisper_weight_tree(
    weights: WhisperModelWeights,
    config: WhisperTurboConfig,
) -> tuple[torch.device, torch.dtype]:
    """Validate typed ownership, shapes, and one uniform inference placement."""

    named = named_whisper_tensors(weights)
    expected = expected_whisper_checkpoint_shapes(config)
    if set(named) != set(expected):
        raise WhisperCheckpointError(
            "Typed Whisper weight ownership does not match the configured layer counts"
        )
    first: Tensor | None = None
    for name, expected_shape in expected.items():
        tensor = named[name]
        if tuple(tensor.shape) != expected_shape:
            raise WhisperCheckpointError(
                f"Whisper tensor {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}"
            )
        if not tensor.dtype.is_floating_point:
            raise WhisperCheckpointError(f"Whisper tensor {name!r} must be floating")
        if tensor.layout is not torch.strided or not tensor.is_contiguous():
            raise WhisperCheckpointError(
                f"Whisper tensor {name!r} must be contiguous strided storage"
            )
        if tensor.requires_grad:
            raise WhisperCheckpointError(
                f"Whisper inference tensor {name!r} must not require gradients"
            )
        if first is None:
            first = tensor
        elif tensor.device != first.device or tensor.dtype is not first.dtype:
            raise WhisperCheckpointError(
                "Whisper inference weights must share one device and dtype"
            )
    if first is None:  # pragma: no cover - positive layer/config invariants
        raise WhisperCheckpointError("Whisper weight tree is empty")
    if weights.decoder.output_projection is not weights.decoder.token_embedding:
        raise WhisperCheckpointError("Whisper output projection must remain tied")
    return first.device, first.dtype


class _TensorReader:
    def __init__(
        self,
        tensors: Mapping[str, Tensor],
        expected_shapes: Mapping[str, tuple[int, ...]],
        *,
        checkpoint_dtype: torch.dtype,
    ) -> None:
        self._tensors = tensors
        self._expected_shapes = expected_shapes
        self._checkpoint_dtype = checkpoint_dtype

        actual = set(tensors)
        expected = set(expected_shapes)
        optional = {"proj_out.weight"}
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected - optional)
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected: " + ", ".join(unexpected))
            raise WhisperCheckpointError(
                "Whisper checkpoint key mismatch (" + "; ".join(details) + ")"
            )

    def tensor(self, name: str) -> Tensor:
        tensor = self._tensors[name]
        expected_shape = self._expected_shapes[name]
        if tuple(tensor.shape) != expected_shape:
            raise WhisperCheckpointError(
                f"Whisper tensor {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}"
            )
        if tensor.dtype is not self._checkpoint_dtype:
            raise WhisperCheckpointError(
                f"Whisper tensor {name!r} has dtype {tensor.dtype}, "
                f"expected {self._checkpoint_dtype}"
            )
        if tensor.layout is not torch.strided or not tensor.is_contiguous():
            raise WhisperCheckpointError(
                f"Whisper tensor {name!r} must be contiguous strided storage"
            )
        return tensor.detach()

    def optional_tied_projection(self, token_embedding: Tensor) -> None:
        projection = self._tensors.get("proj_out.weight")
        if projection is None:
            return
        expected_shape = self._expected_shapes["model.decoder.embed_tokens.weight"]
        if tuple(projection.shape) != expected_shape:
            raise WhisperCheckpointError(
                f"Whisper tensor 'proj_out.weight' has shape {tuple(projection.shape)}, "
                f"expected {expected_shape}"
            )
        if projection.dtype is not self._checkpoint_dtype:
            raise WhisperCheckpointError(
                "Whisper tensor 'proj_out.weight' has the wrong dtype"
            )
        if not torch.equal(projection, token_embedding):
            raise WhisperCheckpointError(
                "Whisper proj_out.weight must be exactly tied to embed_tokens.weight"
            )


def _linear(reader: _TensorReader, prefix: str, *, bias: bool) -> LinearWeights:
    return LinearWeights(
        weight=reader.tensor(f"{prefix}.weight"),
        bias=reader.tensor(f"{prefix}.bias") if bias else None,
    )


def _layer_norm(reader: _TensorReader, prefix: str) -> LayerNormWeights:
    return LayerNormWeights(
        weight=reader.tensor(f"{prefix}.weight"),
        bias=reader.tensor(f"{prefix}.bias"),
    )


def _attention(reader: _TensorReader, prefix: str) -> AttentionWeights:
    return AttentionWeights(
        query=_linear(reader, f"{prefix}.q_proj", bias=True),
        key=_linear(reader, f"{prefix}.k_proj", bias=False),
        value=_linear(reader, f"{prefix}.v_proj", bias=True),
        output=_linear(reader, f"{prefix}.out_proj", bias=True),
    )


def _build_weight_tree(
    tensors: Mapping[str, Tensor],
    config: WhisperTurboConfig,
    *,
    checkpoint_dtype: torch.dtype,
) -> WhisperModelWeights:
    reader = _TensorReader(
        tensors,
        expected_whisper_checkpoint_shapes(config),
        checkpoint_dtype=checkpoint_dtype,
    )

    encoder_layers = []
    for index in range(config.encoder_layers):
        prefix = f"model.encoder.layers.{index}"
        encoder_layers.append(
            EncoderLayerWeights(
                self_attention_layer_norm=_layer_norm(
                    reader, f"{prefix}.self_attn_layer_norm"
                ),
                self_attention=_attention(reader, f"{prefix}.self_attn"),
                final_layer_norm=_layer_norm(reader, f"{prefix}.final_layer_norm"),
                fc1=_linear(reader, f"{prefix}.fc1", bias=True),
                fc2=_linear(reader, f"{prefix}.fc2", bias=True),
            )
        )

    token_embedding = reader.tensor("model.decoder.embed_tokens.weight")
    decoder_layers = []
    for index in range(config.decoder_layers):
        prefix = f"model.decoder.layers.{index}"
        decoder_layers.append(
            DecoderLayerWeights(
                self_attention_layer_norm=_layer_norm(
                    reader, f"{prefix}.self_attn_layer_norm"
                ),
                self_attention=_attention(reader, f"{prefix}.self_attn"),
                cross_attention_layer_norm=_layer_norm(
                    reader, f"{prefix}.encoder_attn_layer_norm"
                ),
                cross_attention=_attention(reader, f"{prefix}.encoder_attn"),
                final_layer_norm=_layer_norm(reader, f"{prefix}.final_layer_norm"),
                fc1=_linear(reader, f"{prefix}.fc1", bias=True),
                fc2=_linear(reader, f"{prefix}.fc2", bias=True),
            )
        )

    reader.optional_tied_projection(token_embedding)
    return WhisperModelWeights(
        encoder=WhisperEncoderWeights(
            conv1=Conv1dWeights(
                weight=reader.tensor("model.encoder.conv1.weight"),
                bias=reader.tensor("model.encoder.conv1.bias"),
                stride=1,
                padding=1,
            ),
            conv2=Conv1dWeights(
                weight=reader.tensor("model.encoder.conv2.weight"),
                bias=reader.tensor("model.encoder.conv2.bias"),
                stride=2,
                padding=1,
            ),
            position_embedding=reader.tensor("model.encoder.embed_positions.weight"),
            layers=tuple(encoder_layers),
            final_layer_norm=_layer_norm(reader, "model.encoder.layer_norm"),
        ),
        decoder=WhisperDecoderWeights(
            token_embedding=token_embedding,
            position_embedding=reader.tensor("model.decoder.embed_positions.weight"),
            layers=tuple(decoder_layers),
            final_layer_norm=_layer_norm(reader, "model.decoder.layer_norm"),
        ),
    )


def load_whisper_safetensors(
    path: str | Path,
    config: WhisperTurboConfig,
    *,
    checkpoint_dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> WhisperModelWeights:
    """Load one exact HF Whisper safetensors file into typed owners.

    The pinned production checkpoint is entirely FP16 and omits
    ``proj_out.weight`` because it is tied. Tests may explicitly request FP32.
    """

    if checkpoint_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("checkpoint_dtype must be float16, bfloat16, or float32")
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Whisper safetensors file not found: {checkpoint_path}"
        )

    from safetensors import safe_open

    try:
        with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("format") != "pt":
                raise WhisperCheckpointError(
                    "Whisper safetensors metadata must declare format='pt'"
                )
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    except WhisperCheckpointError:
        raise
    except Exception as exc:
        raise WhisperCheckpointError(
            f"Could not read Whisper safetensors checkpoint: {exc}"
        ) from exc

    weights = _build_weight_tree(
        tensors,
        config,
        checkpoint_dtype=checkpoint_dtype,
    )
    if device is not None or dtype is not None:
        weights = weights.to(device=device, dtype=dtype)
    return weights


__all__ = [
    "AttentionWeights",
    "Conv1dWeights",
    "DecoderLayerWeights",
    "EncoderLayerWeights",
    "LayerNormWeights",
    "LinearWeights",
    "WhisperCheckpointError",
    "WhisperDecoderWeights",
    "WhisperEncoderWeights",
    "WhisperModelWeights",
    "expected_whisper_checkpoint_shapes",
    "load_whisper_safetensors",
    "named_whisper_tensors",
    "validate_whisper_weight_tree",
]
