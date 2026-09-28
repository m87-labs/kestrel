"""Strict configuration parsing for the inference-only DINOv2 integration.

Field names mirror Hugging Face DINOv2 and BiT processor JSON files. The
implementation is independent of ``transformers`` at runtime.

Configuration semantics adapted from Hugging Face Transformers (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _unknown_keys(cls: type, data: Mapping[str, Any]) -> set[str]:
    return set(data) - {field.name for field in fields(cls)}


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _require_float_tuple(value: Any, name: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise TypeError(f"{name} must contain exactly {length} numbers")
    return tuple(_require_float(item, f"{name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class Dinov2Config:
    """The fields present in the pinned ``facebook/dinov2-small`` config."""

    architectures: tuple[str, ...]
    attention_probs_dropout_prob: float
    drop_path_rate: float
    hidden_act: str
    hidden_dropout_prob: float
    hidden_size: int
    image_size: int
    initializer_range: float
    layer_norm_eps: float
    layerscale_value: float
    mlp_ratio: int
    model_type: str
    num_attention_heads: int
    num_channels: int
    num_hidden_layers: int
    patch_size: int
    qkv_bias: bool
    torch_dtype: str
    transformers_version: str
    use_swiglu_ffn: bool

    @property
    def intermediate_size(self) -> int:
        return self.hidden_size * self.mlp_ratio

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def base_grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_position_embeddings(self) -> int:
        return self.base_grid_size**2 + 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dinov2Config":
        data = _require_mapping(data, "model config")
        unknown = _unknown_keys(cls, data)
        if unknown:
            raise ValueError(f"unsupported DINOv2 config fields: {sorted(unknown)}")
        missing = {field.name for field in fields(cls)} - set(data)
        if missing:
            raise ValueError(f"missing DINOv2 config fields: {sorted(missing)}")

        architectures = data["architectures"]
        if not isinstance(architectures, (list, tuple)) or not all(
            isinstance(item, str) for item in architectures
        ):
            raise TypeError("architectures must be a sequence of strings")

        string_fields = ("hidden_act", "model_type", "torch_dtype", "transformers_version")
        for name in string_fields:
            if not isinstance(data[name], str):
                raise TypeError(f"{name} must be a string")

        return cls(
            architectures=tuple(architectures),
            attention_probs_dropout_prob=_require_float(
                data["attention_probs_dropout_prob"], "attention_probs_dropout_prob"
            ),
            drop_path_rate=_require_float(data["drop_path_rate"], "drop_path_rate"),
            hidden_act=data["hidden_act"],
            hidden_dropout_prob=_require_float(
                data["hidden_dropout_prob"], "hidden_dropout_prob"
            ),
            hidden_size=_require_int(data["hidden_size"], "hidden_size"),
            image_size=_require_int(data["image_size"], "image_size"),
            initializer_range=_require_float(data["initializer_range"], "initializer_range"),
            layer_norm_eps=_require_float(data["layer_norm_eps"], "layer_norm_eps"),
            layerscale_value=_require_float(data["layerscale_value"], "layerscale_value"),
            mlp_ratio=_require_int(data["mlp_ratio"], "mlp_ratio"),
            model_type=data["model_type"],
            num_attention_heads=_require_int(
                data["num_attention_heads"], "num_attention_heads"
            ),
            num_channels=_require_int(data["num_channels"], "num_channels"),
            num_hidden_layers=_require_int(data["num_hidden_layers"], "num_hidden_layers"),
            patch_size=_require_int(data["patch_size"], "patch_size"),
            qkv_bias=_require_bool(data["qkv_bias"], "qkv_bias"),
            torch_dtype=data["torch_dtype"],
            transformers_version=data["transformers_version"],
            use_swiglu_ffn=_require_bool(data["use_swiglu_ffn"], "use_swiglu_ffn"),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Dinov2Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive")

    def validate_v1(self) -> None:
        """Reject checkpoint variants outside the shipped ViT-S/14 slice."""

        self.validate()
        expected: dict[str, Any] = {
            "architectures": ("Dinov2Model",),
            "attention_probs_dropout_prob": 0.0,
            "drop_path_rate": 0.0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 384,
            "image_size": 518,
            "layer_norm_eps": 1e-6,
            "layerscale_value": 1.0,
            "mlp_ratio": 4,
            "model_type": "dinov2",
            "num_attention_heads": 6,
            "num_channels": 3,
            "num_hidden_layers": 12,
            "patch_size": 14,
            "qkv_bias": True,
            "use_swiglu_ffn": False,
        }
        mismatches = [
            f"{name}={getattr(self, name)!r} (expected {value!r})"
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if mismatches:
            raise ValueError("unsupported DINOv2 V1 architecture: " + "; ".join(mismatches))


@dataclass(frozen=True)
class Dinov2ProcessorConfig:
    """The exact image preprocessing contract stored with the checkpoint."""

    crop_height: int
    crop_width: int
    do_center_crop: bool
    do_convert_rgb: bool
    do_normalize: bool
    do_rescale: bool
    do_resize: bool
    image_mean: tuple[float, float, float]
    image_processor_type: str
    image_std: tuple[float, float, float]
    resample: int
    rescale_factor: float
    shortest_edge: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dinov2ProcessorConfig":
        data = _require_mapping(data, "processor config")
        json_fields = {
            "crop_size",
            "do_center_crop",
            "do_convert_rgb",
            "do_normalize",
            "do_rescale",
            "do_resize",
            "image_mean",
            "image_processor_type",
            "image_std",
            "resample",
            "rescale_factor",
            "size",
        }
        unknown = set(data) - json_fields
        missing = json_fields - set(data)
        if unknown:
            raise ValueError(f"unsupported DINOv2 processor fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing DINOv2 processor fields: {sorted(missing)}")

        crop_size = _require_mapping(data["crop_size"], "crop_size")
        size = _require_mapping(data["size"], "size")
        if set(crop_size) != {"height", "width"}:
            raise ValueError("crop_size must contain exactly height and width")
        if set(size) != {"shortest_edge"}:
            raise ValueError("size must contain exactly shortest_edge")
        if not isinstance(data["image_processor_type"], str):
            raise TypeError("image_processor_type must be a string")

        return cls(
            crop_height=_require_int(crop_size["height"], "crop_size.height"),
            crop_width=_require_int(crop_size["width"], "crop_size.width"),
            do_center_crop=_require_bool(data["do_center_crop"], "do_center_crop"),
            do_convert_rgb=_require_bool(data["do_convert_rgb"], "do_convert_rgb"),
            do_normalize=_require_bool(data["do_normalize"], "do_normalize"),
            do_rescale=_require_bool(data["do_rescale"], "do_rescale"),
            do_resize=_require_bool(data["do_resize"], "do_resize"),
            image_mean=_require_float_tuple(data["image_mean"], "image_mean", length=3),
            image_processor_type=data["image_processor_type"],
            image_std=_require_float_tuple(data["image_std"], "image_std", length=3),
            resample=_require_int(data["resample"], "resample"),
            rescale_factor=_require_float(data["rescale_factor"], "rescale_factor"),
            shortest_edge=_require_int(size["shortest_edge"], "size.shortest_edge"),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Dinov2ProcessorConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate_v1(self) -> None:
        expected: dict[str, Any] = {
            "crop_height": 224,
            "crop_width": 224,
            "do_center_crop": True,
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": (0.485, 0.456, 0.406),
            "image_processor_type": "BitImageProcessor",
            "image_std": (0.229, 0.224, 0.225),
            "resample": 3,
            "shortest_edge": 256,
        }
        mismatches = [
            f"{name}={getattr(self, name)!r} (expected {value!r})"
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if not math.isclose(self.rescale_factor, 1.0 / 255.0, rel_tol=0.0, abs_tol=1e-18):
            mismatches.append(
                f"rescale_factor={self.rescale_factor!r} (expected {1.0 / 255.0!r})"
            )
        if mismatches:
            raise ValueError("unsupported DINOv2 V1 processor: " + "; ".join(mismatches))


__all__ = ["Dinov2Config", "Dinov2ProcessorConfig"]
