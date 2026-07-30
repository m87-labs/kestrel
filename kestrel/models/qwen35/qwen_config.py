"""Local Qwen 3.5 config objects loaded from Hugging Face config JSON."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


def _known_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


@dataclass
class Qwen3_5TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_parameters: dict[str, Any] | None = None
    attention_bias: bool = False
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    full_attention_interval: int = 4
    mamba_ssm_dtype: str = "float32"
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    num_experts_per_tok: int | None = None
    num_experts: int | None = None
    expert_weight_format: str = "bf16"
    model_type: str = "qwen3_5_text"
    _attn_implementation: str = "sdpa"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3_5TextConfig":
        cfg = cls(**_known_kwargs(cls, data))
        if cfg.rope_parameters is None:
            cfg.rope_parameters = {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            }
        if cfg.layer_types is None:
            cfg.layer_types = [
                "linear_attention"
                if (idx + 1) % cfg.full_attention_interval
                else "full_attention"
                for idx in range(cfg.num_hidden_layers)
            ]
        if cfg.is_moe:
            cfg.moe_intermediate_size = cfg.moe_intermediate_size or 512
            cfg.shared_expert_intermediate_size = (
                cfg.shared_expert_intermediate_size or cfg.moe_intermediate_size
            )
            cfg.num_experts_per_tok = cfg.num_experts_per_tok or 8
            cfg.num_experts = cfg.num_experts or 256
        return cfg

    def get_text_config(self, decoder: bool = False) -> "Qwen3_5TextConfig":
        return self

    @property
    def is_moe(self) -> bool:
        return self.model_type.startswith("qwen3_5_moe") or (
            self.num_experts is not None
            and self.num_experts_per_tok is not None
            and self.moe_intermediate_size is not None
        )


@dataclass
class Qwen3_5VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] | None = None
    model_type: str = "qwen3_5_vision"
    _attn_implementation: str = "sdpa"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3_5VisionConfig":
        return cls(**_known_kwargs(cls, data))


@dataclass
class Qwen3_5Config:
    text_config: Qwen3_5TextConfig
    vision_config: Qwen3_5VisionConfig
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False
    model_type: str = "qwen3_5"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3_5Config":
        text_data = dict(data.get("text_config") or {})
        if data.get("model_type") == "qwen3_5_moe" and "model_type" not in text_data:
            text_data["model_type"] = "qwen3_5_moe_text"
        text = Qwen3_5TextConfig.from_dict(text_data)
        quantization_config = data.get("quantization_config") or {}
        if (
            quantization_config.get("quant_method") == "fp8"
            and quantization_config.get("fmt") == "e4m3"
        ):
            text.expert_weight_format = "fp8_e4m3"
        vision = Qwen3_5VisionConfig.from_dict(data.get("vision_config") or {})
        return cls(
            text_config=text,
            vision_config=vision,
            image_token_id=int(data.get("image_token_id", cls.image_token_id)),
            video_token_id=int(data.get("video_token_id", cls.video_token_id)),
            vision_start_token_id=int(
                data.get("vision_start_token_id", cls.vision_start_token_id)
            ),
            vision_end_token_id=int(
                data.get("vision_end_token_id", cls.vision_end_token_id)
            ),
            tie_word_embeddings=bool(
                data.get("tie_word_embeddings", cls.tie_word_embeddings)
            ),
            model_type=str(data.get("model_type", cls.model_type)),
        )

    def get_text_config(self, decoder: bool = False) -> Qwen3_5TextConfig:
        return self.text_config


__all__ = ["Qwen3_5Config", "Qwen3_5TextConfig", "Qwen3_5VisionConfig"]
