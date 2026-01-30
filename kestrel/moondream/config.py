"""Configuration dataclasses for the Moondream text stack.

This module is adapted from the open-source Moondream project
(https://github.com/vikhyat/moondream) which is licensed under Apache-2.0.
"""


from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TextMoeConfig:
    num_experts: int = 64
    start_layer: int = 4
    experts_per_token: int = 8
    expert_inner_dim: int = 1024


@dataclass(frozen=True)
class TextConfig:
    dim: int = 2048
    ff_dim: int = 8192
    n_layers: int = 24
    vocab_size: int = 51200
    max_context: int = 32768
    n_heads: int = 32
    n_kv_heads: int = 32
    prefix_attn: int = 730
    group_size: Optional[int] = None
    moe: Optional[TextMoeConfig] = field(default_factory=TextMoeConfig)
    tau_attn: bool = True
    rope_theta: float = 1_500_000.0


@dataclass(frozen=True)
class TokenizerConfig:
    bos_id: int = 0
    eos_id: int = 0
    answer_id: int = 3
    thinking_id: int = 4
    coord_id: int = 5
    size_id: int = 6
    start_ground_points_id: int = 7
    end_ground_id: int = 9
    templates: Dict[str, Optional[Dict[str, List[int]]]] = field(
        default_factory=lambda: {
            "caption": {
                "short": [1, 32708, 2, 12492, 3],
                "normal": [1, 32708, 2, 6382, 3],
                "long": [1, 32708, 2, 4059, 3],
            },
            "query": {"prefix": [1, 15381, 2], "suffix": [3]},
            "detect": {"prefix": [1, 7235, 476, 2], "suffix": [3]},
            "point": {"prefix": [1, 2581, 2], "suffix": [3]},
            "segment": {"prefix": [1, 17374, 2], "suffix": [3]},
        }
    )


@dataclass(frozen=True)
class RegionConfig:
    dim: int = 2048
    coord_feat_dim: int = 256
    coord_out_dim: int = 1024
    size_feat_dim: int = 512
    size_out_dim: int = 2048
    decoder_arch: str = "linear"
    use_ln: bool = True


@dataclass(frozen=True)
class MoondreamTextConfig:
    text: TextConfig = TextConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    region: RegionConfig = RegionConfig()

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MoondreamTextConfig":
        text_dict = dict(config_dict.get("text", {}))
        moe_cfg = text_dict.get("moe")
        if moe_cfg is not None and not isinstance(moe_cfg, TextMoeConfig):
            text_dict["moe"] = TextMoeConfig(**moe_cfg)
        text_cfg = TextConfig(**text_dict)

        tokenizer_dict = dict(config_dict.get("tokenizer", {}))
        tokenizer_cfg = TokenizerConfig(**tokenizer_dict)

        region_dict = dict(config_dict.get("region", {}))
        region_cfg = RegionConfig(**region_dict)
        return cls(text=text_cfg, tokenizer=tokenizer_cfg, region=region_cfg)

    def to_dict(self) -> Dict:
        text_dict = self.text.__dict__.copy()
        moe_cfg = text_dict.get("moe")
        if isinstance(moe_cfg, TextMoeConfig):
            text_dict["moe"] = moe_cfg.__dict__.copy()
        return {
            "text": text_dict,
            "tokenizer": self.tokenizer.__dict__.copy(),
            "region": self.region.__dict__.copy(),
        }


# Shared configs between MD2 and MD3
_VISION_CONFIG = {
    "enc_dim": 1152,
    "enc_patch_size": 14,
    "enc_n_layers": 27,
    "enc_ff_dim": 4304,
    "enc_n_heads": 16,
    "proj_out_dim": 2048,
    "crop_size": 378,
    "in_channels": 3,
    "max_crops": 12,
    "overlap_margin": 4,
    "proj_inner_dim": 8192,
}

_TOKENIZER_CONFIG = {
    "bos_id": 0,
    "eos_id": 0,
    "answer_id": 3,
    "thinking_id": 4,
    "coord_id": 5,
    "size_id": 6,
    "start_ground_points_id": 7,
    "end_ground_id": 9,
    "templates": {
        "caption": {
            "short": [1, 32708, 2, 12492, 3],
            "normal": [1, 32708, 2, 6382, 3],
            "long": [1, 32708, 2, 4059, 3],
        },
        "query": {"prefix": [1, 15381, 2], "suffix": [3]},
        "detect": {"prefix": [1, 7235, 476, 2], "suffix": [3]},
        "point": {"prefix": [1, 2581, 2], "suffix": [3]},
        "segment": {"prefix": [1, 17374, 2], "suffix": [3]},
    },
}

_REGION_CONFIG = {
    "dim": 2048,
    "coord_feat_dim": 256,
    "coord_out_dim": 1024,
    "size_feat_dim": 512,
    "size_out_dim": 2048,
}

DEFAULT_MOONDREAM3_CONFIG = {
    "text": {
        "dim": 2048,
        "ff_dim": 8192,
        "n_layers": 24,
        "vocab_size": 51200,
        "max_context": 32768,
        "n_heads": 32,
        "n_kv_heads": 32,
        "prefix_attn": 730,
        "group_size": None,
        "moe": {
            "num_experts": 64,
            "start_layer": 4,
            "experts_per_token": 8,
            "expert_inner_dim": 1024,
        },
        "tau_attn": True,
        "rope_theta": 1_500_000.0,
    },
    "tokenizer": _TOKENIZER_CONFIG,
    "vision": _VISION_CONFIG,
    "region": _REGION_CONFIG,
}

# Moondream 2 uses Phi-1.5 text backbone without MoE or TAU attention.
# Latest MD2 (2025-06-21) uses the same Starmie tokenizer as MD3.
DEFAULT_MOONDREAM2_CONFIG = {
    "text": {
        "dim": 2048,
        "ff_dim": 8192,
        "n_layers": 24,
        "vocab_size": 51200,
        "max_context": 2048,
        "n_heads": 32,
        "n_kv_heads": 32,
        "prefix_attn": 730,
        "group_size": None,
        "moe": None,
        "tau_attn": False,
        "rope_theta": 10_000.0,
    },
    "vision": _VISION_CONFIG,
    "tokenizer": {
        **_TOKENIZER_CONFIG,
        "templates": {
            **_TOKENIZER_CONFIG["templates"],
            "segment": None,  # MD2 doesn't support segment
        },
    },
    "region": {
        **_REGION_CONFIG,
        "decoder_arch": "mlp",
        "use_ln": False,
    },
}


@dataclass(frozen=True)
class VisionConfig:
    enc_dim: int = 1152
    enc_patch_size: int = 14
    enc_n_layers: int = 27
    enc_ff_dim: int = 4304
    enc_n_heads: int = 16
    proj_out_dim: int = 2048
    crop_size: int = 378
    in_channels: int = 3
    max_crops: int = 12
    overlap_margin: int = 4
    proj_inner_dim: int = 8192


@dataclass(frozen=True)
class MoondreamConfig:
    text: TextConfig = TextConfig()
    vision: VisionConfig = VisionConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    region: RegionConfig = RegionConfig()

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MoondreamConfig":
        text_dict = dict(config_dict.get("text", {}))
        moe_cfg = text_dict.get("moe")
        if moe_cfg is not None and not isinstance(moe_cfg, TextMoeConfig):
            text_dict["moe"] = TextMoeConfig(**moe_cfg)
        text_cfg = TextConfig(**text_dict)

        tokenizer_dict = dict(config_dict.get("tokenizer", {}))
        tokenizer_cfg = TokenizerConfig(**tokenizer_dict)

        vision_dict = dict(config_dict.get("vision", {}))
        vision_cfg = VisionConfig(**vision_dict)
        region_dict = dict(config_dict.get("region", {}))
        region_cfg = RegionConfig(**region_dict)
        return cls(text=text_cfg, vision=vision_cfg, tokenizer=tokenizer_cfg, region=region_cfg)

    def to_dict(self) -> Dict:
        text_dict = self.text.__dict__.copy()
        moe_cfg = text_dict.get("moe")
        if isinstance(moe_cfg, TextMoeConfig):
            text_dict["moe"] = moe_cfg.__dict__.copy()
        return {
            "text": text_dict,
            "vision": self.vision.__dict__.copy(),
            "tokenizer": self.tokenizer.__dict__.copy(),
            "region": self.region.__dict__.copy(),
        }


def load_config() -> MoondreamConfig:
    """Return the default MoondreamConfig."""
    return MoondreamConfig.from_dict(deepcopy(DEFAULT_MOONDREAM3_CONFIG))

__all__ = [
    "TextMoeConfig",
    "TextConfig",
    "TokenizerConfig",
    "RegionConfig",
    "VisionConfig",
    "MoondreamTextConfig",
    "MoondreamConfig",
    "DEFAULT_MOONDREAM2_CONFIG",
    "DEFAULT_MOONDREAM3_CONFIG",
    "load_config",
]
