"""Configuration dataclasses for the Moondream text stack.

This module is adapted from the open-source Moondream project
(https://github.com/vikhyat/moondream) which is licensed under Apache-2.0.
"""

from __future__ import annotations

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
    max_context: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 32
    prefix_attn: int = 730
    group_size: Optional[int] = None
    moe: Optional[TextMoeConfig] = field(default_factory=TextMoeConfig)


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
        }
    )


@dataclass(frozen=True)
class MoondreamTextConfig:
    text: TextConfig = TextConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MoondreamTextConfig":
        text_dict = dict(config_dict.get("text", {}))
        moe_cfg = text_dict.get("moe")
        if moe_cfg is not None and not isinstance(moe_cfg, TextMoeConfig):
            text_dict["moe"] = TextMoeConfig(**moe_cfg)
        text_cfg = TextConfig(**text_dict)

        tokenizer_dict = dict(config_dict.get("tokenizer", {}))
        tokenizer_cfg = TokenizerConfig(**tokenizer_dict)
        return cls(text=text_cfg, tokenizer=tokenizer_cfg)

    def to_dict(self) -> Dict:
        text_dict = self.text.__dict__.copy()
        moe_cfg = text_dict.get("moe")
        if isinstance(moe_cfg, TextMoeConfig):
            text_dict["moe"] = moe_cfg.__dict__.copy()
        return {"text": text_dict, "tokenizer": self.tokenizer.__dict__.copy()}


DEFAULT_MOONDREAM3_CONFIG = {
    "text": {
        "dim": 2048,
        "ff_dim": 8192,
        "n_layers": 24,
        "vocab_size": 51200,
        "max_context": 4096,
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
    },
    "tokenizer": {
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
        },
    },
}

__all__ = [
    "TextMoeConfig",
    "TextConfig",
    "TokenizerConfig",
    "MoondreamTextConfig",
    "DEFAULT_MOONDREAM3_CONFIG",
]
