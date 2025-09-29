"""Probe layer-0 tau/rotary parity between Kestrel and reference Moondream."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXTERNAL_MOONDREAM = PROJECT_ROOT / "external" / "moondream"
if EXTERNAL_MOONDREAM.exists() and str(EXTERNAL_MOONDREAM) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_MOONDREAM))

from tokenizers import Tokenizer

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.moondream.config import DEFAULT_MOONDREAM3_CONFIG
from kestrel.moondream.rope import apply_rotary_emb as internal_apply_rotary
from kestrel.moondream.text import layer_norm as internal_layer_norm
from kestrel.moondream.text import text_encoder as internal_text_encoder
from kestrel.moondream.runtime import MoondreamRuntime

from moondream.torch.config import MoondreamConfig, TextMoeConfig as ExternalTextMoeConfig
from moondream.torch.moondream import MoondreamModel
from moondream.torch.rope import apply_rotary_emb as external_apply_rotary
from moondream.torch.text import layer_norm as external_layer_norm
from moondream.torch.text import text_encoder as external_text_encoder
from moondream.torch.weights import load_weights_into_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return mapping[name]


def _load_config_dict(config_path: Path | None) -> Dict:
    if config_path is None:
        return deepcopy(DEFAULT_MOONDREAM3_CONFIG)
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _prompt_ids(config_dict: Dict, tokenizer: Tokenizer, prompt: str) -> list[int]:
    tok_cfg = config_dict.get("tokenizer", {})
    bos_id = tok_cfg.get("bos_id", 0)
    prefix = tok_cfg.get("templates", {}).get("query", {}).get("prefix", [])
    suffix = tok_cfg.get("templates", {}).get("query", {}).get("suffix", [])
    return [bos_id, *prefix, *tokenizer.encode(prompt).ids, *suffix]


def _stats(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    diff = (a.to(torch.float32).cpu() - b.to(torch.float32).cpu()).abs()
    print(f"{name}: max={diff.max().item():.6f} mean={diff.mean().item():.6f} median={diff.median().item():.6f}")


def _prepare_cfg(config_dict: Dict) -> Dict:
    cfg = deepcopy(config_dict)
    text_cfg = cfg.get("text", {})
    if isinstance(text_cfg.get("moe"), dict):
        text_cfg = dict(text_cfg)
        text_cfg["moe"] = ExternalTextMoeConfig(**text_cfg["moe"])
        cfg["text"] = text_cfg
    return cfg


def main() -> None:
    args = _parse_args()
    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)

    config_dict = _load_config_dict(args.config)
    tokenizer_id = args.tokenizer or "moondream/starmie-v1"
    tokenizer = Tokenizer.from_pretrained(tokenizer_id)
    prompt_ids = _prompt_ids(config_dict, tokenizer, args.prompt)

    with torch.no_grad():
        # Reference path
        cfg_ext = _prepare_cfg(config_dict)
        model_ext = MoondreamModel(MoondreamConfig.from_dict(cfg_ext), dtype=dtype, setup_caches=False).to(device)
        load_weights_into_model(str(args.weights), model_ext)
        model_ext.eval()

        prompt = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)
        embeds_ext = external_text_encoder(prompt, model_ext.text)
        block_ext = model_ext.text.blocks[0]
        ln_ext = external_layer_norm(embeds_ext.to(block_ext.ln.weight.dtype), block_ext.ln).to(torch.float32)
        qkv_ext = block_ext.attn.qkv(ln_ext.to(dtype)).to(torch.float32)
