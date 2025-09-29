"""Compare Kestrel paged inference against the reference Moondream implementation."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import List

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
from kestrel.moondream.runtime import MoondreamRuntime

from moondream.torch.config import MoondreamConfig, TextMoeConfig as ExternalTextMoeConfig
from moondream.torch.moondream import MoondreamModel
from moondream.torch.text import text_encoder
from moondream.torch.weights import load_weights_into_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=None)
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def _load_config_dict(config_path: Path | None) -> dict:
    if config_path is None:
        return deepcopy(DEFAULT_MOONDREAM3_CONFIG)
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _prompt_token_ids(config_dict: dict, tokenizer: Tokenizer, prompt: str) -> list[int]:
    tok_cfg = config_dict.get("tokenizer", {})
    bos_id = tok_cfg.get("bos_id", 0)
    prefix = tok_cfg.get("templates", {}).get("query", {}).get("prefix", [])
    suffix = tok_cfg.get("templates", {}).get("query", {}).get("suffix", [])
    prompt_ids = tokenizer.encode(prompt).ids
    return [bos_id, *prefix, *prompt_ids, *suffix]


def _run_reference(
    config_dict: dict,
    weights_path: Path,
    tokenizer_id: str,
    prompt_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    cfg_dict = deepcopy(config_dict)
    text_cfg = cfg_dict.get("text", {})
    if isinstance(text_cfg.get("moe"), dict):
        text_cfg = dict(text_cfg)
        text_cfg["moe"] = ExternalTextMoeConfig(**text_cfg["moe"])
        cfg_dict["text"] = text_cfg

    with torch.no_grad():
        config = MoondreamConfig.from_dict(cfg_dict)
        model = MoondreamModel(config, dtype=dtype, setup_caches=True).to(device)
        load_weights_into_model(str(weights_path), model)
        model.tokenizer = Tokenizer.from_pretrained(tokenizer_id)
        model.eval()

        prompt = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)
        max_context = model.config.text.max_context
        attn_mask = torch.tril(torch.ones(1, 1, max_context, max_context, dtype=torch.bool, device=device))

        logits_prefill, hidden, next_token, pos = model._prefill_prompt(
            prompt,
            pos=0,
            temperature=0.0,
            top_p=1.0,
            attn_mask=attn_mask,
        )
        ref_prefill = logits_prefill.to("cpu")

        mask = torch.zeros_like(attn_mask)
        mask[:, :, :pos] = 1
        pos_ids = torch.tensor([pos], device=device, dtype=torch.long)

        decode_logits: List[torch.Tensor] = []
        generated: List[int] = []

        while len(generated) < max_new_tokens:
            token_id = next_token.view(-1)[0].item()
            if token_id == model.config.tokenizer.eos_id:
                break
            generated.append(token_id)

            token_emb = text_encoder(next_token, model.text)
            mask[:, :, pos] = 1
            logits_step, hidden = model._decode_one_tok(token_emb, mask, pos_ids)
            decode_logits.append(logits_step.to("cpu"))

            pos += 1
            pos_ids[0] = pos
            next_token = torch.argmax(logits_step, dim=-1, keepdim=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ref_prefill, decode_logits, generated


def main() -> None:
    args = _parse_args()
    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)

    config_dict = _load_config_dict(args.config)
    tokenizer_id = args.tokenizer or "moondream/starmie-v1"
    tokenizer = Tokenizer.from_pretrained(tokenizer_id)
    prompt_ids = _prompt_token_ids(config_dict, tokenizer, args.prompt)

    ref_prefill, ref_decode, generated_ids = _run_reference(
        config_dict=config_dict,
        weights_path=args.weights,
        tokenizer_id=tokenizer_id,
        prompt_ids=prompt_ids,
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
    )

    model_paths = ModelPaths(weights=args.weights, config_json=args.config, tokenizer=args.tokenizer)
    runtime_cfg = RuntimeConfig(
        model_paths=model_paths,
        device=args.device,
        dtype=dtype,
        max_batch_size=2,
        page_size=args.page_size,
        max_seq_length=args.max_seq_length,
    )
    runtime = MoondreamRuntime(runtime_cfg)

    prompt_tensor = torch.tensor(prompt_ids, device=runtime.device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        state, kestrel_prefill, _ = runtime.start_sequence(
            prompt_tokens=prompt_tensor,
            max_new_tokens=args.max_new_tokens,
        )

    diff_prefill = (kestrel_prefill.to("cpu") - ref_prefill).abs()
    print("Prefill logits diff:", {
        "max": diff_prefill.max().item(),
        "mean": diff_prefill.mean().item(),
        "median": diff_prefill.median().item(),
    })

    decode_stats: list[tuple[float, float, float]] = []
    try:
        for step, token_id in enumerate(generated_ids[: args.max_new_tokens]):
            token_tensor = torch.tensor([token_id], device=runtime.device, dtype=torch.long)
            with torch.no_grad():
                kestrel_logits = runtime.decode(state, token_tensor).to("cpu")
            ref_logits = ref_decode[step]
            diff = (kestrel_logits - ref_logits).abs()
            stats = (diff.max().item(), diff.mean().item(), diff.median().item())
            decode_stats.append(stats)
            print(
                f"Step {step} logits diff:",
                {"max": stats[0], "mean": stats[1], "median": stats[2]},
            )
    finally:
        runtime.release_sequence(state)

    text = tokenizer.decode(generated_ids) if generated_ids else ""
    print("Generated tokens:", generated_ids)
    print("Generated text:", text)

    if decode_stats:
        max_abs = max(s[0] for s in decode_stats)
        mean_of_means = sum(s[1] for s in decode_stats) / len(decode_stats)
        mean_of_medians = sum(s[2] for s in decode_stats) / len(decode_stats)
        print(
            "Decode summary:",
            {
                "steps": len(decode_stats),
                "max_abs": max_abs,
                "mean_abs": mean_of_means,
                "median_abs": mean_of_medians,
            },
        )


if __name__ == "__main__":
    main()
