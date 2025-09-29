"""Inspect layer-wise KV cache differences between Kestrel and the reference Moondream implementation."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXTERNAL_MOONDREAM = PROJECT_ROOT / "external" / "moondream"
if EXTERNAL_MOONDREAM.exists() and str(EXTERNAL_MOONDREAM) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_MOONDREAM))

from tokenizers import Tokenizer

import torch.nn.functional as F

import kestrel.moondream.text as internal_text
import moondream.torch.text as external_text

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.moondream.config import DEFAULT_MOONDREAM3_CONFIG
from kestrel.moondream.runtime import MoondreamRuntime, SequenceState

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
    parser.add_argument("--max-new-tokens", type=int, default=16)
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


def _stats(diff: torch.Tensor) -> dict[str, float]:
    d = diff.abs()
    if not d.is_floating_point():
        d = d.to(torch.float32)
    return {"max": d.max().item(), "mean": d.mean().item(), "median": d.median().item()}


_CAPTURE_MODE: Optional[Tuple[str, Optional[int]]] = None


class AttnCaptureStore:
    def __init__(self, label: str) -> None:
        self.label = label
        self.reset()

    def reset(self) -> None:
        self.prefill: Dict[int, Dict[str, torch.Tensor]] = {}
        self.decode: List[Dict[int, Dict[str, torch.Tensor]]] = []

    def begin_decode_step(self) -> int:
        self.decode.append({})
        return len(self.decode) - 1

    def record(
        self,
        stage: str,
        step_idx: Optional[int],
        layer_idx: Optional[int],
        payload: Dict[str, torch.Tensor],
    ) -> None:
        if layer_idx is None:
            return
        if stage == "prefill":
            self.prefill[layer_idx] = payload
            return

        if stage == "decode":
            idx = 0 if step_idx is None else step_idx
            while len(self.decode) <= idx:
                self.decode.append({})
            self.decode[idx][layer_idx] = payload


@contextmanager
def _capture_mode(stage: Optional[str], step_idx: Optional[int]) -> None:
    global _CAPTURE_MODE
    prev = _CAPTURE_MODE
    if stage is None:
        _CAPTURE_MODE = None
    else:
        _CAPTURE_MODE = (stage, step_idx)
    try:
        yield
    finally:
        _CAPTURE_MODE = prev


def _annotate_layers(blocks: torch.nn.ModuleList) -> None:
    for layer_idx, block in enumerate(blocks):
        setattr(block.attn, "_capture_layer_idx", layer_idx)


def _make_internal_attn_wrapper(store: AttnCaptureStore):
    def wrapped(
        x: torch.Tensor,
        module: torch.nn.Module,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[torch.nn.Module],
        attn_mask: Optional[torch.Tensor],
        n_heads: int,
        n_kv_heads: int,
        position_ids: torch.Tensor,
        lora: Optional[dict] = None,
        flex_block_mask_slice=None,
    ) -> torch.Tensor:
        bsz, q_len, d_model = x.shape
        head_dim = d_model // n_heads

        if position_ids.ndim > 1:
            position_ids = position_ids.view(-1)

        qkv_out = module.qkv(x)
        if lora is not None:
            qkv_out += F.linear(F.linear(x, lora["qkv"]["A"]), lora["qkv"]["B"])

        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

        capture: Dict[str, torch.Tensor] = {
            "q_pre_tau": q.detach().to(torch.float32).cpu(),
            "v_pre_tau": v.detach().to(torch.float32).cpu(),
            "position_ids": position_ids.detach().cpu(),
        }

        if hasattr(module, "tau") and module.tau is not None:
            tok_feat = F.gelu(qkv_out)
            tok_q = torch.tanh(torch.matmul(tok_feat, module.tau["wq"].t())).permute(0, 2, 1)
            tok_v = torch.tanh(torch.matmul(tok_feat, module.tau["wv"].t())).permute(0, 2, 1)
            pos = position_ids.to(q.dtype) + 1
            tau_pos = 1 + (torch.sigmoid(module.tau["alpha"][:, None] * pos.log()) - 0.5)
            tau_q = (tok_q + tau_pos[None]).unsqueeze(-1)
            tau_v = (tok_v + tau_pos[None]).unsqueeze(-1)
            q = q * tau_q
            v = v * tau_v
            capture["tau_pos"] = tau_pos.detach().to(torch.float32).cpu()
            capture["tau_q"] = tau_q.detach().to(torch.float32).cpu()
            capture["tau_v"] = tau_v.detach().to(torch.float32).cpu()

        capture["q_post_tau"] = q.detach().to(torch.float32).cpu()
        capture["v_post_tau"] = v.detach().to(torch.float32).cpu()

        freqs_cos = freqs_cis[..., 0][position_ids, :].unsqueeze(0).unsqueeze(0)
        freqs_sin = freqs_cis[..., 1][position_ids, :].unsqueeze(0).unsqueeze(0)
        capture["freqs_cos"] = freqs_cos.detach().to(torch.float32).cpu()
        capture["freqs_sin"] = freqs_sin.detach().to(torch.float32).cpu()

        q_rot = internal_text.apply_rotary_emb(q.to(torch.float32), freqs_cis, position_ids, n_heads)
        k_rot = internal_text.apply_rotary_emb(k.to(torch.float32), freqs_cis, position_ids, n_kv_heads)
        q_for_attn = q_rot.to(q.dtype)
        k_for_attn = k_rot.to(k.dtype)

        capture["q_post_rope"] = q_rot.detach().cpu()
        capture["k_post_rope"] = k_rot.detach().cpu()

        if kv_cache is not None:
            k_for_attn, v = kv_cache.update(position_ids, k_for_attn, v)

        attn_out = F.scaled_dot_product_attention(
            q_for_attn,
            k_for_attn,
            v,
            attn_mask=attn_mask,
            enable_gqa=n_heads != n_kv_heads,
        )

        out = attn_out.transpose(1, 2).reshape(bsz, q_len, d_model)
        out_proj = module.proj(out)
        if lora is not None:
            lora_out = F.linear(F.linear(x, lora["proj"]["A"]), lora["proj"]["B"])
            out = out_proj + lora_out
        else:
            out = out_proj

        mode = _CAPTURE_MODE
        if mode is not None:
            stage, step_idx = mode
            layer_idx = getattr(module, "_capture_layer_idx", None)
            store.record(stage, step_idx, layer_idx, capture)

        return out

    return wrapped


def _make_external_attn_wrapper(store: AttnCaptureStore):
    def wrapped(
        x: torch.Tensor,
        module: torch.nn.Module,
        freqs_cis: torch.Tensor,
        kv_cache: torch.nn.Module,
        attn_mask: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        position_ids: torch.Tensor,
        lora: Optional[dict] = None,
        flex_block_mask_slice=None,
    ) -> torch.Tensor:
        bsz, q_len, d_model = x.shape
        head_dim = d_model // n_heads

        qkv_out = module.qkv(x)
        if lora is not None:
            qkv_out += F.linear(F.linear(x, lora["qkv"]["A"]), lora["qkv"]["B"])

        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

        capture: Dict[str, torch.Tensor] = {
            "q_pre_tau": q.detach().to(torch.float32).cpu(),
            "v_pre_tau": v.detach().to(torch.float32).cpu(),
            "position_ids": position_ids.detach().cpu(),
        }

        if hasattr(module, "tau") and module.tau is not None:
            tok_feat = F.gelu(qkv_out)
            tok_q = torch.tanh(torch.matmul(tok_feat, module.tau["wq"].t())).permute(0, 2, 1)
            tok_v = torch.tanh(torch.matmul(tok_feat, module.tau["wv"].t())).permute(0, 2, 1)
            pos = position_ids.to(q.dtype) + 1
            tau_pos = 1 + (torch.sigmoid(module.tau["alpha"][:, None] * pos.log()) - 0.5)
            tau_q = (tok_q + tau_pos[None]).unsqueeze(-1)
            tau_v = (tok_v + tau_pos[None]).unsqueeze(-1)
            q = q * tau_q
            v = v * tau_v
            capture["tau_pos"] = tau_pos.detach().to(torch.float32).cpu()
            capture["tau_q"] = tau_q.detach().to(torch.float32).cpu()
            capture["tau_v"] = tau_v.detach().to(torch.float32).cpu()

        capture["q_post_tau"] = q.detach().to(torch.float32).cpu()
        capture["v_post_tau"] = v.detach().to(torch.float32).cpu()

        freqs_cos = freqs_cis[..., 0][position_ids, :].unsqueeze(0).unsqueeze(0)
        freqs_sin = freqs_cis[..., 1][position_ids, :].unsqueeze(0).unsqueeze(0)
        capture["freqs_cos"] = freqs_cos.detach().to(torch.float32).cpu()
        capture["freqs_sin"] = freqs_sin.detach().to(torch.float32).cpu()

        q_rot = external_text.apply_rotary_emb(q.to(torch.float32), freqs_cis, position_ids, n_heads)
        k_rot = external_text.apply_rotary_emb(k.to(torch.float32), freqs_cis, position_ids, n_kv_heads)
        q_for_attn = q_rot.to(q.dtype)
        k_for_attn = k_rot.to(k.dtype)

        capture["q_post_rope"] = q_rot.detach().cpu()
        capture["k_post_rope"] = k_rot.detach().cpu()

        k_for_attn, v = kv_cache.update(position_ids, k_for_attn, v)

        if flex_block_mask_slice is not None:
            torch._assert(n_heads == n_kv_heads, "gqa not supported yet")
            attn_out = external_text.flex_attention(
                q_for_attn, k_for_attn, v, block_mask=flex_block_mask_slice
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q_for_attn,
                k_for_attn,
                v,
                attn_mask=attn_mask,
                enable_gqa=n_heads != n_kv_heads,
            )

        out = attn_out.transpose(1, 2).reshape(bsz, q_len, d_model)
        out_proj = module.proj(out)
        if lora is not None:
            lora_out = F.linear(F.linear(x, lora["proj"]["A"]), lora["proj"]["B"])
            out = out_proj + lora_out
        else:
            out = out_proj

        mode = _CAPTURE_MODE
        if mode is not None:
            stage, step_idx = mode
            layer_idx = getattr(module, "_capture_layer_idx", None)
            store.record(stage, step_idx, layer_idx, capture)

        return out

    return wrapped


def _install_attn_wrappers(
    internal_store: AttnCaptureStore, external_store: AttnCaptureStore
) -> None:
    internal_text.attn = _make_internal_attn_wrapper(internal_store)
    external_text.attn = _make_external_attn_wrapper(external_store)


def _tensor_stats(current: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    if current.shape != reference.shape:
        raise ValueError(
            f"Tensor shape mismatch: current={current.shape}, reference={reference.shape}"
        )
    return _stats(current - reference)


def _print_capture_diffs(
    external_store: AttnCaptureStore, internal_store: AttnCaptureStore
) -> None:
    metric_labels = [
        ("q_pre_tau", "Q pre-tau"),
        ("q_post_tau", "Q post-tau"),
        ("q_post_rope", "Q post-rope"),
        ("k_post_rope", "K post-rope"),
        ("v_post_tau", "V post-tau"),
        ("freqs_cos", "RoPE cos"),
        ("freqs_sin", "RoPE sin"),
        ("position_ids", "Position IDs"),
    ]

    print("Prefill attention deltas:")
    for layer_idx in sorted(external_store.prefill):
        if layer_idx not in internal_store.prefill:
            continue
        ext_payload = external_store.prefill[layer_idx]
        int_payload = internal_store.prefill[layer_idx]
        print(f"  Layer {layer_idx}:")
        for key, label in metric_labels:
            if key not in ext_payload or key not in int_payload:
                continue
            stats = _tensor_stats(int_payload[key], ext_payload[key])
            print(f"    {label}: {stats}")
            if layer_idx == 0 and key in {"q_post_rope", "freqs_cos", "freqs_sin"}:
                diff = (int_payload[key] - ext_payload[key]).view(-1)
                sample = diff[:8].tolist()
                nonzero = (diff != 0).sum().item()
                print(f"      sample diff {key}: {sample}")
                print(f"      nonzero count {key}: {nonzero}")
                if nonzero > 0:
                    idx = (diff != 0).nonzero(as_tuple=False)
                    top = idx[:5].view(-1)
                    values = diff[top].tolist()
                    ext_vals = ext_payload[key].view(-1)[top].tolist()
                    int_vals = int_payload[key].view(-1)[top].tolist()
                    print(
                        f"      first nonzero indices {key}: {top.tolist()} diff: {values}"
                    )
                    print(f"        external vals: {ext_vals}")
                    print(f"        internal vals: {int_vals}")

    max_steps = min(len(external_store.decode), len(internal_store.decode))
    for step in range(max_steps):
        print(f"Decode step {step} attention deltas:")
        ext_step = external_store.decode[step]
        int_step = internal_store.decode[step]
        for layer_idx in sorted(ext_step):
            if layer_idx not in int_step:
                continue
            ext_payload = ext_step[layer_idx]
            int_payload = int_step[layer_idx]
            print(f"  Layer {layer_idx}:")
            for key, label in metric_labels:
                if key not in ext_payload or key not in int_payload:
                    continue
                stats = _tensor_stats(int_payload[key], ext_payload[key])
                print(f"    {label}: {stats}")


def _gather_external_kv(model: MoondreamModel, length: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    k_list: list[torch.Tensor] = []
    v_list: list[torch.Tensor] = []
    for block in model.text.blocks:
        k_list.append(block.kv_cache.k_cache[:, :, :length, :].detach().cpu())
        v_list.append(block.kv_cache.v_cache[:, :, :length, :].detach().cpu())
    return k_list, v_list


def _gather_internal_kv(runtime: MoondreamRuntime, batch_idx: int, length: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    page_size = runtime.page_table.page_size
    page_table = runtime.page_table.page_table[batch_idx].detach().cpu()
    logical = torch.arange(length, dtype=torch.long)
    block_idx = logical // page_size
    offset = logical % page_size
    physical_page = page_table[block_idx]
    physical_index = physical_page * page_size + offset

    k_list: list[torch.Tensor] = []
    v_list: list[torch.Tensor] = []
    for block in runtime.model.text.blocks:
        k_cache = block.kv_cache.cache.k_cache
        v_cache = block.kv_cache.cache.v_cache
        gathered_k = k_cache[physical_page, offset].detach().cpu()
        gathered_v = v_cache[physical_page, offset].detach().cpu()
        k_list.append(gathered_k.permute(1, 0, 2).contiguous())
        v_list.append(gathered_v.permute(1, 0, 2).contiguous())
    return k_list, v_list


def _run_reference(
    *,
    config_dict: dict,
    weights_path: Path,
    tokenizer_id: str,
    prompt_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    capture_store: AttnCaptureStore,
) -> tuple[
    tuple[list[torch.Tensor], list[torch.Tensor]],
    tuple[list[tuple[list[torch.Tensor], list[torch.Tensor]]], List[int]],
]:
    capture_store.reset()
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
        _annotate_layers(model.text.blocks)

        prompt = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)
        max_context = model.config.text.max_context
        attn_mask = torch.tril(torch.ones(1, 1, max_context, max_context, dtype=torch.bool, device=device))

        with _capture_mode("prefill", 0):
            logits_prefill, hidden, next_token, pos = model._prefill_prompt(
                prompt,
                pos=0,
                temperature=0.0,
                top_p=1.0,
                attn_mask=attn_mask,
                lora=None,
            )
        pref_kv = _gather_external_kv(model, pos)

        per_step_kv: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
        generated: list[int] = []

        while len(generated) < max_new_tokens:
            token_id = next_token.view(-1)[0].item()
            if token_id == model.config.tokenizer.eos_id:
                break
            generated.append(token_id)

            token_emb = text_encoder(next_token, model.text)
            attn_mask[:, :, pos] = 1
            pos_ids = torch.tensor([pos], device=device, dtype=torch.long)
            decode_idx = capture_store.begin_decode_step()
            with _capture_mode("decode", decode_idx):
                logits_step, hidden = model._decode_one_tok(
                    token_emb, attn_mask, pos_ids, lora=None
                )
            next_token = torch.argmax(logits_step, dim=-1, keepdim=True)
            pos += 1
            per_step_kv.append(_gather_external_kv(model, pos))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pref_kv, (per_step_kv, generated)


def _run_internal(
    runtime: MoondreamRuntime,
    prompt_ids: list[int],
    reference_tokens: list[int],
    max_new_tokens: int,
    capture_store: AttnCaptureStore,
) -> tuple[
    SequenceState,
    tuple[list[torch.Tensor], list[torch.Tensor]],
    list[tuple[list[torch.Tensor], list[torch.Tensor]]],
]:
    capture_store.reset()
    prompt = torch.tensor(prompt_ids, device=runtime.device, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        _annotate_layers(runtime.model.text.blocks)
        with _capture_mode("prefill", 0):
            state, _, _ = runtime.start_sequence(
                prompt_tokens=prompt, max_new_tokens=max_new_tokens
            )

    pref_kv = _gather_internal_kv(runtime, state.batch_idx, state.length)
    per_step_kv: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []

    try:
        for token_id in reference_tokens[:max_new_tokens]:
            token_tensor = torch.tensor([token_id], device=runtime.device, dtype=torch.long)
            with torch.no_grad():
                decode_idx = capture_store.begin_decode_step()
                with _capture_mode("decode", decode_idx):
                    runtime.decode(state, token_tensor)
            per_step_kv.append(_gather_internal_kv(runtime, state.batch_idx, state.length))
    finally:
        runtime.release_sequence(state)

    return state, pref_kv, per_step_kv


def main() -> None:
    args = _parse_args()
    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)

    config_dict = _load_config_dict(args.config)
    tokenizer_id = args.tokenizer or "moondream/starmie-v1"
    tokenizer = Tokenizer.from_pretrained(tokenizer_id)
    prompt_ids = _prompt_token_ids(config_dict, tokenizer, args.prompt)

    external_capture = AttnCaptureStore("external")
    internal_capture = AttnCaptureStore("internal")
    _install_attn_wrappers(internal_capture, external_capture)

    pref_kv_ext, (decode_kv_ext, generated_ids) = _run_reference(
        config_dict=config_dict,
        weights_path=args.weights,
        tokenizer_id=tokenizer_id,
        prompt_ids=prompt_ids,
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        capture_store=external_capture,
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

    _, pref_kv_int, decode_kv_int = _run_internal(
        runtime,
        prompt_ids,
        generated_ids,
        args.max_new_tokens,
        internal_capture,
    )

    ext_pref_k, ext_pref_v = pref_kv_ext
    int_pref_k, int_pref_v = pref_kv_int
    for idx, (ek, ik, ev, iv) in enumerate(zip(ext_pref_k, int_pref_k, ext_pref_v, int_pref_v)):
        print(f"Prefill layer {idx} K diff: {_stats(ik - ek)}")
        print(f"Prefill layer {idx} V diff: {_stats(iv - ev)}")

    for step, (kv_ext, kv_int) in enumerate(zip(decode_kv_ext, decode_kv_int)):
        ext_k, ext_v = kv_ext
        int_k, int_v = kv_int
        print(f"Decode step {step}:")
        for layer_idx, (ek, ik, ev, iv) in enumerate(zip(ext_k, int_k, ext_v, int_v)):
            print(f"  Layer {layer_idx} K diff: {_stats(ik - ek)}")
            print(f"  Layer {layer_idx} V diff: {_stats(iv - ev)}")

    _print_capture_diffs(external_capture, internal_capture)

    print("Generated tokens:", generated_ids)
    print("Generated text:", tokenizer.decode(generated_ids) if generated_ids else "")


if __name__ == "__main__":
    main()
