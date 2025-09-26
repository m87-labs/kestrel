#!/usr/bin/env python3
"""Profile prefill phases of the Moondream runtime with fine-grained timing."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.models import MoondreamTextRuntime
from kestrel.moondream import text as text_mod
from kestrel.moondream import vision as vision_mod
from kestrel.moondream.layers import (
    LayerNormWeights,
    LinearWeights,
    MLPWeights,
    gelu_approx,
    layer_norm,
    linear,
)
from kestrel.moondream.layers import moe_mlp as original_moe_mlp
from kestrel.moondream.layers import mlp as original_dense_mlp
from kestrel.moondream.text import attn as original_attn
from kestrel.moondream.text import (
    lm_head,
    text_decoder as original_text_decoder,
    text_encoder as original_text_encoder,
)


@dataclasses.dataclass
class PhaseSample:
    cpu_ms: float
    gpu_ms: float

    @property
    def gpu_util(self) -> float:
        if self.cpu_ms <= 0.0:
            return float("nan")
        return self.gpu_ms / self.cpu_ms


class PhaseTimer:
    def __init__(self) -> None:
        self._records: dict[str, list[PhaseSample]] = defaultdict(list)

    @contextlib.contextmanager
    def record(self, name: str):
        cuda_available = torch.cuda.is_available() and torch.cuda.current_device() >= 0
        start_event = end_event = None
        if cuda_available:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        cpu_start = time.perf_counter()
        try:
            yield
        finally:
            cpu_end = time.perf_counter()
            cpu_ms = (cpu_end - cpu_start) * 1000.0
            gpu_ms = float("nan")
            if cuda_available and start_event is not None and end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                gpu_ms = start_event.elapsed_time(end_event)
            self._records[name].append(PhaseSample(cpu_ms=cpu_ms, gpu_ms=gpu_ms))

    def merge_into(self, target: dict[str, list[PhaseSample]]) -> None:
        for name, samples in self._records.items():
            target[name].extend(samples)


# ---------------------------------------------------------------------------
# Instrumentation plumbing

_CURRENT_TIMER: PhaseTimer | None = None


def _phase(name: str):
    timer = _CURRENT_TIMER
    if timer is None:
        return contextlib.nullcontext()
    return timer.record(name)


# Vision hooks ----------------------------------------------------------------

_original_prepare_crops = vision_mod.prepare_crops
_original_vision_encoder = vision_mod.vision_encoder
_original_reconstruct_from_crops = vision_mod.reconstruct_from_crops
_original_vision_projection = vision_mod.vision_projection


def _prepare_crops_instrumented(*args, **kwargs):
    with _phase("prefill.vision.prepare_crops"):
        return _original_prepare_crops(*args, **kwargs)


def _vision_encoder_instrumented(*args, **kwargs):
    with _phase("prefill.vision.encoder"):
        return _original_vision_encoder(*args, **kwargs)


def _reconstruct_from_crops_instrumented(*args, **kwargs):
    with _phase("prefill.vision.reconstruct"):
        return _original_reconstruct_from_crops(*args, **kwargs)


def _vision_projection_instrumented(*args, **kwargs):
    with _phase("prefill.vision.projection"):
        return _original_vision_projection(*args, **kwargs)


vision_mod.prepare_crops = _prepare_crops_instrumented  # type: ignore[assignment]
vision_mod.vision_encoder = _vision_encoder_instrumented  # type: ignore[assignment]
vision_mod.reconstruct_from_crops = _reconstruct_from_crops_instrumented  # type: ignore[assignment]
vision_mod.vision_projection = _vision_projection_instrumented  # type: ignore[assignment]


# Text encoder hook -----------------------------------------------------------

def _text_encoder_instrumented(input_ids: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
    with _phase("prefill.text.embedding"):
        return original_text_encoder(input_ids, module)


def _lm_head_instrumented(hidden: torch.Tensor, module: torch.nn.Module, indices: Optional[torch.Tensor] = None):
    with _phase("prefill.text.lm_head"):
        return lm_head(hidden, module, indices)


text_mod.text_encoder = _text_encoder_instrumented  # type: ignore[assignment]
text_mod.lm_head = _lm_head_instrumented  # type: ignore[assignment]


# Attention hook --------------------------------------------------------------

def _attn_instrumented(
    x: torch.Tensor,
    module: torch.nn.Module,
    freqs_cis: torch.Tensor,
    kv_cache: Optional[torch.nn.Module],
    attn_mask: Optional[torch.Tensor],
    n_heads: int,
    n_kv_heads: int,
    position_ids: torch.Tensor,
    lora: Optional[dict] = None,
) -> torch.Tensor:
    timer = _CURRENT_TIMER
    if timer is None:
        return original_attn(
            x,
            module,
            freqs_cis,
            kv_cache,
            attn_mask,
            n_heads,
            n_kv_heads,
            position_ids,
            lora=lora,
        )

    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    if position_ids.ndim == 1:
        position_matrix = position_ids.view(-1, 1)
    elif position_ids.ndim == 2:
        position_matrix = position_ids
    else:
        raise ValueError(f"Unsupported position_ids shape: {position_ids.shape}")

    with timer.record("prefill.attn.qkv"):
        qkv_out = module.qkv(x)
    if lora is not None:
        with timer.record("prefill.attn.lora_qkv"):
            qkv_out = qkv_out + F.linear(F.linear(x, lora["qkv"]["A"]), lora["qkv"]["B"])

    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)

    with timer.record("prefill.attn.reshape"):
        q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    if hasattr(module, "tau") and module.tau is not None:
        with timer.record("prefill.attn.tau"):
            tok_feat = F.gelu(qkv_out)
            tok_q = torch.tanh(torch.matmul(tok_feat, module.tau["wq"].t())).permute(0, 2, 1)
            tok_v = torch.tanh(torch.matmul(tok_feat, module.tau["wv"].t())).permute(0, 2, 1)

            pos = position_matrix.to(q.dtype) + 1
            pos_log = pos.log().unsqueeze(1)
            alpha = module.tau["alpha"].view(1, -1, 1)
            tau_pos = 1 + (torch.sigmoid(alpha * pos_log) - 0.5)

            tau_q = (tok_q + tau_pos).unsqueeze(-1)
            tau_v = (tok_v + tau_pos).unsqueeze(-1)
            q = q * tau_q
            v = v * tau_v

    with timer.record("prefill.attn.rotary_q"):
        q = text_mod.apply_rotary_emb(q.to(torch.float32), freqs_cis, position_ids, n_heads).to(q.dtype)
    with timer.record("prefill.attn.rotary_k"):
        k = text_mod.apply_rotary_emb(k.to(torch.float32), freqs_cis, position_ids, n_kv_heads).to(k.dtype)

    if kv_cache is not None:
        with timer.record("prefill.attn.kv_cache_update"):
            k, v = kv_cache.update(position_ids, k, v)

    with timer.record("prefill.attn.sdpa"):
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, enable_gqa=n_heads != n_kv_heads
        )

    with timer.record("prefill.attn.out_proj"):
        out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
        out = module.proj(out)

    if lora is not None:
        with timer.record("prefill.attn.lora_proj"):
            out = out + F.linear(F.linear(x, lora["proj"]["A"]), lora["proj"]["B"])

    return out


def _dense_mlp_instrumented(x: torch.Tensor, w: MLPWeights, lora: Optional[dict] = None) -> torch.Tensor:
    timer = _CURRENT_TIMER
    if timer is None:
        return original_dense_mlp(x, w, lora=lora)

    with timer.record("prefill.mlp.fc1"):
        x0 = linear(x, w.fc1)
    if lora is not None:
        with timer.record("prefill.mlp.lora_fc1"):
            x1 = F.linear(F.linear(x, lora["fc1"]["A"]), lora["fc1"]["B"])
            x = x0 + x1
    else:
        x = x0

    with timer.record("prefill.mlp.activation"):
        x = gelu_approx(x)

    with timer.record("prefill.mlp.fc2"):
        x0 = linear(x, w.fc2)
    if lora is not None:
        with timer.record("prefill.mlp.lora_fc2"):
            x1 = F.linear(F.linear(x, lora["fc2"]["A"]), lora["fc2"]["B"])
            x = x0 + x1
    else:
        x = x0

    return x


def _moe_mlp_instrumented(
    x: torch.Tensor,
    mlp_module: torch.nn.Module,
    experts_per_token: int,
    *,
    mode: str = "decode",
) -> torch.Tensor:
    timer = _CURRENT_TIMER
    if timer is None or mode != "prefill":
        return original_moe_mlp(x, mlp_module, experts_per_token, mode=mode)

    B, T, C = x.shape
    x_flat = x.reshape(-1, C)

    router = mlp_module["router"]
    scatter_mlp = mlp_module["mlp"]

    with timer.record("prefill.moe.router_linear"):
        router_logits = router(x_flat)

    with timer.record("prefill.moe.topk"):
        topk_logits, topk_idxs = torch.topk(router_logits, experts_per_token, dim=-1)

    with timer.record("prefill.moe.softmax"):
        topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32).to(x.dtype)

    num_tokens, top_k = topk_idxs.shape

    with timer.record("prefill.moe.scatter_mlp"):
        mlp_out = scatter_mlp(x_flat, topk_weights, topk_idxs).view(B, T, C)
    return mlp_out


def _text_decoder_instrumented(
    x: torch.Tensor,
    module: torch.nn.Module,
    attn_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    config,
    lora: Optional[dict] = None,
    *,
    flashinfer_ctx=None,
    flashinfer_metadata=None,
    use_flashinfer: bool = False,
    use_graph: bool = False,
    mode: str = "decode",
) -> torch.Tensor:
    timer = _CURRENT_TIMER
    if timer is None or mode != "prefill":
        return original_text_decoder(
            x,
            module,
            attn_mask,
            position_ids,
            config,
            lora=lora,
            flashinfer_ctx=flashinfer_ctx,
            flashinfer_metadata=flashinfer_metadata,
            use_flashinfer=use_flashinfer,
            use_graph=use_graph,
            mode=mode,
        )

    for i, block in enumerate(module.blocks):
        if lora is not None:
            layer_lora = lora["text"]["blocks"][str(i)]
            mlp_lora = layer_lora["mlp"]
            attn_lora = layer_lora["attn"]
        else:
            mlp_lora = None
            attn_lora = None

        ln_weights = LayerNormWeights(weight=block.ln.weight, bias=block.ln.bias)
        with timer.record(f"prefill.layer{i:02d}.layer_norm"):
            x_norm = layer_norm(x, ln_weights)

        with timer.record(f"prefill.layer{i:02d}.attn"):
            attn_out = _attn_instrumented(
                x_norm,
                block.attn,
                module.freqs_cis,
                block.kv_cache,
                attn_mask,
                config.n_heads,
                config.n_kv_heads,
                position_ids,
                lora=attn_lora,
            )

        if config.moe is not None and i >= config.moe.start_layer:
            with timer.record(f"prefill.layer{i:02d}.moe"):
                mlp_out = _moe_mlp_instrumented(
                    x_norm,
                    block.mlp,
                    config.moe.experts_per_token,
                    mode=mode,
                )
        else:
            mlp_weights = MLPWeights(
                fc1=LinearWeights(
                    weight=block.mlp["fc1"].weight, bias=block.mlp["fc1"].bias
                ),
                fc2=LinearWeights(
                    weight=block.mlp["fc2"].weight, bias=block.mlp["fc2"].bias
                ),
            )
            with timer.record(f"prefill.layer{i:02d}.mlp"):
                mlp_out = _dense_mlp_instrumented(x_norm, mlp_weights, lora=mlp_lora)

        with timer.record(f"prefill.layer{i:02d}.residual"):
            x = x + attn_out + mlp_out

    return x


# Apply text instrumentation
text_mod.attn = _attn_instrumented  # type: ignore[assignment]
text_mod.text_decoder = _text_decoder_instrumented  # type: ignore[assignment]
text_mod.moe_mlp = _moe_mlp_instrumented  # type: ignore[assignment]
text_mod.mlp = _dense_mlp_instrumented  # type: ignore[assignment]


_original_prefill_method = MoondreamTextRuntime._prefill


def _prefill_wrapper(self, inputs_embeds, attn_mask, position_ids):
    with _phase("prefill.core.total"):
        return _original_prefill_method(self, inputs_embeds, attn_mask, position_ids)


MoondreamTextRuntime._prefill = _prefill_wrapper  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Profiling driver

def _load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _build_runtime(
    *,
    weights: Path,
    device: str,
    dtype: str,
    max_batch_size: int,
    max_seq_length: int,
    enable_cuda_graphs: bool,
    enable_compile: bool,
) -> MoondreamTextRuntime:
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    resolved_dtype = dtype_map.get(dtype.lower())
    if resolved_dtype is None:
        raise ValueError(f"Unsupported dtype {dtype}")

    paths = ModelPaths(weights=weights)
    rt_cfg = RuntimeConfig(
        model_paths=paths,
        device=device,
        dtype=resolved_dtype,
        enable_cuda_graphs=enable_cuda_graphs,
        enable_compile=enable_compile,
        max_batch_size=max_batch_size,
        max_seq_length=max_seq_length,
    )
    return MoondreamTextRuntime(rt_cfg)


def _tokenize_prompt(runtime: MoondreamTextRuntime, prompt: str) -> torch.Tensor:
    prompt_tokens = runtime.build_prompt_tokens(prompt)
    return prompt_tokens


@dataclasses.dataclass
class SummaryRow:
    name: str
    count: int
    cpu_ms_mean: float
    cpu_ms_total: float
    cpu_ms_std: float
    gpu_ms_mean: float
    gpu_ms_total: float
    gpu_util_mean: float
    cpu_share: float


def _summarize(records: Dict[str, List[PhaseSample]]) -> list[SummaryRow]:
    total_cpu = sum(sample.cpu_ms for samples in records.values() for sample in samples)
    summary: list[SummaryRow] = []
    for name, samples in records.items():
        cpu_vals = [s.cpu_ms for s in samples]
        gpu_vals = [s.gpu_ms for s in samples]
        util_vals = [s.gpu_util for s in samples if not math.isnan(s.gpu_util)]
        cpu_mean = statistics.mean(cpu_vals) if cpu_vals else 0.0
        cpu_total = sum(cpu_vals)
        cpu_std = statistics.pstdev(cpu_vals) if len(cpu_vals) > 1 else 0.0
        gpu_mean = statistics.mean(gpu_vals) if gpu_vals else float("nan")
        gpu_total = sum(gpu_vals)
        gpu_util_mean = statistics.mean(util_vals) if util_vals else float("nan")
        cpu_share = cpu_total / total_cpu if total_cpu > 0 else 0.0
        summary.append(
            SummaryRow(
                name=name,
                count=len(samples),
                cpu_ms_mean=cpu_mean,
                cpu_ms_total=cpu_total,
                cpu_ms_std=cpu_std,
                gpu_ms_mean=gpu_mean,
                gpu_ms_total=gpu_total,
                gpu_util_mean=gpu_util_mean,
                cpu_share=cpu_share,
            )
        )
    summary.sort(key=lambda row: row.cpu_ms_total, reverse=True)
    return summary


def _group_by_category(summary: Iterable[SummaryRow]) -> dict[str, SummaryRow]:
    buckets: dict[str, list[SummaryRow]] = defaultdict(list)
    for row in summary:
        if row.name.startswith("prefill.layer"):
            parts = row.name.split(".")
            if len(parts) >= 3:
                category = f"prefill.{parts[2]}"
            else:
                category = "prefill.layer"
        elif row.name.startswith("prefill.attn"):
            category = "prefill.attn"
        elif row.name.startswith("prefill.moe"):
            category = "prefill.moe"
        elif row.name.startswith("prefill.vision"):
            category = "prefill.vision"
        elif row.name.startswith("prefill.text"):
            category = "prefill.text"
        else:
            category = "other"
        buckets[category].append(row)

    category_rows: dict[str, SummaryRow] = {}
    for category, rows in buckets.items():
        cpu_total = sum(r.cpu_ms_total for r in rows)
        gpu_total = sum(r.gpu_ms_total for r in rows)
        count = sum(r.count for r in rows)
        cpu_vals = [r.cpu_ms_total for r in rows]
        cpu_mean = cpu_total / count if count > 0 else 0.0
        cpu_std = statistics.pstdev(cpu_vals) if len(cpu_vals) > 1 else 0.0
        gpu_mean = gpu_total / count if count > 0 else float("nan")
        util_vals = [r.gpu_util_mean for r in rows if not math.isnan(r.gpu_util_mean)]
        gpu_util_mean = statistics.mean(util_vals) if util_vals else float("nan")
        category_rows[category] = SummaryRow(
            name=category,
            count=count,
            cpu_ms_mean=cpu_mean,
            cpu_ms_total=cpu_total,
            cpu_ms_std=cpu_std,
            gpu_ms_mean=gpu_mean,
            gpu_ms_total=gpu_total,
            gpu_util_mean=gpu_util_mean,
            cpu_share=0.0,
        )
    total_cpu = sum(row.cpu_ms_total for row in category_rows.values())
    if total_cpu > 0:
        for row in category_rows.values():
            row.cpu_share = row.cpu_ms_total / total_cpu
    return category_rows


def run_profile(args) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    runtime = _build_runtime(
        weights=args.weights,
        device=args.device,
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
        max_seq_length=args.max_seq_length,
        enable_cuda_graphs=args.enable_cuda_graphs,
        enable_compile=args.enable_compile,
    )

    image: Optional[Image.Image] = None
    if args.image is not None:
        image = _load_image(args.image)

    prompt_tokens = _tokenize_prompt(runtime, args.prompt)

    # Warm-up (no instrumentation)
    state, _ = runtime.start_sequence(prompt_tokens=prompt_tokens, image=image)
    runtime.release_sequence(state)

    all_records: dict[str, list[PhaseSample]] = defaultdict(list)

    for run_idx in range(args.runs):
        timer = PhaseTimer()
        global _CURRENT_TIMER
        _CURRENT_TIMER = timer
        try:
            with timer.record("prefill.start_sequence.total"):
                state, _ = runtime.start_sequence(prompt_tokens=prompt_tokens, image=image)
        finally:
            _CURRENT_TIMER = None
        runtime.release_sequence(state)
        timer.merge_into(all_records)

    summary = _summarize(all_records)
    categories = _group_by_category(summary)

    print("=== Prefill Phase Summary (per-call totals across runs) ===")
    print(
        f"{'Phase':50s} {'Count':>5s} {'CPU ms (total)':>14s} {'CPU ms (mean)':>13s} {'GPU ms (total)':>14s} {'GPU util':>9s} {'CPU share':>10s}"
    )
    for row in summary:
        gpu_util = f"{row.gpu_util_mean*100:.1f}%" if not math.isnan(row.gpu_util_mean) else "n/a"
        print(
            f"{row.name:50s} {row.count:5d} {row.cpu_ms_total:14.3f} {row.cpu_ms_mean:13.3f} {row.gpu_ms_total:14.3f} {gpu_util:>9s} {row.cpu_share*100:9.2f}%"
        )

    print("\n=== Aggregated Categories ===")
    print(
        f"{'Category':30s} {'CPU ms (total)':>14s} {'GPU ms (total)':>14s} {'GPU util':>9s} {'CPU share':>10s}"
    )
    for name, row in sorted(categories.items(), key=lambda item: item[1].cpu_ms_total, reverse=True):
        gpu_util = f"{row.gpu_util_mean*100:.1f}%" if not math.isnan(row.gpu_util_mean) else "n/a"
        print(
            f"{name:30s} {row.cpu_ms_total:14.3f} {row.gpu_ms_total:14.3f} {gpu_util:>9s} {row.cpu_share*100:9.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Moondream prefill phases")
    parser.add_argument("--weights", type=Path, required=True, help="Path to Moondream weights")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to evaluate")
    parser.add_argument("--image", type=Path, help="Optional image path for multimodal prefill")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (default: cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Computation dtype")
    parser.add_argument("--max-batch-size", dest="max_batch_size", type=int, default=2)
    parser.add_argument("--max-seq-length", dest="max_seq_length", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=3, help="Number of measured runs after warm-up")
    parser.add_argument("--enable-cuda-graphs", action="store_true", help="Enable CUDA graphs (decode path)")
    parser.add_argument("--enable-compile", action="store_true", help="Enable torch.compile for prefill")
    return parser.parse_args()


if __name__ == "__main__":
    run_profile(parse_args())
