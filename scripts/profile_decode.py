#!/usr/bin/env python3
"""Profile a single decode forward pass through the LLM.

This script profiles one decode step to identify optimization targets.
It instruments each component with NVTX markers for nsys profiling.

## Quick Start (nsys)

    cd ~/code/kestrel
    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --stats=true \
        -o /tmp/decode_profile --force-overwrite=true \
        uv run python scripts/profile_decode.py --weights <path_to_weights>

Notes:
- CUDA graphs are always enabled; this script is meant to mirror production decode.
- The NVTX range "profile_window" wraps the steady-state decode iterations.
- The script prints a "PROFILE:" line with per-iteration timing (avg/p50/p90/min/max).
- For kernel breakdown within steady-state decode:
  nsys stats --report cuda_gpu_kern_sum --filter-nvtx profile_window /tmp/decode_profile.nsys-rep

## Comparison (BF16 vs FP8)

Use identical settings for a fair comparison (same batch/prompt/iters).

    # BF16
    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --stats=true \
        -o /tmp/decode_profile_bf16 --force-overwrite=true \
        uv run python scripts/profile_decode.py --weights ~/code/kestrel/model.pt \
            --decode-iters 200 --profile-iters 100

    # FP8
    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --stats=true \
        -o /tmp/decode_profile_fp8 --force-overwrite=true \
        uv run python scripts/profile_decode.py --weights ~/code/kestrel/model_fp8.pt \
            --decode-iters 200 --profile-iters 100

    # Extract steady-state kernel summaries for side-by-side comparison
    nsys stats --report cuda_gpu_kern_sum --filter-nvtx profile_window /tmp/decode_profile_bf16.nsys-rep
    nsys stats --report cuda_gpu_kern_sum --filter-nvtx profile_window /tmp/decode_profile_fp8.nsys-rep

Key output sections:
- 'cuda_gpu_kern_sum': Per-kernel timing

## Expected Components (24 layers: 4 dense + 20 MoE)

| Component | Count | Notes |
|-----------|-------|-------|
| layernorm | 25 | 24 pre-attn + 1 final |
| attention | 24 | QKV + rotary + FA3 + out_proj |
| dense_mlp | 4 | Layers 0-3 |
| moe_mlp | 20 | Layers 4-23 |
| lm_head | 1 | Final projection |
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Prefer locally precompiled CuTe kernels if available (keeps compiled extensions from the wheel).
_LOCAL_PRECOMPILED = (
    REPO_ROOT / "kestrel-kernels" / "python" / "kestrel_kernels" / "precompiled"
)
if _LOCAL_PRECOMPILED.exists():
    try:
        from kestrel_kernels import precompile as _kk_precompile

        _kk_precompile.PRECOMPILED_DIR = _LOCAL_PRECOMPILED
    except Exception:
        pass

_FLASH_ATTN_PRECOMPILED_DISABLED = False


def _force_flash_attn_jit() -> None:
    """Disable precompiled FlashAttention kernels (force JIT) for this process."""
    global _FLASH_ATTN_PRECOMPILED_DISABLED
    if _FLASH_ATTN_PRECOMPILED_DISABLED:
        return
    from kestrel_kernels.flash_attn.cute import interface as fa_interface

    def _no_precompiled(*_args, **_kwargs):
        return None

    fa_interface._try_load_precompiled_flash_attn = _no_precompiled  # type: ignore[assignment]
    fa_interface._flash_attn_fwd.compile_cache.clear()
    _FLASH_ATTN_PRECOMPILED_DISABLED = True


def _maybe_retry_flash_attn(exc: Exception) -> bool:
    """Return True if we should retry after forcing FlashAttention JIT."""
    msg = str(exc)
    if "Mismatched type on argument #4" in msg and not _FLASH_ATTN_PRECOMPILED_DISABLED:
        print("Detected precompiled FlashAttention mismatch; forcing JIT compile and retrying.")
        _force_flash_attn_jit()
        return True
    return False


def patch_text_decoder_with_nvtx():
    """Monkey-patch text_decoder to add NVTX markers around each component."""
    from kestrel.moondream import text as text_module
    from kestrel.moondream.layers import layer_norm, mlp, moe_mlp, LayerNormWeights, MLPWeights, LinearWeights

    def instrumented_text_decoder(
        x, module, attn_mask, position_ids, config, *,
        slot_mapping, use_prefix_attn=False, mode="decode",
        page_table=None, fa3_seqused_k=None,
        lora_workspace=None, lora_slot_ids=None, single_lora_id=None,
    ):
        for i, block in enumerate(module.blocks):
            # LayerNorm
            nvtx.range_push(f"layer{i}_ln")
            ln_weights = LayerNormWeights(weight=block.ln.weight, bias=block.ln.bias)
            x_norm = layer_norm(x, ln_weights)
            nvtx.range_pop()

            # Attention
            nvtx.range_push(f"layer{i}_attn")
            attn_out = text_module.attn(
                x_norm,
                block.attn,
                module.cos_sin_cache,
                block.kv_cache,
                attn_mask,
                config.n_heads,
                config.n_kv_heads,
                position_ids,
                mode=mode,
                slot_mapping=slot_mapping,
                use_prefix_attn=use_prefix_attn,
                page_table=page_table,
                fa3_seqused_k=fa3_seqused_k,
            )
            nvtx.range_pop()

            # MLP (dense or MoE)
            if config.moe is not None and i >= config.moe.start_layer:
                nvtx.range_push(f"layer{i}_moe")
                moe_workspace = lora_workspace.moe_layer(i) if lora_workspace else None
                mlp_out = moe_mlp(
                    x_norm,
                    block.mlp,
                    config.moe.experts_per_token,
                    mode=mode,
                    lora_workspace=moe_workspace,
                    lora_slot_ids=lora_slot_ids,
                    single_lora_id=single_lora_id,
                )
                nvtx.range_pop()
            else:
                nvtx.range_push(f"layer{i}_dense_mlp")
                mlp_weights = MLPWeights(
                    fc1=LinearWeights(
                        weight=block.mlp["fc1"].weight, bias=block.mlp["fc1"].bias
                    ),
                    fc2=LinearWeights(
                        weight=block.mlp["fc2"].weight, bias=block.mlp["fc2"].bias
                    ),
                )
                dense_workspace = lora_workspace.dense_layer(i) if lora_workspace else None
                mlp_out = mlp(
                    x_norm,
                    mlp_weights,
                    lora_workspace=dense_workspace,
                    lora_slot_ids=lora_slot_ids,
                )
                nvtx.range_pop()

            x = x + attn_out + mlp_out

        return x

    text_module.text_decoder = instrumented_text_decoder
    # Ensure runtime module uses the patched symbol (runtime imports text_decoder at module scope).
    try:
        from kestrel.moondream import runtime as runtime_module
        runtime_module.text_decoder = instrumented_text_decoder
    except Exception:
        pass


def patch_lm_head_with_nvtx():
    """Monkey-patch lm_head to add NVTX markers."""
    from kestrel.moondream import text as text_module
    from kestrel.moondream.layers import layer_norm, LayerNormWeights
    import torch.nn.functional as F

    def instrumented_lm_head(hidden, module, indices=None):
        nvtx.range_push("lm_head")
        hidden_last = hidden[:, -1, :]

        nvtx.range_push("lm_head_ln")
        post_ln = LayerNormWeights(weight=module.post_ln.weight, bias=module.post_ln.bias)
        hidden_norm = layer_norm(hidden_last, post_ln)
        nvtx.range_pop()

        nvtx.range_push("lm_head_proj")
        if indices is not None:
            weights = module.lm_head.weight[indices]
            bias = module.lm_head.bias[indices]
            logits = F.linear(hidden_norm, weights, bias)
        else:
            logits = module.lm_head(hidden_norm)
        nvtx.range_pop()

        nvtx.range_pop()  # lm_head
        return logits

    text_module.lm_head = instrumented_lm_head
    # Ensure runtime module uses the patched symbol (runtime imports lm_head at module scope).
    try:
        from kestrel.moondream import runtime as runtime_module
        runtime_module.lm_head = instrumented_lm_head
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Profile decode forward pass")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--batch-size", type=int, default=33, help="Decode batch size")
    parser.add_argument("--prompt-len", type=int, default=740, help="Prompt length (prefill tokens)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup decode iterations")
    parser.add_argument("--decode-iters", type=int, default=50, help="Total decode iterations")
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=20,
        help="Number of steady-state decode iterations to profile (last N iterations)",
    )
    args = parser.parse_args()

    # Apply NVTX instrumentation before importing runtime
    patch_text_decoder_with_nvtx()
    patch_lm_head_with_nvtx()

    from kestrel.config import RuntimeConfig
    from kestrel.moondream.runtime import MoondreamRuntime, SequenceState, TextToken

    device = torch.device("cuda")

    # Create runtime config
    # CUDA graphs require captured batch sizes >= requested batch size.
    # Graph sizes are generated up to (max_batch_size - 1), so allocate +1.
    max_batch_size = args.batch_size + 1
    runtime_cfg = RuntimeConfig(
        model_path=args.weights.expanduser(),
        max_batch_size=max_batch_size,
        enable_cuda_graphs=True,
    )

    print(f"Loading model from {args.weights}...")
    runtime = MoondreamRuntime(runtime_cfg)
    config = runtime.config.text
    moe_desc = (
        f"MoE starts at layer {config.moe.start_layer}" if config.moe is not None else "no MoE"
    )
    print(
        f"Model loaded. Config: {config.n_layers} layers, hidden={config.dim}, {moe_desc}"
    )

    # Create prompt tokens (BOS + some tokens)
    bos_id = runtime.config.tokenizer.bos_id
    prompt_tokens = [TextToken(bos_id)] + [
        TextToken(100 + i) for i in range(max(0, args.prompt_len - 1))
    ]

    total_decode_iters = max(1, args.decode_iters)
    profile_iters = max(1, min(args.profile_iters, total_decode_iters))
    max_new_tokens = args.warmup + total_decode_iters + 4

    print(
        f"Starting {args.batch_size} sequences with {len(prompt_tokens)} prompt tokens..."
    )
    seq_states: list[SequenceState] = []
    prefill_ok = True
    with torch.inference_mode():
        for _ in range(args.batch_size):
            try:
                seq_state, _ = runtime.start_sequence(
                    prompt_tokens, max_new_tokens=max_new_tokens
                )
                seq_states.append(seq_state)
            except Exception as exc:
                if _maybe_retry_flash_attn(exc):
                    try:
                        seq_state, _ = runtime.start_sequence(
                            prompt_tokens, max_new_tokens=max_new_tokens
                        )
                        seq_states.append(seq_state)
                        continue
                    except Exception as exc_retry:
                        print(
                            "Prefill failed after JIT retry "
                            f"({exc_retry}). Falling back to synthetic KV cache."
                        )
                prefill_ok = False
                break

    if not prefill_ok or len(seq_states) != args.batch_size:
        for seq in seq_states:
            runtime.release_sequence(seq)
        print("Falling back to synthetic KV cache for profiling.")
        seq_states = []
        with torch.inference_mode():
            for _ in range(args.batch_size):
                batch_idx = runtime.page_table.allocate()
                runtime.page_table.reserve(batch_idx, args.prompt_len + max_new_tokens)
                seq_states.append(
                    SequenceState(
                        batch_idx=batch_idx,
                        length=args.prompt_len,
                        max_length=args.prompt_len + max_new_tokens,
                        prompt_length=args.prompt_len,
                    )
                )
        prefill_ok = False

    print(f"Prefill done. Sequence at position {seq_states[0].length}")

    # Get decode slot
    slot = runtime.decode_slots[0]

    # Prepare static metadata buffers (batch idx + lora slots)
    for i, seq in enumerate(seq_states):
        slot.meta.batch_idx.cpu[i] = seq.batch_idx
        slot.meta.lora_slot_ids.cpu[i] = seq.lora_slot

    # Static decode inputs (all text tokens)
    token_ids = torch.full(
        (args.batch_size,),
        500,
        device=device,
        dtype=torch.long,
    )
    coord_values = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=slot.decode_coord_values.dtype,
    )
    size_values = torch.zeros(
        (args.batch_size, 2),
        device=device,
        dtype=slot.decode_size_values.dtype,
    )

    def run_decode():
        # Update per-step positions in pinned CPU buffer
        for i, seq in enumerate(seq_states):
            slot.meta.input_pos.cpu[i] = seq.length

        # H2D copies for metadata
        slot.meta.batch_idx.copy_to_gpu(args.batch_size)
        slot.meta.input_pos.copy_to_gpu(args.batch_size)
        slot.meta.lora_slot_ids.copy_to_gpu(args.batch_size)

        # Stage decode inputs
        slot.decode_token_ids[: args.batch_size].copy_(token_ids, non_blocking=True)
        slot.decode_coord_values[: args.batch_size].copy_(coord_values, non_blocking=True)
        slot.decode_size_values[: args.batch_size].copy_(size_values, non_blocking=True)

        # Run decode (uses runtime's unified decode path)
        try:
            runtime.decode_with_slot(slot, batch_size=args.batch_size)
        except Exception as exc:
            if _maybe_retry_flash_attn(exc):
                runtime.decode_with_slot(slot, batch_size=args.batch_size)
            else:
                raise

        # Advance sequence lengths to simulate token generation
        for seq in seq_states:
            seq.length += 1

    # Warmup
    print(f"Running {args.warmup} warmup decode iterations...")
    with torch.inference_mode(), torch.cuda.stream(slot.compute_stream):
        for _ in range(args.warmup):
            run_decode()
    torch.cuda.synchronize()

    profile_start = total_decode_iters - profile_iters
    print(
        "Running profiled decode iterations (CUDA graph replay)... "
        f"total={total_decode_iters}, profiled={profile_iters}"
    )

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(profile_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(profile_iters)]

    with torch.inference_mode(), torch.cuda.stream(slot.compute_stream):
        profiled_idx = 0
        for i in range(total_decode_iters):
            if i == profile_start:
                nvtx.range_push("profile_window")
            if i >= profile_start:
                start_events[profiled_idx].record()
            run_decode()
            if i >= profile_start:
                end_events[profiled_idx].record()
                profiled_idx += 1
        nvtx.range_pop()

    torch.cuda.synchronize()

    times_ms = [
        start.elapsed_time(end) for start, end in zip(start_events, end_events)
    ]
    times_ms_sorted = sorted(times_ms)
    avg_ms = sum(times_ms) / len(times_ms)
    p50_ms = times_ms_sorted[len(times_ms_sorted) // 2]
    p90_idx = max(0, int(0.9 * (len(times_ms_sorted) - 1)))
    p90_ms = times_ms_sorted[p90_idx]
    min_ms = times_ms_sorted[0]
    max_ms = times_ms_sorted[-1]

    print(
        "PROFILE: "
        f"batch={args.batch_size}, prompt_len={args.prompt_len}, "
        f"profile_iters={profile_iters}, "
        f"avg_ms={avg_ms:.3f}, p50_ms={p50_ms:.3f}, "
        f"p90_ms={p90_ms:.3f}, min_ms={min_ms:.3f}, max_ms={max_ms:.3f}"
    )

    # Cleanup
    if prefill_ok:
        for seq in seq_states:
            runtime.release_sequence(seq)
    else:
        for seq in seq_states:
            runtime.page_table.erase(seq.batch_idx, 0)

    print("Done. Run with nsys to see timing breakdown.")


if __name__ == "__main__":
    main()
