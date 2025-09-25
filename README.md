# Kestrel – Flex-Nano Moondream Status

## Implemented Functionality

- **Multimodal Moondream runtime** — `kestrel/moondream/` assembles both the text decoder and vision encoder, mirrors upstream tokenizer templates, and loads weights directly from the production checkpoints (torch or safetensors). Rotary tables stay in fp32 for parity.

- **Paged KV cache integration** — `kestrel/models/moondream_text.py` wires the shared `PagedKVCache` into every transformer block, handles sequence accounting, and exposes `start_sequence` / `decode_batch` / `release` for greedy or sampling-based flows.

- **Vision prefix support** — Image crops are generated and stitched entirely on device, the vision stack runs under `torch.inference_mode`, and image embeddings are inserted with bidirectional attention while subsequent text stays causal. `MoondreamTextRuntime.greedy_generate(..., image=...)` now matches the reference model on parity checks.

- **Runtime configuration & guards** — `kestrel/config.py` exposes `RuntimeConfig` knobs for device, dtype, page size, sequence limits, and compiler flags. Invalid combinations (e.g., `max_batch_size < 2` or seq length not divisible by page size) are rejected early.

- **Scheduler + async engine** — `kestrel/scheduler/` implements a flex-nano–style prefill/decode loop with request queues, while `kestrel/engine.py` batches submissions on an asyncio worker. Execution metrics now include true processing latency, time-to-first-token (TTFT), decode latency, and per-request token counts.

- **torch.compile & CUDA graphs** — Prefill uses `torch.compile(dynamic=True)` by default (with fallbacks). Decode captures CUDA graphs per batch size; both can be disabled via config or CLI flags.

- **Benchmarking & diagnostics**
  - `examples/benchmark_scheduler.py`: fires batched workloads (text or image+text) and reports throughput plus latency breakdowns. Accepts `--image`/`--image-dir` to benchmark multimodal traffic.
  - `examples/compare_vision.py`: runs reference vs Kestrel inference for the same image/prompt. Use `--mode reference|kestrel` to avoid loading both models concurrently.
  - `examples/inspect_kv.py`, `examples/probe_tau.py`: quickly spot regression in cache contents or τ gating.

- **Usage pattern** — After syncing to a GPU box (e.g., `./sync.sh belka`) and running `uv sync` (or activating the existing venv on belka), parity and benchmarking remain reproducible:
  ```bash
  # Vision + text parity check (reference run)
  uv run python examples/compare_vision.py \
      --mode reference \
      --weights ~/code/moondream/model.pt \
      --image external/moondream/assets/demo-1.jpg \
      --prompt "Describe the image." \
      --device cuda --dtype bfloat16 --max-new-tokens 64
  ```

### Sampling & Benchmarking How-To

- **Sampling smoke test**

  ```bash
  uv run python -m kestrel.main schedule \
      "Tell me about the oceans." \
      "How do rockets work?" \
      --weights ~/code/moondream/model.pt \
      --max-batch-size 8 \
      --max-new-tokens 256 \
      --device cuda --dtype bfloat16 --stream
  ```

  This exercises the asynchronous engine end-to-end; expect full responses (no immediate EOS) on the first decode step.

- **Scheduler benchmark**
  ```bash
  uv run python examples/benchmark_scheduler.py \
      --weights ~/code/moondream/model.pt \
      --device cuda --dtype bfloat16 \
      --num-prompts 32 --max-new-tokens 512 \
      --max-batch-size 8 --max-seq-length 4096 \
      --image external/moondream/assets/demo-1.jpg
  ```
  Add `--image` (repeatable) or `--image-dir` to exercise vision-conditioned prompts; images cycle if fewer than prompts. The script prints prefill/decode throughput and latency per round; use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on belka to match our recorded runs.

## Pending Work

- [done] **Sampling modes & streaming** — Extend the scheduler/engine beyond greedy decoding: top-k/top-p sampling, temperature control, and streaming token delivery (plus answer-token masking consistent with the reference `MoondreamModel`).
- [done] **Vision prefix parity** — Bring the vision encoder, projection, and per-request image handling to parity with the reference implementation. (Spatial grounding & LoRA variants remain TODO.)
- **Spatial reasoning & LoRA variants** — Reintroduce spatial prompts, point detection, and LoRA adapter support to reach full multimodal parity.
- **Serving surfaces** — Layer HTTP/gRPC entrypoints with request metadata, backpressure, logging, and observability around the async engine.
- **Automated testing & CI** — Stand up pytest coverage (page eviction, scheduler edge cases, τ/rotary parity), static type-checking (pyright), linting, and integrate benchmark smoke tests into CI/perf tracking.

These items track the remaining phases from the original flex-nano-vllm plan and will bring Kestrel from a parity-verified text core to a full serving stack.
