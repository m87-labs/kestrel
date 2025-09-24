# Kestrel – Flex-Nano Moondream Status

## Implemented Functionality

- **Self-contained Moondream text stack** — `kestrel/moondream/` hosts configs, rotary helpers, attention/MLP layers, τ scaling, and weight-loading code. Defaults now mirror the latest upstream tokenizer templates, so prompts match the reference Moondream release when no external JSON is provided.

- **Paged KV cache integration** — `kestrel/models/moondream_text.py` wires the shared `PagedKVCache` (`kestrel/kv_cache.py`) into every transformer block, manages sequence state, and exposes `start_sequence` / `decode` / `decode_batch` / `release` helpers for greedy generation.

- **Runtime configuration & guards** — `kestrel/config.py` provides `ModelPaths` and `RuntimeConfig` to control device, dtype, page size, and batch limits. We now enforce `max_batch_size >= 2` (slot 0 stays reserved in the page table).

- **Parity with reference Moondream (text-only)** — Rotary tables are regenerated in float32, τ gating mirrors the upstream implementation, and parity scripts confirm identical logits with the Hugging Face runner on CUDA bf16 (`examples/compare_text.py`).

- **Scheduler + asynchronous engine** — `kestrel/scheduler/` implements request structs, waiting/running queues, page-table reservation, and a flex-nano–style prefill/decode loop; `kestrel/engine.py` layers on an asyncio coordinator that micro-batches submissions, exposes per-request metrics, and powers the CLI.

- **torch.compile & CUDA graphs** — Prefill runs under `torch.compile` by default (opt out via `enable_compile=False` or the CLI `--disable-compile`). Decode batches lazily capture CUDA graphs per batch size for faster replay; disable with `enable_cuda_graphs=False` or `--disable-cuda-graphs`.

- **Benchmarking tooling** — `examples/benchmark_scheduler.py` fires synthetic prompt traffic through the scheduler, reports throughput/latency, and produces reproducible numbers (see belka runs at batch sizes 2 / 4 / 8).

- **Diagnostics & regression tooling**  
  - `examples/compare_text.py`: end-to-end logit comparison (prefill + decode) against the reference model.  
  - `examples/inspect_kv.py`: captures per-layer K/V, τ, rotary tensors, and prints stats to pinpoint drift.  
  - `examples/probe_tau.py`: focused probe for layer-0 τ/rotary behaviour.

- **Usage pattern** — After syncing to a GPU box (e.g., `./sync.sh belka`) and running `uv sync`, parity and benchmarking remain reproducible:  
  ```bash
  # Text parity / greedy check
  uv run python examples/compare_text.py \
      --weights ~/code/moondream/model.pt \
      --prompt "What is the capital of France?" \
      --device cuda --dtype bfloat16 --max-new-tokens 6
  ```
  A zero diff confirms logits match the reference.

### Sampling & Benchmarking How-To

- **Sampling smoke test**  
  ```bash
  uv run python -m kestrel.main schedule \
      "Tell me about the oceans." \
      "How do rockets work?" \
      --weights ~/code/moondream/model.pt \
      --max-batch-size 8 \
      --max-new-tokens 128 \
      --device cuda --dtype bfloat16
  ```
  This exercises the asynchronous engine end-to-end; expect full responses (no immediate EOS) on the first decode step.

- **Scheduler benchmark**  
  ```bash
  uv run python examples/benchmark_scheduler.py \
      --weights ~/code/moondream/model.pt \
      --device cuda --dtype bfloat16 \
      --num-prompts 32 --max-new-tokens 512 \
      --max-batch-size 8 --max-seq-length 4096
  ```
  The script prints prefill/decode throughput and latency per round; use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on belka to match our recorded runs.

## Pending Work

- **Sampling modes & streaming** — Extend the scheduler/engine beyond greedy decoding: top-k/top-p sampling, temperature control, and streaming token delivery (plus answer-token masking consistent with the reference `MoondreamModel`).
- **Vision & multimodal integration** — Re-enable the Moondream vision encoder, spatial prompts, point detection, and LoRA variant handling to reach image-text parity.
- **Serving surfaces** — Layer HTTP/gRPC entrypoints with request metadata, backpressure, logging, and observability around the async engine.
- **Automated testing & CI** — Stand up pytest coverage (page eviction, scheduler edge cases, τ/rotary parity), static type-checking (pyright), linting, and integrate benchmark smoke tests into CI/perf tracking.

These items track the remaining phases from the original flex-nano-vllm plan and will bring Kestrel from a parity-verified text core to a full serving stack.
