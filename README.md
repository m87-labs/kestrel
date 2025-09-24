# Kestrel – Flex-Nano Moondream Status

## Implemented Functionality

- **Self-contained Moondream text stack** — `kestrel/moondream/` hosts configs, rotary helpers, attention/MLP layers, τ scaling, and weight-loading code. Defaults map to the Moondream 3 preview when no external JSON is provided.

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

- **Usage pattern** — After syncing to a GPU box (e.g., `./sync.sh belka`) and running `uv sync`, parity and benchmarking are available via:  
  ```bash
  uv run python examples/compare_text.py \
      --weights ~/code/moondream/model.pt \
      --prompt "What is the capital of France?" \
      --device cuda --dtype bfloat16 --max-new-tokens 6

  uv run python examples/benchmark_scheduler.py \
      --weights ~/code/moondream/model.pt \
      --dtype float16 --device cuda \
      --max-seq-length 2048 --max-batch-size 4
  ```
  A zero diff on the compare script signals parity; the benchmark reports throughput/latency per round.

## Pending Work

1. **Coordinator / orchestration** — Introduce an engine layer (e.g., `kestrel/engine.py`) plus a richer CLI that loads configs/tokenizer once, owns the runtime + scheduler lifecycle, and exposes a clean submission API.
2. **Sampling modes & streaming** — Add top-k/top-p sampling, streaming token callbacks, and point-generation support to match the reference UX.
4. **Vision & multimodal integration** — Re-enable the Moondream vision encoder, spatial prompting, and LoRA variant handling so we can reach image-text parity.
5. **Serving surfaces** — Build HTTP/gRPC surfaces with request metadata, backpressure, auth/logging, and basic observability counters.
6. **Testing & CI** — Stand up pytest coverage (page eviction, scheduler edge cases, τ/rotary parity), type-checking (pyright), linting, and hook the benchmark into CI or perf dashboards.

These items track the remaining phases from the original flex-nano-vllm plan and will bring Kestrel from a parity-verified text core to a full serving stack.
