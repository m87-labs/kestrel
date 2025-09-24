# Kestrel – Flex-Nano Moondream Status

## Implemented Functionality

- **Self-contained Moondream text stack** — `kestrel/moondream/` hosts configs, rotary helpers, attention/MLP layers, and weight-loading code. Defaults map to the Moondream 3 preview when no external JSON is provided.

- **Paged KV cache integration** — `kestrel/models/moondream_text.py` wires the shared `PagedKVCache` (`kestrel/kv_cache.py`) into every transformer block, manages sequence state, and exposes `start_sequence` / `decode` / `release` helpers for greedy generation.

- **Runtime configuration** — `kestrel/config.py` provides `ModelPaths` and `RuntimeConfig` to control device, dtype, page size, batch limits, and optional overrides for config/tokenizer assets.

- **Parity with reference Moondream (text-only)** — The rotary tables are regenerated in float32, and parity scripts confirm identical logits with the upstream Hugging Face implementation on CUDA bf16 (e.g., prompt “What is the capital of France?”). KV caches, τ scaling, and rotary outputs match layer-by-layer.

- **Diagnostics & regression tooling**  
  - `examples/compare_text.py`: end-to-end logit comparison (prefill + decode) against the reference model.  
  - `examples/inspect_kv.py`: captures per-layer K/V, τ, rotary tensors, and prints stats to pinpoint drift.  
  - `examples/probe_tau.py`: focused probe for layer-0 τ/rotary behaviour.

- **Usage pattern** — After syncing to a GPU box (e.g., `./sync.sh belka`) and running `uv sync`, the parity check is:  
  ```bash
  uv run python examples/compare_text.py \
      --weights ~/code/moondream/model.pt \
      --prompt "What is the capital of France?" \
      --device cuda --dtype bfloat16 --max-new-tokens 6
  ```
  A zero diff on all steps signals success. `inspect_kv.py` can be invoked with the same arguments for deeper instrumentation.

## Pending Work

1. **Scheduler & batching** — Port flex-nano’s inference queues (waiting/running/done deques, reservation checks, preemption) and build a batched prefill/decode loop that drives `MoondreamTextRuntime`.
2. **Execution orchestration** — Introduce a coordinator (e.g., `kestrel/engine.py`) plus a real CLI in `kestrel/main.py` to load tokenizers, launch the runtime, and accept multiple prompts.
3. **Sampling modes & streaming** — Add greedy/top-k/top-p utilities, streaming callbacks, and point-generation support mirroring the reference.
4. **Vision & multimodal integration** — Re-enable the Moondream vision encoder, spatial prompting, and LoRA variant handling so image-text parity is possible.
5. **Serving surfaces** — Expose HTTP/gRPC (or similar) endpoints, configuration hot-reload, auth/logging, and basic observability counters.
6. **Testing & benchmarks** — Stand up pytest coverage (page eviction, scheduler, flex-attention parity on toy weights), type-checking (pyright), linting, and throughput benchmarks comparing against the original Moondream runner.

These items track the remaining phases from the original flex-nano-vllm plan and will bring Kestrel from a parity-verified text core to a full serving stack.
