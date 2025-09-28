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
  - `examples/benchmark_http_server.py`: ramps HTTP load against `/v1/query`, increasing RPS until overload while tracking observed throughput, latency, TTFT, and error rates.
  - `examples/benchmark_decode_scaling.py`: measures decoder throughput directly against the runtime (no HTTP), prefilling once and timing decode iterations as batch size scales.
  - `examples/compare_vision.py`: runs reference vs Kestrel inference for the same image/prompt. Use `--mode reference|kestrel` to avoid loading both models concurrently.
  - `examples/inspect_kv.py`, `examples/probe_tau.py`: quickly spot regression in cache contents or τ gating.

- **HTTP serving interface** — `kestrel/server/http.py` exposes a Starlette-based ASGI app and `kestrel.main serve` CLI entrypoint. `/v1/query` accepts base64 data URLs for `image_url`, runs the shared inference engine, and returns text answers with prompt/decode token metrics.

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

### HTTP server

- **Launch**

  ```bash
  uv run python -m kestrel.main serve \
      --weights ~/code/moondream/model.pt \
      --device cuda --dtype bfloat16 \
      --default-max-new-tokens 128 \
      --host 0.0.0.0 --port 8080
  ```

  The server primes the shared engine on startup and then serves concurrent POST requests at `http://<host>:<port>/v1/query`. `/healthz` returns 200 once the warmup finishes.

- **Request shape**

  ```json
  {
    "question": "Describe the image",
    "image_url": "data:image/png;base64,<...>",
    "settings": {
      "temperature": 0.0,
      "top_p": 1.0
    }
  }
  ```

  `image_url` must be a base64 blob (raw or `data:image/...;base64,<payload>`). Responses include the generated `answer`, `request_id`, `finish_reason`, and engine timings (`ttft_s`, `decode_tokens_per_s`, etc.).

- **Load testing**

  ```bash
  uv run python examples/benchmark_http_server.py \
      --url http://127.0.0.1:8080/v1/query \
      --image external/moondream/assets/demo-1.jpg \
      --stage-duration 30 --start-concurrency 1 --concurrency-step 2 --max-concurrency 64
  ```

  The script warms the endpoint, then increases concurrency until overload (triggered by error rate, latency, or TTFT thresholds). Each stage prints concurrency, observed throughput, success/error counts, and p95 latency/TTFT; use `--output` to capture full JSON metrics for dashboards.

- **Reproducible benchmark workflow**

  1. **Sync & dependencies** – `./sync.sh <host>` to push the repo, then on the GPU box run `uv sync` inside `~/code/kestrel` so Starlette/Uvicorn/httpx land.
  2. **Launch server** – for each batch size (or other config) start the ASGI server in tmux:
     ```bash
     tmux new-session -d -s kestrel_http \
         'cd ~/code/kestrel && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python -m kestrel.main serve \
            --weights ~/code/moondream/model.pt \
            --device cuda --dtype bfloat16 \
            --max-batch-size ${BATCH} \
            --default-max-new-tokens 256 \
            --host 0.0.0.0 --port 8080'
     ```
     Wait for `Application startup complete` via `tmux capture-pane -pt kestrel_http` before driving load.
  3. **Run the HTTP ramp** – hit `/v1/query` with the load generator, storing results per configuration:
     ```bash
     uv run python examples/benchmark_http_server.py \
         --url http://127.0.0.1:8080/v1/query \
         --image external/moondream/assets/demo-1.jpg \
         --stage-duration 30 \
         --start-concurrency 1 --concurrency-step 2 --max-concurrency 32 \
         --latency-threshold-ms 4000 --ttft-threshold-ms 2500 --error-threshold 0.10 \
         --output /tmp/http_benchmark_bs${BATCH}.json
     ```
     Adjust the thresholds to match the GPU’s SLA. Restart the server between sweeps to pick up the next batch size.
  4. **Collect artifacts** – pull `/tmp/http_benchmark_bs*.json` back (`scp host:/tmp/http_benchmark_bs*.json reports/`).
  5. **Render HTML summary** – run the helper snippet below to emit `reports/http_benchmark_report.html` with per-stage p50/p95 latency, TTFT, decode/processing splits, and token throughput. Archive the HTML together with the raw JSON for future comparisons.

     ```bash
     python3 - <<'PY'
     import json
     from pathlib import Path

     def to_ms(value):
         return None if value is None else value * 1000.0

     def fmt(value, digits=2):
         if value is None:
             return '—'
         return f"{value:.{digits}f}"

     def fmt_pair(p50, p95):
         return f"{fmt(p50,1)}/{fmt(p95,1)}"

     report_dir = Path('reports')
     json_files = sorted(report_dir.glob('http_benchmark_bs*.json'), key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0))
     if not json_files:
         raise SystemExit('No benchmark JSON files found in reports/')

     parameters_note = None
     sections = []
     for json_path in json_files:
         data = json.loads(json_path.read_text())
         results = data.get('results', [])
         batch_size = int(''.join(filter(str.isdigit, json_path.stem)) or 0)
         if parameters_note is None:
             params = {
                 'stage_duration': data.get('stage_duration_s'),
                 'start_rps': data.get('start_rps'),
                 'rps_step': data.get('rps_step'),
                 'max_rps': data.get('max_rps'),
                 'latency_threshold_ms': data.get('latency_threshold_ms'),
                 'ttft_threshold_ms': data.get('ttft_threshold_ms'),
                 'error_threshold': data.get('error_threshold'),
             }
             parameters_note = (
                 f"stage_duration={params['stage_duration']} s, start_rps={params['start_rps']}, "
                 f"rps_step={params['rps_step']}, max_rps={params['max_rps']}, "
                 f"latency_threshold={params['latency_threshold_ms']} ms, "
                 f"ttft_threshold={params['ttft_threshold_ms']} ms, "
                 f"error_threshold={params['error_threshold']*100:.1f}%"
             )
         rows = []
         for stage in results:
             latency = stage.get('latency', {})
             ttft = stage.get('ttft', {})
             decode = stage.get('decode_latency', {})
             proc = stage.get('processing_latency', {})
             tokens = stage.get('tokens', {})
             rows.append(
                 "<tr>"
                 f"<td>{fmt(stage.get('target_rps'),1)}</td>"
                 f"<td>{fmt(stage.get('throughput_rps'),2)}</td>"
                 f"<td>{int(stage.get('concurrency_limit', 0))}</td>"
                 f"<td>{stage.get('requests_success')}/{stage.get('requests_total')}</td>"
                 f"<td>{fmt(stage.get('error_rate', 0)*100,1)}</td>"
                 f"<td>{fmt_pair(to_ms(latency.get('p50')), to_ms(latency.get('p95')))}</td>"
                 f"<td>{fmt_pair(to_ms(ttft.get('p50')), to_ms(ttft.get('p95')))}</td>"
                 f"<td>{fmt_pair(to_ms(decode.get('p50')), to_ms(decode.get('p95')))}</td>"
                 f"<td>{fmt_pair(to_ms(proc.get('p50')), to_ms(proc.get('p95')))}</td>"
                 f"<td>{fmt(tokens.get('input_per_s'),1)}</td>"
                 f"<td>{fmt(tokens.get('output_per_s'),1)}</td>"
                 f"<td>{'Yes' if stage.get('overloaded') else 'No'}</td>"
                 f"<td>{'; '.join(stage.get('overload_reasons') or []) or '—'}</td>"
                 "</tr>"
             )
         sections.append(
             f"<h2>Max Batch Size {batch_size}</h2>"
             "<table><thead><tr>"
             "<th>Target RPS</th><th>Observed RPS</th><th>Concurrency</th>"
             "<th>Success/Total</th><th>Error %</th><th>Latency p50/p95 (ms)</th>"
             "<th>TTFT p50/p95 (ms)</th><th>Decode p50/p95 (ms)</th><th>Processing p50/p95 (ms)</th>"
             "<th>Input tok/s</th><th>Output tok/s</th><th>Overloaded</th><th>Overload Reason</th>"
             "</tr></thead><tbody>" + ''.join(rows) + "</tbody></table>"
         )

     html = f"""
     <!DOCTYPE html>
     <html lang='en'>
     <head>
     <meta charset='utf-8'>
     <title>Kestrel HTTP Benchmark Report</title>
     <style>
     body {{ font-family: Arial, sans-serif; margin: 2rem; }}
     table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
     th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: center; }}
     th {{ background: #f0f0f0; }}
     h1 {{ margin-bottom: 0.5rem; }}
     p.note {{ margin-bottom: 2rem; font-style: italic; }}
     </style>
     </head>
     <body>
     <h1>Kestrel HTTP Ramp Benchmark</h1>
     <p class='note'>Parameters: {parameters_note}</p>
     {''.join(sections)}
     <p>Sources: {', '.join(p.name for p in json_files)}</p>
     </body>
     </html>
     """

     (report_dir / 'http_benchmark_report.html').write_text(html.strip(), encoding='utf-8')
     PY
     ```

## Pending Work

- [done] **Sampling modes & streaming** — Extend the scheduler/engine beyond greedy decoding: top-k/top-p sampling, temperature control, and streaming token delivery (plus answer-token masking consistent with the reference `MoondreamModel`).
- [done] **Vision prefix parity** — Bring the vision encoder, projection, and per-request image handling to parity with the reference implementation. (Spatial grounding & LoRA variants remain TODO.)
- **Spatial reasoning & LoRA variants** — Reintroduce spatial prompts, point detection, and LoRA adapter support to reach full multimodal parity.
- **Serving surfaces** — Build on the new HTTP server with auth, observability, request prioritization, and gRPC support for downstream integrations.
- **Automated testing & CI** — Stand up pytest coverage (page eviction, scheduler edge cases, τ/rotary parity), static type-checking (pyright), linting, and integrate benchmark smoke tests into CI/perf tracking.

These items track the remaining phases from the original flex-nano-vllm plan and will bring Kestrel from a parity-verified text core to a full serving stack.
