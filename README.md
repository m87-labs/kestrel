# Kestrel Inference Engine

## Design Overview

- **Skill‑first architecture** – Every request is routed through a `SkillSpec`. Skills build prompts, describe decoding phases, and format results. The default `QuerySkill` emits plain text; structured skills (e.g., `PointSkill`) reuse the same interfaces without touching scheduler internals.
- **Runtime core** – `MoondreamRuntime` owns memory planning, paged KV caches, FlashInfer decode, and image encoding. It exposes generic helpers (`prefill_prompt`, `run_structured_phase`, `append_tokens`) that skills consume.
- **Generation scheduler** – `kestrel/scheduler/GenerationScheduler` batches prefill/decode steps across requests, tracks per-request phase state, and streams tokens. It is skill-agnostic: all decoding/formatting decisions come from the attached skill object.
- **Async engine** – `kestrel/engine.InferenceEngine` mediates between clients and the scheduler. It resolves skills, builds `SkillRequest`s, gathers metrics, and exposes high-level entrypoints such as `engine.query(...)`.
- **Serving surfaces** – `kestrel.main serve` launches a Starlette app (`kestrel/server/http.py`) that shares the async engine, providing HTTP endpoints with consistent metrics and streaming behaviour. CLI helpers keep local smoke tests close to production traffic patterns.

## API Overview

### Python Engine

```python
from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine

cfg = RuntimeConfig(
    weights_path="~/code/moondream/model.pt",
    device="cuda",
    dtype="bfloat16",
    max_batch_size=4,
    max_seq_length=4096,
)
engine = await InferenceEngine.create(cfg)

result = await engine.query(
    question="Describe the image.",
    image=pyvips.Image.new_from_file("demo.jpg"),
    max_new_tokens=128,
    temperature=0.0,
)
print(result.text)
```

- `InferenceEngine.query(...)` routes to the `query` skill (the default text+image path).
- `InferenceEngine.submit(...)` remains available for advanced callers who want to pass raw prompt tokens or experiment with additional skills.

### Skills

- `kestrel.skills.base.SkillSpec` defines the contract for prompt construction, structured phase recipes, and result formatting.
- `kestrel.skills.query.QuerySkill` is registered by default. Additional skills can be registered via `SkillRegistry([...])` and passed to `InferenceEngine.create(..., skills=registry)`.

### CLI

- `uv run python -m kestrel.main serve` – launch the HTTP server (see usage examples below for full command).
- `uv run python -m kestrel.main schedule ...` – push one-off prompts through the async engine for smoke testing or benchmarking.
- `examples/` – self-contained scripts for parity checks, benchmarks, and diagnostics. They all run through the shared runtime so results match production.

### HTTP Endpoints

`POST /v1/query`

```json
{
  "question": "Describe the image",
  "image_url": "data:image/png;base64,<...>",
  "max_new_tokens": 128,
  "settings": {
    "temperature": 0.0,
    "top_p": 1.0
  }
}
```

Response fields include the generated `answer`, `finish_reason`, and timing metrics (`processing_latency_s`, `ttft_s`, `decode_latency_s`, token counts/throughput). Streaming responses emit Server-Sent Events with the same structure.

`POST /v1/point`

```json
{
  "object": "burger",
  "image_url": "data:image/png;base64,<...>",
  "max_points": 16
}
```

Responses include `points` (normalised `[x, y]` pairs), `finish_reason`, request metadata, and the same timing metrics as `/v1/query`.

## Usage Examples

### Vision + Text Parity Check (reference run)

```bash
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

  Exercises the asynchronous engine end-to-end; expect full responses (no immediate EOS) on the first decode step.

- **Scheduler benchmark**

  ```bash
  uv run python examples/benchmark_scheduler.py \
      --weights ~/code/moondream/model.pt \
      --device cuda --dtype bfloat16 \
      --num-prompts 32 --max-new-tokens 512 \
      --max-batch-size 8 --max-seq-length 4096 \
      --image external/moondream/assets/demo-1.jpg
  ```

  Add `--image` (repeatable) or `--image-dir` to exercise vision-conditioned prompts; images cycle if fewer than prompts. The script prints prefill/decode throughput and latency per round; use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on GPU hosts.

### HTTP Server

- **Launch**

  ```bash
  uv run python -m kestrel.main serve \
      --weights ~/code/moondream/model.pt \
      --device cuda --dtype bfloat16 \
      --max-batch-size 64 \
      --default-max-new-tokens 512 \
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

- **Load testing workflow**

  ```bash
  uv run python examples/benchmark_http_server.py \
      --url http://127.0.0.1:8080/v1/query \
      --image external/moondream/assets/demo-1.jpg \
      --stage-duration 30 --start-concurrency 1 --concurrency-step 4 --max-concurrency 64
  ```

  Increase concurrency until overload while tracking observed throughput, latency, TTFT, and error rates. Use `--output` to capture JSON metrics; the helper script in `examples/benchmark_http_server.py` can render an HTML summary from those artifacts.

- **Point detection example**

  ```bash
  curl -s http://127.0.0.1:8080/v1/point \
      -H 'content-type: application/json' \
      -d '{
            "object": "burger",
            "image_url": "data:image/jpeg;base64,'"$(base64 -w0 external/moondream/assets/demo-1.jpg)"'"
          }'
  ```

  Returns normalised coordinates under the `points` key.
