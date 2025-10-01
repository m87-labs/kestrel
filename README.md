# Kestrel Inference Engine

## Design Overview

- **Skill‑first architecture** – Every request is routed through a `SkillSpec`. Skills build prompts, describe decoding phases, and format results. The default `QuerySkill` emits plain text; structured skills (e.g., `PointSkill`) reuse the same interfaces without touching scheduler internals.
- **Runtime core** – `MoondreamRuntime` owns memory planning, paged KV caches, FlashInfer decode, and image encoding. It exposes generic helpers (`prefill_prompt`, `run_structured_phase`, `append_tokens`) that skills consume.
- **Generation scheduler** – `kestrel/scheduler/GenerationScheduler` batches prefill/decode steps across requests, tracks per-request phase state, and streams tokens. It is skill-agnostic: all decoding/formatting decisions come from the attached skill object.
- **Async engine** – `kestrel/engine.InferenceEngine` mediates between clients and the scheduler. It resolves skills, builds skill-specific request objects, gathers metrics, and exposes high-level entrypoints such as `engine.query(...)`.
- **Serving surfaces** – `kestrel.main serve` launches a Starlette app (`kestrel/server/http.py`) that shares the async engine, providing HTTP endpoints with consistent metrics and streaming behaviour. CLI helpers keep local smoke tests close to production traffic patterns.

## API Overview

### Python Engine

```python
from pathlib import Path

import torch

from kestrel.config import ModelPaths, RuntimeConfig
from kestrel.engine import InferenceEngine
import pyvips

cfg = RuntimeConfig(
    model_paths=ModelPaths(
        weights=Path("~/code/moondream/model.pt").expanduser(),
    ),
    device="cuda",
    dtype=torch.bfloat16,
    max_batch_size=4,
)
engine = await InferenceEngine.create(cfg)

image = pyvips.Image.new_from_file("demo.jpg")
result = await engine.query(
    image=image,
    question="Describe the image.",
    settings={
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
    },
)
print(result.output["answer"])

point_result = await engine.point(
    image,
    "cat",
    settings={"max_objects": 2},
)
print(point_result.output["points"])

detect_result = await engine.detect(
    image,
    "cat",
    settings={"max_objects": 4},
)
print(detect_result.output["objects"])

caption_result = await engine.caption(
    image,
    length="normal",
    settings={
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 64,
    },
)
print(caption_result.output["caption"])
```

- `InferenceEngine.query(...)`, `.caption(...)`, `.point(...)`, and `.detect(...)` mirror the `moondream` reference API (async, with optional `settings` dictionaries). When `stream=True` is passed to `query` or `caption`, the helpers now return an `EngineStream` async iterator that yields incremental text chunks while decoding continues. Reasoning traces remain opt-in via the `reasoning` flag.
- Direct helpers like `engine.query(...)`, `engine.point(...)`, `engine.detect(...)`, and `engine.caption(...)` remain the supported public surface.

#### Streaming usage

```python
query_stream = await engine.query(
    question="Summarize the scene.",
    reasoning=False,
    stream=True,
    settings={"max_tokens": 64},
)
async for update in query_stream:
    print(f"answer += {update.text}")
query_result = await query_stream.result()
print("final", query_result.output["answer"])

caption_stream = await engine.caption(
    image=image,
    length="short",
    stream=True,
    settings={"temperature": 0.2, "max_tokens": 80},
)
chunks = []
async for update in caption_stream:
    chunks.append(update.text)
caption_result = await caption_stream.result()
assert "".join(chunks) == caption_result.output["caption"]
```

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
  "settings": {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 128
  }
}
```

Response fields include the generated `answer`, `finish_reason`, and timing metrics (`processing_latency_s`, `ttft_s`, `decode_latency_s`, token counts/throughput). When `"stream": true` is supplied, the endpoint upgrades to Server-Sent Events and emits incremental `chunk` payloads; the final event carries the completed answer alongside metrics.

`POST /v1/point`

```json
{
  "object": "burger",
  "image_url": "data:image/png;base64,<...>",
  "settings": {
    "max_objects": 2
  }
}
```

Responses include `points` (normalised `[x, y]` pairs), `finish_reason`, request metadata, and the same timing metrics as `/v1/query`.

`POST /v1/detect`

```json
{
  "object": "burger",
  "image_url": "data:image/png;base64,<...>",
  "settings": {
    "max_objects": 10
  }
}
```

Returns `objects` (each `{ "x_min", "y_min", "x_max", "y_max" }`), the finish reason, and latency metrics matching `/v1/query`.

`POST /v1/caption`

```json
{
  "image_url": "data:image/png;base64,<...>",
  "length": "normal",
  "settings": {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 64
  }
}
```

Responses include the generated `caption`, finish reason, and the standard metrics block.

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
