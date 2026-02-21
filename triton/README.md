# Kestrel Triton Inference Server Backend

Deploy Moondream via [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) using the Python backend. Triton handles HTTP/gRPC protocol while Kestrel owns all batching and scheduling internally.

## Architecture

- **Single `moondream` Triton model** handles all skills (query, caption, detect, point) via a `SKILL` input tensor
- **Decoupled mode** — streaming sends multiple responses; non-streaming sends one with `COMPLETE_FINAL`
- **Triton batching bypassed** (`max_batch_size: 0`) — Kestrel's internal scheduler owns all batching

## Prerequisites

- NVIDIA Triton Inference Server (24.08+ recommended)
- Python 3.10+
- `kestrel` installed (`pip install kestrel`)
- GPU with sufficient VRAM for the model
- A Moondream API key (`MOONDREAM_API_KEY`)

## Running Triton

The `triton/model_repository/` directory is ready to use directly as a Triton model repository.

```bash
docker run --gpus all --rm -it \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ./triton/model_repository:/models \
  -e MOONDREAM_API_KEY=your-api-key \
  -e KESTREL_MODEL=moondream3-preview \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  bash -c "pip install kestrel && tritonserver --model-repository=/models"
```

To use a local model checkpoint instead of downloading:

```bash
docker run --gpus all --rm -it \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ./triton/model_repository:/models \
  -v /path/to/model/weights:/model_weights \
  -e MOONDREAM_API_KEY=your-api-key \
  -e KESTREL_MODEL_PATH=/model_weights \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  bash -c "pip install kestrel && tritonserver --model-repository=/models"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOONDREAM_API_KEY` | (required) | Moondream API key for model access |
| `KESTREL_MODEL_PATH` | (none) | Local path to model weights |
| `KESTREL_MODEL` | `moondream3-preview` | Model name (for download) |
| `KESTREL_DEVICE` | `cuda` | Device (`cuda`, `cpu`) |
| `KESTREL_MAX_BATCH_SIZE` | `4` | Maximum concurrent batch size |

## Client Examples

All requests go to the single `moondream` model. The `SKILL` input tensor selects which skill to invoke.

Since the model uses decoupled mode (for streaming support), all requests must use the streaming gRPC API (`async_stream_infer`), even for non-streaming calls.

```python
import json
import threading
import numpy as np
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient("localhost:8001")

with open("image.jpg", "rb") as f:
    image_bytes = f.read()

def string_input(name, value):
    inp = grpcclient.InferInput(name, [1], "BYTES")
    inp.set_data_from_numpy(np.array([value], dtype=np.object_))
    return inp

def bytes_input(name, data):
    inp = grpcclient.InferInput(name, [1], "BYTES")
    inp.set_data_from_numpy(np.array([data], dtype=np.object_))
    return inp

def bool_input(name, value):
    inp = grpcclient.InferInput(name, [1], "BOOL")
    inp.set_data_from_numpy(np.array([value], dtype=np.bool_))
    return inp

def infer(inputs):
    """Send a request and collect all response chunks."""
    chunks, done = [], threading.Event()
    def cb(result, error):
        if error:
            raise RuntimeError(error)
        data = json.loads(result.as_numpy("TEXT_OUTPUT")[0].decode("utf-8"))
        chunks.append(data)
        if data.get("completed", True):
            done.set()
    client.start_stream(callback=cb)
    client.async_stream_infer("moondream", inputs)
    done.wait()
    client.stop_stream()
    return chunks
```

### Query

```python
chunks = infer([
    string_input("SKILL", "query"),
    string_input("QUESTION", "What is in this image?"),
    bytes_input("IMAGE", image_bytes),
])
print(chunks[0]["answer"])
```

### Streaming Caption

```python
chunks = infer([
    string_input("SKILL", "caption"),
    bytes_input("IMAGE", image_bytes),
    bool_input("STREAM", True),
])
# Intermediate chunks have {"chunk": "...", "completed": false}
# Final chunk has {"caption": "full text", "completed": true}
print(chunks[-1]["caption"])
```

### Detect / Point

```python
chunks = infer([
    string_input("SKILL", "detect"),
    string_input("OBJECT", "car"),
    bytes_input("IMAGE", image_bytes),
])
print(chunks[0]["objects"])  # [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8}]

chunks = infer([
    string_input("SKILL", "point"),
    string_input("OBJECT", "person"),
    bytes_input("IMAGE", image_bytes),
])
print(chunks[0]["points"])  # [{"x": 0.5, "y": 0.3}]
```

## Output JSON Format

### Query
```json
{"request_id": "42", "finish_reason": "stop", "answer": "...", "metrics": {"input_tokens": 128, "output_tokens": 64, "prefill_time_ms": 12.5, "decode_time_ms": 45.2, "ttft_ms": 15.0}}
```

With reasoning enabled, an additional `"reasoning"` field is included.

### Caption
```json
{"request_id": "43", "finish_reason": "stop", "caption": "...", "metrics": {...}}
```

### Detect
```json
{"request_id": "44", "finish_reason": "stop", "objects": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.8}], "metrics": {...}}
```

### Point
```json
{"request_id": "45", "finish_reason": "stop", "points": [{"x": 0.5, "y": 0.3}], "metrics": {...}}
```

### Streaming Protocol

Streaming responses (query, caption) send multiple `TEXT_OUTPUT` responses via Triton's decoupled mode:

**Intermediate chunks:**
```json
{"chunk": "partial text", "completed": false, "token_index": 5}
```

**Final response (query):**
```json
{"chunk": "remaining", "completed": true, "request_id": "42", "finish_reason": "stop", "answer": "full text", "metrics": {...}}
```

**Final response (caption):**
```json
{"chunk": "remaining", "completed": true, "request_id": "43", "finish_reason": "stop", "caption": "full text", "metrics": {...}}
```

## Testing

Run the included test client:

```bash
pip install tritonclient[grpc] numpy
python triton/client.py --url localhost:8001 --image path/to/test.jpg
```
