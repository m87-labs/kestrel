# Kestrel

![Kestrel Overview](https://raw.githubusercontent.com/m87-labs/kestrel/main/assets/kestrel-overview.png)

High-performance inference engine for the [Moondream](https://moondream.ai) vision-language model.

Kestrel provides async, micro-batched serving with streaming support, paged KV caching, and optimized CUDA kernels. It's designed for production deployments where throughput and latency matter.

## Features

- **Async micro-batching** — Cooperative scheduler batches heterogeneous requests without compromising per-request latency
- **Streaming** — Real-time token streaming for query and caption tasks
- **Multi-task** — Visual Q&A, captioning, point detection, object detection, and segmentation
- **Paged KV cache** — Efficient memory management for high concurrency
- **Prefix caching** — Radix tree-based caching for repeated prompts and images
- **LoRA adapters** — Parameter-efficient fine-tuning support with automatic cloud loading

## Requirements

- Python 3.10+
- NVIDIA Hopper GPU or newer (e.g. H100)
- `MOONDREAM_API_KEY` environment variable (get this from [moondream.ai](https://moondream.ai))

## Installation

```bash
pip install kestrel huggingface_hub
```

## Model Access

Kestrel supports both Moondream 3 and Moondream 2:

| Model | Repository | Notes |
|-------|------------|-------|
| Moondream 2 | [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) | Public, no approval needed |
| Moondream 3 | [moondream/moondream3-preview](https://huggingface.co/moondream/moondream3-preview) | Requires access approval |

For Moondream 3, request access (automatically granted) then authenticate with `huggingface-cli login` or set `HF_TOKEN`.

## Quick Start

```python
import asyncio

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine


async def main():
    # Download model weights (cached after first run)
    # For Moondream 2:
    model_path = hf_hub_download("vikhyatk/moondream2", filename="model.safetensors")
    model = "moondream2"

    # For Moondream 3:
    # model_path = hf_hub_download("vikhyatk/moondream3-preview", filename="model_fp8.pt")
    # model = "moondream3-preview"

    # Configure the engine
    cfg = RuntimeConfig(model_path, model=model)

    # Create the engine (loads model and warms up)
    engine = await InferenceEngine.create(cfg)

    # Load an image (as numpy array)
    image = np.array(Image.open("photo.jpg").convert("RGB"))

    # Visual question answering
    result = await engine.query(
        image=image,
        question="What's in this image?",
        settings={"temperature": 0.2, "max_tokens": 512},
    )
    print(result.output["answer"])

    # Clean up
    await engine.shutdown()


asyncio.run(main())
```

## Tasks

Kestrel supports several vision-language tasks through dedicated methods on the engine.

### Query (Visual Q&A)

Ask questions about an image:

```python
result = await engine.query(
    image=image,
    question="How many people are in this photo?",
    settings={
        "temperature": 0.2,  # Lower = more deterministic
        "top_p": 0.9,
        "max_tokens": 512,
    },
)
print(result.output["answer"])
```

### Caption

Generate image descriptions:

```python
result = await engine.caption(
    image,
    length="normal",  # "short", "normal", or "long"
    settings={"temperature": 0.2, "max_tokens": 512},
)
print(result.output["caption"])
```

### Point

Locate objects as normalized (x, y) coordinates:

```python
result = await engine.point(image, "person")
print(result.output["points"])
# [{"x": 0.5, "y": 0.3}, {"x": 0.8, "y": 0.4}]
```

Coordinates are normalized to [0, 1] where (0, 0) is top-left.

### Detect

Detect objects as bounding boxes:

```python
result = await engine.detect(
    image,
    "car",
    settings={"max_objects": 10},
)
print(result.output["objects"])
# [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6}, ...]
```

Bounding box coordinates are normalized to [0, 1].

### Segment

Generate a segmentation mask (Moondream 3 only):

```python
result = await engine.segment(image, "dog")
seg = result.output["segments"][0]
print(seg["svg_path"])  # SVG path data for the mask
print(seg["bbox"])      # {"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}
```

Note: Segmentation requires Moondream 3 and separate model weights. Contact [moondream.ai](https://moondream.ai) for access.

## Streaming

For longer responses, you can stream tokens as they're generated:

```python
image = np.array(Image.open("photo.jpg").convert("RGB"))

stream = await engine.query(
    image=image,
    question="Describe this scene in detail.",
    stream=True,
    settings={"max_tokens": 1024},
)

# Print tokens as they arrive
async for chunk in stream:
    print(chunk.text, end="", flush=True)

# Get the final result with metrics
result = await stream.result()
print(f"\n\nGenerated {result.metrics.output_tokens} tokens")
```

Streaming is supported for `query` and `caption` methods.

## Response Format

All methods return an `EngineResult` with these fields:

```python
result.output          # Dict with task-specific output ("answer", "caption", "points", etc.)
result.finish_reason   # "stop" (natural end) or "length" (hit max_tokens)
result.metrics         # Timing and token counts
```

The `metrics` object contains:

```python
result.metrics.input_tokens     # Number of input tokens (including image)
result.metrics.output_tokens    # Number of generated tokens
result.metrics.prefill_time_ms  # Time to process input
result.metrics.decode_time_ms   # Time to generate output
result.metrics.ttft_ms          # Time to first token
```

## Using Finetunes

If you've created a finetuned model through the [Moondream API](https://moondream.ai), you can use it by passing the adapter ID:

```python
result = await engine.query(
    image=image,
    question="What's in this image?",
    settings={"adapter": "01J5Z3NDEKTSV4RRFFQ69G5FAV@1000"},
)
```

The adapter ID format is `{finetune_id}@{step}` where:
- `finetune_id` is the ID of your finetune job
- `step` is the training step/checkpoint to use

Adapters are automatically downloaded and cached on first use.

## Configuration

### RuntimeConfig

```python
RuntimeConfig(
    model_path="/path/to/model.pt",
    max_batch_size=8,  # Max concurrent requests (default: 4)
)
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MOONDREAM_API_KEY` | Required. Get this from [moondream.ai](https://moondream.ai). |

## License

Free for evaluation and non-commercial use. Commercial use requires a license from [Moondream](https://moondream.ai).
