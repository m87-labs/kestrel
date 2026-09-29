# Kestrel

![Kestrel Overview](https://raw.githubusercontent.com/m87-labs/kestrel/main/assets/kestrel-overview.png)

High-performance inference engine for multimodal models.

Kestrel is the inference engine behind [Photon](https://moondream.ai/p/photon), Moondream's on-device deployment option. Most Moondream users should install via `pip install moondream`; this repository provides the engine directly and supports additional model families.

Kestrel provides async, micro-batched inference with streaming support, paged KV caching, and optimized CUDA and Metal kernels. It's designed for production deployments where throughput and latency matter.

## Features

- **Async micro-batching** — Cooperative scheduler batches heterogeneous requests without compromising per-request latency
- **Streaming** — Real-time token and transcription progress
- **Multi-task** — Vision-language generation, spatial reasoning, transcription, and speech synthesis
- **Paged KV cache** — Efficient memory management for high concurrency
- **Prefix caching** — Radix tree-based caching for repeated prompts and images
- **LoRA adapters** — Parameter-efficient fine-tuning support with automatic cloud loading

## Requirements

- Python 3.10–3.14.
- One of:
  - **NVIDIA GPU** on Linux x86_64 / aarch64 or Windows x86_64. Optimized kernels for
    SM80 (A100), SM86 (A10, RTX 30-series), SM87 (Jetson Orin), SM89 (L4, L40S, RTX 4090),
    SM90 (H100, H200, GH200), SM100 (B200), SM110 (Jetson Thor), SM120 (RTX PRO 6000).
    Other CUDA GPUs may work but have not been tested.
  - **Apple Silicon Mac** (M-series) on macOS 13 (Ventura) or later, with native Metal kernels.
- `MOONDREAM_API_KEY` (optional) — only needed for Moondream finetuned-model inference (get a key from [moondream.ai](https://moondream.ai))

## Installation

```bash
pip install kestrel
```

For Jetson Orin (JetPack 6) or Jetson Thor (JetPack 7), see the
[Jetson setup guide](https://github.com/m87-labs/kestrel/blob/main/docs/jetson.md).

## Model Access

Kestrel supports these model families:

| Model | Repository | Notes |
|-------|------------|-------|
| Moondream 2 | [vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2) | Public, no approval needed |
| Moondream 3 | [moondream/moondream3-preview](https://huggingface.co/moondream/moondream3-preview) | Public, no approval needed |
| Moondream 3.1 9B A2B | [moondream/moondream3.1-9B-A2B](https://huggingface.co/moondream/moondream3.1-9B-A2B) | Public, no approval needed |
| DINOv2 ViT-S/14 | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) | Image embeddings; one-launch encoder on H100 BF16 |
| Qwen 3.5 | [Qwen 3.5 collection](https://huggingface.co/collections/Qwen/qwen35) | 0.8B, 2B, 4B, 9B, 27B, and 35B-A3B; Base variants where published |
| Qwen 3.6 | [Qwen 3.6 collection](https://huggingface.co/collections/Qwen/qwen36) | 27B and 35B-A3B; BF16 and FP8 checkpoints |
| Gemma 4 | [Gemma 4 collection](https://huggingface.co/collections/google/gemma-4) | E2B, E4B, 26B-A4B, and 31B base/instruction variants |
| Whisper large-v3-turbo | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | Transcription, translation, long-form audio, and word timestamps |
| Qwen3-ASR | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Transcription, long-form/live audio, language hints, prompting, and forced-aligned word timestamps |
| Parakeet TDT 0.6B v3 | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Multilingual transcription, long-form/live audio, and native word/character timestamps |
| parakeet-redux | [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) | The ternary Parakeet: 178 MB, the same capabilities, and runs on the CPU and Apple silicon as well as CUDA |
| parakeet-ultra | [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) | The full-precision Parakeet trained further: the same capabilities, better on every benchmark than the original |
| Qwen3-TTS CustomVoice | [0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice), [1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Streaming 24 kHz speech synthesis with voice and language selection |
| Kokoro-82M | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | 24 kHz phoneme-to-speech with voice blends, speed control, and segment streaming |

## Quick Start

```python
import asyncio

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine


async def main():
    # Weights are automatically downloaded from HuggingFace on first run.
    # Use a registered model name or Hugging Face repository ID.
    cfg = RuntimeConfig(model="google/gemma-4-E2B-it")

    # Create the engine (loads model and warms up). No API key needed for
    # local inference; pass api_key="..." only for finetuned models.
    engine = await InferenceEngine.create(cfg)

    # Load an image (JPEG, PNG, or WebP bytes)
    image = open("photo.jpg", "rb").read()

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

## Image embeddings

`dinov2-small` serves the single-pass `embed` task. Its output contains FP32
`last_hidden_state` (`[1, 257, 384]`) and `pooler_output` (`[1, 384]`) tensors.
The H100 BF16 path loads the shipped encoder; other devices use the inference
model with the same output contract.

```python
import asyncio
from PIL import Image
from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine


async def main():
    engine = await InferenceEngine.create(RuntimeConfig(model="dinov2-small"))
    try:
        result = await engine.model().embed(image=Image.open("photo.jpg"))
        print(result.output["pooler_output"].shape)
    finally:
        await engine.shutdown()


asyncio.run(main())
```

## Speech transcription

Whisper, Qwen3-ASR, and Parakeet use the same model-bound `transcribe`
capability. Choose a Hugging Face repository ID from the model table; Kestrel
downloads its weights and selects that model's optimized runtime.

```python
import asyncio
from pathlib import Path

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine

MODEL = "Qwen/Qwen3-ASR-0.6B"


async def main():
    engine = await InferenceEngine.create(
        RuntimeConfig(
            model=MODEL,
            max_batch_size=4,
        )
    )
    model = engine.model(MODEL)
    try:
        result = await model.transcribe(
            audio=Path("meeting.m4a"),
            timestamps="word",
        )
        print(result.output["text"])
        for segment in result.output["segments"]:
            for word in segment.get("words", []):
                print(
                    f"{word['start']:7.2f}  {word['end']:7.2f}  {word['word']}"
                )
    finally:
        await engine.shutdown()


asyncio.run(main())
```

Kestrel accepts encoded paths, bytes, bounded binary streams, raw mono PCM,
and asynchronous PCM iterators. Long paths are decoded incrementally.
`moondream/parakeet-redux` also runs on the CPU and on Apple silicon: pass
`device="cpu"` or `device="mps"`. See
[Speech-to-text models](https://github.com/m87-labs/kestrel/blob/main/docs/asr.md)
for the shared interface and model
capability matrix, model-specific options, formats, and resource limits.

For speech synthesis with Qwen3-TTS or Kokoro, see
[Text-to-speech models](https://github.com/m87-labs/kestrel/blob/main/docs/tts.md).

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
Point prompts can also include normalized spatial references:

```python
result = await engine.point(
    image,
    "gaze",
    spatial_refs=[[0.42, 0.18]],  # e.g. the subject's head or eye location
)
```

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
image = open("photo.jpg", "rb").read()

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
    model="moondream3-preview",  # or "moondream2" / "moondream3.1-9B-A2B"
    max_batch_size=4,            # Max concurrent requests
    decode_path="auto",          # "auto", "native", or fail-closed "generated"
)
```

`decode_path="native"` disables generated decode construction on runtimes that
provide a native decode path. Qwen 3.5/3.6 requires compatible bundled generated
programs for both `"auto"` and `"generated"`, covering every active batch size
up to `max_batch_size`; construction or decode fails instead of falling back.
Gemma 4 applies the same fail-closed coverage rule when `"generated"` is
selected. Moondream currently supports only the default `"auto"` policy.

To run from local files instead of the registered HuggingFace weights or
tokenizer, keep `model` set to the matching registered architecture and pass
local paths:

```python
RuntimeConfig(
    model="moondream3.1-9B-A2B",
    model_path="/models/moondream/model.safetensors",
    tokenizer_path="/models/moondream/tokenizer.json",
)
```

`model_path` points to the local checkpoint file and skips the automatic
HuggingFace weight download. `tokenizer_path` is optional and only applies to
models whose runtime uses a tokenizer; tokenizer-free models do not need one.
When provided, it can point directly to a `tokenizer.json` file or to a
directory containing `tokenizer.json`. When omitted, Kestrel uses the tokenizer
declared by the registered model. Local files must match the selected model
architecture and checkpoint format.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MOONDREAM_API_KEY` | Optional. Only needed for finetuned-model inference. Get this from [moondream.ai](https://moondream.ai). |
| `HF_HOME` | Override HuggingFace cache directory for downloaded weights (default: `~/.cache/huggingface`). |
| `HF_TOKEN` | Hugging Face token for private or gated model repositories. Alternatively, run `huggingface-cli login`. |
| `OMP_WAIT_POLICY` | Set to `passive` before torch loads when transcribing on the CPU, or torch's idle OpenMP workers spin on the cores the kernels use. |

## Triton Inference Server

Kestrel can be deployed as a
[Triton Inference Server](https://github.com/triton-inference-server/server)
backend. See the
[Triton setup guide](https://github.com/m87-labs/kestrel/blob/main/triton_server/README.md).

## Benchmarks

Throughput and latency for the `query` skill are tracked in
[PERFORMANCE.md](https://github.com/m87-labs/kestrel/blob/main/PERFORMANCE.md),
with results broken out by GPU.

## Telemetry

Kestrel reports basic usage telemetry to help us decide which hardware platforms
to prioritize for support and optimization. Each report includes the model in
use, your GPU type and memory, aggregate request/error and token counts, your
machine's hostname, and timestamps. Prompts, images, and model outputs are never
sent.

## License

Local inference is free and requires no API key. Finetuned-model inference
requires a Moondream API key — see [moondream.ai/pricing](https://moondream.ai/pricing).

For the DINOv2 Hopper route, supplied GPU `pixel_values` must already be contiguous
BF16 on the model device. CPU preprocessing casts before transfer. The native
Torch fallback continues to accept floating-point tensors and converts them to
its configured device and dtype.

### RF-DETR detection

RF-DETR runs through the single-pass detection API on Hopper with BF16:

```python
from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine

engine = await InferenceEngine.create(RuntimeConfig(model="rfdetr-nano"))
result = await engine.model("rfdetr-nano").detect(image=image, threshold=0.5)
print(result.output["objects"])
await engine.shutdown()
```

Available model names are `rfdetr-nano`, `rfdetr-small`, `rfdetr-medium`,
`rfdetr-base`, `rfdetr-large`, `rfdetr-xlarge`, and `rfdetr-2xlarge`.
Large means the current single-P4 model, not DeprecatedLarge. Each variant uses
its checkpoint's fixed resolution and batch size one. Unsupported devices or
precisions fail at startup; this route requires the corresponding artifacts in
`kestrel-kernels`.

`image` accepts RGB PIL images, encoded image bytes, HWC NumPy arrays, or CPU CHW
tensors. Floating-point pixels must be in `[0, 1]`. Alternatively, `pixel_values`
accepts preprocessed `[1, 3, resolution, resolution]` tensors. GPU inputs must
already be contiguous BF16 on the model device. Resizing and ImageNet normalization
run on CPU before the input transfer; the detector itself uses one GPU launch.

Each object contains normalized `x_min`, `y_min`, `x_max`, `y_max`, `score`,
`class_id`, and `label`. Classes use the original sparse COCO IDs, including
`90` for toothbrush; unused IDs have an empty label. This is fixed-vocabulary
detection, not an open-vocabulary `object=` prompt. `threshold` defaults to `0.5`;
`max_objects` caps results at 300. There is no NMS.

Use `model_path` to supply a checkpoint. Nano through Large download the released
COCO checkpoint when no path is provided. XL and 2XL currently require an explicit
path to their released checkpoint. Custom class vocabularies and segmentation
checkpoints are not supported by these declared executables.
