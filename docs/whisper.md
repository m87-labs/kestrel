# Whisper transcription

Kestrel serves `openai/whisper-large-v3-turbo` through the same model-bound
capability interface as its other model families. The public package owns the
model configuration, checkpoint loading, audio preparation, decoding policy,
long-form orchestration, timestamps, result schema, and optimized execution.
Generated decode programs are resolved through Kestrel's shared packed-kernel
runtime, just like the other CUDA model families.

There is no eager serving fallback. Creating the model without compatible
packed prefill, sampling, and generated-decode artifacts fails closed.

## Basic usage

```python
from pathlib import Path

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine

WHISPER_MODEL = "openai/whisper-large-v3-turbo"

engine = await InferenceEngine.create(
    RuntimeConfig(model=WHISPER_MODEL)
)
whisper = engine.model(WHISPER_MODEL)
try:
    result = await whisper.transcribe(
        audio=Path("interview.mp3"),
        timestamps="word",
    )
    print(result.output["text"])
finally:
    await engine.shutdown()
```

The JSON-compatible final output includes text, detected or requested language,
task, source and clip durations, diagnostics, and timestamped segments when
requested.

Binding the handle by repository ID keeps capability routing explicit. The
same `whisper` handle remains unambiguous when the engine is configured with
additional model runtimes.

## Inputs

`audio` accepts:

- a filesystem path;
- encoded bytes;
- a bounded seekable or non-seekable binary stream;
- a one-dimensional NumPy array or CPU Torch tensor containing mono PCM, with
  an explicit `sample_rate`;
- an asynchronous iterator of one-dimensional mono PCM chunks, with an
  explicit `sample_rate`.

Encoded files may be WAV/WAVEX, FLAC, MP3, Ogg Vorbis, Opus, M4A, MP4, MOV,
or WebM. Channel layouts are interpreted from container metadata and downmixed
to mono before transcription. Unsupported or ambiguous layouts fail closed.

Paths are decoded incrementally and never materialized as one unbounded
waveform. Encoded bytes and binary streams are snapshotted into a bounded
64 MiB buffer before incremental decoding; stream reads begin at the current
position. A one-shot raw PCM array is limited to 30 seconds. For longer raw
audio, pass an asynchronous iterator of chunks containing at most 1,048,576
samples each. Encoded files and live sessions are limited to 24 hours.

## Options

Arguments to `transcribe` belong to the model prompt:

| Argument | Default | Meaning |
| --- | --- | --- |
| `audio` | required | Encoded audio, raw PCM, or live PCM iterator |
| `sample_rate` | none | Required for raw or live PCM; forbidden for encoded audio |
| `language` | auto | Source language code, such as `"en"` or `"es"` |
| `task` | `"transcribe"` | `"transcribe"` or English `"translate"` |
| `timestamps` | `"segment"` | `"none"`, `"segment"`, or `"word"` |
| `initial_prompt` | none | Persistent text context for every decode window |
| `condition_on_previous_text` | `True` | Carry bounded text context between windows |
| `clip_start_seconds` | `0.0` | Start offset for an encoded source |
| `clip_end_seconds` | source end | Exclusive end of the requested clip |
| `stream` | `False` | Return progressive snapshots instead of one awaited result |

Sampling and fallback controls belong in `settings`:

```python
result = await whisper.transcribe(
    audio=Path("lecture.flac"),
    language="en",
    initial_prompt="Kestrel, CUDA, Blackwell",
    settings={
        "temperature": 0.0,
        "max_tokens": 444,
        "best_of": 1,
    },
)
```

The optional quality thresholds are `compression_ratio_threshold`,
`logprob_threshold`, and `no_speech_threshold`. Kestrel retries only within its
bounded temperature/candidate policy and reports the selected attempt's
diagnostics.

## Progressive files

Passing `stream=True` returns a bounded `CapabilityStream`. Updates are current
snapshots, not an append-only queue; a slow consumer receives the newest state
without allowing progress events to grow memory without bound.

```python
stream = await whisper.transcribe(
    audio=Path("meeting.m4a"),
    timestamps="segment",
    stream=True,
)
async for update in stream:
    print(update.text)
result = await stream.result()
```

## Live PCM

An asynchronous iterator enables live transcription. Every chunk must be
non-empty, one-dimensional mono PCM represented by a NumPy array or CPU Torch
tensor. The sample rate is fixed for the session.

```python
async def microphone_chunks():
    while True:
        chunk = await microphone.read()
        if chunk is None:
            return
        yield chunk


stream = await whisper.transcribe(
    audio=microphone_chunks(),
    sample_rate=48_000,
    timestamps="segment",
    stream=True,
)
async for update in stream:
    render_partial_transcript(update.output)
final = await stream.result()
```

Live PCM does not accept clip ranges. The iterator is closed on success,
failure, or cancellation when it provides `aclose()`.

## Native execution

Kestrel owns the complete Whisper runtime. Prefill uses packed Kestrel kernels
inside fixed-shape CUDA graphs, and decode uses the shared generated-decode
binder over artifacts packaged with `kestrel-kernels`. The compiler source is
not imported by the installed runtime.
