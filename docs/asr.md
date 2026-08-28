# Speech-to-text models

Kestrel serves four speech-to-text checkpoints through the same model-bound
`transcribe` capability:

- `openai/whisper-large-v3-turbo`
- `Qwen/Qwen3-ASR-0.6B-hf`
- `Qwen/Qwen3-ASR-1.7B-hf`
- `nvidia/parakeet-tdt-0.6b-v3`

## Basic usage

```python
from pathlib import Path

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine

MODEL = "Qwen/Qwen3-ASR-0.6B-hf"

engine = await InferenceEngine.create(RuntimeConfig(model=MODEL))
model = engine.model(MODEL)
try:
    result = await model.transcribe(
        audio=Path("interview.mp3"),
        timestamps="word",
    )
    print(result.output["text"])
finally:
    await engine.shutdown()
```

`audio` may be an encoded path, encoded bytes, a bounded binary stream, raw
mono NumPy or CPU Torch PCM with `sample_rate`, or an asynchronous iterator of
raw PCM chunks. Files are decoded incrementally and may be clipped with
`clip_start_seconds` and `clip_end_seconds`.

Encoded files may be WAV/WAVEX, FLAC, MP3, Ogg Vorbis, Opus, M4A, MP4, MOV,
or WebM. Container channel layouts are downmixed to mono. Paths are decoded
incrementally; encoded bytes and streams are snapshotted into a bounded 64 MiB
buffer, beginning at the stream's current position. Inputs and live sessions
are limited to 24 hours.

All three model families support long recordings, progressive file
transcription, and live PCM:

```python
stream = await model.transcribe(
    audio=microphone_chunks(),
    sample_rate=48_000,
    timestamps="segment",
    stream=True,
)
async for update in stream:
    render(update.output)
final = await stream.result()
```

Progress updates are current snapshots. Their `provisional` field becomes
`False` on the final update. The bounded `CapabilityStream` replaces a pending
snapshot when its consumer falls behind, so progress reporting cannot grow an
unbounded event queue. Calling `result()` without iterating remains supported.

Every live chunk must be a non-empty, one-dimensional NumPy array or CPU Torch
tensor. The sample rate is fixed for the session, each chunk is limited to
1,048,576 samples, and live PCM does not accept clip ranges. An iterator that
provides `aclose()` is closed on success, failure, or cancellation.

## Model capabilities

| Capability | Whisper large-v3-turbo | Qwen3-ASR | Parakeet TDT |
| --- | --- | --- | --- |
| Language | automatic or forced | automatic or forced | automatic; code not reported |
| English translation | yes | no | no |
| Initial prompt | yes | yes | no |
| Segment timestamps | yes | yes | yes |
| Word timestamps | alignment | forced alignment | native token durations |
| Character timestamps | no | no | native token durations |
| Decoding | temperature fallback | temperature and top-p | deterministic |

## Whisper large-v3-turbo

Whisper accepts these prompt options:

| Argument | Default | Meaning |
| --- | --- | --- |
| `audio` | required | Encoded audio, raw PCM, or live PCM iterator |
| `sample_rate` | none | Required for raw or live PCM; forbidden for encoded audio |
| `language` | auto | Source language code, such as `"en"` or `"es"` |
| `task` | `"transcribe"` | `"transcribe"` or English `"translate"` |
| `timestamps` | `"segment"` | `"none"`, `"segment"`, or `"word"` |
| `initial_prompt` | none | Persistent text context for every decode window |
| `condition_on_previous_text` | `True` | Carry bounded text context between windows |
| `clip_start_seconds` | `0.0` | Start offset within the source |
| `clip_end_seconds` | source end | Exclusive end of the requested clip |
| `stream` | `False` | Return progressive snapshots |

Sampling and fallback controls belong in `settings`:

```python
result = await model.transcribe(
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

Optional quality thresholds are `compression_ratio_threshold`,
`logprob_threshold`, and `no_speech_threshold`. Kestrel retries only within its
bounded temperature/candidate policy and returns the selected attempt's
diagnostics. A one-shot raw PCM array is limited to 30 seconds; use a live PCM
iterator for longer raw input.

Whisper has no eager serving fallback. Creating the model without compatible
packed prefill, sampling, and generated-decode artifacts fails closed.

## Qwen3-ASR

Qwen accepts `language`, `initial_prompt`, and
`condition_on_previous_text`. Its `settings` may contain `max_tokens`,
`temperature`, and `top_p`. Word timestamps use the pinned
`Qwen/Qwen3-ForcedAligner-0.6B-hf` checkpoint.
The aligner supports Chinese, Cantonese, English, German, Spanish, French,
Italian, Portuguese, Russian, Korean, and Japanese; use segment or no
timestamps for Qwen's other transcription languages.

The forced aligner can also timestamp an existing transcript directly:

```python
ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
engine = await InferenceEngine.create(RuntimeConfig(model=ALIGNER))
try:
    result = await engine.model(ALIGNER).align(
        audio=Path("interview.mp3"),
        text="Existing transcript",
        language="en",
    )
finally:
    await engine.shutdown()
```

## Parakeet TDT

Parakeet accepts `timestamps="none"`, `"segment"`, `"word"`, or
`"character"`. It does not accept language forcing, prompting, translation,
temperature, or top-p. `settings.max_tokens` remains available as a decode
bound. The multilingual checkpoint chooses the language implicitly and returns
`language: null`; it does not expose a language-classification result.
Live Parakeet sessions produce bounded-context previews every two seconds after
four seconds of initial context. Completed 180-second blocks and the final
result use the same full-context path as file transcription, so previews may be
revised without changing committed transcription quality.

All implementations are inference-only. Qwen and Parakeet audio features stay
on the GPU after the input waveform is transferred. Whisper and Qwen decoding
use Kestrel's bundled generated-decode programs. Installed runtimes do not
import compiler source.
