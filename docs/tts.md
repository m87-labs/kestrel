# Speech synthesis models

Kestrel supports three checkpoints through `model().synthesize(...)`:

- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `hexgrad/Kokoro-82M`

All three checkpoints support streaming through `model().synthesize(...)`.
Qwen3-TTS accepts text, voice, and language selection; its 1.7B checkpoint also accepts
`instructions`. Kokoro accepts a phoneme string, `speed`, and comma-separated
voice blends. Each returns mono
24 kHz floating-point PCM in `output["audio"]` and the sample rate in
`output["sample_rate"]`.
Qwen3-TTS serving requires a CUDA GPU and BF16; Kokoro supports CPU and CUDA.
To run Kokoro on CPU even when a GPU is present, set
`RuntimeConfig(model=model_id, device="cpu")`.

```python
from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
engine = await InferenceEngine.create(RuntimeConfig(model=model_id))
try:
    result = await engine.model(model_id).synthesize(
        text="Good morning!",
        voice="Ryan",
        language="English",
    )
    pcm = result.output["audio"]
    sample_rate = result.output["sample_rate"]
finally:
    await engine.shutdown()
```

For incremental audio, pass `stream=True` inside the engine's lifetime. Every
update contains a nonempty audio payload. `stream.result()` returns the complete
waveform; call `stream.aclose()` if the consumer stops early.

```python
engine = await InferenceEngine.create(RuntimeConfig(model=model_id))
try:
    stream = await engine.model(model_id).synthesize(
        text="A longer passage to read aloud.",
        voice="Ryan",
        stream=True,
    )
    async for update in stream:
        play(update.output["audio"], update.output["sample_rate"])
    result = await stream.result()
finally:
    await engine.shutdown()
```

Kokoro uses the same call with `model_id = "hexgrad/Kokoro-82M"`, but takes
`phonemes=` instead of `text=`. Pass a Unicode string using the Kokoro
checkpoint's phoneme vocabulary, for example `phonemes="həlˈO"` with the
default `af_heart` voice. Kestrel segments long phoneme strings and streams
completed segments. Its optional `speed` must be positive. Kokoro does not
convert text to phonemes; callers can use a frontend such as
[Misaki](https://github.com/hexgrad/misaki) before calling Kestrel. Misaki and
spaCy are not Kestrel dependencies.

Qwen3-TTS CustomVoice supports both model sizes, automatic or explicit
language selection, and streamed or complete output. Its `settings` mapping
supports talker and code-predictor sampling controls; Kokoro does not use
sampling settings. The 0.6B checkpoint can occasionally repeat or stutter on
sampled speech. Treat synthesized speech as generated content and review it
before publishing or using it in a consequential setting.
