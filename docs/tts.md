# Text-to-speech models

Kestrel supports three checkpoints through `model().synthesize(...)`:

- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `hexgrad/Kokoro-82M`

The models share a text, voice, and streaming interface. Qwen3-TTS accepts
language selection and the 1.7B checkpoint also accepts `instructions`.
Kokoro accepts `speed` and comma-separated voice blends. Each returns mono
24 kHz floating-point PCM in `output["audio"]` and the sample rate in
`output["sample_rate"]`.
Qwen3-TTS serving requires a CUDA GPU and BF16; Kokoro supports CPU and CUDA.

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

Kokoro uses the same calls with `model_id = "hexgrad/Kokoro-82M"`; its default
voice is `af_heart`, its default language is `en-us`, and its optional `speed`
must be positive. It segments long text and streams completed segments.
For English text, install the optional dependencies with
`pip install 'kestrel[kokoro]'`, then run
`python -m spacy download en_core_web_sm`. Misaki currently supports Python
3.10–3.12. Japanese and Mandarin require Misaki's corresponding language
extras and, for Japanese, UniDic data.

Qwen3-TTS CustomVoice supports both model sizes, automatic or explicit
language selection, and streamed or complete output. Its `settings` mapping
supports talker and code-predictor sampling controls; Kokoro does not use
sampling settings. The 0.6B checkpoint can occasionally repeat or stutter on
sampled speech. Treat synthesized speech as generated content and review it
before publishing or using it in a consequential setting.
