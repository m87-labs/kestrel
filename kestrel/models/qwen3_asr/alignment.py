"""Inference-only Qwen3 forced alignment.

The aligner is a separate Qwen checkpoint: audio and transcript are consumed in
one causal prefill and a small classifier predicts an 80 ms timestamp bin at
each ``<timestamp>`` token.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from kestrel.device import empty_cache, resolve_device
from kestrel.runtime import ExecutionShape
from tokenizers import Tokenizer
from torch import Tensor, nn

from kestrel.models.asr.audio import decode_audio
from kestrel.models.asr.checkpoint import resolve_checkpoint
from kestrel.models.asr.contract import TranscriptionRequest, Word

from .config import AudioEncoderConfig, ForcedAlignerConfig
from .features import qwen3_asr_features
from .model import AudioEncoder, TextDecoder
from .tokenizer import language_code, resolve_language


MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
REVISION = "c07281df297b9905d24a508279258cccf987a064"
_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
)
_STATE_PREFIXES = (
    ("model.audio_tower.", "thinker.audio_tower."),
    ("model.multi_modal_projector.linear_1.", "thinker.audio_tower.proj1."),
    ("model.multi_modal_projector.linear_2.", "thinker.audio_tower.proj2."),
    ("model.language_model.", "thinker.model."),
    ("score.", "thinker.lm_head."),
)


class AlignerAudioEncoder(AudioEncoder):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__(config)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def forward(
        self,
        features: Tensor,
        feature_mask: Tensor,
        window_lengths: tuple[int, ...] | None = None,
    ) -> Tensor:
        encoded = super().forward(features, feature_mask, window_lengths)
        return self.proj2(F.gelu(self.proj1(encoded)))


class ForcedAlignerThinker(nn.Module):
    def __init__(self, config: ForcedAlignerConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_tower = AlignerAudioEncoder(config.audio)
        self.model = TextDecoder(config.text)
        self.lm_head = nn.Linear(
            config.text.hidden_size, config.classify_num, bias=False
        )

    def forward(
        self,
        input_ids: Tensor,
        features: Tensor,
        feature_mask: Tensor,
        window_lengths: tuple[int, ...] | None = None,
    ) -> Tensor:
        embeddings = self.model.embed_tokens(input_ids)
        audio = self.audio_tower(features, feature_mask, window_lengths).to(
            embeddings.dtype
        )
        placeholder = input_ids == self.config.audio_token_id
        if int(placeholder.sum()) != audio.shape[0]:
            raise ValueError("forced-aligner prompt and encoded audio lengths disagree")
        embeddings = embeddings.masked_scatter(placeholder.unsqueeze(-1), audio)
        hidden, _ = self.model(input_embeddings=embeddings, use_cache=False)
        return self.lm_head(hidden).float()


class Qwen3ForcedAlignerModel(nn.Module):
    def __init__(self, config: ForcedAlignerConfig) -> None:
        super().__init__()
        self.config = config
        self.thinker = ForcedAlignerThinker(config)

    def reset_nonpersistent_buffers(self) -> None:
        self.thinker.audio_tower.reset_nonpersistent_buffers()
        self.thinker.model.reset_nonpersistent_buffers()


class ForcedAlignerTokenizer:
    def __init__(self, root: Path, config: ForcedAlignerConfig) -> None:
        self.backend = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.audio_start = self._id("<|audio_start|>")
        self.audio_pad = self._id("<|audio_pad|>")
        self.audio_end = self._id("<|audio_end|>")
        self.timestamp = self._id("<timestamp>")
        if (
            self.audio_pad != config.audio_token_id
            or self.timestamp != config.timestamp_token_id
        ):
            raise ValueError(
                "forced-aligner tokenizer and model special tokens disagree"
            )

    def _id(self, token: str) -> int:
        value = self.backend.token_to_id(token)
        if value is None:
            raise ValueError(f"forced-aligner tokenizer is missing {token!r}")
        return value

    def prompt_ids(self, words: list[str], audio_tokens: int) -> list[int]:
        ids = [self.audio_start, *([self.audio_pad] * audio_tokens), self.audio_end]
        for word in words:
            ids.extend(self.backend.encode(word, add_special_tokens=False).ids)
            ids.extend((self.timestamp, self.timestamp))
        return ids


@dataclass(frozen=True, slots=True)
class LoadedForcedAligner:
    model: Qwen3ForcedAlignerModel
    tokenizer: ForcedAlignerTokenizer


def load_forced_aligner(
    checkpoint: str | Path = MODEL_ID,
    *,
    revision: str = REVISION,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
) -> LoadedForcedAligner:
    root = resolve_checkpoint(
        checkpoint,
        revision=revision,
        filenames=_FILES,
        local_files_only=local_files_only,
    )
    config = ForcedAlignerConfig.from_json_file(root / "config.json")
    with torch.device("meta"):
        model = Qwen3ForcedAlignerModel(config)
    from safetensors.torch import load_file

    state = load_file(str(root / "model.safetensors"), device="cpu")
    renamed: dict[str, Tensor] = {}
    for name, value in state.items():
        for source, target in _STATE_PREFIXES:
            if name.startswith(source):
                renamed[target + name.removeprefix(source)] = value
                break
        else:
            raise ValueError(f"unsupported forced-aligner checkpoint tensor {name!r}")
    model.load_state_dict(renamed, strict=True, assign=True)
    model.reset_nonpersistent_buffers()
    model.to(device=device, dtype=dtype).eval()
    return LoadedForcedAligner(model, ForcedAlignerTokenizer(root, config))


def _kept(character: str) -> bool:
    return character == "'" or unicodedata.category(character)[0] in ("L", "N")


def _cjk(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2CEAF
    )


def alignment_words(text: str, language: str) -> list[str]:
    """Split transcript into stable alignment units without language packages."""

    pieces: list[str] = []
    buffered: list[str] = []

    def flush() -> None:
        if buffered:
            pieces.append("".join(buffered))
            buffered.clear()

    character_granularity = language.lower() in ("chinese", "cantonese", "japanese")
    for character in text:
        if not _kept(character):
            flush()
        elif _cjk(character) or (character_granularity and not character.isascii()):
            flush()
            pieces.append(character)
        else:
            buffered.append(character)
    flush()
    return pieces


def _repair_monotonic(values: list[int]) -> list[int]:
    if not values:
        return []
    length = [1] * len(values)
    parent = [-1] * len(values)
    for right in range(1, len(values)):
        for left in range(right):
            if values[left] <= values[right] and length[left] + 1 > length[right]:
                length[right] = length[left] + 1
                parent[right] = left
    keep = [False] * len(values)
    index = max(range(len(values)), key=length.__getitem__)
    while index >= 0:
        keep[index] = True
        index = parent[index]
    result = values.copy()
    start = 0
    while start < len(values):
        if keep[start]:
            start += 1
            continue
        end = start
        while end < len(values) and not keep[end]:
            end += 1
        left = result[start - 1] if start else None
        right = result[end] if end < len(values) else None
        count = end - start
        for offset in range(count):
            if left is None:
                result[start + offset] = right  # type: ignore[assignment]
            elif right is None:
                result[start + offset] = left
            elif count <= 2:
                result[start + offset] = left if offset + 1 <= count - offset else right
            else:
                result[start + offset] = int(
                    left + (right - left) * (offset + 1) / (count + 1)
                )
        start = end
    return result


@torch.inference_mode()
def align_transcript(
    loaded: LoadedForcedAligner,
    waveform: np.ndarray | Tensor,
    text: str,
    language: str,
    *,
    offset_seconds: float = 0.0,
) -> tuple[Word, ...]:
    config = loaded.model.config
    duration = len(waveform) / 16_000
    if duration > 300:
        raise ValueError("Qwen3 forced alignment supports audio up to 300 seconds")
    canonical = next(
        (
            item
            for item in config.supported_languages
            if item.lower() == language.lower()
        ),
        None,
    )
    if canonical is None:
        raise ValueError(
            f"Qwen3 forced alignment does not support language {language!r}"
        )
    words = alignment_words(text, canonical)
    if not words:
        return ()
    audio_tokens, window_lengths = loaded.model.thinker.audio_tower.feature_layout(
        len(waveform) // 160
    )
    input_ids = loaded.tokenizer.prompt_ids(words, audio_tokens)
    if len(input_ids) > config.text.max_position_embeddings:
        raise ValueError(
            "audio and transcript exceed the forced aligner's context window"
        )

    device = next(loaded.model.parameters()).device
    dtype = next(loaded.model.parameters()).dtype
    waveform = torch.as_tensor(waveform, dtype=torch.float32, device=device)
    features, mask = qwen3_asr_features(waveform)
    logits = loaded.model.thinker(
        torch.tensor([input_ids], dtype=torch.long, device=device),
        features.to(dtype),
        mask,
        window_lengths,
    )
    positions = torch.tensor(input_ids, device=device) == config.timestamp_token_id
    bins = logits[0, positions].argmax(-1).cpu().tolist()
    bins = _repair_monotonic(bins)
    scale = config.timestamp_segment_ms / 1000.0
    clip_end = offset_seconds + duration
    return tuple(
        Word(
            word,
            round(min(clip_end, offset_seconds + bins[index * 2] * scale), 3),
            round(min(clip_end, offset_seconds + bins[index * 2 + 1] * scale), 3),
        )
        for index, word in enumerate(words)
    )


class Qwen3ForcedAlignerRuntime:
    """Single-pass runtime for the standalone forced-aligner checkpoint."""

    execution_shape = ExecutionShape.SINGLE_PASS
    batch_capacity = 1

    def __init__(
        self,
        cfg: object,
        *,
        compute_stream: object | None = None,
        kv_pool: object | None = None,
        max_lora_rank: int | None = None,
        aligner: LoadedForcedAligner | None = None,
    ) -> None:
        del kv_pool, max_lora_rank
        self._model_name = getattr(cfg, "model", MODEL_ID)
        self.device = resolve_device(
            cfg.resolved_device()
            if hasattr(cfg, "resolved_device")
            else getattr(cfg, "device", "cuda")
        )
        self.dtype = (
            cfg.resolved_dtype()
            if hasattr(cfg, "resolved_dtype")
            else getattr(cfg, "dtype", torch.bfloat16)
        )
        self.compute_stream = compute_stream
        if aligner is None:
            aligner = load_forced_aligner(
                getattr(cfg, "model_path", None) or self._model_name,
                revision=REVISION,
                device=self.device,
                dtype=self.dtype,
            )
        self.aligner = aligner

    @property
    def model_name(self) -> str:
        return self._model_name

    def tasks(self) -> tuple[str, ...]:
        return ("align",)

    def preprocess_image_async(self, image: object) -> None:
        del image
        raise ValueError("Qwen3 forced alignment does not accept images")

    @torch.inference_mode()
    def forward(
        self,
        task: str,
        inputs: Sequence[object],
    ) -> tuple[dict[str, object], ...]:
        if task != "align":
            raise ValueError("Qwen3ForcedAlignerRuntime only accepts align requests")
        outputs = []
        for value in inputs:
            if not isinstance(value, Mapping):
                raise TypeError("align inputs must be a mapping")
            if any(not isinstance(name, str) for name in value):
                raise TypeError("align field names must be strings")
            unknown = sorted(
                set(value)
                - {
                    "audio",
                    "sample_rate",
                    "text",
                    "language",
                    "clip_start_seconds",
                    "clip_end_seconds",
                }
            )
            if unknown:
                raise ValueError(f"Unsupported align option(s): {', '.join(unknown)}")
            if "audio" not in value:
                raise ValueError("audio must be provided for alignment")
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                raise TypeError("text must be a non-empty string")
            request = TranscriptionRequest.from_prompt(
                {
                    name: value[name]
                    for name in (
                        "audio",
                        "sample_rate",
                        "language",
                        "clip_start_seconds",
                        "clip_end_seconds",
                    )
                    if name in value
                }
            )
            if request.language is None:
                raise ValueError("language must be provided for alignment")
            language = resolve_language(request.language)
            assert language is not None
            audio = decode_audio(
                request.audio,  # type: ignore[arg-type]
                sample_rate=request.sample_rate,
                clip_start_seconds=request.clip_start_seconds,
                clip_end_seconds=request.clip_end_seconds,
                max_duration_seconds=300,
            )
            words = align_transcript(
                self.aligner,
                audio.waveform,
                text.strip(),
                language,
                offset_seconds=audio.clip_start_seconds,
            )
            outputs.append(
                {
                    "text": text.strip(),
                    "language": language_code(language),
                    "duration_seconds": audio.duration_seconds,
                    "source_duration_seconds": audio.source_duration_seconds,
                    "clip_start_seconds": audio.clip_start_seconds,
                    "clip_end_seconds": (
                        audio.clip_start_seconds + audio.duration_seconds
                    ),
                    "words": [word.as_dict() for word in words],
                }
            )
        return tuple(outputs)

    def shutdown(self) -> None:
        empty_cache(self.device)


__all__ = [
    "LoadedForcedAligner",
    "MODEL_ID",
    "Qwen3ForcedAlignerRuntime",
    "Qwen3ForcedAlignerModel",
    "REVISION",
    "align_transcript",
    "alignment_words",
    "load_forced_aligner",
]
