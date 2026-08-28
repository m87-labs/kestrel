"""Kestrel runtime for Parakeet TDT transcription."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from kestrel.device import empty_cache, resolve_device
from kestrel.runtime import ExecutionShape

from kestrel.models.asr.audio import AudioChunks, DecodedAudio
from kestrel.models.asr.contract import (
    Character,
    DecodeSettings,
    Segment,
    TranscriptionRequest,
    TranscriptionResult,
    Word,
)

from .contract import parse_request
from .decode_graph import _TdtBatchGraphDecoder
from .generated_decode import _TdtBatchGeneratedDecoder
from .features import parakeet_features
from .model import ParakeetTdt, TdtState
from .tokenizer import ParakeetTokenizer
from .weights import MODEL_ID, load_parakeet_tdt


def _timed_segments(
    words: tuple[Word, ...],
    *,
    include_words: bool,
    max_duration_seconds: float = 30,
) -> tuple[Segment, ...]:
    segments: list[Segment] = []
    pending: list[Word] = []
    for word in words:
        pending.append(word)
        if (
            word.text.rstrip().endswith((".", "?", "!"))
            or word.end - pending[0].start >= max_duration_seconds
        ):
            grouped = tuple(pending)
            segments.append(
                Segment(
                    " ".join(item.text for item in grouped),
                    grouped[0].start,
                    grouped[-1].end,
                    grouped if include_words else (),
                )
            )
            pending.clear()
    if pending:
        grouped = tuple(pending)
        segments.append(
            Segment(
                " ".join(item.text for item in grouped),
                grouped[0].start,
                grouped[-1].end,
                grouped if include_words else (),
            )
        )
    return tuple(segments)


def _encoder_frames(samples: int, factor: int) -> int:
    frames = samples // 160
    while factor > 1:
        frames = (frames + 1) // 2
        factor //= 2
    return frames


@dataclass(frozen=True, slots=True)
class _StreamState:
    decoder: TdtState
    token_ids: tuple[int, ...]
    durations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _StreamWindow:
    state: _StreamState | None
    start_sample: int
    sample_count: int | None
    duration_seconds: float


class ParakeetTdtRuntime:
    execution_shape = ExecutionShape.SINGLE_PASS
    batch_capacity = 8

    def __init__(
        self,
        cfg: Any,
        *,
        compute_stream: Any = None,
        kv_pool: Any = None,
        max_lora_rank: int | None = None,
        model: ParakeetTdt | None = None,
        tokenizer: ParakeetTokenizer | None = None,
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
            else getattr(cfg, "dtype", torch.float32)
        )
        self.compute_stream = compute_stream
        if model is None or tokenizer is None:
            checkpoint = getattr(cfg, "model_path", None) or self._model_name
            loaded = load_parakeet_tdt(checkpoint, device=self.device, dtype=self.dtype)
            model, tokenizer = loaded.model, loaded.tokenizer
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.decode_path = getattr(cfg, "decode_path", "auto")
        if self.decode_path not in {"auto", "native", "generated"}:
            raise ValueError("decode_path must be 'auto', 'native', or 'generated'")
        self._batch_decoder = None
        generated_supported = (
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and self.dtype == torch.bfloat16
        )
        if self.decode_path == "generated" and not generated_supported:
            raise RuntimeError(
                "Parakeet generated decode requires CUDA with BF16 weights"
            )
        if self.decode_path != "native" and generated_supported:
            stream = compute_stream or torch.cuda.current_stream(self.device)
            self._batch_decoder = _TdtBatchGeneratedDecoder.create(
                self.model,
                max_batch=self.batch_capacity,
                compute_stream=stream,
                required=self.decode_path == "generated",
            )
        if (
            self._batch_decoder is None
            and bool(getattr(cfg, "enable_cuda_graphs", True))
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            stream = compute_stream or torch.cuda.current_stream(self.device)
            self._batch_decoder = _TdtBatchGraphDecoder(
                self.model,
                max_batch=self.batch_capacity,
                compute_stream=stream,
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    def tasks(self) -> tuple[str, ...]:
        return ("transcribe",)

    def preprocess_image_async(self, image: object) -> None:
        del image
        raise ValueError("Parakeet does not accept images")

    def _batch_audio_features(
        self,
        rows: Sequence[tuple[int, DecodedAudio]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups: dict[int, list[int]] = {}
        for group_index, (_request_index, audio) in enumerate(rows):
            groups.setdefault(audio.waveform.size, []).append(group_index)

        feature_rows: dict[int, torch.Tensor] = {}
        mask_rows: dict[int, torch.Tensor] = {}
        for indices in groups.values():
            waveforms = torch.from_numpy(
                np.stack([rows[index][1].waveform for index in indices])
            ).to(self.device)
            features, masks = parakeet_features(waveforms)
            for batch_index, row_index in enumerate(indices):
                feature_rows[row_index] = features[batch_index]
                mask_rows[row_index] = masks[batch_index]

        completed_features = [feature_rows[index] for index in range(len(rows))]
        completed_masks = [mask_rows[index] for index in range(len(rows))]
        width = max(row.shape[0] for row in completed_features)
        features = torch.stack(
            [F.pad(row, (0, 0, 0, width - row.shape[0])) for row in completed_features]
        )
        masks = torch.stack(
            [F.pad(row, (0, width - row.shape[0])) for row in completed_masks]
        )
        return features.to(self.dtype), masks

    def _append_chunk_result(
        self,
        request: TranscriptionRequest,
        audio: DecodedAudio,
        token_ids: list[int],
        durations: list[int],
        text_parts: list[str],
        segments: list[Segment],
        *,
        frame_seconds: float,
    ) -> None:
        text = self.tokenizer.decode(token_ids)
        text_parts.append(text.strip())
        if request.timestamps == "none" or not text:
            return
        clip_end = audio.clip_start_seconds + audio.duration_seconds
        if request.timestamps == "character":
            characters = tuple(
                Character(
                    item.text,
                    min(clip_end, item.start + audio.clip_start_seconds),
                    min(clip_end, item.end + audio.clip_start_seconds),
                )
                for item in self.tokenizer.characters(
                    token_ids, durations, frame_seconds
                )
            )
            if characters:
                segments.append(
                    Segment(
                        text,
                        characters[0].start,
                        characters[-1].end,
                        characters=characters,
                    )
                )
            else:
                segments.append(Segment(text, audio.clip_start_seconds, clip_end))
            return
        timed_words = tuple(
            Word(
                word.text,
                min(clip_end, word.start + audio.clip_start_seconds),
                min(clip_end, word.end + audio.clip_start_seconds),
            )
            for word in self.tokenizer.words(token_ids, durations, frame_seconds)
        )
        if timed_words:
            segments.extend(
                _timed_segments(
                    timed_words,
                    include_words=request.timestamps == "word",
                )
            )
        else:
            segments.append(Segment(text, audio.clip_start_seconds, clip_end))

    def _run_stream_group(
        self,
        rows: Sequence[tuple[int, DecodedAudio]],
        windows: Sequence[_StreamWindow],
        requests: Sequence[TranscriptionRequest],
        features: torch.Tensor,
        mask: torch.Tensor,
        *,
        max_tokens: int,
    ) -> tuple[dict[str, object], ...]:
        encoded, valid = self.model.encode(features, mask)
        values = []
        factor = self.model.config.encoder.subsampling_factor
        for row, window, request, row_encoded, row_valid in zip(
            rows, windows, requests, encoded, valid, strict=True
        ):
            _index, audio = row
            previous = window.state
            generated = self.model.generate_encoded(
                row_encoded[None],
                row_valid[None],
                max_tokens=max_tokens,
                start_frame=_encoder_frames(window.start_sample, factor),
                frame_count=(
                    None
                    if window.sample_count is None
                    else _encoder_frames(window.sample_count, factor)
                ),
                state=None if previous is None else previous.decoder,
            )
            length = int(generated.lengths[0])
            token_ids = generated.sequences[0, 1:length].tolist()
            durations = generated.durations[0, 1:length].tolist()
            if previous is None:
                all_token_ids = (self.tokenizer.blank_token_id, *token_ids)
                all_durations = (0, *durations)
            else:
                all_token_ids = (*previous.token_ids, *token_ids)
                all_durations = (*previous.durations, *durations)
            if generated.state is None:
                raise RuntimeError("stateful TDT decoding returned no state")
            state = _StreamState(generated.state, all_token_ids, all_durations)
            text_parts: list[str] = []
            segments: list[Segment] = []
            logical_audio = DecodedAudio(
                audio.waveform,
                window.duration_seconds,
                window.duration_seconds,
                0.0,
            )
            self._append_chunk_result(
                request,
                logical_audio,
                list(all_token_ids),
                list(all_durations),
                text_parts,
                segments,
                frame_seconds=generated.encoder_frame_seconds,
            )
            value = TranscriptionResult(
                text=" ".join(text_parts),
                language=None,
                duration_seconds=window.duration_seconds,
                source_duration_seconds=window.duration_seconds,
                clip_start_seconds=0.0,
                segments=tuple(segments),
            ).as_dict()
            value["_stream_state"] = state
            values.append(value)
        return tuple(values)

    @torch.inference_mode()
    def forward(
        self, task: str, inputs: Sequence[Any]
    ) -> tuple[dict[str, object] | Exception, ...]:
        if task != "transcribe":
            raise ValueError("ParakeetTdtRuntime only accepts transcribe requests")

        parsed: list[
            tuple[TranscriptionRequest, DecodeSettings, _StreamWindow | None] | None
        ] = [None] * len(inputs)
        results: list[dict[str, object] | Exception | None] = [None] * len(inputs)
        for index, value in enumerate(inputs):
            try:
                if not isinstance(value, Mapping):
                    raise ValueError("Parakeet transcribe inputs must be mappings")
                prompt = dict(value)
                stream_window = prompt.pop("_stream_window", None)
                if stream_window is not None and not isinstance(
                    stream_window, _StreamWindow
                ):
                    raise TypeError("invalid internal Parakeet stream window")
                request, settings = parse_request(prompt, prompt.pop("settings", None))
                if request.stream:
                    raise ValueError(
                        "Parakeet stream=True must run through the model capability handle"
                    )
                parsed[index] = (request, settings, stream_window)
            except Exception as exc:
                results[index] = exc

        with ExitStack() as stack:
            sources: list[AudioChunks | None] = [None] * len(inputs)
            for index, item in enumerate(parsed):
                if item is None:
                    continue
                request, _settings, _stream_window = item
                try:
                    sources[index] = stack.enter_context(
                        AudioChunks(
                            request.audio,  # type: ignore[arg-type]
                            sample_rate=request.sample_rate,
                            clip_start_seconds=request.clip_start_seconds,
                            clip_end_seconds=request.clip_end_seconds,
                            target_sample_rate=16_000,
                            max_duration_seconds=24 * 60 * 60,
                        )
                    )
                except Exception as exc:
                    results[index] = exc
            iterators = [
                iter(source.chunks(180)) if source is not None else None
                for source in sources
            ]
            text_parts: list[list[str]] = [[] for _ in inputs]
            segments: list[list[Segment]] = [[] for _ in inputs]

            while any(iterator is not None for iterator in iterators):
                chunks = []
                for index, iterator in enumerate(iterators):
                    if iterator is None:
                        continue
                    try:
                        chunks.append((index, next(iterator)))
                    except StopIteration:
                        iterators[index] = None
                    except Exception as exc:
                        iterators[index] = None
                        results[index] = exc
                groups: dict[tuple[int, bool], list[tuple[int, DecodedAudio]]] = {}
                for index, audio in chunks:
                    item = parsed[index]
                    assert item is not None
                    groups.setdefault(
                        (item[1].max_tokens, item[2] is not None), []
                    ).append((index, audio))
                for (max_tokens, is_stream), group in groups.items():
                    valid_group = []
                    for index, audio in group:
                        if audio.waveform.size < 320:
                            iterators[index] = None
                            results[index] = ValueError(
                                "Parakeet audio is too short to normalize"
                            )
                        else:
                            valid_group.append((index, audio))
                    if not valid_group:
                        continue
                    features, mask = self._batch_audio_features(valid_group)
                    if is_stream:
                        stream_windows = []
                        stream_requests = []
                        for index, _audio in valid_group:
                            item = parsed[index]
                            assert item is not None and item[2] is not None
                            stream_requests.append(item[0])
                            stream_windows.append(item[2])
                        stream_results = self._run_stream_group(
                            valid_group,
                            stream_windows,
                            stream_requests,
                            features,
                            mask,
                            max_tokens=max_tokens,
                        )
                        for (index, _audio), value in zip(
                            valid_group, stream_results, strict=True
                        ):
                            results[index] = value
                        continue
                    if (
                        self._batch_decoder is None
                        or features.shape[0] < self._batch_decoder.minimum_batch
                    ):
                        output = self.model.generate(
                            features,
                            mask,
                            max_tokens=max_tokens,
                        )
                    else:
                        encoded, valid = self.model.encode(features, mask)
                        output = self._batch_decoder.generate(
                            encoded,
                            valid,
                            max_tokens=max_tokens,
                        )
                    packed = torch.cat(
                        (
                            output.lengths[:, None],
                            output.sequences,
                            output.durations,
                        ),
                        dim=1,
                    ).tolist()
                    for (index, audio), row in zip(valid_group, packed, strict=True):
                        length = row[0]
                        token_ids = row[1 : 1 + length]
                        durations = row[1 + output.sequences.shape[1] :][:length]
                        item = parsed[index]
                        assert item is not None
                        self._append_chunk_result(
                            item[0],
                            audio,
                            token_ids,
                            durations,
                            text_parts[index],
                            segments[index],
                            frame_seconds=output.encoder_frame_seconds,
                        )
            for index, source in enumerate(sources):
                if source is not None and results[index] is None:
                    results[index] = TranscriptionResult(
                        text=" ".join(part for part in text_parts[index] if part),
                        language=None,
                        duration_seconds=source.duration_seconds,
                        source_duration_seconds=source.source_duration_seconds,
                        clip_start_seconds=source.clip_start_seconds,
                        segments=tuple(segments[index]),
                    ).as_dict()

        finalized = []
        for result in results:
            assert result is not None
            finalized.append(result)
        return tuple(finalized)

    def shutdown(self) -> None:
        empty_cache(self.device)


__all__ = ["ParakeetTdtRuntime"]
