"""Autoregressive Kestrel runtime for Qwen3-ASR transcription."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from kestrel.device import (
    empty_cache,
    make_event,
    make_stream,
    resolve_device,
    stream_context,
)
from kestrel.kv_cache import (
    KVMemoryPool,
    PageTable,
    PagedKVLayerSpec,
    allocate_paged_kv_layers,
)
from kestrel.runtime import ExecutionShape
from kestrel.runtime.decode_slot import DecodeSlot, create_decode_slot
from kestrel.runtime.paged_resources import decode_slot_rows
from kestrel.runtime.preprocessing import derive_preprocessing_workers
from kestrel.runtime.sampling import SamplingHooks
from kestrel.runtime.staging import AsyncPreprocessor
from kestrel.runtime.state import PreparedSequence
from kestrel.runtime.tokens import TextToken, Token
from kestrel.runtime.uncached_paged import UncachedPagedRuntime
from kestrel.utils import CpuGpuBuffer

from kestrel.models.asr.audio import MAX_SHORT_AUDIO_SECONDS, DecodedAudio, decode_audio
from kestrel.models.asr.contract import TranscriptionRequest

from .alignment import LoadedForcedAligner, align_transcript, load_forced_aligner
from .features import qwen3_asr_features
from .generated_decode import create_generated_decode
from .model import Qwen3AsrForConditionalGeneration
from .tokenizer import Qwen3AsrTokenizer
from .weights import QWEN3_ASR_MODELS, load_qwen3_asr


_PREFILL_SLOTS = 2
_DECODE_SLOTS = 2
_MAX_CHUNK_SECONDS = {"none": 1_200, "segment": 30, "word": 180}


@dataclass(frozen=True, slots=True)
class PreparedQwenAudio:
    request: TranscriptionRequest
    audio: DecodedAudio
    audio_tokens: int
    window_lengths: tuple[int, ...]


@dataclass(slots=True)
class QwenPrefillSlot:
    slot_id: int
    batch_indices: CpuGpuBuffer
    step_done_event: Any
    commit_done_event: Any

    @property
    def batch_idx(self) -> torch.Tensor:
        return self.batch_indices.gpu


class Qwen3AsrRuntime(UncachedPagedRuntime):
    """Reference audio prefill with compiler-generated batched text decode."""

    execution_shape = ExecutionShape.AUTOREGRESSIVE
    image_prefix_length = 0
    spec = None
    sampling_hooks = SamplingHooks()

    def __init__(
        self,
        cfg: Any,
        *,
        compute_stream: Any = None,
        kv_pool: KVMemoryPool,
        max_lora_rank: int | None = None,
        model: Qwen3AsrForConditionalGeneration | None = None,
        tokenizer: Qwen3AsrTokenizer | None = None,
        aligner: LoadedForcedAligner | None = None,
    ) -> None:
        del max_lora_rank
        self._model_name = getattr(cfg, "model", next(iter(QWEN3_ASR_MODELS)))
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
        if self.device.type != "cuda" or self.dtype is not torch.bfloat16:
            raise ValueError("Qwen3-ASR serving requires CUDA bfloat16")
        if kv_pool is None:
            raise TypeError("Qwen3AsrRuntime requires the engine-owned kv_pool")
        if kv_pool.device != self.device:
            raise ValueError("kv_pool and Qwen3-ASR must use the same device")

        if model is None or tokenizer is None:
            checkpoint = getattr(cfg, "model_path", None) or self._model_name
            loaded = load_qwen3_asr(
                checkpoint,
                revision=QWEN3_ASR_MODELS.get(self._model_name),
                device=self.device,
                dtype=self.dtype,
            )
            model, tokenizer = loaded.model, loaded.tokenizer
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._aligner = aligner

        self.decode_path = getattr(cfg, "decode_path", "auto")
        if self.decode_path == "native":
            raise ValueError("Qwen3-ASR serving requires generated decode")
        if self.decode_path not in {"auto", "generated"}:
            raise ValueError("decode_path must be 'auto' or 'generated'")
        self.max_batch_size = int(getattr(cfg, "max_batch_size", 1))
        if not 1 <= self.max_batch_size <= 8:
            raise ValueError("Qwen3-ASR max_batch_size must lie in [1, 8]")
        self.max_batch_slots = self.max_batch_size + 2
        text = self.model.config.text
        self.max_seq_length = text.max_position_embeddings
        self.vocab_size = text.vocab_size
        self.eos_token_ids = self.model.config.eos_token_ids

        self._compute_stream = compute_stream or make_stream(self.device)
        self._copy_stream = make_stream(self.device)
        self._kv_pool = kv_pool
        self.graph_capture_lock = threading.RLock()
        self.active_sequences: dict[int, Any] = {}
        self._prepared_audio: dict[int, PreparedQwenAudio] = {}

        requested_pages = int(
            getattr(cfg, "kv_cache_pages", text.max_position_embeddings + 1)
        )
        if requested_pages <= 1:
            raise ValueError("kv_cache_pages must leave room beyond reserved page zero")
        self._kv_cache_pages = requested_pages
        self.page_table = PageTable(
            n_pages=self._kv_cache_pages,
            page_size=1,
            max_batch_size=self.max_batch_slots,
            device=str(self.device),
            h2d_stream=self._compute_stream,
        )
        self._paged_kv = allocate_paged_kv_layers(
            layer_specs=(PagedKVLayerSpec(text.num_key_value_heads, text.head_dim),)
            * text.num_hidden_layers,
            page_table=self.page_table,
            pool=kv_pool,
            dtype=self.dtype,
        )

        positions = torch.arange(
            text.max_position_embeddings, device=self.device, dtype=torch.float32
        )
        frequencies = torch.outer(
            positions,
            self.model.model.language_model.inverse_frequency.float(),
        )
        rotary = torch.cat((frequencies, frequencies), dim=-1)
        self._rope_cosine = rotary.cos().contiguous()
        self._rope_sine = rotary.sin().contiguous()

        self._prefill_slots = tuple(
            QwenPrefillSlot(
                slot_id=index,
                batch_indices=CpuGpuBuffer(
                    self.max_batch_size,
                    dtype=torch.int64,
                    device=self.device,
                    pin_memory=True,
                    with_numpy=False,
                    zero=False,
                ),
                step_done_event=make_event(self.device),
                commit_done_event=make_event(self.device),
            )
            for index in range(_PREFILL_SLOTS)
        )
        self.prefill_slots: Sequence[QwenPrefillSlot] = self._prefill_slots
        self._free_prefill_slot_ids = set(range(_PREFILL_SLOTS))

        decode_rows = decode_slot_rows(self.max_batch_size)
        self.decode_slots: Sequence[DecodeSlot] = tuple(
            create_decode_slot(
                slot_id=index,
                device=self.device,
                dtype=self.dtype,
                max_batch_slots=decode_rows,
                kv_cache_pages=self._kv_cache_pages,
                vocab_size=self.vocab_size,
                hidden_dim=text.hidden_size,
                position_shape=(decode_rows, 1),
                compute_stream=self._compute_stream,
                copy_stream=self._copy_stream,
            )
            for index in range(_DECODE_SLOTS)
        )
        self.generated_decode = create_generated_decode(self)
        if self.generated_decode is None:
            raise RuntimeError("Qwen3-ASR requires bundled generated-decode programs")

        self._audio_preprocessor = AsyncPreprocessor(
            self._prepare_audio,
            workers=derive_preprocessing_workers(self.max_batch_size),
        )

    def _prepare_audio(self, request: object) -> PreparedQwenAudio:
        if not isinstance(request, TranscriptionRequest):
            raise TypeError("Qwen3-ASR encoder input must be a TranscriptionRequest")
        audio = decode_audio(
            request.audio,  # type: ignore[arg-type]
            sample_rate=request.sample_rate,
            clip_start_seconds=request.clip_start_seconds,
            clip_end_seconds=request.clip_end_seconds,
            max_duration_seconds=_MAX_CHUNK_SECONDS[request.timestamps],
        )
        audio_tokens, window_lengths = self.model.model.audio_tower.feature_layout(
            audio.waveform.size // 160
        )
        return PreparedQwenAudio(
            request,
            audio,
            audio_tokens,
            window_lengths,
        )

    def _batch_audio_features(
        self, audio_rows: Sequence[PreparedQwenAudio]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups: dict[int, list[int]] = {}
        for index, audio in enumerate(audio_rows):
            groups.setdefault(audio.audio.waveform.size, []).append(index)

        feature_rows: dict[int, torch.Tensor] = {}
        mask_rows: dict[int, torch.Tensor] = {}
        for indices in groups.values():
            waveforms = torch.from_numpy(
                np.stack([audio_rows[index].audio.waveform for index in indices])
            ).to(self.device)
            features, masks = qwen3_asr_features(waveforms)
            for group_index, batch_index in enumerate(indices):
                feature_rows[batch_index] = features[group_index]
                mask_rows[batch_index] = masks[group_index]

        completed_features = [feature_rows[index] for index in range(len(audio_rows))]
        completed_masks = [mask_rows[index] for index in range(len(audio_rows))]
        max_length = max(row.shape[-1] for row in completed_features)
        features = torch.stack(
            [
                torch.nn.functional.pad(row, (0, max_length - row.shape[-1]))
                for row in completed_features
            ]
        )
        masks = torch.stack(
            [
                torch.nn.functional.pad(row, (0, max_length - row.shape[-1]))
                for row in completed_masks
            ]
        )
        return features.to(self.dtype), masks

    def preprocess_encoder_input_async(self, encoder_input: object) -> Any:
        if isinstance(encoder_input, TranscriptionRequest):
            audio = encoder_input.audio
            short_pcm = (
                isinstance(audio, (np.ndarray, torch.Tensor))
                and encoder_input.sample_rate == 16_000
                and encoder_input.clip_start_seconds == 0
                and encoder_input.clip_end_seconds is None
                and audio.ndim == 1
                and int(audio.shape[0]) <= MAX_SHORT_AUDIO_SECONDS * 16_000
            )
            if short_pcm:
                # A completed future lets one already-decoded PCM burst reach
                # admission together without delaying an individual request.
                completed: Future[Any] = Future()
                try:
                    completed.set_result(self._prepare_audio(encoder_input))
                except Exception as exc:
                    completed.set_exception(exc)
                return completed
        return self._audio_preprocessor.submit(encoder_input)

    def preprocess_image_async(self, image: object) -> None:
        del image
        raise ValueError("Qwen3-ASR does not accept images")

    def acquire_prefill_slot(self, slot_id: int | None = None) -> QwenPrefillSlot:
        if slot_id is None:
            if not self._free_prefill_slot_ids:
                raise RuntimeError("Qwen3-ASR prefill slots are exhausted")
            selected = min(self._free_prefill_slot_ids)
        else:
            selected = int(slot_id)
        if selected not in self._free_prefill_slot_ids:
            raise RuntimeError(f"Qwen3-ASR prefill slot {selected} is unavailable")
        self._free_prefill_slot_ids.remove(selected)
        return self._prefill_slots[selected]

    def release_prefill_slot(self, slot: Any) -> None:
        if not isinstance(slot, QwenPrefillSlot):
            raise TypeError("Qwen3AsrRuntime received a foreign prefill slot")
        slot_id = int(slot.slot_id)
        if (
            slot_id not in range(len(self._prefill_slots))
            or self._prefill_slots[slot_id] is not slot
        ):
            raise ValueError("Qwen3AsrRuntime received a foreign prefill slot")
        if slot_id in self._free_prefill_slot_ids:
            raise RuntimeError(f"Qwen3-ASR prefill slot {slot_id} was released twice")
        self._free_prefill_slot_ids.add(slot_id)

    def prepare_sequence(
        self,
        prompt_tokens: Sequence[Token],
        *,
        image: np.ndarray | None = None,
        image_crops: Any | None = None,
        encoder_input: object | None = None,
        max_new_tokens: int | None = None,
        lora_slot: int = 0,
        image_hash: bytes | None = None,
        adapter_id: str | None = None,
    ) -> PreparedSequence:
        if image is not None or image_crops is not None or image_hash is not None:
            raise ValueError("Qwen3-ASR does not accept images")
        if lora_slot != 0 or adapter_id is not None:
            raise ValueError("Qwen3-ASR does not support adapters")
        if not isinstance(encoder_input, PreparedQwenAudio):
            raise TypeError("Qwen3-ASR prepare_sequence requires prepared audio")
        if (
            len(prompt_tokens) != 1
            or not isinstance(prompt_tokens[0], TextToken)
            or prompt_tokens[0].token_id != self.tokenizer.audio_token_id
        ):
            raise ValueError("Qwen3-ASR expects one audio prompt marker")
        if max_new_tokens is None:
            new_tokens = 4096
        elif (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("Qwen3-ASR max_new_tokens must be positive")
        else:
            new_tokens = max_new_tokens

        request = encoder_input.request
        prompt_ids = self.tokenizer.prompt_ids(
            encoder_input.audio_tokens,
            language=request.language,
            initial_prompt=request.initial_prompt,
        )
        target_length = len(prompt_ids) + new_tokens
        if target_length > self.max_seq_length:
            raise ValueError(
                "audio prompt plus settings.max_tokens exceeds the "
                "Qwen3-ASR context window"
            )
        prepared = self._prepare_uncached_sequence(
            tokens=[TextToken(token_id=token_id) for token_id in prompt_ids],
            target_length=target_length,
            image_length=0,
            lora_slot=0,
            adapter_id=None,
            image_hash=None,
        )
        row = int(prepared.state.batch_idx)
        try:
            self._prepared_audio[row] = encoder_input
        except BaseException:
            super().abort_prepared_sequence(prepared)
            raise
        return prepared

    def _copy_prefill_cache(
        self,
        row: int,
        cache: Any,
        batch_index: int,
        length: int,
    ) -> None:
        pages = torch.as_tensor(
            self.page_table.page_table_cpu[row][:length],
            device=self.device,
            dtype=torch.long,
        )
        for paged, (key, value) in zip(self._paged_kv, cache, strict=True):
            paged.k_cache[:, :, 0].index_copy_(
                0,
                pages,
                key[batch_index, :, :length].transpose(0, 1),
            )
            paged.v_cache[:, :, 0].index_copy_(
                0,
                pages,
                value[batch_index, :, :length].transpose(0, 1),
            )

    @staticmethod
    def _validate_empty_optional_batch(
        name: str,
        values: Sequence[object | None] | None,
        batch_size: int,
    ) -> None:
        if values is not None and (
            len(values) != batch_size or any(value is not None for value in values)
        ):
            raise ValueError(f"Qwen3-ASR does not support {name}")

    @torch.inference_mode()
    def launch_prepared_batch(
        self,
        prepared_sequences: Sequence[PreparedSequence],
        prefill_slot: Any,
        *,
        images: Sequence[np.ndarray | None] | None = None,
        image_crops_list: Sequence[Any] | None = None,
        encoder_inputs: Sequence[object | None] | None = None,
    ) -> torch.Tensor:
        batch_size = len(prepared_sequences)
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError("Qwen3-ASR prefill batch is outside runtime capacity")
        if not isinstance(prefill_slot, QwenPrefillSlot):
            raise TypeError("Qwen3AsrRuntime received a foreign prefill slot")
        slot_id = int(prefill_slot.slot_id)
        if (
            slot_id not in range(len(self._prefill_slots))
            or self._prefill_slots[slot_id] is not prefill_slot
        ):
            raise ValueError("Qwen3AsrRuntime received a foreign prefill slot")
        if slot_id in self._free_prefill_slot_ids:
            raise RuntimeError("Qwen3-ASR prefill slot must be acquired before launch")
        self._validate_empty_optional_batch("images", images, batch_size)
        self._validate_empty_optional_batch(
            "image_crops_list", image_crops_list, batch_size
        )
        if encoder_inputs is None or len(encoder_inputs) != batch_size:
            raise ValueError("encoder_inputs must match the Qwen3-ASR prefill batch")

        rows = [int(prepared.state.batch_idx) for prepared in prepared_sequences]
        if len(set(rows)) != batch_size:
            raise ValueError("Qwen3-ASR prefill batch contains a duplicate state row")
        self.page_table.commit_block_table(rows)
        token_rows = []
        retained_audio_rows = []
        prompt_lengths = []
        for prepared, supplied_audio, row in zip(
            prepared_sequences, encoder_inputs, rows, strict=True
        ):
            retained_audio = self._prepared_audio.get(row)
            if (
                not isinstance(retained_audio, PreparedQwenAudio)
                or supplied_audio is not retained_audio
            ):
                raise RuntimeError(
                    f"Qwen3-ASR prepared audio ownership mismatch for row {row}"
                )
            token_ids = [
                token.token_id
                for token in prepared.tokens_list
                if isinstance(token, TextToken)
            ]
            if len(token_ids) != len(prepared.tokens_list):
                raise TypeError("Qwen3-ASR prompt contains a non-text token")
            if prepared.state.length != len(token_ids):
                raise RuntimeError("Qwen3-ASR prepared prompt length drifted")
            token_rows.append(token_ids)
            retained_audio_rows.append(retained_audio)
            prompt_lengths.append(len(token_ids))

        with stream_context(self._compute_stream):
            max_prompt_length = max(prompt_lengths)
            input_ids = torch.full(
                (batch_size, max_prompt_length),
                self.model.config.pad_token_id,
                dtype=torch.long,
            )
            for index, token_ids in enumerate(token_rows):
                input_ids[index, : len(token_ids)] = torch.tensor(token_ids)
            input_ids = input_ids.to(self.device)

            features, feature_mask = self._batch_audio_features(retained_audio_rows)
            output = self.model.prefill(
                input_ids,
                features,
                feature_mask,
                window_lengths=tuple(
                    length
                    for audio in retained_audio_rows
                    for length in audio.window_lengths
                ),
                last_indices=torch.tensor(
                    prompt_lengths,
                    dtype=torch.long,
                    device=self.device,
                )
                - 1,
            )
            for index, (row, length) in enumerate(
                zip(rows, prompt_lengths, strict=True)
            ):
                self._copy_prefill_cache(row, output.cache, index, length)
            prefill_slot.batch_indices.cpu[:batch_size] = torch.tensor(
                rows, dtype=torch.int64
            )
            prefill_slot.batch_indices.copy_to_gpu(batch_size)
        return output.logits[:, 0].to(self.dtype)

    def finalize_prepared_sequence_after_prefill(
        self,
        prepared: PreparedSequence,
    ) -> None:
        row = int(prepared.state.batch_idx)
        if row not in self._prepared_audio:
            raise RuntimeError(f"Qwen3-ASR prepared audio row {row} is missing")
        super().finalize_prepared_sequence_after_prefill(prepared)
        self._prepared_audio.pop(row)

    @torch.inference_mode()
    def decode_with_slot(self, slot: DecodeSlot, batch_size: int) -> None:
        if batch_size == 0:
            return
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError("Qwen3-ASR decode batch is outside runtime capacity")
        slot_id = int(slot.slot_id)
        if (
            slot_id not in range(len(self.decode_slots))
            or self.decode_slots[slot_id] is not slot
        ):
            raise ValueError("Qwen3AsrRuntime received a foreign decode slot")
        with stream_context(self._compute_stream):
            self.generated_decode.run(slot, batch_size)

    def align_words(
        self,
        audio: DecodedAudio,
        text: str,
        language: str,
    ) -> Any:
        if self._aligner is None:
            self._aligner = load_forced_aligner(device=self.device, dtype=self.dtype)
        return align_transcript(
            self._aligner,
            audio.waveform,
            text,
            language,
            offset_seconds=audio.clip_start_seconds,
        )

    def _release_runtime_state(self, batch_idx: int) -> None:
        self._prepared_audio.pop(batch_idx, None)

    def shutdown(self) -> None:
        self._audio_preprocessor.shutdown()
        empty_cache(self.device)


__all__ = ["PreparedQwenAudio", "Qwen3AsrRuntime", "QwenPrefillSlot"]
