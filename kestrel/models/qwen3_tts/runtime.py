"""Autoregressive Kestrel runtime for Qwen3-TTS CustomVoice synthesis."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from kestrel_kernels.sampling import apply_token_mask_and_repetition_

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
from kestrel.runtime.paged_resources import (
    bound_kv_cache_pages,
    decode_slot_rows,
)
from kestrel.runtime.preprocessing import derive_preprocessing_workers
from kestrel.runtime.sampling import SamplingHooks
from kestrel.runtime.staging import AsyncPreprocessor
from kestrel.runtime.state import PreparedSequence
from kestrel.runtime.tokens import TextToken, Token
from kestrel.runtime.uncached_paged import UncachedPagedRuntime
from kestrel.utils import CpuGpuBuffer

from .codec import IncrementalCodecState
from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPO_ID,
    NUM_CODE_GROUPS,
    SAMPLES_PER_FRAME,
)
from .contract import DEFAULT_SAMPLING, CustomVoiceRequest
from .codec_replay import TorchCodec
from .generated_decode import (
    TALKER_GENERATED_PREFILL_LENGTH,
    create_generated_decode,
    create_predictor_generated_decode,
    create_talker_generated_prefill,
)
from .input_layout import TalkerTokenLayout, build_talker_token_layout
from .loader import LoadedQwen3TTS, load_qwen3_tts
from .skill import Qwen3TTSSynthesisState


_PREFILL_SLOTS = 2
_DECODE_SLOTS = 2
_PROMPT_MARKER_ID = 0
_PREDICTOR_TOP_K = DEFAULT_SAMPLING.subtalker_top_k
_PREDICTOR_TEMPERATURE = DEFAULT_SAMPLING.subtalker_temperature
_PREDICTOR_TOP_P = DEFAULT_SAMPLING.subtalker_top_p
# Start with 80 ms of audible reserve after suppressing the fixed bootstrap
# frame, refill it immediately, then amortize warm codec work. H100 at 10 RPS:
# (2, 4, 8, 12) caused 544 underruns; (2, 2, 8, 12) caused 38.
# Tried 2/2/8/12: H100 1.7B p95 64.2→63.4 ms, underruns 1→10; keeping 3 first.
_CODEC_CHUNKS = (3, 2, 8, 12)
_MAX_CODEC_CHUNK = max(_CODEC_CHUNKS)


def _uses_generated_prefill(lengths: Sequence[int]) -> bool:
    return all(
        TALKER_GENERATED_PREFILL_LENGTH - 1 <= length <= TALKER_GENERATED_PREFILL_LENGTH
        for length in lengths
    )


def _validate_generation_budget(layout: TalkerTokenLayout, max_new_tokens: int) -> None:
    continuation = len(layout.continuation_text_token_ids)
    if continuation and max_new_tokens <= continuation:
        raise ValueError(
            "Qwen3-TTS settings.max_tokens is too small to consume all streaming text"
        )


def _rotary_tables(
    positions: int,
    head_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(
        torch.arange(positions, device=device, dtype=torch.float32),
        frequencies,
    )
    rotary = torch.cat((angles, angles), dim=-1)
    return rotary.cos().contiguous(), rotary.sin().contiguous()


@dataclass(frozen=True, slots=True)
class PreparedSynthesis:
    request: CustomVoiceRequest
    layout: TalkerTokenLayout


@dataclass(slots=True, eq=False)
class Qwen3TTSPrefillSlot:
    pool_index: int
    slot_id: int
    compute_stream: Any
    batch_indices: CpuGpuBuffer
    packed_batch_indices: CpuGpuBuffer
    step_done_event: Any
    commit_done_event: Any
    scratch: dict[str, torch.Tensor]

    @property
    def batch_idx(self) -> torch.Tensor:
        return self.batch_indices.gpu


@dataclass(slots=True)
class _CodecLane:
    row: int
    decoder_state: IncrementalCodecState
    suppress_bootstrap: bool
    buffered_frames: int = 0
    chunk_index: int = 0
    decoded_frames: int = 0


@dataclass(slots=True)
class _CodecBuffer:
    pcm_cpu: torch.Tensor
    copy_done: Any


@dataclass(slots=True)
class _CodecJob:
    buffer: _CodecBuffer
    states: tuple[Qwen3TTSSynthesisState, ...]
    frame_count: int
    skip_samples: tuple[int, ...]


class Qwen3TTSRuntime(UncachedPagedRuntime):
    """Generated Talker and streaming codec serving with eager prompt setup."""

    execution_shape = ExecutionShape.AUTOREGRESSIVE
    image_prefix_length = 0
    spec = None

    def __init__(
        self,
        cfg: Any,
        *,
        compute_stream: Any = None,
        kv_pool: KVMemoryPool,
        max_lora_rank: int | None = None,
        loaded: LoadedQwen3TTS | None = None,
    ) -> None:
        del max_lora_rank
        self._model_name = getattr(cfg, "model", DEFAULT_REPO_ID)
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
            raise ValueError("Qwen3-TTS serving requires CUDA bfloat16")
        if kv_pool is None:
            raise TypeError("Qwen3TTSRuntime requires the engine-owned kv_pool")
        if kv_pool.device != self.device:
            raise ValueError("kv_pool and Qwen3-TTS must use the same device")

        if loaded is None:
            from kestrel.models.registry import get_spec

            model_spec = get_spec(self._model_name)
            checkpoint = getattr(cfg, "model_path", None) or model_spec.repo_id
            if checkpoint is None:
                raise ValueError("Qwen3-TTS model spec must declare repo_id")
            loaded = load_qwen3_tts(
                checkpoint,
                revision=model_spec.revision,
                device=self.device,
                dtype=self.dtype,
            )
        self.config = loaded.config
        self.talker = loaded.talker.eval()
        self.code_predictor = loaded.code_predictor.eval()
        self.codec = loaded.codec.eval()
        self.text_processor = loaded.text_processor

        self.decode_path = getattr(cfg, "decode_path", "auto")
        if self.decode_path == "native":
            raise ValueError("Qwen3-TTS serving requires generated Talker decode")
        if self.decode_path not in {"auto", "generated"}:
            raise ValueError("decode_path must be 'auto' or 'generated'")
        self.max_batch_size = int(getattr(cfg, "max_batch_size", 1))
        if not 1 <= self.max_batch_size <= 32:
            raise ValueError("Qwen3-TTS max_batch_size must lie in [1, 32]")
        self.max_batch_slots = self.max_batch_size * 2
        talker = self.config.talker
        self.max_seq_length = int(talker.max_position_embeddings)
        self.vocab_size = int(talker.vocab_size)
        self.eos_token_ids = (int(talker.codec_eos_token_id),)

        self._compute_stream = compute_stream or make_stream(self.device)
        self._copy_stream = make_stream(self.device)
        self._kv_pool = kv_pool
        self.graph_capture_lock = threading.RLock()
        self.active_sequences: dict[int, Any] = {}
        self._prepared_synthesis: dict[int, PreparedSynthesis] = {}
        self._text_continuations: dict[int, TalkerTokenLayout] = {}

        self._kv_cache_pages = bound_kv_cache_pages(
            int(getattr(cfg, "kv_cache_pages", talker.max_position_embeddings + 1)),
            page_size=1,
            max_batch_size=self.max_batch_size,
            max_seq_length=self.max_seq_length,
        )
        self.page_table = PageTable(
            n_pages=self._kv_cache_pages,
            page_size=1,
            max_batch_size=self.max_batch_slots,
            device=str(self.device),
            h2d_stream=self._compute_stream,
        )
        self._paged_kv = allocate_paged_kv_layers(
            layer_specs=(PagedKVLayerSpec(talker.num_key_value_heads, talker.head_dim),)
            * talker.num_hidden_layers,
            page_table=self.page_table,
            pool=kv_pool,
            dtype=self.dtype,
        )

        predictor = self.config.code_predictor
        decode_rows = decode_slot_rows(self.max_batch_size)
        self._predictor_page_table = PageTable(
            n_pages=1 + decode_rows * NUM_CODE_GROUPS,
            page_size=1,
            max_batch_size=decode_rows + 1,
            device=str(self.device),
            h2d_stream=self._compute_stream,
        )
        predictor_rows = []
        for _ in range(decode_rows):
            row = self._predictor_page_table.allocate()
            self._predictor_page_table.reserve(row, NUM_CODE_GROUPS)
            predictor_rows.append(row)
        self._predictor_page_table.commit_block_table(predictor_rows)
        self._predictor_batch_idx = torch.tensor(
            predictor_rows,
            dtype=torch.int64,
            device=self.device,
        )
        self._predictor_paged_kv = allocate_paged_kv_layers(
            layer_specs=(
                PagedKVLayerSpec(
                    predictor.num_key_value_heads,
                    predictor.head_dim,
                ),
            )
            * predictor.num_hidden_layers,
            page_table=self._predictor_page_table,
            pool=kv_pool,
            dtype=self.dtype,
        )

        self._rope_cosine, self._rope_sine = _rotary_tables(
            talker.max_position_embeddings,
            talker.head_dim,
            float(talker.rope_theta),
            self.device,
        )
        (
            self._predictor_rope_cosine,
            self._predictor_rope_sine,
        ) = _rotary_tables(
            NUM_CODE_GROUPS,
            predictor.head_dim,
            float(predictor.rope_theta),
            self.device,
        )
        with torch.inference_mode():
            self._projected_text_embedding = (
                self.talker.text_projection(self.talker.model.text_embedding.weight)
                .contiguous()
                .detach()
            )
        del self.talker.model.text_embedding
        del self.talker.text_projection
        self._pending_embeddings = torch.empty(
            (self.max_batch_slots, talker.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        allowed_talker_tokens = torch.zeros(self.vocab_size, dtype=torch.uint8)
        allowed_talker_tokens[:2048] = 1
        allowed_talker_tokens[self.eos_token_ids[0]] = 1
        self._allowed_talker_tokens = allowed_talker_tokens.to(self.device)
        self._seen_talker_tokens = torch.zeros(
            (self.max_batch_slots, self.vocab_size),
            dtype=torch.uint8,
            device=self.device,
        )
        self._repetition_penalties_by_batch = CpuGpuBuffer(
            self.max_batch_slots,
            dtype=torch.float32,
            device=self.device,
            pin_memory=True,
            zero=False,
        )
        self._predictor_temperatures_by_batch = CpuGpuBuffer(
            self.max_batch_slots,
            dtype=torch.float32,
            device=self.device,
            pin_memory=True,
            zero=False,
        )
        self._predictor_top_ks_by_batch = CpuGpuBuffer(
            self.max_batch_slots,
            dtype=torch.int32,
            device=self.device,
            pin_memory=True,
            zero=False,
        )
        self._predictor_top_ps_by_batch = CpuGpuBuffer(
            self.max_batch_slots,
            dtype=torch.float32,
            device=self.device,
            pin_memory=True,
            zero=False,
        )
        self._custom_predictor_slots: set[int] = set()
        self._codec_frames = torch.empty(
            (self.max_batch_slots, NUM_CODE_GROUPS, _MAX_CODEC_CHUNK),
            dtype=torch.int64,
            device=self.device,
        )
        self._codec_lanes: dict[Qwen3TTSSynthesisState, _CodecLane] = {}
        # One commit can form at most one codec group per active row.
        self._free_codec_buffers = [
            _CodecBuffer(
                pcm_cpu=torch.empty(
                    (decode_rows, _MAX_CODEC_CHUNK * SAMPLES_PER_FRAME),
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=True,
                ),
                copy_done=make_event(self.device),
            )
            for _ in range(self.max_batch_size)
        ]
        self._pending_codec_jobs: deque[_CodecJob] = deque()
        self._unpublished_codec_states: set[Qwen3TTSSynthesisState] = set()
        self._prefill_slots = tuple(
            self._create_prefill_slot(index, decode_rows)
            for index in range(_PREFILL_SLOTS)
        )
        self.prefill_slots: Sequence[Qwen3TTSPrefillSlot] = self._prefill_slots
        self._free_prefill_slot_ids = set(range(_PREFILL_SLOTS))

        decode_slots = []
        for index in range(_DECODE_SLOTS):
            slot = create_decode_slot(
                slot_id=index,
                device=self.device,
                dtype=self.dtype,
                max_batch_slots=decode_rows,
                kv_cache_pages=self._kv_cache_pages,
                vocab_size=self.vocab_size,
                hidden_dim=talker.hidden_size,
                position_shape=(decode_rows, 1),
                scratch_specs={
                    "decode_embeddings": (
                        (decode_rows, talker.hidden_size),
                        self.dtype,
                    ),
                    **self._predictor_scratch_specs(decode_rows),
                },
                compute_stream=self._compute_stream,
                copy_stream=self._copy_stream,
            )
            self._attach_decode_staging(slot, decode_rows)
            decode_slots.append(slot)
        self.decode_slots: Sequence[DecodeSlot] = tuple(decode_slots)

        self._predictor_rng = torch.Generator(device=self.device)
        self._predictor_rng.seed()
        self.sampling_hooks = SamplingHooks(
            process_logits=self._process_logits,
            require_packed_sampling=True,
            post_sample=self._post_sample,
            materialize_tokens=self._materialize_tokens,
            prepare_decode_inputs=self._prepare_decode_inputs,
            can_dispatch=self._can_dispatch_skill,
            advance_auxiliary=self._advance_auxiliary,
        )
        self.generated_decode = create_generated_decode(self)
        self._generated_prefill = create_talker_generated_prefill(self)
        (
            self._predictor_prime,
            self._predictor_residual,
        ) = create_predictor_generated_decode(self)
        self._codec = TorchCodec(
            self.codec, self._compute_stream, self.graph_capture_lock,
            self.max_batch_size)
        self._text_preprocessor = AsyncPreprocessor(
            self._prepare_synthesis,
            workers=derive_preprocessing_workers(self.max_batch_size),
        )

    def _create_prefill_slot(
        self,
        slot_id: int,
        slot_capacity: int,
    ) -> Qwen3TTSPrefillSlot:
        talker = self.config.talker
        packed_capacity = (
            1 << (slot_capacity * TALKER_GENERATED_PREFILL_LENGTH - 1).bit_length()
        )
        scratch = {
            **{
                name: torch.empty(shape, dtype=dtype, device=self.device)
                for name, (shape, dtype) in self._predictor_scratch_specs(
                    slot_capacity
                ).items()
            },
            "prompt_embeddings": torch.empty(
                (
                    packed_capacity,
                    talker.hidden_size,
                ),
                dtype=self.dtype,
                device=self.device,
            ),
            "prefill_input_pos": torch.arange(
                packed_capacity,
                dtype=torch.int32,
                device=self.device,
            ).remainder_(TALKER_GENERATED_PREFILL_LENGTH),
            "prefill_hidden": torch.empty(
                (packed_capacity, talker.hidden_size),
                dtype=self.dtype,
                device=self.device,
            ),
            "prefill_logits": torch.empty(
                (packed_capacity, talker.vocab_size),
                dtype=self.dtype,
                device=self.device,
            ),
        }
        self._initialize_predictor_scratch(scratch)
        return Qwen3TTSPrefillSlot(
            pool_index=slot_id,
            slot_id=_DECODE_SLOTS + slot_id,
            compute_stream=self._compute_stream,
            batch_indices=CpuGpuBuffer(
                slot_capacity,
                dtype=torch.int64,
                device=self.device,
                pin_memory=True,
                with_numpy=False,
                zero=False,
            ),
            packed_batch_indices=CpuGpuBuffer(
                packed_capacity,
                dtype=torch.int64,
                device=self.device,
                pin_memory=True,
                zero=False,
            ),
            step_done_event=make_event(self.device),
            commit_done_event=make_event(self.device),
            scratch=scratch,
        )

    def _predictor_scratch_specs(
        self,
        capacity: int,
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        talker = self.config.talker
        predictor = self.config.code_predictor
        return {
            "predictor_code0": ((capacity,), torch.int64),
            "predictor_past_hidden": ((capacity, talker.hidden_size), self.dtype),
            "predictor_frames": ((capacity, NUM_CODE_GROUPS), torch.int64),
            "predictor_codec_sum": ((capacity, talker.hidden_size), self.dtype),
            "predictor_seed": ((capacity, predictor.hidden_size), self.dtype),
            "predictor_uniforms": (
                (capacity, NUM_CODE_GROUPS - 1),
                torch.float32,
            ),
            "predictor_temperature": ((capacity,), torch.float32),
            "predictor_top_k": ((capacity,), torch.int32),
            "predictor_top_p": ((capacity,), torch.float32),
            "predictor_input_pos": ((capacity,), torch.int32),
        }

    @staticmethod
    def _initialize_predictor_scratch(scratch: dict[str, torch.Tensor]) -> None:
        scratch["predictor_temperature"].fill_(_PREDICTOR_TEMPERATURE)
        scratch["predictor_top_k"].fill_(_PREDICTOR_TOP_K)
        scratch["predictor_top_p"].fill_(_PREDICTOR_TOP_P)
        scratch["predictor_input_pos"].zero_()

    def _attach_decode_staging(self, slot: DecodeSlot, capacity: int) -> None:
        self._initialize_predictor_scratch(slot.scratch)
        slot.text_token_ids = CpuGpuBuffer(
            capacity,
            dtype=torch.int64,
            device=self.device,
            pin_memory=True,
            zero=False,
        )

    def _prepare_synthesis(self, value: object) -> PreparedSynthesis:
        if not isinstance(value, CustomVoiceRequest):
            raise TypeError("Qwen3-TTS encoder input must be a CustomVoiceRequest")
        if self.config.model_size == "0b6" and value.instructions:
            raise ValueError("Qwen3-TTS 0.6B does not support instructions")
        encoded = self.text_processor.encode(value)
        return PreparedSynthesis(value, build_talker_token_layout(self.config, encoded))

    def preprocess_encoder_input_async(
        self, encoder_input: object
    ) -> Future[PreparedSynthesis]:
        # A worker round trip costs an admission tick for short text. Keep
        # larger prompts off the scheduler thread (35k characters took 12 ms).
        if (
            isinstance(encoder_input, CustomVoiceRequest)
            and len(encoder_input.text) + len(encoder_input.instructions or "") <= 512
        ):
            prepared: Future[PreparedSynthesis] = Future()
            prepared.set_result(self._prepare_synthesis(encoder_input))
            return prepared
        return self._text_preprocessor.submit(encoder_input)

    def preprocess_image_async(self, image: object) -> None:
        del image
        raise ValueError("Qwen3-TTS does not accept images")

    def acquire_prefill_slot(self, slot_id: int | None = None) -> Qwen3TTSPrefillSlot:
        if slot_id is None:
            if not self._free_prefill_slot_ids:
                raise RuntimeError("Qwen3-TTS prefill slots are exhausted")
            selected = min(self._free_prefill_slot_ids)
        else:
            selected = int(slot_id)
        if selected not in self._free_prefill_slot_ids:
            raise RuntimeError(f"Qwen3-TTS prefill slot {selected} is unavailable")
        self._free_prefill_slot_ids.remove(selected)
        return self._prefill_slots[selected]

    def release_prefill_slot(self, slot: Any) -> None:
        if not isinstance(slot, Qwen3TTSPrefillSlot):
            raise TypeError("Qwen3TTSRuntime received a foreign prefill slot")
        slot_id = int(slot.pool_index)
        if (
            slot_id not in range(len(self._prefill_slots))
            or self._prefill_slots[slot_id] is not slot
        ):
            raise ValueError("Qwen3TTSRuntime received a foreign prefill slot")
        if slot_id in self._free_prefill_slot_ids:
            raise RuntimeError(f"Qwen3-TTS prefill slot {slot_id} was released twice")
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
            raise ValueError("Qwen3-TTS does not accept images")
        if lora_slot != 0 or adapter_id is not None:
            raise ValueError("Qwen3-TTS does not support adapters")
        if not isinstance(encoder_input, PreparedSynthesis):
            raise TypeError("Qwen3-TTS prepare_sequence requires prepared text")
        if (
            len(prompt_tokens) != 1
            or not isinstance(prompt_tokens[0], TextToken)
            or prompt_tokens[0].token_id != _PROMPT_MARKER_ID
        ):
            raise ValueError("Qwen3-TTS expects one synthesis prompt marker")
        if max_new_tokens is None:
            new_tokens = DEFAULT_MAX_TOKENS
        elif (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 2
        ):
            raise ValueError("Qwen3-TTS max_new_tokens must be at least 2")
        else:
            new_tokens = max_new_tokens

        layout = encoder_input.layout
        _validate_generation_budget(layout, new_tokens)
        target_length = layout.sequence_length + new_tokens
        if target_length > self.max_seq_length:
            raise ValueError(
                "synthesis prompt plus settings.max_tokens exceeds the "
                "Qwen3-TTS context window"
            )
        prepared = self._prepare_uncached_sequence(
            tokens=[TextToken(token_id=value) for value in layout.text_token_ids],
            target_length=target_length,
            image_length=0,
            lora_slot=0,
            adapter_id=None,
            image_hash=None,
        )
        row = int(prepared.state.batch_idx)
        try:
            predictor_temperature, predictor_top_k, predictor_top_p = (
                encoder_input.request.sampling.subtalker_parameters
            )
            self._predictor_temperatures_by_batch.np[row] = predictor_temperature
            self._predictor_top_ks_by_batch.np[row] = predictor_top_k
            self._predictor_top_ps_by_batch.np[row] = predictor_top_p
            self._repetition_penalties_by_batch.np[row] = (
                encoder_input.request.sampling.repetition_penalty
            )
            with stream_context(self._compute_stream):
                self._seen_talker_tokens[row].zero_()
                self._repetition_penalties_by_batch.gpu[row : row + 1].copy_(
                    self._repetition_penalties_by_batch.cpu[row : row + 1],
                    non_blocking=True,
                )
                self._predictor_temperatures_by_batch.gpu[row : row + 1].copy_(
                    self._predictor_temperatures_by_batch.cpu[row : row + 1],
                    non_blocking=True,
                )
                self._predictor_top_ks_by_batch.gpu[row : row + 1].copy_(
                    self._predictor_top_ks_by_batch.cpu[row : row + 1],
                    non_blocking=True,
                )
                self._predictor_top_ps_by_batch.gpu[row : row + 1].copy_(
                    self._predictor_top_ps_by_batch.cpu[row : row + 1],
                    non_blocking=True,
                )
            self._prepared_synthesis[row] = encoder_input
            self._text_continuations[row] = layout
        except BaseException:
            super().abort_prepared_sequence(prepared)
            raise
        return prepared

    @staticmethod
    def _validate_empty_optional_batch(
        name: str,
        values: Sequence[object | None] | None,
        batch_size: int,
    ) -> None:
        if values is not None and (
            len(values) != batch_size or any(value is not None for value in values)
        ):
            raise ValueError(f"Qwen3-TTS does not support {name}")

    def _copy_prefill_cache(
        self,
        rows: Sequence[int],
        lengths: Sequence[int],
        cache: Any,
    ) -> None:
        pages = torch.tensor(
            [
                page
                for row, length in zip(rows, lengths, strict=True)
                for page in self.page_table.page_table_cpu[row][:length]
            ],
            device=self.device,
            dtype=torch.long,
        )
        for paged, (key, value) in zip(self._paged_kv, cache, strict=True):
            assert paged is not None
            paged.k_cache[:, :, 0].index_copy_(
                0,
                pages,
                key[0].transpose(0, 1),
            )
            paged.v_cache[:, :, 0].index_copy_(
                0,
                pages,
                value[0].transpose(0, 1),
            )

    def _launch_generated_prompt(
        self,
        retained: Sequence[PreparedSynthesis],
        prefill_slot: Qwen3TTSPrefillSlot,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = [item.layout.sequence_length for item in retained]
        if any(
            length
            not in {
                TALKER_GENERATED_PREFILL_LENGTH - 1,
                TALKER_GENERATED_PREFILL_LENGTH,
            }
            for length in lengths
        ):
            raise ValueError("generated Qwen3-TTS prefill requires 9 or 10 rows")
        talker = self.config.talker
        # Tried pinned staging: paired TTFA 0.998–1.018x on H100,
        # 0.978–0.988x on B200; keeping direct input construction.
        text_ids = torch.tensor(
            [
                item.layout.text_token_ids
                + (self.config.tts_pad_token_id,)
                * (TALKER_GENERATED_PREFILL_LENGTH - item.layout.sequence_length)
                for item in retained
            ],
            dtype=torch.long,
            device=self.device,
        )
        codec_ids = torch.tensor(
            [
                item.layout.codec_token_ids
                + (talker.codec_pad_id,)
                * (TALKER_GENERATED_PREFILL_LENGTH - item.layout.sequence_length)
                for item in retained
            ],
            dtype=torch.long,
            device=self.device,
        )
        codec_mask = torch.tensor(
            [
                item.layout.codec_embedding_mask
                + (False,)
                * (TALKER_GENERATED_PREFILL_LENGTH - item.layout.sequence_length)
                for item in retained
            ],
            dtype=torch.bool,
            device=self.device,
        )
        batch_size = len(retained)
        active_rows = batch_size * TALKER_GENERATED_PREFILL_LENGTH
        prompt = prefill_slot.scratch["prompt_embeddings"][:active_rows].view(
            batch_size,
            TALKER_GENERATED_PREFILL_LENGTH,
            -1,
        )
        prompt.copy_(self.talker.model.codec_embedding(codec_ids))
        prompt.mul_(codec_mask.unsqueeze(-1))
        prompt.add_(F.embedding(text_ids, self._projected_text_embedding))
        self._generated_prefill.run(prefill_slot, active_rows)
        if len(set(lengths)) == 1:
            last_rows: slice | torch.Tensor = slice(
                lengths[0] - 1,
                active_rows,
                TALKER_GENERATED_PREFILL_LENGTH,
            )
        else:
            last_rows = torch.tensor(
                [
                    index * TALKER_GENERATED_PREFILL_LENGTH + length - 1
                    for index, length in enumerate(lengths)
                ],
                dtype=torch.long,
                device=self.device,
            )
        return (
            prefill_slot.scratch["prefill_hidden"][last_rows],
            prefill_slot.scratch["prefill_logits"][last_rows],
        )

    def _launch_eager_prompt(
        self,
        retained: Sequence[PreparedSynthesis],
        lengths: Sequence[int],
        rows: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_length = sum(lengths)
        text_ids = torch.empty((1, total_length), dtype=torch.long)
        codec_ids = torch.empty((1, total_length), dtype=torch.long)
        codec_mask = torch.empty((1, total_length), dtype=torch.bool)
        position_ids = torch.empty((1, total_length), dtype=torch.long)
        offsets = [0]
        for synthesis, length in zip(retained, lengths, strict=True):
            layout = synthesis.layout
            start = offsets[-1]
            end = start + length
            text_ids[0, start:end] = torch.tensor(layout.text_token_ids)
            codec_ids[0, start:end] = torch.tensor(layout.codec_token_ids)
            codec_mask[0, start:end] = torch.tensor(layout.codec_embedding_mask)
            position_ids[0, start:end] = torch.arange(length)
            offsets.append(end)

        text_ids = text_ids.to(self.device)
        codec_ids = codec_ids.to(self.device)
        codec_mask = codec_mask.to(self.device)
        position_ids = position_ids.to(self.device)
        cu_seq_lens = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        input_embeddings = F.embedding(
            text_ids,
            self._projected_text_embedding,
        )
        input_embeddings.add_(
            self.talker.model.codec_embedding(codec_ids) * codec_mask.unsqueeze(-1)
        )
        output = self.talker(
            input_embeddings,
            position_ids=position_ids,
            cu_seq_lens=cu_seq_lens,
        )
        last_indices = cu_seq_lens[1:].long() - 1
        self._copy_prefill_cache(rows, lengths, output.past_key_values)
        return (
            output.last_hidden_state[0, last_indices],
            output.logits[0, last_indices],
        )

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
            raise ValueError("Qwen3-TTS prefill batch is outside runtime capacity")
        if not isinstance(prefill_slot, Qwen3TTSPrefillSlot):
            raise TypeError("Qwen3TTSRuntime received a foreign prefill slot")
        if prefill_slot not in self._prefill_slots:
            raise ValueError("Qwen3TTSRuntime received a foreign prefill slot")
        if prefill_slot.pool_index in self._free_prefill_slot_ids:
            raise RuntimeError("Qwen3-TTS prefill slot must be acquired before launch")
        self._validate_empty_optional_batch("images", images, batch_size)
        self._validate_empty_optional_batch(
            "image_crops_list", image_crops_list, batch_size
        )
        if encoder_inputs is None or len(encoder_inputs) != batch_size:
            raise ValueError("encoder_inputs must match the Qwen3-TTS prefill batch")

        rows = [int(prepared.state.batch_idx) for prepared in prepared_sequences]
        if len(set(rows)) != batch_size:
            raise ValueError("Qwen3-TTS prefill batch contains a duplicate state row")
        retained = []
        lengths = []
        for prepared, supplied, row in zip(
            prepared_sequences, encoder_inputs, rows, strict=True
        ):
            synthesis = self._prepared_synthesis.get(row)
            if (
                not isinstance(synthesis, PreparedSynthesis)
                or supplied is not synthesis
            ):
                raise RuntimeError(
                    f"Qwen3-TTS prepared text ownership mismatch for row {row}"
                )
            if prepared.state.length != synthesis.layout.sequence_length:
                raise RuntimeError("Qwen3-TTS prepared prompt length drifted")
            retained.append(synthesis)
            lengths.append(synthesis.layout.sequence_length)

        self.page_table.commit_block_table(rows)
        prefill_slot.batch_indices.cpu[:batch_size] = torch.tensor(
            rows, dtype=torch.int64
        )
        generated_prompt = _uses_generated_prefill(lengths)
        if generated_prompt:
            active_rows = batch_size * TALKER_GENERATED_PREFILL_LENGTH
            prefill_slot.packed_batch_indices.np[:active_rows] = np.repeat(
                np.asarray(rows, dtype=np.int64),
                TALKER_GENERATED_PREFILL_LENGTH,
            )
        with stream_context(self._compute_stream):
            prefill_slot.batch_indices.copy_to_gpu(batch_size)
            if generated_prompt:
                prefill_slot.packed_batch_indices.copy_to_gpu(active_rows)
                hidden_rows, logits = self._launch_generated_prompt(
                    retained,
                    prefill_slot,
                )
            else:
                hidden_rows, logits = self._launch_eager_prompt(
                    retained,
                    lengths,
                    rows,
                )
            for prepared, hidden in zip(prepared_sequences, hidden_rows, strict=True):
                prepared.state.last_hidden = hidden.detach()
        return logits.to(self.dtype)

    def finalize_prepared_sequence_after_prefill(
        self,
        prepared: PreparedSequence,
    ) -> None:
        row = int(prepared.state.batch_idx)
        if row not in self._prepared_synthesis:
            raise RuntimeError(f"Qwen3-TTS prepared state row {row} is missing")
        super().finalize_prepared_sequence_after_prefill(prepared)
        self._prepared_synthesis.pop(row)

    def _process_logits(
        self,
        logits: torch.Tensor,
        *,
        sequences: Sequence[Any],
        batch_idx: torch.Tensor,
    ) -> None:
        eos = self.eos_token_ids[0]
        apply_token_mask_and_repetition_(
            logits,
            self._allowed_talker_tokens,
            self._seen_talker_tokens,
            self._repetition_penalties_by_batch.gpu,
            batch_idx,
            require_packed=logits.is_cuda,
        )

        for row, sequence in enumerate(sequences):
            if len(sequence.skill_state.tokens) < 2:
                logits[row, eos] = float("-inf")

        top_ks = [
            sequence.skill_state.synthesis.sampling.top_k
            if sequence.skill_state.synthesis.sampling.do_sample
            else 0
            for sequence in sequences
        ]
        if len(set(top_ks)) == 1:
            top_k = min(top_ks[0], self.vocab_size)
            if 0 < top_k < self.vocab_size:
                values, indices = torch.topk(logits, top_k, dim=-1)
                logits.fill_(float("-inf"))
                logits.scatter_(1, indices, values)
        else:
            for row, requested_top_k in enumerate(top_ks):
                top_k = min(requested_top_k, self.vocab_size)
                if not 0 < top_k < self.vocab_size:
                    continue
                values, indices = torch.topk(logits[row], top_k)
                logits[row].fill_(float("-inf"))
                logits[row].scatter_(0, indices, values)

    def _stage_predictor_sampling(
        self,
        slot: Any,
        sequences: Sequence[Any],
        batch_idx: torch.Tensor,
    ) -> None:
        custom = any(
            sequence.skill_state.synthesis.sampling.subtalker_parameters
            != (_PREDICTOR_TEMPERATURE, _PREDICTOR_TOP_K, _PREDICTOR_TOP_P)
            for sequence in sequences
        )
        slot_key = id(slot)
        scratch = slot.scratch
        if custom:
            torch.index_select(
                self._predictor_temperatures_by_batch.gpu,
                0,
                batch_idx,
                out=scratch["predictor_temperature"][: len(sequences)],
            )
            torch.index_select(
                self._predictor_top_ks_by_batch.gpu,
                0,
                batch_idx,
                out=scratch["predictor_top_k"][: len(sequences)],
            )
            torch.index_select(
                self._predictor_top_ps_by_batch.gpu,
                0,
                batch_idx,
                out=scratch["predictor_top_p"][: len(sequences)],
            )
            self._custom_predictor_slots.add(slot_key)
        elif slot_key in self._custom_predictor_slots:
            scratch["predictor_temperature"].fill_(_PREDICTOR_TEMPERATURE)
            scratch["predictor_top_k"].fill_(_PREDICTOR_TOP_K)
            scratch["predictor_top_p"].fill_(_PREDICTOR_TOP_P)
            self._custom_predictor_slots.remove(slot_key)

    @torch.inference_mode()
    def _generate_frame(
        self,
        slot: Any,
        hidden_last: torch.Tensor,
        sampled_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(sampled_ids.shape[0])
        scratch = slot.scratch
        code0 = scratch["predictor_code0"][:batch_size]
        torch.clamp_max(sampled_ids, 2047, out=code0)
        frames = scratch["predictor_frames"][:batch_size]
        frames[:, 0].copy_(code0)
        if not hasattr(slot, "hidden_last"):
            scratch["predictor_past_hidden"][:batch_size].copy_(hidden_last)
        scratch["predictor_uniforms"][:batch_size].uniform_(
            generator=self._predictor_rng
        )
        self._predictor_prime.run(slot, batch_size)
        self._predictor_residual.run(slot, batch_size)
        return frames, scratch["predictor_codec_sum"][:batch_size]

    def _post_sample(
        self,
        slot: Any,
        *,
        sampled_ids: torch.Tensor,
        hidden_last: torch.Tensor | None,
        sequences: Sequence[Any],
        batch_idx: torch.Tensor,
        temperatures: torch.Tensor | None,
        top_ps: torch.Tensor | None,
        token_logprobs: torch.Tensor | None,
        ready_event: Any,
    ) -> tuple[Any, int] | None:
        del temperatures, top_ps, token_logprobs, ready_event
        if hidden_last is None:
            raise RuntimeError("Qwen3-TTS post_sample requires Talker hidden state")
        if all(sequence.finalized for sequence in sequences):
            return None
        batch_size = int(sampled_ids.shape[0])
        self._seen_talker_tokens[batch_idx, sampled_ids] = 1
        self._stage_predictor_sampling(slot, sequences, batch_idx)
        frames, codec_sum = self._generate_frame(slot, hidden_last, sampled_ids)
        self._pending_embeddings.index_copy_(0, batch_idx, codec_sum)
        return slot, batch_size

    def _decode_codec_group(
        self,
        states: Sequence[Qwen3TTSSynthesisState],
        lanes: Sequence[_CodecLane],
        frame_count: int,
        *,
        stream_output_ready: Callable[[Any], bool] | None = None,
    ) -> None:
        if len(states) != len(lanes):
            raise RuntimeError("Qwen3-TTS codec states and lanes do not match")
        if any(not 0 < frame_count <= lane.buffered_frames for lane in lanes):
            raise RuntimeError("Qwen3-TTS codec output has an invalid frame count")
        buffer = self._acquire_codec_buffer(
            stream_output_ready=stream_output_ready
        )
        sample_count = frame_count * SAMPLES_PER_FRAME
        warm = bool(lanes[0].decoder_state.frame_position)
        with stream_context(self._compute_stream):
            codes = self._codec_frames[[lane.row for lane in lanes], :, :frame_count]
            pcm = self._codec.run(
                codes,
                tuple(lane.decoder_state for lane in lanes),
            )
            # Tried a separate copy stream: D2H was ~0.03 ms; keeping it inline.
            buffer.pcm_cpu[: len(states), :sample_count].copy_(
                pcm,
                non_blocking=True,
            )
            buffer.copy_done.record(self._compute_stream)

        skip_samples = []
        for lane in lanes:
            lane.buffered_frames -= frame_count
            lane.chunk_index += 1
            skip_samples.append(
                SAMPLES_PER_FRAME
                if lane.decoded_frames == 0 and lane.suppress_bootstrap
                else 0
            )
            lane.decoded_frames += frame_count
        job = _CodecJob(
            buffer=buffer,
            states=tuple(states),
            frame_count=frame_count,
            skip_samples=tuple(skip_samples),
        )
        if warm:
            self._pending_codec_jobs.append(job)
        else:
            # Tried deferring cold completion: ready PCM stayed hidden behind
            # the next 6.5 ms decode step. Publish the first chunk immediately.
            buffer.copy_done.synchronize()
            self._finish_codec_job(
                job, stream_output_ready=stream_output_ready
            )

    def _finish_codec_job(
        self,
        job: _CodecJob,
        *,
        stream_output_ready: Callable[[Any], bool] | None = None,
    ) -> None:
        sample_count = job.frame_count * SAMPLES_PER_FRAME
        for output_row, (state, skip) in enumerate(
            zip(job.states, job.skip_samples, strict=True)
        ):
            chunk = job.buffer.pcm_cpu[output_row, skip:sample_count].numpy()
            had_audio = state.has_audio
            state.append_pcm(chunk)
            if not had_audio:
                # Silent chunks have not built a playback reserve. Keep the
                # short refill until onset, including immediately after it.
                self._codec_lanes[state].chunk_index = 1
            if stream_output_ready is not None:
                stream_output_ready(state)
            else:
                self._unpublished_codec_states.add(state)
        self._free_codec_buffers.append(job.buffer)

    def _poll_codec_jobs(
        self,
        *,
        stream_output_ready: Callable[[Any], bool] | None = None,
    ) -> None:
        if stream_output_ready is not None and self._unpublished_codec_states:
            for state in self._unpublished_codec_states:
                stream_output_ready(state)
            self._unpublished_codec_states.clear()
        while (
            self._pending_codec_jobs
            and self._pending_codec_jobs[0].buffer.copy_done.query()
        ):
            self._finish_codec_job(
                self._pending_codec_jobs.popleft(),
                stream_output_ready=stream_output_ready,
            )

    def _acquire_codec_buffer(
        self,
        *,
        stream_output_ready: Callable[[Any], bool] | None = None,
    ) -> _CodecBuffer:
        self._poll_codec_jobs(stream_output_ready=stream_output_ready)
        if not self._free_codec_buffers:
            self._pending_codec_jobs[0].buffer.copy_done.synchronize()
            self._poll_codec_jobs(stream_output_ready=stream_output_ready)
        if not self._free_codec_buffers:
            raise RuntimeError("Qwen3-TTS codec pipeline failed to release a buffer")
        return self._free_codec_buffers.pop()

    def _drain_codec_states(
        self,
        states: Sequence[Qwen3TTSSynthesisState],
    ) -> None:
        targets = set(states)
        last = next(
            (
                index
                for index in range(len(self._pending_codec_jobs) - 1, -1, -1)
                if targets.intersection(self._pending_codec_jobs[index].states)
            ),
            -1,
        )
        if last < 0:
            return
        self._pending_codec_jobs[last].buffer.copy_done.synchronize()
        for _ in range(last + 1):
            self._finish_codec_job(self._pending_codec_jobs.popleft())

    @staticmethod
    def _next_codec_chunk(lane: _CodecLane) -> int:
        # Tried one-frame onset refill: B2000.6 paired p95 116.2→119.7ms,
        # underruns 2→3; keep the regular two-frame refill.
        return _CODEC_CHUNKS[min(lane.chunk_index, len(_CODEC_CHUNKS) - 1)]

    def _codec_lane(
        self,
        state: Qwen3TTSSynthesisState,
        row: int,
    ) -> _CodecLane:
        lane = self._codec_lanes.get(state)
        if lane is None:
            lane = _CodecLane(
                row=row,
                decoder_state=IncrementalCodecState(),
                suppress_bootstrap=state.suppresses_fixed_bootstrap_audio,
            )
            self._codec_lanes[state] = lane
        elif lane.row != row:
            raise RuntimeError("Qwen3-TTS request changed codec rows")
        return lane

    def _release_synthesis_audio(self, state: Qwen3TTSSynthesisState) -> None:
        self._flush_codec_states((state,))
        self._unpublished_codec_states.discard(state)
        self._codec_lanes.pop(state, None)

    def _flush_codec_states(
        self,
        states: Sequence[Qwen3TTSSynthesisState],
    ) -> None:
        ready: dict[
            tuple[int, bool],
            list[tuple[Qwen3TTSSynthesisState, _CodecLane]],
        ] = {}
        for state in states:
            lane = self._codec_lanes.get(state)
            if lane is not None and lane.buffered_frames:
                key = (lane.buffered_frames, bool(lane.decoder_state.frame_position))
                ready.setdefault(key, []).append((state, lane))
        for (frame_count, _warm), group in ready.items():
            grouped_states, lanes = zip(*group, strict=True)
            self._decode_codec_group(grouped_states, lanes, frame_count)
        self._drain_codec_states(states)

    def _can_dispatch_skill(self, state: Any, *, inflight_steps: int) -> bool:
        if not isinstance(state, Qwen3TTSSynthesisState):
            raise TypeError("Qwen3-TTS request has the wrong skill state")
        lane = self._codec_lanes.get(state)
        return (
            lane is None
            or lane.buffered_frames + inflight_steps
            < self._next_codec_chunk(lane)
        )

    def _advance_auxiliary(
        self,
        *,
        force: bool,
        stream_output_ready: Callable[[Any], bool],
    ) -> bool:
        del force
        self._poll_codec_jobs(stream_output_ready=stream_output_ready)
        groups: dict[
            tuple[int, bool],
            list[tuple[Qwen3TTSSynthesisState, _CodecLane]],
        ] = {}
        for state, lane in self._codec_lanes.items():
            frame_count = self._next_codec_chunk(lane)
            if lane.buffered_frames >= frame_count:
                key = (frame_count, bool(lane.decoder_state.frame_position))
                groups.setdefault(key, []).append((state, lane))
        if not groups:
            return False

        def deadline(item: tuple[Qwen3TTSSynthesisState, _CodecLane]) -> float:
            value = item[0].next_output_deadline(self)
            return value if value is not None else float("inf")

        def group_deadline(
            group: list[tuple[Qwen3TTSSynthesisState, _CodecLane]],
        ) -> float:
            return min(map(deadline, group))

        selected = min(groups.values(), key=group_deadline)
        # Tried capping codec groups at 16 at H100 RPS10: 5 underruns / 1.21 s
        # and 50.02 xRT versus 1 / 16.0 ms and 50.38 xRT; keep it uncapped.
        selected = sorted(selected, key=deadline)[: self.max_batch_size]
        states, lanes = zip(*selected, strict=True)
        self._decode_codec_group(
            states,
            lanes,
            self._next_codec_chunk(lanes[0]),
            stream_output_ready=stream_output_ready,
        )
        return True

    def _materialize_tokens(
        self,
        token_ids_cpu: torch.Tensor,
        sequences: Sequence[Any],
        batch_idx: torch.Tensor,
        step_handle: tuple[Any, int] | None,
    ) -> list[Token]:
        if step_handle is None:
            return [
                TextToken(token_id=int(value))
                for value in token_ids_cpu.view(-1).tolist()
            ]
        self._poll_codec_jobs()
        slot, batch_size = step_handle
        if batch_size != len(sequences):
            raise RuntimeError("Qwen3-TTS audio row count does not match sequences")
        frames = slot.scratch["predictor_frames"][:batch_size]
        token_ids = [int(value) for value in token_ids_cpu.view(-1).tolist()]
        del batch_idx
        physical_rows = [int(sequence.state.batch_idx) for sequence in sequences]

        eos = self.eos_token_ids[0]
        ready: dict[
            tuple[int, bool],
            list[tuple[Qwen3TTSSynthesisState, _CodecLane]],
        ] = {}
        terminal_states: list[Qwen3TTSSynthesisState] = []
        active_lanes: list[tuple[int, _CodecLane]] = []
        for source_row, (token_id, sequence, physical_row) in enumerate(
            zip(token_ids, sequences, physical_rows, strict=True)
        ):
            if token_id == eos or sequence.finalized:
                continue
            state = sequence.skill_state
            if not isinstance(state, Qwen3TTSSynthesisState):
                raise TypeError("Qwen3-TTS request has the wrong skill state")
            lane = self._codec_lane(state, physical_row)
            if lane.buffered_frames >= self._codec_frames.shape[2]:
                raise RuntimeError("Qwen3-TTS codec chunk exceeded its buffer")
            active_lanes.append((source_row, lane))

        with stream_context(self._compute_stream):
            offsets = {lane.buffered_frames for _source, lane in active_lanes}
            if len(active_lanes) == batch_size and len(offsets) == 1:
                offset = offsets.pop()
                physical_batch = (
                    slot.batch_indices.gpu[:batch_size]
                    if isinstance(slot, Qwen3TTSPrefillSlot)
                    else slot.meta.batch_idx.gpu[:batch_size]
                )
                self._codec_frames[:, :, offset].index_copy_(
                    0,
                    physical_batch,
                    frames,
                )
            else:
                for source_row, lane in active_lanes:
                    self._codec_frames[lane.row, :, lane.buffered_frames].copy_(
                        frames[source_row]
                    )
        for _source_row, lane in active_lanes:
            lane.buffered_frames += 1

        for token_id, sequence in zip(token_ids, sequences, strict=True):
            if sequence.finalized:
                continue
            state = sequence.skill_state
            if not isinstance(state, Qwen3TTSSynthesisState):
                raise TypeError("Qwen3-TTS request has the wrong skill state")
            lane = self._codec_lanes.get(state)
            if token_id == eos:
                terminal_states.append(state)
                frame_count = 0 if lane is None else lane.buffered_frames
                if frame_count:
                    key = (frame_count, bool(lane.decoder_state.frame_position))
                    ready.setdefault(key, []).append((state, lane))
                continue
            if lane is None:
                raise RuntimeError("Qwen3-TTS codec lane is missing")
            terminal = state.token_count + 1 >= state.request.max_new_tokens
            if terminal:
                terminal_states.append(state)
            if terminal:
                key = (
                    lane.buffered_frames,
                    bool(lane.decoder_state.frame_position),
                )
                ready.setdefault(key, []).append((state, lane))
        for (frame_count, _warm), group in ready.items():
            states, lanes = zip(*group, strict=True)
            self._decode_codec_group(states, lanes, frame_count)
        if terminal_states:
            self._drain_codec_states(terminal_states)
        return [TextToken(token_id=token_id) for token_id in token_ids]

    def _prepare_decode_inputs(
        self,
        slot: DecodeSlot,
        batch_idx: torch.Tensor,
        batch_size: int,
    ) -> None:
        batch_indices = slot.meta.batch_idx.np
        positions = slot.meta.input_pos.np
        next_ids = slot.text_token_ids.np
        for index in range(batch_size):
            state_row = int(batch_indices[index])
            try:
                layout = self._text_continuations[state_row]
            except KeyError as error:
                raise RuntimeError(
                    f"Qwen3-TTS text continuation row {state_row} is missing"
                ) from error
            step = int(positions[index]) - layout.sequence_length
            if step < 0:
                raise RuntimeError("Qwen3-TTS decode position precedes its prefill")
            continuation = layout.continuation_text_token_ids
            next_ids[index] = (
                continuation[step]
                if step < len(continuation)
                else layout.text_pad_token_id
            )
        slot.text_token_ids.copy_to_gpu(batch_size)
        torch.index_select(
            self._pending_embeddings,
            0,
            batch_idx,
            out=slot.scratch["decode_embeddings"][:batch_size],
        )

    @torch.inference_mode()
    def decode_with_slot(self, slot: DecodeSlot, batch_size: int) -> None:
        if batch_size == 0:
            return
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError("Qwen3-TTS decode batch is outside runtime capacity")
        slot_id = int(slot.slot_id)
        if (
            slot_id not in range(len(self.decode_slots))
            or self.decode_slots[slot_id] is not slot
        ):
            raise ValueError("Qwen3TTSRuntime received a foreign decode slot")
        with stream_context(self._compute_stream):
            self.generated_decode.run(slot, batch_size)

    def _release_runtime_state(self, batch_idx: int) -> None:
        self._prepared_synthesis.pop(batch_idx, None)
        self._text_continuations.pop(batch_idx, None)
        states = tuple(
            state for state, lane in self._codec_lanes.items() if lane.row == batch_idx
        )
        self._flush_codec_states(states)
        for state in states:
            self._unpublished_codec_states.discard(state)
            self._codec_lanes.pop(state, None)

    def shutdown(self) -> None:
        self._text_preprocessor.shutdown()
        if self._pending_codec_jobs:
            self._pending_codec_jobs[-1].buffer.copy_done.synchronize()
            while self._pending_codec_jobs:
                self._finish_codec_job(self._pending_codec_jobs.popleft())
        empty_cache(self.device)


__all__ = ["PreparedSynthesis", "Qwen3TTSPrefillSlot", "Qwen3TTSRuntime"]
