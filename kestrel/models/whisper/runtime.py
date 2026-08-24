"""Optimized autoregressive runtime integration for Whisper Turbo.

This module owns scheduler-facing lifecycle and stable state.  Production
forward work is delegated exclusively to the public custom-prefill session and
Kestrel's shared generated-decode runtime. The eager model in :mod:`model` is
never imported here.
"""

from __future__ import annotations

import json
import math
import threading
import traceback
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from kestrel.device import make_event, make_stream, stream_context
from kestrel.kv_cache import (
    KVMemoryPool,
    PageTable,
    PagedKVLayerSpec,
    allocate_paged_kv_layers,
)
from kestrel.runtime.decode_slot import DecodeSlot, create_decode_slot
from kestrel.runtime.paged_resources import decode_slot_rows
from kestrel.runtime.preprocessing import derive_preprocessing_workers
from kestrel.runtime.sampling import SamplingHooks
from kestrel.runtime.staging import AsyncPreprocessor
from kestrel.runtime.state import PrefillClassification, PreparedSequence
from kestrel.runtime.tokens import TextToken, Token
from kestrel.runtime.uncached_paged import UncachedPagedRuntime
from kestrel.utils import CpuGpuBuffer, PackedBuffer
from kestrel_kernels import get_runtime
from kestrel_kernels.cubin_runtime import capture_packed_artifact_receipts
from kestrel_kernels.fallback import is_cute_jit_enabled

from .alignment import (
    TranscriptAnalysis,
    TranscriptScores,
    align_transcript_words,
    no_speech_probability,
)
from .assets import CHECKPOINT_REVISION, MODEL_NAME, REPO_ID, WhisperAssets
from .audio import AudioSource, PreparedAudio, prepare_audio
from .config import WhisperPreprocessorConfig, WhisperTurboConfig
from .generated_decode import create_generated_decode
from .prefill_session import NativeWhisperPrefillSession
from .runtime_abi import (
    WhisperExecutionBindings,
    WhisperCrossArenas,
    WhisperDecodeBuffers,
    WhisperPrefillBuffers,
    WhisperSelfKVArenas,
    validate_resident_buffers,
)
from .tokenizer import WhisperTokenizer
from .weights import (
    WhisperModelWeights,
    load_whisper_safetensors,
    validate_whisper_weight_tree,
)

_CUTE_JIT_ENABLED_AT_RUNTIME_IMPORT = is_cute_jit_enabled()
_SAMPLING = get_runtime().sampling


_PREFILL_SLOT_COUNT = 2
_DECODE_SLOT_COUNT = 2
_CONTROL_TOKEN_CAPACITY = 4
_DEFAULT_TRANSCRIPT_TOKENS = 444
_CONSTRAINT_PLAN_WIDTH = 8


def _validated_packed_receipts(
    receipts: tuple[dict[str, object], ...],
    *,
    expected_families: set[str],
    component: str,
) -> list[dict[str, object]]:
    required = {
        "schema_version",
        "family",
        "variant_key",
        "architecture",
        "payload_kind",
        "payload_sha256",
        "archive_path",
        "archive_size_bytes",
        "archive_sha256",
        "archive_root",
    }
    normalized = [dict(receipt) for receipt in receipts]
    if not normalized or any(not required.issubset(receipt) for receipt in normalized):
        raise RuntimeError(f"Whisper {component} packed provenance is incomplete")
    families = {str(receipt["family"]) for receipt in normalized}
    if families != expected_families:
        raise RuntimeError(
            f"Whisper {component} packed provenance families are {sorted(families)}, "
            f"expected {sorted(expected_families)}"
        )
    for receipt in normalized:
        for field in ("payload_sha256", "archive_sha256"):
            digest = receipt[field]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(
                    f"Whisper {component} packed provenance has invalid {field}"
                )
        if (
            receipt["schema_version"] != 1
            or not isinstance(receipt["variant_key"], str)
            or not receipt["variant_key"]
            or not isinstance(receipt["architecture"], str)
            or not receipt["architecture"]
            or not isinstance(receipt["payload_kind"], str)
            or not receipt["payload_kind"]
            or not isinstance(receipt["archive_path"], str)
            or not receipt["archive_path"]
            or not isinstance(receipt["archive_size_bytes"], int)
            or isinstance(receipt["archive_size_bytes"], bool)
            or int(receipt["archive_size_bytes"]) <= 0
            or not isinstance(receipt["archive_root"], str)
            or not receipt["archive_root"]
        ):
            raise RuntimeError(
                f"Whisper {component} packed provenance has malformed identity fields"
            )
    return sorted(
        normalized,
        key=lambda receipt: (
            str(receipt["family"]),
            str(receipt["variant_key"]),
            str(receipt["architecture"]),
        ),
    )


def _native_provenance(
    bindings: WhisperExecutionBindings,
    decode_session: Any,
) -> dict[str, Any]:
    capacities = tuple(
        capacity
        for capacity in (1, 2, 4, 8)
        if capacity <= 1 << (int(bindings.max_batch_size) - 1).bit_length()
    )
    identities = sorted(
        (dict(receipt) for receipt in decode_session.artifact_receipts),
        key=lambda receipt: int(receipt["capacity"]),
    )
    if tuple(int(identity["capacity"]) for identity in identities) != capacities:
        raise RuntimeError(
            "Whisper generated decode did not resolve every required AOT capacity"
        )
    archives = sorted(
        {
            (
                str(identity["archive_path"]),
                int(identity["archive_size_bytes"]),
                str(identity["archive_sha256"]),
                str(identity["archive_root"]),
            )
            for identity in identities
        }
    )
    prefill_slots = sorted(slot.slot_id for slot in bindings.prefill_buffers)
    return {
        "prefill": {
            "backend": "cuda-graph",
            "implementation_modules": [
                "kestrel.models.whisper.prefill_stem",
                "kestrel.models.whisper.prefill_encoder",
                "kestrel.models.whisper.prefill_decoder_prefix",
            ],
            "native_kernels_required": True,
            "aot_required": True,
            "graph_coverage": [
                {"slot_id": slot_id, "batch_size": batch_size}
                for slot_id in prefill_slots
                for batch_size in range(1, int(bindings.max_batch_size) + 1)
            ],
        },
        "decode": {
            "backend": "generated-aot",
            "aot_family": "whisper_large_v3_turbo",
            "aot_required": True,
            "batch_capacities": list(capacities),
            "slot_ids": sorted(slot.slot_id for slot in bindings.decode_buffers),
            "bundle_archives": [
                {
                    "path": path,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                    "root": root,
                }
                for path, size_bytes, sha256, root in archives
            ],
            "artifact_identities": identities,
        },
    }


@dataclass(frozen=True, slots=True)
class WhisperRuntimeComponents:
    """Explicit dependency injection for CPU lifecycle/integration tests.

    Production construction never receives this object: it resolves the pinned
    assets and builds the public native sessions directly. Tests may substitute
    enqueue-only sessions while exercising the real scheduler and slot lifecycle.
    """

    config: WhisperTurboConfig
    preprocessor_config: WhisperPreprocessorConfig
    tokenizer: WhisperTokenizer
    weights: WhisperModelWeights
    session_factory: Any | None = None


@dataclass(slots=True)
class WhisperPrefillSlot:
    """Public prefill-slot fields plus internal pointer-stable staging."""

    slot_id: int
    batch_idx: Tensor
    step_done_event: Any
    commit_done_event: Any
    features: CpuGpuBuffer
    metadata: PackedBuffer
    logits: Tensor

    def execution_buffers(self) -> WhisperPrefillBuffers:
        return WhisperPrefillBuffers(
            slot_id=self.slot_id,
            input_features=self.features.gpu,
            control_token_ids=self.metadata.control_token_ids.gpu,
            prefix_lengths=self.metadata.prefix_lengths.gpu,
            batch_idx=self.batch_idx,
            slot_mapping=self.metadata.slot_mapping.gpu,
            logits_out=self.logits,
        )


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Whisper {name} must be a positive integer")
    return int(value)


def _resolved_device(cfg: Any) -> torch.device:
    if hasattr(cfg, "resolved_device"):
        return torch.device(cfg.resolved_device())
    return torch.device(getattr(cfg, "device", "cuda"))


def _resolved_dtype(cfg: Any) -> torch.dtype:
    if hasattr(cfg, "resolved_dtype"):
        return cfg.resolved_dtype()
    return getattr(cfg, "dtype", torch.bfloat16)


def _load_production_components(
    cfg: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> WhisperRuntimeComponents:
    if getattr(cfg, "model", MODEL_NAME) != MODEL_NAME:
        raise ValueError(
            f"WhisperRuntime only serves {MODEL_NAME!r}, got "
            f"{getattr(cfg, 'model', None)!r}"
        )
    if device.type != "cuda":
        raise ValueError("The optimized Whisper runtime requires a CUDA device")
    if dtype is not torch.bfloat16:
        raise ValueError("The optimized Whisper runtime requires bfloat16")
    if _CUTE_JIT_ENABLED_AT_RUNTIME_IMPORT or is_cute_jit_enabled():
        raise RuntimeError(
            "Production Whisper forbids KESTREL_CUTE_JIT; install packed AOT "
            "GELU, FlashAttention, sampling, and generated-decode artifacts"
        )

    if (
        getattr(cfg, "model_path", None) is not None
        or getattr(cfg, "tokenizer_path", None) is not None
    ):
        raise ValueError(
            "Production Whisper does not accept local checkpoint overrides; "
            f"assets are resolved from {REPO_ID!r} at immutable revision "
            f"{CHECKPOINT_REVISION!r}"
        )

    assets = WhisperAssets()
    config = WhisperTurboConfig.from_assets(assets)
    preprocessor_config = WhisperPreprocessorConfig.from_assets(assets)
    tokenizer = WhisperTokenizer.from_assets(assets)
    weights = load_whisper_safetensors(
        assets.path("model.safetensors"),
        config,
        checkpoint_dtype=torch.float16,
        device=device,
        dtype=dtype,
    )

    return WhisperRuntimeComponents(
        config=config,
        preprocessor_config=preprocessor_config,
        tokenizer=tokenizer,
        weights=weights,
    )


class WhisperRuntime(UncachedPagedRuntime):
    """Pinned Whisper Turbo runtime with custom prefill and generated decode."""

    spec = None
    image_prefix_length = 0

    def __init__(
        self,
        cfg: Any,
        *,
        max_lora_rank: Optional[int] = None,
        kv_pool: KVMemoryPool,
        compute_stream: torch.cuda.Stream | None = None,
        _components: WhisperRuntimeComponents | None = None,
    ) -> None:
        del max_lora_rank
        try:
            self._initialize(
                cfg,
                kv_pool=kv_pool,
                compute_stream=compute_stream,
                _components=_components,
            )
        except BaseException as exc:
            self._abort_failed_construction()
            traceback.clear_frames(exc.__traceback__)
            # A retained startup exception must not keep the failed runtime or
            # caller-supplied components alive through this executing frame.
            del cfg, kv_pool, compute_stream, _components, self
            raise

    def _abort_failed_construction(self) -> None:
        """Best-effort shutdown followed by unconditional ownership release."""

        owned = self.__dict__
        for name in (
            "_prefill_session",
            "_audio_preprocessor",
        ):
            resource = owned.get(name)
            if resource is None:
                continue
            try:
                resource.shutdown()
            except BaseException:
                pass
        # This is a failed, unpublished instance. Clearing the complete mapping
        # is the durable transaction rollback: it releases arenas, slots,
        # streams, partial sessions, and the sampling hook's bound-self cycle.
        owned.clear()

    def _abort_failed_warmup(self) -> BaseException | None:
        """Close and detach a fully built runtime that was never published."""

        shutdown_error: BaseException | None = None
        try:
            self.shutdown()
        except BaseException as exc:
            traceback.clear_frames(exc.__traceback__)
            shutdown_error = exc.with_traceback(None)
        finally:
            # Normal shutdown keeps resident state attached for safe idempotent
            # lifecycle calls. A failed factory warmup cannot publish this
            # instance, so release every remaining arena, slot, stream, and
            # bound-method cycle immediately instead.
            self.__dict__.clear()
        return shutdown_error

    def _initialize(
        self,
        cfg: Any,
        *,
        kv_pool: KVMemoryPool,
        compute_stream: torch.cuda.Stream | None,
        _components: WhisperRuntimeComponents | None,
    ) -> None:
        from kestrel.runtime import ExecutionShape

        self.execution_shape = ExecutionShape.AUTOREGRESSIVE
        self._model_name = MODEL_NAME
        self.device = _resolved_device(cfg)
        self.dtype = _resolved_dtype(cfg)
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("Whisper runtime dtype must be a torch.dtype")
        self.max_batch_size = _positive_int(
            "max_batch_size", getattr(cfg, "max_batch_size", 1)
        )
        if self.max_batch_size > 8:
            raise ValueError("Whisper generated decode supports max_batch_size <= 8")
        self.max_batch_slots = self.max_batch_size + 2
        # Page-table row zero and physical page zero are engine-reserved no-op
        # resources. Generated B1/B2/B4/B8 launches pad compact batches with
        # that same row, while the two extra scheduler rows remain available to
        # the optimistic prefill/decode pipeline.
        self._padding_batch_idx = 0
        self._decode_batch_capacities = tuple(
            capacity
            for capacity in (1, 2, 4, 8)
            if capacity <= 1 << (self.max_batch_size - 1).bit_length()
        )
        self.page_size = _positive_int("page_size", getattr(cfg, "page_size", 1))
        if self.page_size != 1:
            raise ValueError("Whisper generated decode requires page_size=1")
        requested_kv_cache_pages = _positive_int(
            "kv_cache_pages", getattr(cfg, "kv_cache_pages", 65536)
        )
        if getattr(cfg, "enable_prefix_cache", False):
            warnings.warn(
                "Whisper prefix caching is disabled because encoder conditioning "
                "is not part of the decoder-prefix cache key",
                RuntimeWarning,
                stacklevel=2,
            )

        production = _components is None
        if production and not bool(getattr(cfg, "enable_cuda_graphs", True)):
            raise ValueError(
                "Optimized Whisper serving requires CUDA graphs for custom prefill"
            )
        components = _components or _load_production_components(
            cfg,
            device=self.device,
            dtype=self.dtype,
        )
        self._config = components.config
        self._preprocessor_config = components.preprocessor_config
        self.tokenizer = components.tokenizer
        if self.tokenizer.controls.vocab_size != self._config.vocab_size:
            raise ValueError(
                "Whisper tokenizer and model configuration vocabularies disagree"
            )
        weight_device, weight_dtype = validate_whisper_weight_tree(
            components.weights, self._config
        )
        if weight_device != self.device or weight_dtype is not self.dtype:
            raise ValueError(
                "Whisper runtime weights must match the configured device and dtype"
            )
        if production and self._config != WhisperTurboConfig():
            raise ValueError("Production Whisper runtime geometry must remain pinned")
        self._require_native = production

        self.max_seq_length = self._config.max_target_positions
        self.vocab_size = self._config.vocab_size
        self.eos_token_ids = (self._config.eos_token_id,)
        max_addressable_pages = (self.max_batch_slots - 1) * self.max_seq_length + 1
        self.requested_kv_cache_pages = requested_kv_cache_pages
        self.effective_kv_cache_pages = min(
            requested_kv_cache_pages,
            max_addressable_pages,
        )
        self._kv_cache_pages = self.effective_kv_cache_pages
        self.sampling_hooks = SamplingHooks(
            process_logits=self._process_logits,
            adjust_sampling_params=self._adjust_sampling_params,
            score_sampled_tokens=self._score_sampled_tokens,
            require_packed_greedy_logprobs=self._require_native,
        )

        self._compute_stream = (
            compute_stream if compute_stream is not None else make_stream(self.device)
        )
        self._copy_stream = make_stream(self.device)
        self.graph_capture_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._warmup_lock = threading.Lock()
        self._closed = False
        self._prefill_warmed = False
        self._constraints_warmed = False
        self._decode_warmed = False
        self._sampling_artifact_receipts: tuple[dict[str, object], ...] = ()
        self._prepared_audio: dict[int, PreparedAudio] = {}
        self._owned_cross_rows: set[int] = set()
        self.active_sequences: dict[int, Any] = {}

        if kv_pool is None:
            raise TypeError("WhisperRuntime requires the engine-owned kv_pool")
        self._kv_pool = kv_pool
        if self._kv_pool.device != self.device:
            raise ValueError(
                f"kv_pool.device ({self._kv_pool.device}) must match Whisper "
                f"device ({self.device})"
            )

        self.page_table = PageTable(
            n_pages=self._kv_cache_pages,
            page_size=1,
            max_batch_size=self.max_batch_slots,
            device=str(self.device),
            prefix_cache=None,
            h2d_stream=self._compute_stream,
        )
        paged_self_kv = allocate_paged_kv_layers(
            layer_specs=tuple(
                PagedKVLayerSpec(
                    n_heads=self._config.decoder_attention_heads,
                    head_dim=self._config.decoder_head_dim,
                )
                for _ in range(self._config.decoder_layers)
            ),
            page_table=self.page_table,
            pool=self._kv_pool,
            dtype=self.dtype,
        )
        if any(cache is None for cache in paged_self_kv):  # pragma: no cover
            raise RuntimeError("Whisper decoder self-KV allocation is incomplete")
        self._paged_self_kv = tuple(
            cache for cache in paged_self_kv if cache is not None
        )
        self.self_kv = WhisperSelfKVArenas(
            keys=tuple(cache.k_cache.squeeze(2) for cache in self._paged_self_kv),
            values=tuple(cache.v_cache.squeeze(2) for cache in self._paged_self_kv),
        )
        self.self_kv.validate(
            self._config,
            n_pages=self._kv_cache_pages,
            device=self.device,
            dtype=self.dtype,
        )
        self.cross_kv = WhisperCrossArenas.allocate(
            self._config,
            self.max_batch_slots,
            device=self.device,
            dtype=self.dtype,
        )
        self.cross_kv.validate(
            self._config,
            self.max_batch_slots,
            device=self.device,
            dtype=self.dtype,
        )
        # Padded generated rows use row zero. Every real row is populated by
        # custom prefill before it can enter decode; the no-op row must never
        # contain stale conditioning from allocator reuse.
        with stream_context(self._compute_stream):
            self.cross_kv.keys[:, self._padding_batch_idx].zero_()
            self.cross_kv.values[:, self._padding_batch_idx].zero_()

        self._prefill_slots = tuple(
            self._create_prefill_slot(slot_id) for slot_id in range(_PREFILL_SLOT_COUNT)
        )
        self.prefill_slots: Sequence[WhisperPrefillSlot] = self._prefill_slots
        self._free_prefill_slot_ids = set(range(_PREFILL_SLOT_COUNT))

        decode_rows = decode_slot_rows(self.max_batch_size)
        self._decode_slots = tuple(
            create_decode_slot(
                slot_id=slot_id,
                device=self.device,
                dtype=self.dtype,
                max_batch_slots=decode_rows,
                kv_cache_pages=self._kv_cache_pages,
                vocab_size=self.vocab_size,
                hidden_dim=self._config.d_model,
                position_shape=(decode_rows, 1),
                compute_stream=self._compute_stream,
                copy_stream=self._copy_stream,
            )
            for slot_id in range(_DECODE_SLOT_COUNT)
        )
        self.decode_slots: Sequence[DecodeSlot] = self._decode_slots

        # Sampling can overlap two resident pipeline slots. Each slot therefore
        # owns its own pinned constraint source and GPU destination: mutating a
        # shared pinned tensor for the next sample could race an earlier async
        # H2D that has been enqueued but not yet consumed. The generic hook
        # already receives the resident batch_idx view, which identifies its
        # slot without exposing Whisper concepts in the public API. Scheduler
        # reuse of that slot is fenced by its commit_done_event, recorded after
        # sampling on the same compute stream, so this CPU source remains
        # immutable until the H2D and constrained kernel have completed.
        resident_batch_idx = tuple(
            slot.batch_idx for slot in self._prefill_slots
        ) + tuple(slot.meta.batch_idx.gpu for slot in self._decode_slots)
        self._logits_constraint_buffers = tuple(
            CpuGpuBuffer(
                self.max_batch_size,
                _CONSTRAINT_PLAN_WIDTH,
                dtype=torch.int32,
                device=self.device,
                pin_memory=self.device.type == "cuda",
            )
            for _ in resident_batch_idx
        )
        self._logits_constraints_by_batch_idx_ptr: dict[int, CpuGpuBuffer] = {}
        for batch_idx_tensor, constraints in zip(
            resident_batch_idx, self._logits_constraint_buffers, strict=True
        ):
            pointer = int(batch_idx_tensor.data_ptr())
            if pointer in self._logits_constraints_by_batch_idx_ptr:
                raise RuntimeError("Whisper resident batch_idx pointers must be unique")
            self._logits_constraints_by_batch_idx_ptr[pointer] = constraints

        prefill_buffers = tuple(
            slot.execution_buffers() for slot in self._prefill_slots
        )
        decode_buffers = tuple(
            WhisperDecodeBuffers(
                slot_id=slot.slot_id,
                token_ids=slot.decode_token_ids,
                input_pos=slot.meta.input_pos.gpu,
                batch_idx=slot.meta.batch_idx.gpu,
                logits_out=slot.logits,
            )
            for slot in self._decode_slots
        )
        validate_resident_buffers(
            config=self._config,
            device=self.device,
            dtype=self.dtype,
            max_batch_size=self.max_batch_size,
            prefill_buffers=prefill_buffers,
            decode_buffers=decode_buffers,
        )
        bindings = WhisperExecutionBindings(
            cross_kv=self.cross_kv,
            self_kv=self.self_kv,
            prefill_buffers=prefill_buffers,
            decode_buffers=decode_buffers,
            max_batch_size=self.max_batch_size,
            compute_stream=self._compute_stream,
        )

        if components.session_factory is None:
            capability = torch.cuda.get_device_capability(self.device)
            if capability[0] not in (9, 10):
                raise RuntimeError(
                    "Optimized Whisper serving currently supports Hopper and "
                    f"Blackwell, got compute capability {capability[0]}."
                    f"{capability[1]}"
                )
            self._prefill_session = NativeWhisperPrefillSession(
                bindings,
                components.weights,
                require_packed=True,
            )
            self._decode_session = create_generated_decode(self, components.weights)
            backend_provenance = _native_provenance(bindings, self._decode_session)
        else:
            self._prefill_session = components.session_factory.create_prefill(bindings)
            self._decode_session = components.session_factory.create_decode(bindings)
            backend_provenance = components.session_factory.native_provenance(
                bindings, self._decode_session
            )
        self._audio_preprocessor = AsyncPreprocessor(
            partial(prepare_audio, config=self._preprocessor_config),
            workers=derive_preprocessing_workers(self.max_batch_size),
        )
        if not isinstance(backend_provenance, dict):
            raise TypeError("Whisper native provenance must be a dict")
        if production and not backend_provenance:
            raise RuntimeError(
                "Production Whisper execution did not declare native provenance"
            )
        if production and (
            backend_provenance.get("prefill", {}).get("aot_required") is not True
            or backend_provenance.get("decode", {}).get("aot_required") is not True
        ):
            raise RuntimeError(
                "Production Whisper provenance must require AOT prefill and decode"
            )
        # Fail during construction if provenance leaks tensors, devices, or
        # another non-serializable implementation detail into this API.
        self._native_provenance_spec = json.loads(json.dumps(backend_provenance))
        self._native_provenance: dict[str, Any] = {}
        self._alignment_decoder = components.weights.decoder

    @property
    def native_provenance(self) -> dict[str, Any]:
        """Return an immutable-by-copy certificate for warmed native serving."""

        return json.loads(json.dumps(self._native_provenance))

    @torch.inference_mode()
    def analyze_transcript(
        self,
        *,
        batch_idx: int,
        language: str,
        task: str,
        prefix_token_ids: Sequence[int],
        text_token_ids: Sequence[int],
        avg_logprob: float,
        duration_seconds: float,
        include_words: bool,
    ) -> TranscriptAnalysis:
        """Score no-speech and optionally align words while its row is owned."""

        if (
            isinstance(batch_idx, bool)
            or not isinstance(batch_idx, int)
            or batch_idx not in self._owned_cross_rows
        ):
            raise RuntimeError("Whisper decoder analysis requires an owned state row")
        if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
            raise ValueError("Whisper decoder analysis requires a positive duration")
        if not math.isfinite(avg_logprob) or avg_logprob > 1e-6:
            raise ValueError("Whisper selected-token average logprob is invalid")
        decoder = self._alignment_decoder
        if decoder is None:
            raise RuntimeError("Whisper decoder analysis is shut down")
        controls = self.tokenizer.controls
        text_ids = tuple(int(token_id) for token_id in text_token_ids)
        if any(not 0 <= token_id < controls.eos_id for token_id in text_ids):
            raise ValueError("Whisper decoder analysis received a non-text token")
        # Encoder positions represent 20 ms of source audio. Keep one real frame
        # for a non-empty admitted clip and exclude right-padding from DTW.
        frames = min(
            self._config.max_source_positions,
            max(1, int(math.ceil(duration_seconds * 50.0))),
        )
        with stream_context(self._compute_stream):
            no_speech_prob = no_speech_probability(
                decoder=decoder,
                tokenizer=self.tokenizer,
                prefix_token_ids=prefix_token_ids,
                cross_keys=self.cross_kv.keys[:, batch_idx : batch_idx + 1],
                cross_values=self.cross_kv.values[:, batch_idx : batch_idx + 1],
                config=self._config,
            )
            words = (
                align_transcript_words(
                    decoder=decoder,
                    tokenizer=self.tokenizer,
                    language=language,
                    task=task,
                    text_token_ids=text_ids,
                    cross_keys=self.cross_kv.keys[:, batch_idx : batch_idx + 1],
                    cross_values=self.cross_kv.values[:, batch_idx : batch_idx + 1],
                    num_frames=frames,
                    config=self._config,
                )
                if include_words
                else ()
            )
        return TranscriptAnalysis(
            words=words,
            scores=TranscriptScores(
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
            ),
        )

    def _create_prefill_slot(self, slot_id: int) -> WhisperPrefillSlot:
        pin = self.device.type == "cuda"
        features = CpuGpuBuffer(
            self.max_batch_size,
            self._config.num_mel_bins,
            self._config.max_source_positions * 2,
            dtype=self.dtype,
            device=self.device,
            pin_memory=pin,
            with_numpy=False,
            zero=False,
        )
        metadata = PackedBuffer(
            [
                (
                    "control_token_ids",
                    (self.max_batch_size, _CONTROL_TOKEN_CAPACITY),
                    torch.int64,
                ),
                ("prefix_lengths", (self.max_batch_size,), torch.int32),
                ("batch_idx", (self.max_batch_size,), torch.int64),
                (
                    "slot_mapping",
                    (self.max_batch_size, _CONTROL_TOKEN_CAPACITY),
                    torch.int64,
                ),
            ],
            device=self.device,
            pin_memory=pin,
        )
        return WhisperPrefillSlot(
            slot_id=slot_id,
            batch_idx=metadata.batch_idx.gpu,
            step_done_event=make_event(
                self.device, enable_timing=False, blocking=False
            ),
            commit_done_event=make_event(
                self.device, enable_timing=False, blocking=False
            ),
            features=features,
            metadata=metadata,
            logits=torch.empty(
                (self.max_batch_size, self.vocab_size),
                dtype=self.dtype,
                device=self.device,
            ),
        )

    def _stage_logits_constraints(
        self,
        *,
        sequences: Sequence[Any],
        batch_idx: Tensor,
    ) -> Tensor:
        """Stage batch-aligned timestamp plans for generic logits processing."""

        batch_size = len(sequences)
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError("Whisper logits-constraint batch is outside capacity")
        if (
            batch_idx.ndim != 1
            or int(batch_idx.shape[0]) != batch_size
            or batch_idx.dtype is not torch.int64
            or batch_idx.device != self.device
        ):
            raise ValueError(
                "Whisper logits constraints require device INT64 batch_idx [B]"
            )

        from .skill import WhisperTranscribeState

        try:
            staging = self._logits_constraints_by_batch_idx_ptr[
                int(batch_idx.data_ptr())
            ]
        except KeyError as exc:
            raise RuntimeError(
                "Whisper sampling batch_idx must alias a resident prefill/decode slot"
            ) from exc
        plans_cpu = staging.cpu
        plans_cpu[:batch_size].zero_()
        for row, sequence in enumerate(sequences):
            state = getattr(sequence, "skill_state", None)
            if not isinstance(state, WhisperTranscribeState):
                raise TypeError(
                    "Whisper logits constraints require WhisperTranscribeState"
                )
            plan = state.timestamp_plan()
            if plan is None:
                continue
            ranges = tuple(plan.suppress_ranges)
            if len(ranges) > 3:
                raise RuntimeError(
                    "Whisper timestamp grammar exceeded three suppression ranges"
                )
            flags = (1 if ranges else 0) | (
                2 if plan.detect_timestamp_from_logprob else 0
            )
            split = int(state.controls.timestamp_begin_id)
            if plan.detect_timestamp_from_logprob and not (0 < split < self.vocab_size):
                raise RuntimeError(
                    "Whisper timestamp grammar produced an invalid partition split"
                )
            plans_cpu[row, 0] = flags
            plans_cpu[row, 1] = split
            for index, (start, end) in enumerate(ranges):
                if not 0 <= start <= end <= self.vocab_size:
                    raise RuntimeError(
                        "Whisper timestamp grammar produced an invalid range"
                    )
                plans_cpu[row, 2 + 2 * index] = start
                plans_cpu[row, 3 + 2 * index] = end
        return staging.copy_to_gpu(batch_size)

    def _process_logits(
        self,
        logits: Tensor,
        *,
        sequences: Sequence[Any],
        batch_idx: Tensor,
    ) -> None:
        """Apply Whisper timestamp plans before ordinary Kestrel sampling.

        All-zero plan rows deliberately travel through the same native kernel:
        automatic-language and forced-control phases are unchanged, while
        transcript rows carry timestamp ranges and the partition-mass rule.
        """

        constraints = self._stage_logits_constraints(
            sequences=sequences,
            batch_idx=batch_idx,
        )
        _SAMPLING.apply_logits_constraints_(
            logits,
            constraints,
            require_packed=self._require_native,
        )

    def _score_sampled_tokens(
        self,
        logits: Tensor,
        *,
        sampled_ids: Tensor,
        token_logprobs: Tensor,
        sequences: Sequence[Any],
        batch_idx: Tensor,
        temperatures: Tensor,
        top_ps: Tensor,
    ) -> None:
        """Stage untempered selected-token model scores after all masks."""

        del sequences, batch_idx, temperatures, top_ps
        if (
            logits.ndim != 2
            or sampled_ids.shape != (logits.shape[0],)
            or token_logprobs.shape != (logits.shape[0],)
            or sampled_ids.dtype is not torch.int64
            or token_logprobs.dtype is not torch.float32
            or sampled_ids.device != logits.device
            or token_logprobs.device != logits.device
        ):
            raise ValueError("Whisper sampled-token scoring received an invalid ABI")
        scores = logits.to(dtype=torch.float32)
        selected = scores.gather(1, sampled_ids.unsqueeze(1)).squeeze(1)
        token_logprobs.copy_(selected - torch.logsumexp(scores, dim=1))

    def _adjust_sampling_params(
        self,
        temperatures: Tensor,
        top_ps: Tensor,
        *,
        sequences: Sequence[Any],
        batch_idx: Tensor,
    ) -> None:
        """Keep automatic language identification greedy on the device."""

        if (
            temperatures.shape != (len(sequences),)
            or top_ps.shape != temperatures.shape
            or temperatures.dtype is not torch.float32
            or top_ps.dtype is not torch.float32
            or temperatures.device != self.device
            or top_ps.device != self.device
            or batch_idx.shape != temperatures.shape
            or batch_idx.dtype is not torch.int64
            or batch_idx.device != self.device
        ):
            raise ValueError("Whisper sampling parameters received an invalid ABI")
        from .skill import WhisperTranscribeState

        language_rows = []
        for row, sequence in enumerate(sequences):
            state = getattr(sequence, "skill_state", None)
            if not isinstance(state, WhisperTranscribeState):
                raise TypeError(
                    "Whisper sampling parameters require WhisperTranscribeState"
                )
            if state.phase == "language":
                language_rows.append(row)
        if not language_rows:
            return
        rows = torch.tensor(language_rows, dtype=torch.int64, device=self.device)
        temperatures.index_fill_(0, rows, 0.0)
        top_ps.index_fill_(0, rows, 1.0)

    def preprocess_image_async(self, image: np.ndarray | bytes) -> Any:
        del image
        raise ValueError("WhisperRuntime does not support images")

    def preprocess_encoder_input_async(self, encoder_input: object) -> Any:
        if not isinstance(encoder_input, AudioSource):
            raise TypeError(
                "Whisper encoder preprocessing requires a validated AudioSource"
            )
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("WhisperRuntime is shut down")
            return self._audio_preprocessor.submit(encoder_input)

    def acquire_prefill_slot(self, slot_id: int | None = None) -> WhisperPrefillSlot:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("WhisperRuntime is shut down")
            if slot_id is None:
                if not self._free_prefill_slot_ids:
                    raise RuntimeError("Whisper prefill slots are exhausted")
                selected = min(self._free_prefill_slot_ids)
            else:
                selected = int(slot_id)
                if selected not in range(len(self._prefill_slots)):
                    raise ValueError(f"Unknown Whisper prefill slot {selected}")
                if selected not in self._free_prefill_slot_ids:
                    raise RuntimeError(
                        f"Whisper prefill slot {selected} is already acquired"
                    )
            self._free_prefill_slot_ids.remove(selected)
            return self._prefill_slots[selected]

    def release_prefill_slot(self, slot: Any) -> None:
        if not isinstance(slot, WhisperPrefillSlot):
            raise TypeError("WhisperRuntime received a foreign prefill slot")
        slot_id = int(slot.slot_id)
        with self._lifecycle_lock:
            if (
                slot_id not in range(len(self._prefill_slots))
                or self._prefill_slots[slot_id] is not slot
            ):
                raise ValueError("WhisperRuntime received a foreign prefill slot")
            if slot_id in self._free_prefill_slot_ids:
                raise RuntimeError(f"Whisper prefill slot {slot_id} was released twice")
            self._free_prefill_slot_ids.add(slot_id)

    def classify_prefill(
        self,
        prompt_tokens: Sequence[Token],
        *,
        has_image: bool = False,
        image_hash: bytes | None = None,
        adapter_id: str | None = None,
    ) -> PrefillClassification:
        if has_image or image_hash is not None:
            raise ValueError("Whisper transcription does not accept images")
        if adapter_id is not None:
            raise ValueError("Whisper transcription does not support adapters")
        return super().classify_prefill(
            prompt_tokens,
            has_image=False,
            image_hash=None,
            adapter_id=None,
        )

    def _control_ids(self, prompt_tokens: Sequence[Token]) -> list[int]:
        if not prompt_tokens:
            raise ValueError("Whisper control prefix must not be empty")
        if any(not isinstance(token, TextToken) for token in prompt_tokens):
            raise TypeError("Whisper control prefix must contain only TextToken")
        token_ids = [int(token.token_id) for token in prompt_tokens]
        controls = self.tokenizer.controls
        valid = False
        if len(token_ids) == 1:
            valid = token_ids == [controls.decoder_start_id]
        elif len(token_ids) in (3, 4):
            valid = (
                token_ids[0] == controls.decoder_start_id
                and token_ids[1] in controls.language_token_ids
                and token_ids[2] in (controls.transcribe_id, controls.translate_id)
                and (len(token_ids) == 3 or token_ids[3] == controls.no_timestamps_id)
            )
        if (
            not valid
            and len(token_ids) in (3, 4)
            and token_ids[0] == controls.prev_sot_id
        ):
            valid = all(
                0 <= token_id < controls.vocab_size and token_id != controls.eos_id
                for token_id in token_ids[1:]
            )
        if not valid:
            raise ValueError(
                "Whisper control prefix must be SOT alone or "
                "SOT/language/task[/no-timestamps], or a validated initial-prompt prefix"
            )
        return token_ids

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
            raise ValueError("Whisper transcription does not accept images")
        if lora_slot != 0 or adapter_id is not None:
            raise ValueError("Whisper transcription does not support adapters")
        if not isinstance(encoder_input, PreparedAudio):
            raise TypeError("Whisper prepare_sequence requires PreparedAudio")
        token_ids = self._control_ids(prompt_tokens)
        if max_new_tokens is None:
            new_tokens = _DEFAULT_TRANSCRIPT_TOKENS
        elif (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise ValueError("Whisper max_new_tokens must be a non-negative integer")
        else:
            new_tokens = int(max_new_tokens)
        target_length = len(token_ids) + new_tokens

        prepared = self._prepare_uncached_sequence(
            tokens=list(prompt_tokens),
            target_length=target_length,
            image_length=0,
            lora_slot=0,
            adapter_id=None,
            image_hash=None,
        )
        row = int(prepared.state.batch_idx)
        try:
            if row in self._owned_cross_rows or row in self._prepared_audio:
                raise RuntimeError(f"Whisper state row {row} is already owned")
            self._owned_cross_rows.add(row)
            self._prepared_audio[row] = encoder_input
        except BaseException:
            super().abort_prepared_sequence(prepared)
            raise
        return prepared

    @staticmethod
    def _validate_optional_batch(
        name: str,
        values: Sequence[object | None] | None,
        batch_size: int,
    ) -> None:
        if values is None:
            return
        if len(values) != batch_size:
            raise ValueError(f"{name} length must match prepared_sequences")
        if any(value is not None for value in values):
            raise ValueError(f"WhisperRuntime does not support {name}")

    def _stage_prefill(
        self,
        slot: WhisperPrefillSlot,
        prepared_sequences: Sequence[PreparedSequence],
        encoder_inputs: Sequence[object | None],
    ) -> None:
        batch_size = len(prepared_sequences)
        controls_cpu = slot.metadata.control_token_ids.cpu
        lengths_cpu = slot.metadata.prefix_lengths.cpu
        batch_idx_cpu = slot.metadata.batch_idx.cpu
        mapping_cpu = slot.metadata.slot_mapping.cpu
        controls_cpu.zero_()
        lengths_cpu.zero_()
        batch_idx_cpu.zero_()
        mapping_cpu.zero_()

        rows: list[int] = []
        for compact_row, (prepared, supplied_audio) in enumerate(
            zip(prepared_sequences, encoder_inputs)
        ):
            row = int(prepared.state.batch_idx)
            if row in rows:
                raise ValueError("Whisper prefill batch contains a duplicate state row")
            rows.append(row)
            retained_audio = self._prepared_audio.get(row)
            if retained_audio is None or supplied_audio is not retained_audio:
                raise RuntimeError(
                    f"Whisper prepared audio ownership mismatch for row {row}"
                )
            if row not in self._owned_cross_rows:
                raise RuntimeError(f"Whisper cross-K/V row {row} is not owned")
            token_ids = self._control_ids(prepared.tokens_list)
            if prepared.state.length != len(token_ids):
                raise RuntimeError("Whisper prepared prefix length drifted")
            pages = self.page_table.page_table_cpu[row]
            if len(pages) < len(token_ids):
                raise RuntimeError("Whisper prefix self-KV pages were not reserved")

            slot.features.cpu[compact_row].copy_(retained_audio.input_features)
            controls_cpu[compact_row, : len(token_ids)] = torch.tensor(
                token_ids, dtype=torch.int64
            )
            lengths_cpu[compact_row] = len(token_ids)
            batch_idx_cpu[compact_row] = row
            # page_size=1, so each physical page index is also the flat slot.
            mapping_cpu[compact_row, : len(token_ids)] = torch.tensor(
                pages[: len(token_ids)], dtype=torch.int64
            )

        self.page_table.commit_block_table(rows)
        slot.features.copy_to_gpu(batch_size)
        slot.metadata.copy_to_gpu()

    @torch.inference_mode()
    def launch_prepared_batch(
        self,
        prepared_sequences: Sequence[PreparedSequence],
        prefill_slot: Any,
        *,
        images: Sequence[np.ndarray | None] | None = None,
        image_crops_list: Sequence[Any] | None = None,
        encoder_inputs: Sequence[object | None] | None = None,
    ) -> Tensor:
        batch_size = len(prepared_sequences)
        if batch_size == 0:
            raise ValueError("prepared_sequences must be non-empty")
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Whisper prefill batch {batch_size} exceeds "
                f"max_batch_size={self.max_batch_size}"
            )
        self._validate_optional_batch("images", images, batch_size)
        self._validate_optional_batch("image_crops_list", image_crops_list, batch_size)
        if encoder_inputs is None or len(encoder_inputs) != batch_size:
            raise ValueError("encoder_inputs must match the Whisper prefill batch")
        if not isinstance(prefill_slot, WhisperPrefillSlot):
            raise TypeError("WhisperRuntime received a foreign prefill slot")
        slot_id = int(prefill_slot.slot_id)
        if (
            slot_id not in range(len(self._prefill_slots))
            or self._prefill_slots[slot_id] is not prefill_slot
        ):
            raise ValueError("WhisperRuntime received a foreign prefill slot")
        if slot_id in self._free_prefill_slot_ids:
            raise RuntimeError("Whisper prefill slot must be acquired before launch")

        with stream_context(self._compute_stream):
            self._stage_prefill(
                prefill_slot,
                prepared_sequences,
                encoder_inputs,
            )
            # Contract: this call enqueues every decoder-visible write, including
            # cross-K/V and prefix self-KV, before it returns.
            self._prefill_session.launch(slot_id, batch_size)
        return prefill_slot.logits[:batch_size]

    def finalize_prepared_sequence_after_prefill(
        self, prepared: PreparedSequence
    ) -> None:
        row = int(prepared.state.batch_idx)
        if row not in self._owned_cross_rows:
            raise RuntimeError(f"Whisper cross-K/V row {row} is not owned")
        if row not in self._prepared_audio:
            raise RuntimeError(f"Whisper prepared audio row {row} is missing")
        super().finalize_prepared_sequence_after_prefill(prepared)
        # The scheduler synchronizes prefill before this host-only finalizer.
        self._prepared_audio.pop(row)

    def _release_runtime_state(self, batch_idx: int) -> None:
        self._prepared_audio.pop(batch_idx, None)
        self._owned_cross_rows.discard(batch_idx)

    @torch.inference_mode()
    def decode_with_slot(self, slot: DecodeSlot, batch_size: int) -> None:
        if batch_size == 0:
            return
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError("Whisper decode batch size is outside runtime capacity")
        slot_id = int(slot.slot_id)
        if (
            slot_id not in range(len(self._decode_slots))
            or self._decode_slots[slot_id] is not slot
        ):
            raise ValueError("WhisperRuntime received a foreign decode slot")
        launch_capacity = next(
            capacity
            for capacity in self._decode_batch_capacities
            if capacity >= batch_size
        )
        # Generated programs are compiled for B1/B2/B4/B8. Tail lanes address
        # the globally reserved no-op row/page and are ignored by the scheduler.
        # Staging and launch share the captured compute stream, so no host sync
        # or device-value read is introduced for non-power-of-two batches.
        with stream_context(self._compute_stream):
            if launch_capacity != batch_size:
                slot.decode_token_ids[batch_size:launch_capacity].zero_()
                slot.meta.input_pos.gpu[batch_size:launch_capacity].zero_()
                slot.meta.batch_idx.gpu[batch_size:launch_capacity].fill_(
                    self._padding_batch_idx
                )
            # Resident slot tensors were bound at session construction. The
            # generated launch reads token_ids/input_pos/batch_idx and writes logits.
            self._decode_session.run(slot, launch_capacity)

    def _warmup_decode(self) -> None:
        slot = self._decode_slots[0]
        with stream_context(self._compute_stream):
            for capacity in self._decode_batch_capacities:
                slot.decode_token_ids[:capacity].zero_()
                slot.meta.input_pos.gpu[:capacity].zero_()
                slot.meta.batch_idx.gpu[:capacity].zero_()
                self._decode_session.run(slot, capacity)
                if self._compute_stream is not None:
                    self._compute_stream.synchronize()

    def warmup(self) -> None:
        """Warm each execution path once; successful paths are not repeated."""

        with self._warmup_lock:
            if self._closed:
                raise RuntimeError("WhisperRuntime is shut down")
            if not self._prefill_warmed:
                self._prefill_session.warmup()
                self._prefill_warmed = True
            if not self._constraints_warmed:
                logits = torch.zeros(
                    (1, self.vocab_size),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                constraints = torch.zeros(
                    (1, _CONSTRAINT_PLAN_WIDTH),
                    dtype=torch.int32,
                    device=self.device,
                )
                with capture_packed_artifact_receipts() as receipt_capture:
                    with stream_context(self._compute_stream):
                        _SAMPLING.apply_logits_constraints_(
                            logits,
                            constraints,
                            require_packed=self._require_native,
                        )
                        if self._require_native:
                            _SAMPLING.greedy_logprobs_from_logits(
                                logits,
                                out=torch.empty(
                                    (1,), dtype=torch.int64, device=self.device
                                ),
                                logprobs_out=torch.empty(
                                    (1,), dtype=torch.float32, device=self.device
                                ),
                                require_packed=True,
                            )
                    if self._compute_stream is not None:
                        # A packed receipt certifies a completed warmup launch,
                        # not merely a host enqueue. Keep this synchronization
                        # inside the transaction so an asynchronous failure
                        # aborts the receipt and leaves the sampling half
                        # retryable.
                        self._compute_stream.synchronize()
                self._sampling_artifact_receipts = receipt_capture.receipts
                self._constraints_warmed = True
            if not self._decode_warmed:
                self._warmup_decode()
                self._decode_warmed = True
            if self._require_native and not self._native_provenance:
                if not self._native_provenance_spec:
                    raise RuntimeError("Whisper native provenance is unavailable")
                prefill_receipts = _validated_packed_receipts(
                    self._prefill_session.artifact_receipts,
                    expected_families={"flash_attn", "gelu", "gelu_add"},
                    component="prefill",
                )
                sampling_receipts = _validated_packed_receipts(
                    self._sampling_artifact_receipts,
                    expected_families={"softmax_greedy"},
                    component="sampling",
                )
                prefill_provenance = {
                    **self._native_provenance_spec["prefill"],
                    "artifact_receipts": prefill_receipts,
                }
                self._native_provenance = {
                    **self._native_provenance_spec,
                    "prefill": prefill_provenance,
                    "sampling": {
                        "backend": (
                            "kestrel_kernels.sampling.apply_logits_constraints_"
                        ),
                        "native_kernel_required": True,
                        "constraint_plan_dtype": "int32",
                        "constraint_plan_width": _CONSTRAINT_PLAN_WIDTH,
                        "artifact_receipts": sampling_receipts,
                    },
                }

    def shutdown(self) -> None:
        """Stop CPU workers, fence queued GPU work, and release native state."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True

        errors: list[BaseException] = []
        try:
            self._audio_preprocessor.shutdown()
        except BaseException as exc:
            errors.append(exc)
        try:
            if self._compute_stream is not None:
                self._compute_stream.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except BaseException as exc:
            errors.append(exc)
        try:
            self._prefill_session.shutdown()
        except BaseException as exc:
            errors.append(exc)
        self._decode_session = None
        self._alignment_decoder = None
        owned_rows = set(self.active_sequences).union(self._owned_cross_rows)
        for row in owned_rows:
            if row not in self.page_table.free_batch_idx:
                self.page_table.erase(row, 0)
        self.active_sequences.clear()
        self._prepared_audio.clear()
        self._owned_cross_rows.clear()
        if errors:
            raise errors[0]


__all__ = [
    "WhisperPrefillSlot",
    "WhisperRuntime",
    "WhisperRuntimeComponents",
]
