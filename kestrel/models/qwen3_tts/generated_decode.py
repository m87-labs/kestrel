"""Bindings for compiler-generated Qwen3-TTS Talker decode."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable

from kestrel.runtime.generated_decode import (
    GeneratedDecode,
    GeneratedDecodeSpec,
    PagedDecodeBindings,
)

from .config import NUM_CODE_GROUPS


_DECODE_CAPACITIES = (1, 2, 4, 8, 16, 32)
_ADMISSION_BANDS = ((32, 24),)
TALKER_GENERATED_PREFILL_LENGTH = 10


def _program_names(
    stem: str,
    suffix: str = "",
    batch_sizes: Iterable[int] = _DECODE_CAPACITIES,
    admission_bands: Iterable[tuple[int, int]] = (),
) -> frozenset[str]:
    names = frozenset(
        f"{stem}{suffix}_b{batch_size}"
        for batch_size in batch_sizes
    )
    return names | frozenset(
        f"{stem}{suffix}_b{capacity}_max{maximum}"
        for capacity, maximum in admission_bands
    )


def _packed_rows(batch_size: int) -> int:
    return int(batch_size) * TALKER_GENERATED_PREFILL_LENGTH


def create_generated_decode(runtime: Any) -> GeneratedDecode:
    weight_sources = dict(
        runtime.talker.named_parameters(remove_duplicate=False)
    )
    weight_sources["projected_text_embedding"] = (
        runtime._projected_text_embedding
    )
    spec = GeneratedDecodeSpec(
        label="Qwen3-TTS Talker",
        weight_root=runtime.talker,
        weight_layer_prefix="model.layers",
        weight_sources=weight_sources,
        bindings=PagedDecodeBindings(
            runtime._paged_kv,
            extra_runtime_inputs=lambda bound: {
                "rope_cosine": bound._rope_cosine,
                "rope_sine": bound._rope_sine,
            },
            extra_slot_inputs=lambda slot, capacity: {
                "input_embedding": slot.scratch["decode_embeddings"][:capacity],
                "additive_embedding_ids": slot.text_token_ids.gpu[:capacity],
            },
        ),
        program_names=_program_names(
            runtime.config.generated_name,
            admission_bands=_ADMISSION_BANDS,
        ),
    )
    return GeneratedDecode.require(
        runtime,
        spec,
        batch_sizes=range(1, runtime.max_batch_size + 1),
    )


@dataclass(frozen=True, slots=True)
class _TalkerPrefillBindings:
    runtime: Any

    def is_eligible(self, _runtime: Any) -> bool:
        return all(
            layer is not None
            and int(layer.k_cache.shape[2]) == int(layer.v_cache.shape[2]) == 1
            for layer in self.runtime._paged_kv
        )

    def runtime_inputs(self, _runtime: Any) -> dict[str, Any]:
        return {
            "page_table": self.runtime.page_table.page_table,
            "mK": [layer.k_cache[:, :, 0] for layer in self.runtime._paged_kv],
            "mV": [layer.v_cache[:, :, 0] for layer in self.runtime._paged_kv],
            "rope_cosine": self.runtime._rope_cosine,
            "rope_sine": self.runtime._rope_sine,
            "kv_len": TALKER_GENERATED_PREFILL_LENGTH,
        }

    @staticmethod
    def slot_inputs(slot: Any, capacity: int) -> dict[str, Any]:
        scratch = slot.scratch
        return {
            "input_embedding": scratch["prompt_embeddings"][:capacity],
            "input_pos": scratch["prefill_input_pos"][:capacity],
            "final_norm": scratch["prefill_hidden"][:capacity],
            "logits": scratch["prefill_logits"][:capacity],
            "batch_idx": slot.packed_batch_indices.gpu[:capacity],
        }

    @staticmethod
    def launch_extents(_slot: Any, batch_size: int) -> dict[str, int]:
        return {
            "active_batch": int(batch_size),
            "kv_len": TALKER_GENERATED_PREFILL_LENGTH,
        }


def create_talker_generated_prefill(runtime: Any) -> GeneratedDecode:
    """Bind the ten-row packed CustomVoice prompt program."""

    prefill_runtime = SimpleNamespace(
        device=runtime.device,
        dtype=runtime.dtype,
        max_batch_size=_packed_rows(runtime.max_batch_size),
        compute_stream=runtime._compute_stream,
        decode_slots=runtime.prefill_slots,
    )
    return GeneratedDecode.require(
        prefill_runtime,
        GeneratedDecodeSpec(
            label="Qwen3-TTS Talker packed prefill",
            weight_root=runtime.talker,
            weight_layer_prefix="model.layers",
            weight_storage=runtime.generated_decode.weight_storage,
            bindings=_TalkerPrefillBindings(runtime),
            program_names=_program_names(
                runtime.config.generated_name,
                "_packed_prefill10",
                range(1, runtime.max_batch_size + 1),
            ),
        ),
        batch_sizes=tuple(
            _packed_rows(batch_size)
            for batch_size in range(1, runtime.max_batch_size + 1)
        ),
    )


@dataclass(frozen=True, slots=True)
class _PredictorBindings:
    runtime: Any
    stage: str

    def is_eligible(self, _runtime: Any) -> bool:
        return all(
            layer is not None
            and int(layer.k_cache.shape[2]) == int(layer.v_cache.shape[2]) == 1
            for layer in self.runtime._predictor_paged_kv
        )

    def runtime_inputs(self, _runtime: Any) -> dict[str, Any]:
        return {
            "page_table": self.runtime._predictor_page_table.page_table,
            "mK": [layer.k_cache[:, :, 0] for layer in self.runtime._predictor_paged_kv],
            "mV": [layer.v_cache[:, :, 0] for layer in self.runtime._predictor_paged_kv],
            "rope_cosine": self.runtime._predictor_rope_cosine,
            "rope_sine": self.runtime._predictor_rope_sine,
            "kv_len": 1 if self.stage == "prime" else NUM_CODE_GROUPS,
        }

    def slot_inputs(self, slot: Any, capacity: int) -> dict[str, Any]:
        scratch = slot.scratch
        common = {
            "batch_idx": self.runtime._predictor_batch_idx[:capacity],
            "codec_sum": scratch["predictor_codec_sum"][:capacity],
            "cp_seed": scratch["predictor_seed"][:capacity],
            "input_pos": scratch["predictor_input_pos"][:capacity],
        }
        if self.stage == "prime":
            return {
                **common,
                "past_hidden": getattr(
                    slot,
                    "hidden_last",
                    scratch["predictor_past_hidden"],
                )[:capacity],
                "code0": scratch["predictor_code0"][:capacity],
            }
        return {
            **common,
            "cp_seed_next": common["cp_seed"],
            "codec_sum_next": common["codec_sum"],
            "frames": scratch["predictor_frames"][:capacity],
            "uniforms": scratch["predictor_uniforms"][:capacity],
            "temperature": scratch["predictor_temperature"][:capacity],
            "top_k": scratch["predictor_top_k"][:capacity],
            "top_p": scratch["predictor_top_p"][:capacity],
        }

    def launch_extents(self, _slot: Any, batch_size: int) -> dict[str, int]:
        return {
            "active_batch": int(batch_size),
            "kv_len": 1 if self.stage == "prime" else NUM_CODE_GROUPS,
        }


def create_predictor_generated_decode(
    runtime: Any,
) -> tuple[GeneratedDecode, GeneratedDecode]:
    """Bind the predictor's one-step prime and fixed 15-step residual loop."""

    slots = (*runtime.decode_slots, *runtime.prefill_slots)
    predictor_runtime = SimpleNamespace(
        device=runtime.device,
        dtype=runtime.dtype,
        max_batch_size=runtime.max_batch_size,
        compute_stream=runtime._compute_stream,
        decode_slots=slots,
    )
    prime_sources = dict(
        runtime.code_predictor.named_parameters(remove_duplicate=False)
    )
    prime_sources["talker_codec_embedding.weight"] = (
        runtime.talker.model.codec_embedding.weight
    )
    prime = GeneratedDecode.require(
        predictor_runtime,
        GeneratedDecodeSpec(
            label="Qwen3-TTS Code Predictor prime",
            weight_root=runtime.code_predictor,
            weight_layer_prefix="model.layers",
            weight_sources=prime_sources,
            bindings=_PredictorBindings(runtime, "prime"),
            program_names=_program_names(
                f"{runtime.config.generated_name}_code_predictor",
                "_prime",
                admission_bands=_ADMISSION_BANDS,
            ),
        ),
        batch_sizes=range(1, runtime.max_batch_size + 1),
    )
    residual = GeneratedDecode.require(
        predictor_runtime,
        GeneratedDecodeSpec(
            label="Qwen3-TTS Code Predictor residual",
            weight_root=runtime.code_predictor,
            weight_layer_prefix="model.layers",
            weight_storage=prime.weight_storage,
            bindings=_PredictorBindings(runtime, "residual"),
            program_names=_program_names(
                f"{runtime.config.generated_name}_code_predictor",
                "_residual",
                admission_bands=_ADMISSION_BANDS,
            ),
        ),
        batch_sizes=range(1, runtime.max_batch_size + 1),
    )
    return prime, residual


__all__ = [
    "TALKER_GENERATED_PREFILL_LENGTH",
    "create_generated_decode",
    "create_predictor_generated_decode",
    "create_talker_generated_prefill",
]
