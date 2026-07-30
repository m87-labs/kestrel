"""Gemma runtime bindings for the compiler-generated decode device program."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class _UnsupportedDecodeConfig(ValueError):
    """The shipped decode program does not cover this model configuration."""


def _compile_from_config(
    config: Any,
    *,
    batch_capacity: int = 1,
    max_kv_len: int,
    num_ctas: int,
    gpu: str,
):
    from mkl.compiler.frontend.models.gemma import compile_gemma4_decode

    if bool(config.enable_moe_block):
        raise _UnsupportedDecodeConfig("generated Gemma decode currently covers dense MLPs")
    if bool(config.attention_bias):
        raise _UnsupportedDecodeConfig("generated Gemma decode requires bias-free attention")
    if bool(config.attention_k_eq_v):
        raise _UnsupportedDecodeConfig("generated Gemma decode requires an explicit V projection")
    if str(config.hidden_activation) != "gelu_pytorch_tanh":
        raise _UnsupportedDecodeConfig(
            f"unsupported Gemma activation {config.hidden_activation!r}")
    ple_hidden = int(config.hidden_size_per_layer_input or 0)
    if ple_hidden <= 0:
        raise _UnsupportedDecodeConfig("generated Gemma decode requires traced PLE inputs")
    layer_types = tuple(map(str, config.layer_types or ()))
    supported_types = {"sliding_attention", "full_attention"}
    if not layer_types or set(layer_types) - supported_types:
        raise _UnsupportedDecodeConfig(
            f"unsupported Gemma layer types {sorted(set(layer_types))}")
    first_shared = len(layer_types) - int(config.num_kv_shared_layers)
    branches = {
        f"{kind}_{'shared' if layer >= first_shared else 'fresh'}"
        for layer, layer_type in enumerate(layer_types)
        for kind in (
            "global" if layer_type == "full_attention" else "local",
        )
    }
    required = {"local_fresh", "global_fresh", "local_shared", "global_shared"}
    if branches != required:
        raise _UnsupportedDecodeConfig(
            "generated Gemma decode requires fresh/shared local/global coverage")

    return compile_gemma4_decode(
        layer_types=list(layer_types),
        num_kv_shared_layers=int(config.num_kv_shared_layers),
        hidden=int(config.hidden_size),
        inter=int(config.intermediate_size),
        nh=int(config.num_attention_heads),
        nkv=int(config.num_key_value_heads),
        global_nkv=int(config.num_key_value_heads),
        local_head_dim=int(config.head_dim),
        global_head_dim=int(config.global_head_dim),
        window=int(config.sliding_window),
        max_kv_len=int(max_kv_len),
        ple_hidden=ple_hidden,
        ple_vocab=int(config.vocab_size_per_layer_input),
        vocab_size=int(config.vocab_size),
        rms_norm_eps=float(config.rms_norm_eps),
        final_logit_softcapping=config.final_logit_softcapping,
        tie_word_embeddings=bool(config.tie_word_embeddings),
        double_wide_mlp=bool(config.use_double_wide_mlp),
        num_ctas=int(num_ctas),
        num_splits=None,
        batch_tile=int(batch_capacity),
        static_extent_bindings={},
        gpu=str(gpu),
    )


def _supports_paged_decode_abi(paged_layers: Any) -> bool:
    return bool(paged_layers) and all(
        layer is None
        or (
            int(layer.k_cache.shape[2]) == 1
            and int(layer.v_cache.shape[2]) == 1
        )
        for layer in paged_layers
    )


def _paged_tensors(
    paged_layers: Any,
    layer_types: list[str],
    *,
    kind: str,
    field: str,
) -> list[torch.Tensor | None]:
    values: list[torch.Tensor | None] = []
    for layer_type, layer in zip(layer_types, paged_layers, strict=True):
        if layer is None or layer_type != kind:
            values.append(None)
            continue
        tensor = getattr(layer, field)
        if int(tensor.shape[2]) != 1:
            raise ValueError(
                "generated decode requires unit KV pages, got "
                f"shape {tuple(tensor.shape)}")
        values.append(tensor[:, :, 0, :])
    return values


def _rope_tables(text_model: Any, *, capacity: int, dtype: torch.dtype, device):
    positions = torch.arange(capacity, dtype=torch.int64, device=device).view(1, -1)
    probe = torch.empty((1, 1, 1), dtype=dtype, device=device)
    tables = {}
    for kind in ("sliding_attention", "full_attention"):
        cos, sin = text_model.rotary_emb(probe, positions, kind)
        # The model rounds the FP32 trigonometric result to its activation dtype.
        # Preserve those rounded values while satisfying the generated FP32 ABI.
        tables[kind] = (cos[0].float().contiguous(), sin[0].float().contiguous())
    return tables


@dataclass
class _SlotInvocation:
    invocation: Any


class Gemma4DecodeMegakernel:
    """Bind Gemma-owned buffers to compiler-generated decode capacity domains."""

    @classmethod
    def try_create(cls, runtime: Any) -> Gemma4DecodeMegakernel | None:
        if runtime.device.type != "cuda":
            return None
        if getattr(runtime, "dtype", torch.bfloat16) is not torch.bfloat16:
            return None
        if not _supports_paged_decode_abi(
            getattr(runtime, "_shared_paged_layers", None)
        ):
            return None

        try:
            from mkl.compiler.frontend.gpu_model import CalibrationUnavailable
            from mkl.compiler.frontend.models.aot import DECODE_BATCH_CAPACITIES
            from mkl.compiler.frontend.validate_program import validate_compiled_tape
            from mkl.megakernel.device_runtime import (
                DeviceRuntimeError,
                resolve_capacity_programs,
                resolve_shipped_aot_bundle,
            )
        except ModuleNotFoundError as exc:
            missing = str(exc.name or "")
            if missing != "mkl" and not missing.startswith("mkl."):
                raise
            return None

        properties = torch.cuda.get_device_properties(runtime.device)
        arch = f"sm{properties.major}{properties.minor}"
        try:
            programs = resolve_capacity_programs(
                DECODE_BATCH_CAPACITIES,
                max_active_extent=int(getattr(runtime, "max_batch_size", 1)),
                compile_program=lambda batch_capacity: _compile_from_config(
                    runtime.model.model.language_model.config,
                    batch_capacity=batch_capacity,
                    max_kv_len=int(runtime.max_seq_length),
                    num_ctas=int(properties.multi_processor_count),
                    gpu=str(properties.name),
                ),
                validate_program=validate_compiled_tape,
                resolve_bundle=resolve_shipped_aot_bundle,
                arch=arch,
            )
        except _UnsupportedDecodeConfig:
            return None
        except CalibrationUnavailable as exc:
            raise DeviceRuntimeError(
                "generated Gemma decode has no calibration for "
                f"{properties.name!r}: {exc}"
            ) from exc
        if not programs:
            return None
        return cls(runtime, programs=programs)

    def __init__(
        self,
        runtime: Any,
        *,
        programs: dict[int, tuple[Any, Any, Any]],
    ) -> None:
        from mkl.compiler.frontend import bind_owned_weight_storage
        from mkl.megakernel.device_runtime import (
            assemble_torch_device_bindings,
            bind_aot_device_program,
        )
        from mkl.megakernel.device_input_preparation import (
            derive_device_input_preparation_plan,
        )

        config = runtime.model.model.language_model.config
        self._programs = dict(sorted(programs.items()))
        first_compiled = next(iter(self._programs.values()))[0]
        self.compiled = first_compiled
        self.text_model = runtime.model.model.language_model
        ambient_stream = torch.cuda.current_stream(runtime.device)
        self._model_weights_ready = torch.cuda.Event()
        self._model_weights_ready.record(ambient_stream)
        runtime.primary_stream.wait_event(self._model_weights_ready)
        self._weight_storage_ready = torch.cuda.Event()
        with torch.cuda.stream(runtime.primary_stream):
            self.weight_storage = bind_owned_weight_storage(
                first_compiled,
                runtime.model,
                device=runtime.device,
                layer_prefix="model.language_model.layers",
            )
            rope_tables = _rope_tables(
                self.text_model,
                capacity=int(runtime.max_seq_length),
                dtype=runtime.dtype,
                device=runtime.device,
            )
            self._weight_storage_ready.record(runtime.primary_stream)
        ambient_stream.wait_event(self._weight_storage_ready)

        layer_types = list(config.layer_types)
        paged_layers = runtime._shared_paged_layers
        local_k = _paged_tensors(
            paged_layers, layer_types, kind="sliding_attention", field="k_cache")
        local_v = _paged_tensors(
            paged_layers, layer_types, kind="sliding_attention", field="v_cache")
        global_k = _paged_tensors(
            paged_layers, layer_types, kind="full_attention", field="k_cache")
        global_v = _paged_tensors(
            paged_layers, layer_types, kind="full_attention", field="v_cache")
        local_cos, local_sin = rope_tables["sliding_attention"]
        global_cos, global_sin = rope_tables["full_attention"]
        authoritative_page_table = runtime.page_table.page_table
        n_pages = int(runtime.page_table.n_pages)

        preparation_plans = []
        self._slots = {}
        for slot in runtime.decode_slots:
            for batch_capacity, program in self._programs.items():
                compiled, validated, aot_bundle = program
                runtime_inputs = {
                    "input_ids": slot.decode_token_ids[:batch_capacity],
                    "final_norm": slot.hidden_last[:batch_capacity],
                    "logits": slot.logits[:batch_capacity],
                    "page_table": authoritative_page_table,
                    "batch_idx": slot.meta.batch_idx.gpu[:batch_capacity],
                    "input_pos": slot.meta.input_pos.gpu[:batch_capacity],
                    "kv_len": 1,
                    "rope_cos_local": local_cos,
                    "rope_sin_local": local_sin,
                    "rope_cos_global": global_cos,
                    "rope_sin_global": global_sin,
                    "mK_local": local_k,
                    "mV_local": local_v,
                    "mK_global": global_k,
                    "mV_global": global_v,
                }
                preparation_plans.append(
                    derive_device_input_preparation_plan(
                        compiled.device_program,
                        ready_inputs=runtime_inputs,
                        preparations=(),
                    )
                )
                extents = {
                    "active_batch": batch_capacity,
                    "n_pages": n_pages,
                    "page_table_capacity": int(authoritative_page_table.shape[1]),
                    "position_capacity": int(runtime.max_seq_length),
                    "state_rows": int(authoritative_page_table.shape[0]),
                }
                bindings = assemble_torch_device_bindings(
                    compiled,
                    bound_weights=self.weight_storage.buffers,
                    runtime_inputs=runtime_inputs,
                    runtime_extents=extents,
                    stream=slot.compute_stream,
                    device=runtime.device,
                )
                invocation = bind_aot_device_program(
                    validated, aot_bundle, bindings.values)
                self._slots[(int(slot.slot_id), batch_capacity)] = _SlotInvocation(
                    invocation=invocation,
                )
        if any(preparation_plans):
            raise RuntimeError(
                "generated Gemma decode unexpectedly requires derived inputs")

    def _capacity_for(self, batch_size: int) -> int | None:
        from mkl.megakernel.device_runtime import select_compiled_capacity

        return select_compiled_capacity(self._programs, batch_size)

    def supports(self, batch_size: int) -> bool:
        return self._capacity_for(batch_size) is not None

    @torch.inference_mode()
    def run(self, slot: Any, batch_size: int = 1) -> None:
        batch_size = int(batch_size)
        batch_capacity = self._capacity_for(batch_size)
        if batch_capacity is None:
            raise ValueError(
                f"no generated decode capacity covers batch size {batch_size}")
        state = self._slots[(int(slot.slot_id), batch_capacity)]
        kv_len = int(slot.meta.input_pos.cpu[:batch_size].max()) + 1
        state.invocation.launch(active_batch=batch_size, kv_len=kv_len)


__all__ = ["Gemma4DecodeMegakernel"]
