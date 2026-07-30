"""Qwen runtime bindings for the compiler-generated decode device program."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class _UnsupportedDecodeConfig(ValueError):
    """The shipped decode program does not cover this model configuration."""


def _layer_kinds(config: Any) -> list[int]:
    mapping = {"linear_attention": 0, "full_attention": 1}
    try:
        return [mapping[str(kind)] for kind in config.layer_types]
    except KeyError as exc:
        raise _UnsupportedDecodeConfig(
            f"unsupported Qwen decode layer type {exc.args[0]!r}"
        ) from exc


def _rope_config(config: Any) -> tuple[float, list[int]]:
    rope = config.rope_parameters or {}
    partial = float(rope.get("partial_rotary_factor", 0.25))
    sections = [int(value) for value in rope.get("mrope_section", ())]
    if len(sections) != 3:
        raise _UnsupportedDecodeConfig(
            f"Qwen decode needs three M-RoPE sections, got {sections}")
    return partial, sections


def _compile_from_config(
    config: Any,
    *,
    batch_capacity: int = 1,
    num_ctas: int,
    gpu: str,
):
    from mkl.compiler.frontend.models.qwen import compile_qwen35

    if config.is_moe:
        raise _UnsupportedDecodeConfig(
            "the current Qwen decode trace covers dense MLP layers"
        )
    partial_rotary, rope_sections = _rope_config(config)
    return compile_qwen35(
        n_layers=int(config.num_hidden_layers),
        hidden=int(config.hidden_size),
        inter=int(config.intermediate_size),
        nh=int(config.num_attention_heads),
        nkv=int(config.num_key_value_heads),
        head_dim=int(config.head_dim),
        num_k_heads=int(config.linear_num_key_heads),
        num_v_heads=int(config.linear_num_value_heads),
        key_head_dim=int(config.linear_key_head_dim),
        value_head_dim=int(config.linear_value_head_dim),
        conv_kernel=int(config.linear_conv_kernel_dim),
        partial_rotary=partial_rotary,
        rope_sections=rope_sections,
        vocab_size=int(config.vocab_size),
        rms_norm_eps=float(config.rms_norm_eps),
        tie_word_embeddings=bool(config.tie_word_embeddings),
        num_ctas=int(num_ctas),
        num_splits=None,
        max_kv_len=int(config.max_position_embeddings),
        batch_tile=int(batch_capacity),
        static_extent_bindings={},
        gpu=str(gpu),
        layer_types=_layer_kinds(config),
    )


def _state_tensors(state_pool: Any, field: str) -> list[torch.Tensor | None]:
    return [
        None if storage is None else getattr(storage, field)
        for storage in state_pool.layers
    ]


def _paged_tensors(
    paged_layers: Any,
    field: str,
) -> list[torch.Tensor | None]:
    result = []
    for layer in paged_layers:
        if layer is None:
            result.append(None)
            continue
        tensor = getattr(layer, field)
        if tensor.shape[2] != 1:
            raise ValueError(
                "the current decode page ABI requires page_size=1, got "
                f"shape {tuple(tensor.shape)}")
        result.append(tensor[:, :, 0, :])
    return result


def _supports_paged_decode_abi(paged_layers: Any) -> bool:
    """Whether every live KV layer has the generated program's unit-page view."""

    return all(
        layer is None
        or (
            int(layer.k_cache.shape[2]) == 1
            and int(layer.v_cache.shape[2]) == 1
        )
        for layer in paged_layers
    )


@dataclass
class _SlotInvocation:
    binding_set: Any
    invocation: Any
    uses_kv_len: bool


class Qwen35DecodeMegakernel:
    """Bind Qwen-owned buffers to compiler-generated decode capacity domains."""

    @classmethod
    def try_create(cls, runtime: Any) -> Qwen35DecodeMegakernel | None:
        """Return the exact shipped program, or ``None`` for native fallback."""

        if getattr(runtime, "device", None) is None or runtime.device.type != "cuda":
            return None
        if getattr(runtime, "dtype", torch.bfloat16) is not torch.bfloat16:
            return None
        if not _supports_paged_decode_abi(
            getattr(runtime, "_shared_paged_layers", ())
        ):
            return None

        try:
            from mkl.compiler.frontend.gpu_model import CalibrationUnavailable
            from mkl.compiler.frontend.models.aot import DECODE_BATCH_CAPACITIES
            from mkl.compiler.frontend.validate_program import (
                validate_compiled_tape,
            )
            from mkl.megakernel.device_runtime import (
                DeviceRuntimeError,
                resolve_capacity_programs,
                resolve_shipped_aot_bundle,
            )
        except ModuleNotFoundError as exc:
            missing = str(exc.name or "")
            if missing != "mkl":
                raise
            return None

        config = runtime.model.model.language_model.config
        device_properties = torch.cuda.get_device_properties(runtime.device)
        num_ctas = device_properties.multi_processor_count
        arch = f"sm{device_properties.major}{device_properties.minor}"
        try:
            programs = resolve_capacity_programs(
                DECODE_BATCH_CAPACITIES,
                max_active_extent=int(getattr(runtime, "max_batch_size", 1)),
                compile_program=lambda batch_capacity: _compile_from_config(
                    config,
                    batch_capacity=batch_capacity,
                    num_ctas=num_ctas,
                    gpu=device_properties.name,
                ),
                validate_program=validate_compiled_tape,
                resolve_bundle=resolve_shipped_aot_bundle,
                arch=arch,
            )
        except _UnsupportedDecodeConfig:
            return None
        except CalibrationUnavailable as exc:
            raise DeviceRuntimeError(
                "generated Qwen decode has no calibration for "
                f"{device_properties.name!r}: {exc}"
            ) from exc
        if not programs:
            return None
        runtime._linear_state_pool.initialize_from_config(
            config, dtype=runtime.dtype
        )
        return cls(runtime, programs=programs)

    def __init__(
        self,
        runtime: Any,
        *,
        programs: dict[int, tuple[Any, Any, Any]],
    ) -> None:
        from mkl.compiler.frontend import (
            bind_owned_weight_storage,
            derive_device_carried_state_contracts,
        )
        from mkl.megakernel.device_runtime import (
            assemble_torch_device_bindings,
            bind_aot_device_program,
        )
        from mkl.megakernel.device_input_preparation import (
            DeviceInputPreparation,
            derive_device_input_preparation_plan,
        )
        from mkl.megakernel.state_runtime import StateRepresentationRequirement

        if runtime.device.type != "cuda":
            raise ValueError("generated decode requires a CUDA runtime")
        config = runtime.model.model.language_model.config
        self._programs = dict(sorted(programs.items()))
        first_compiled = next(iter(self._programs.values()))[0]
        self.compiled = first_compiled
        requirements_by_capacity = {
            batch_capacity: tuple(
                StateRepresentationRequirement(
                    contract.buffer,
                    contract.representation,
                    contract.storage_axis_order,
                    contract.storage_dtype,
                )
                for contract in derive_device_carried_state_contracts(
                    compiled.graph, compiled.device_program)
            )
            for batch_capacity, (compiled, _validated, _bundle)
            in self._programs.items()
        }
        state_buffers_by_capacity = {
            tuple(requirement.buffer for requirement in requirements)
            for requirements in requirements_by_capacity.values()
        }
        if len(state_buffers_by_capacity) != 1:
            raise RuntimeError(
                "generated decode batch domains disagree on carried state buffers")
        self.state_buffers = next(iter(state_buffers_by_capacity))
        self.state_requirements_by_capacity = requirements_by_capacity
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
            self._weight_storage_ready.record(runtime.primary_stream)
        # Decode and native prefill use primary_stream. Also order the ambient stream so
        # setup or a future consumer cannot observe partially copied physical slabs.
        torch.cuda.current_stream(runtime.device).wait_event(
            self._weight_storage_ready)
        bound_weights = self.weight_storage.buffers

        conv_states = _state_tensors(runtime._linear_state_pool, "conv_states")
        recurrent_states_by_capacity = {}
        recurrent_states_by_form = {}
        default_recurrent_states = None
        for batch_capacity, requirements in requirements_by_capacity.items():
            recurrent_requirements = tuple(
                requirement for requirement in requirements
                if requirement.buffer == "gdn_recurrent_state"
            )
            if len(recurrent_requirements) > 1:
                raise RuntimeError(
                    "generated decode declares recurrent state more than once")
            if not recurrent_requirements:
                if default_recurrent_states is None:
                    default_recurrent_states = _state_tensors(
                        runtime._linear_state_pool, "recurrent_states")
                recurrent_states = default_recurrent_states
            else:
                form = recurrent_requirements[0].physical_form
                recurrent_states = recurrent_states_by_form.get(form)
                if recurrent_states is None:
                    recurrent_states = (
                        runtime._linear_state_pool.recurrent_tensors_for_form(form)
                    )
                    recurrent_states_by_form[form] = recurrent_states
            recurrent_states_by_capacity[batch_capacity] = recurrent_states
        if any(
            tensor is None
            for layer, tensor in zip(runtime._linear_state_pool.layers, conv_states)
            if layer is not None
        ):
            raise RuntimeError("Qwen GDN state pool must be initialized before binding")
        m_k = _paged_tensors(runtime._shared_paged_layers, "k_cache")
        m_v = _paged_tensors(runtime._shared_paged_layers, "v_cache")
        rope_inv_freq = self.text_model.rotary_emb.inv_freq
        n_pages = int(runtime.page_table.n_pages)
        authoritative_page_table = runtime.page_table.page_table
        preparations = (
            DeviceInputPreparation(
                "gather_rope_deltas",
                produces=("rope_deltas",),
                requires=("batch_idx",),
            ),
            DeviceInputPreparation(
                "prepare_position_ids",
                produces=("position_ids",),
                requires=("input_pos", "rope_deltas"),
            ),
        )
        preparation_plans = []

        self._slots = {}
        for slot in runtime.decode_slots:
            for batch_capacity, program in self._programs.items():
                compiled, validated, aot_bundle = program
                recurrent_states = recurrent_states_by_capacity[batch_capacity]
                runtime_inputs = {
                    "input_ids": slot.decode_token_ids[:batch_capacity],
                    "final_norm": slot.hidden_last[:batch_capacity],
                    "logits": slot.logits[:batch_capacity],
                    "page_table": authoritative_page_table,
                    "batch_idx": slot.meta.batch_idx.gpu[:batch_capacity],
                    "input_pos": slot.meta.input_pos.gpu[:batch_capacity],
                    "position_ids": slot.position_ids[1:4, :batch_capacity, 0],
                    "rope_inv_freq": rope_inv_freq,
                    "gdn_conv_state": conv_states,
                    "gdn_recurrent_state": recurrent_states,
                    "mK": m_k,
                    "mV": m_v,
                    "kv_len": 1,
                }
                preparation_plans.append(
                    derive_device_input_preparation_plan(
                        compiled.device_program,
                        ready_inputs=(
                            set(runtime_inputs)
                            - {"position_ids", "rope_deltas"}
                        ),
                        preparations=preparations,
                    )
                )
                extents = {
                    "active_batch": batch_capacity,
                    "n_pages": n_pages,
                    "page_table_capacity": int(authoritative_page_table.shape[1]),
                    "state_rows": int(authoritative_page_table.shape[0]),
                }
                binding_set = assemble_torch_device_bindings(
                    compiled,
                    bound_weights=bound_weights,
                    runtime_inputs=runtime_inputs,
                    runtime_extents=extents,
                    stream=slot.compute_stream,
                    device=runtime.device,
                )
                invocation = bind_aot_device_program(
                    validated, aot_bundle, binding_set.values
                )
                self._slots[(int(slot.slot_id), batch_capacity)] = _SlotInvocation(
                    binding_set=binding_set,
                    invocation=invocation,
                    uses_kv_len=any(
                        argument.name == "kv_len"
                        for argument in compiled.device_program.argument_plan.arguments
                    ),
                )
        distinct_plans = {
            tuple(step.name for step in plan)
            for plan in preparation_plans
        }
        if len(distinct_plans) != 1:
            raise RuntimeError(
                "generated decode batch domains disagree on input preparation")
        self._input_preparation_plan = preparation_plans[0]
        self._input_preparation_callbacks = {
            "gather_rope_deltas": self._gather_rope_deltas,
            "prepare_position_ids": self._prepare_position_ids,
        }
        self._decode_rope_deltas = runtime._decode_rope_deltas

    def _capacity_for(self, batch_size: int) -> int | None:
        from mkl.megakernel.device_runtime import select_compiled_capacity

        return select_compiled_capacity(self._programs, batch_size)

    def supports(self, batch_size: int) -> bool:
        return self._capacity_for(batch_size) is not None

    def state_requirements_for(
        self,
        batch_size: int,
    ) -> tuple[Any, ...]:
        batch_capacity = self._capacity_for(int(batch_size))
        if batch_capacity is None:
            raise ValueError(
                f"no generated decode capacity covers batch size {batch_size}")
        return self.state_requirements_by_capacity[batch_capacity]

    def _gather_rope_deltas(self, slot: Any, batch_size: int) -> None:
        torch.index_select(
            self._decode_rope_deltas,
            0,
            slot.meta.batch_idx.gpu[:batch_size],
            out=slot.rope_deltas[:batch_size],
        )

    @staticmethod
    def _prepare_position_ids(slot: Any, batch_size: int) -> None:
        cache_position_ids = slot.meta.input_pos.gpu[:batch_size].view(-1, 1)
        position_ids = slot.position_ids[:, :batch_size, :]
        position_ids.copy_(cache_position_ids)
        position_ids[1:].add_(slot.rope_deltas[:batch_size])

    @torch.inference_mode()
    def run(self, slot: Any, batch_size: int = 1) -> None:
        batch_size = int(batch_size)
        batch_capacity = self._capacity_for(batch_size)
        if batch_capacity is None:
            raise ValueError(
                f"no generated decode capacity covers batch size {batch_size}"
            )
        with torch.cuda.stream(slot.compute_stream):
            for step in self._input_preparation_plan:
                self._input_preparation_callbacks[step.name](slot, batch_size)
            state = self._slots[(int(slot.slot_id), batch_capacity)]
            kv_len = int(slot.meta.input_pos.cpu[:batch_size].max()) + 1
            scalar_overrides = {"kv_len": kv_len} if state.uses_kv_len else {}
            scalar_overrides["active_batch"] = batch_size
            state.invocation.launch(**scalar_overrides)


__all__ = ["Qwen35DecodeMegakernel"]
