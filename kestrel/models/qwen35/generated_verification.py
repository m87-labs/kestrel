"""Prepared AOT causal verification for independent Qwen DFlash sessions."""

import json
from pathlib import Path

import torch
from kestrel_kernels.generated_decode import (
    assemble_bindings,
    materialize_weights,
    resolve_program,
)
from kestrel_kernels.generated_verification import GeneratedGdnPrefixReplay

from .cache import Qwen35InferenceCache

_PROGRAMS = {
    (1, 16, (1, 10, 18, 27, 35, 44, 52, 61)): "qwen35_27b_fp8_dflash_c1_t16",
    (1, 8, (5, 19, 33, 47, 61)): "qwen35_27b_fp8_dflash2_c1_t8",
    (2, 8, (5, 19, 33, 47, 61)): "qwen35_27b_fp8_dflash2_c2_t8",
}


def _validate_hopper_contract(runtime, target, draft_config, sequences):
    taps = tuple(draft_config.target_layer_ids)
    expected = {
        "hidden_size": 2048, "num_hidden_layers": 40, "vocab_size": 248320,
        "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
        "num_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6,
        "partial_rotary_factor": 0.25, "tie_word_embeddings": False,
        "expert_weight_format": "bf16", "dense_weight_format": "bf16",
    }
    dimensions_match = all(getattr(target, key, None) == value
                           for key, value in expected.items())
    if (runtime.max_batch_size not in range(1, 9) or not 1 <= sequences <= runtime.max_batch_size
            or draft_config.block_size != 16 or taps != (1, 6, 11, 16, 22, 27, 32, 37)
            or not dimensions_match or tuple(target.mrope_section) != (11, 11, 10)
            or tuple(target.layer_types) != (("linear_attention",) * 3
                                             + ("full_attention",)) * 10):
        raise ValueError("unsupported Hopper causal verification model or draft geometry")


class Qwen35GeneratedVerification:
    @staticmethod
    def supports(runtime, draft):
        if runtime.max_batch_size not in (1, 2) or runtime.page_size != 1:
            return False
        target = runtime.model.model.language_model.config
        if torch.cuda.get_device_capability(runtime.device) == (9, 0):
            if runtime.max_batch_size != 1 or not runtime._cfg.enable_cuda_graphs or draft.config.selector_rank:
                return False
            try:
                _validate_hopper_contract(runtime, target, draft.config, 1)
            except ValueError:
                return False
            return True
        if (target.hidden_size != 5120 or target.num_hidden_layers != 64
                or target.intermediate_size != 17408 or target.vocab_size != 248320
                or target.dense_weight_format != "fp8_e4m3"):
            return False
        if torch.cuda.get_device_capability(runtime.device) != (10, 0):
            return False
        return all((sequences, draft.config.block_size, tuple(draft.config.target_layer_ids))
                   in _PROGRAMS for sequences in range(1, runtime.max_batch_size + 1))

    def __init__(self, runtime, draft, *, sequences=1, weights=None, verification_rows=None):
        if runtime.max_batch_size not in range(1, 9) or not 1 <= sequences <= runtime.max_batch_size:
            raise ValueError("generated DFlash verification supports up to eight sequences")
        target = runtime.model.model.language_model.config
        taps = tuple(draft.config.target_layer_ids)
        arch = torch.cuda.get_device_capability(runtime.device)
        block = draft.config.block_size if verification_rows is None else verification_rows
        if type(block) is not int or not 1 <= block <= draft.config.block_size:
            raise ValueError("invalid verification row count")
        if arch != (9, 0) and block != draft.config.block_size:
            raise ValueError("short verification spans require Hopper")
        program = _PROGRAMS.get((sequences, draft.config.block_size, taps))
        registration = "qwen35_verification"
        self.segmented = False
        if arch == (9, 0):
            _validate_hopper_contract(runtime, target, draft.config, sequences)
            registration = "qwen35_moe_verification"
            if sequences > 1 and block not in {2: (2, 4, 8), 3: (2,), 4: (2, 4)}.get(sequences, (2,)):
                raise ValueError("unsupported concurrent Hopper verification span")
            if block not in (2, 3, 4, 5, 6, 7, 8, 16):
                raise ValueError("unsupported Hopper verification row count")
            program = f"qwen35_35b_a3b_bf16_dflash_c{sequences}_t{block}"
            self.segmented = True
        if (runtime.page_size != 1 or arch not in ((9, 0), (10, 0)) or program is None
                or (arch == (10, 0) and runtime.max_batch_size not in (1, 2))):
            raise ValueError("generated DFlash verification requires a supported target, "
                             "page size 1, and a supported draft geometry")
        self.runtime = runtime
        self.text = runtime.model.model.language_model
        self.block = block
        self.sequences = sequences
        self.rows = self.block * sequences
        self.capture_layers = taps
        self.registration, self.program_name = registration, program
        self.program = resolve_program(
            registration=registration, program=program,
            arch=f"sm{arch[0]}{arch[1]}", device_sms=torch.cuda.get_device_properties(
                runtime.device).multi_processor_count, verification=True)
        if self.program is None:
            raise RuntimeError("missing AOT Qwen generated verification program")
        if self.program.descriptor.get("verification") != {
                "sequence_length": self.block, "sequence_capacity": sequences}:
            raise ValueError("generated verification artifact has incompatible sequence geometry")
        # This runs before target/prefill graph capture: native parameters may
        # become views into the generated weight slabs.
        if weights is None:
            weights = getattr(runtime, "_verification_weight_storage", None)
        self.weights = weights if weights is not None else materialize_weights(
            runtime.model, self.program.descriptor, layer_prefix="model.language_model.layers")
        torch.cuda.current_stream(runtime.device).synchronize()
        self.layers = tuple(i for i, kind in enumerate(self.text.config.layer_types)
                            if kind == "linear_attention")
        self.invocation = None
        self.pending = False

    def _prepare(self, committed, slots):
        runtime, device = self.runtime, self.runtime.device
        self.stream = torch.cuda.current_stream(device)
        # Physical KV capacity may cover several requests; each qualified
        # Hopper attention sequence still has a 32K context window.
        page_capacity = (min(32768, runtime.page_table.page_table.shape[1])
                         if self.segmented else runtime.page_table.page_table.shape[1])
        inputs = dict(
            batch_idx=torch.arange(self.sequences, dtype=torch.int64, device=device).repeat_interleave(self.block),
            page_table=torch.empty((self.sequences, page_capacity), dtype=torch.int32, device=device).copy_(
                runtime.page_table.page_table[:self.sequences, :page_capacity]),
            rope_delta_table=runtime._decode_rope_deltas[:self.sequences].clone(),
            rope_inv_freq=self.text.rotary_emb.inv_freq,
            gdn_conv_state=[None if i not in self.layers else torch.empty(
                (self.sequences, *layer.conv_states.shape[1:]), dtype=layer.conv_states.dtype, device=device)
                for i, layer in enumerate(committed[0].layers)],
            gdn_recurrent_state=[None if i not in self.layers else torch.empty(
                (self.sequences, *layer.recurrent_states.shape[1:]), dtype=layer.recurrent_states.dtype, device=device)
                for i, layer in enumerate(committed[0].layers)],
            mK=[None if layer is None else layer.k_cache[:, :, 0, :] for layer in runtime._paged_kv],
            mV=[None if layer is None else layer.v_cache[:, :, 0, :] for layer in runtime._paged_kv],
            kv_len=page_capacity,
        )
        dtypes = dict(bf16=torch.bfloat16, fp32=torch.float32, int32=torch.int32, int64=torch.int64)
        for operand in self.program.descriptor["device_program"]["physical_abi"]["operands"]:
            name = operand["logical_name"]
            if operand["owner"] != "binder" and name not in inputs:
                inputs[name] = (0 if operand["transport"] == "scalar" else torch.zeros(
                    tuple(operand["abi_shape"]), dtype=dtypes[operand["dtype"]], device=device))
        n_pages = next(layer.k_cache.shape[0] for layer in runtime._paged_kv if layer is not None)
        self.bindings = assemble_bindings(
            self.program.descriptor, weights=self.weights.buffers, runtime_inputs=inputs,
            runtime_extents=dict(active_batch=self.rows, state_rows=self.sequences, n_pages=n_pages,
                                 page_table_capacity=inputs["page_table"].shape[1]),
            stream=self.stream, device=device)
        if self.segmented:
            # Graph warmup may write KV, so use only the tentative suffix of
            # the current slot. Never warm up at position zero of a live prefix.
            for row, (cache, slot) in enumerate(zip(committed, slots, strict=True)):
                span = slice(row * self.block, (row + 1) * self.block)
                inputs["input_pos"][span].copy_(torch.arange(
                    cache.seq_length, cache.seq_length + self.block,
                    dtype=inputs["input_pos"].dtype, device=device))
                inputs["page_table"][row:row + 1].copy_(runtime.page_table.page_table[slot:slot + 1, :page_capacity])
                inputs["rope_delta_table"][row:row + 1].copy_(runtime._decode_rope_deltas[slot:slot + 1])
            for name in ("gdn_conv_state", "gdn_recurrent_state"):
                for value in inputs[name]:
                    if value is not None:
                        value.zero_()
            from kestrel_kernels.generated_verification_segments import (
                SegmentedVerification,
            )

            self.invocation = SegmentedVerification(
                registration=self.registration, parent_program=self.program_name,
                parent_descriptor=self.program.descriptor, weights=self.weights.buffers,
                inputs=inputs, capture_layers=self.capture_layers, stream=self.stream, device=device)
            self.invocation.capture()
            self.launch = self.invocation.launch
        else:
            self.invocation = self.program.bind(self.bindings)
            # Tried host live-kv overrides: 395.62 vs 395.77 tok/s at C1;
            # keep the fixed launcher (device work already follows input_pos).
            self.launch = self.invocation.prepare_repeated_launch(active_batch=self.rows)
        self.inputs = inputs
        self.positions = torch.arange(self.block, dtype=torch.int32, device=device)
        decay = torch.stack([self.text.layers[i].linear_attn.A_log for i in self.layers]).float()
        bias = torch.stack([self.text.layers[i].linear_attn.dt_bias for i in self.layers]).float()
        self.replays, self.verified, self.accepted = [], [], []
        self.state_destinations = {name: [] for name in ("gdn_conv_state", "gdn_recurrent_state")}
        self.initial_states, self.initial_histories = [], []
        for row, source in enumerate(committed):
            span = slice(row * self.block, (row + 1) * self.block)
            states = [source.layers[i].recurrent_states for i in self.layers]
            histories = [source.layers[i].conv_states for i in self.layers]
            if self.sequences > 1:
                states = [torch.empty_like(value) for value in states]
                histories = [torch.empty_like(value) for value in histories]
            self.initial_states.append(states)
            self.initial_histories.append(histories)
            replay = GeneratedGdnPrefixReplay(
                raw=inputs["gdn_projection_snapshots"][span],
                convolved=inputs["gdn_convolution_snapshots"][span],
                layers=self.layers, decay=decay, bias=bias,
                states=states, histories=histories)
            verified = Qwen35InferenceCache(config=self.text.config, paged_kv=runtime._paged_kv)
            accepted = Qwen35InferenceCache(config=self.text.config, paged_kv=runtime._paged_kv)
            for index, state, history in zip(self.layers, replay.states, replay.histories, strict=True):
                working_state = inputs["gdn_recurrent_state"][index][row:row + 1]
                working_history = inputs["gdn_conv_state"][index][row:row + 1]
                self.state_destinations["gdn_recurrent_state"].append(working_state)
                self.state_destinations["gdn_conv_state"].append(working_history)
                for cache, s, h in ((verified, working_state, working_history), (accepted, state, history)):
                    layer = cache.layers[index]
                    layer.recurrent_states, layer.conv_states = s, h
                    layer.has_previous_state = True
            self.replays.append(replay)
            self.verified.append(verified)
            self.accepted.append(accepted)

    def target(self, tokens, committed, slot):
        candidates = tokens.unsqueeze(0) if isinstance(tokens, torch.Tensor) else [tokens]
        return self.target_many(candidates, [committed], [slot])[0]

    def target_many(self, candidates, committed, slots):
        self._stage_candidates(candidates, committed, slots)
        self.pending = True
        self.pending_rows = set(range(self.sequences))
        self.launch()
        predictions = self.inputs["input_ids"].reshape(self.sequences, self.block).tolist()
        features = self.inputs["verification_features"].reshape(self.sequences, self.block, -1)
        return [(predictions[row], features[row:row + 1], self.verified[row])
                for row in range(self.sequences)]

    def _stage_candidates(self, candidates, committed, slots):
        if self.pending:
            raise RuntimeError("generated verification still has an uncommitted block")
        if isinstance(candidates, torch.Tensor) and (
                candidates.shape != (self.sequences, self.block)
                or candidates.device != torch.device(self.runtime.device)
                or candidates.dtype not in (torch.int32, torch.int64)):
            raise ValueError("verification candidates require device-local integer token rows")
        if (len(candidates) != self.sequences or len(committed) != self.sequences
                or len(slots) != self.sequences or len(set(slots)) != self.sequences
                or any(len(tokens) != self.block for tokens in candidates)):
            raise ValueError("generated verification requires complete independent candidate blocks")
        if getattr(self, "segmented", False):
            table = self.runtime.page_table.page_table
            if any(type(slot) is not int or not 0 <= slot < table.shape[0] for slot in slots):
                raise ValueError("verification slot is outside the page table")
            if any(cache.seq_length < 0 or cache.seq_length + self.block > min(32768, table.shape[1])
                   for cache in committed):
                raise ValueError("verification suffix exceeds the page table capacity")
        if self.invocation is None:
            self._prepare(committed, slots)
        if torch.cuda.current_stream(self.runtime.device) != self.stream:
            raise RuntimeError("generated verification must use its prepared CUDA stream")
        self.sources = tuple(committed)
        self.start_positions = tuple(cache.seq_length for cache in committed)
        for field, name in (("conv_states", "gdn_conv_state"),
                            ("recurrent_states", "gdn_recurrent_state")):
            sources = [getattr(cache.layers[i], field) for cache in committed for i in self.layers]
            if self.sequences > 1:
                # Cohort rows may change owners. Preserve every starting state
                # before any prefix commit can overwrite another row's source.
                initial = self.initial_histories if field == "conv_states" else self.initial_states
                retained = [value for row in initial for value in row]
                torch._foreach_copy_(retained, sources)
                sources = retained
            torch._foreach_copy_(self.state_destinations[name], sources)
        token_rows = (candidates if isinstance(candidates, torch.Tensor) else
                      torch.tensor(candidates, dtype=torch.int32, device=self.runtime.device))
        self.inputs["input_ids"].copy_(token_rows.reshape(-1))
        for row, (cache, slot) in enumerate(zip(committed, slots, strict=True)):
            span = slice(row * self.block, (row + 1) * self.block)
            torch.add(self.positions, cache.seq_length, out=self.inputs["input_pos"][span])
            self.inputs["page_table"][row:row + 1].copy_(self.runtime.page_table.page_table[slot:slot + 1, :self.inputs["page_table"].shape[1]])
            self.inputs["rope_delta_table"][row:row + 1].copy_(self.runtime._decode_rope_deltas[slot:slot + 1])
            self.verified[row].seq_length = cache.seq_length + self.block

    def measure_replay(self, candidates, committed, slots):
        """Measure a fresh tentative replay without creating a commit lease.

        Staging restores recurrent state and proposal IDs on every call. Only
        tentative KV suffixes are written, as during normal graph warmup.
        """
        self._stage_candidates(candidates, committed, slots)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        self.launch()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end)

    def commit(self, context):
        session, verified, features, expected, count = context
        if not self.pending or type(count) is not int or not 1 <= count <= self.block:
            raise ValueError("accepted prefix does not belong to the outstanding verification")
        row = next((i for i in self.pending_rows if self.verified[i] is verified), None)
        if row is None:
            raise ValueError("accepted prefix does not belong to the outstanding verification")
        if torch.cuda.current_stream(self.runtime.device) != self.stream:
            raise RuntimeError("generated verification must use its prepared CUDA stream")
        source, replay, accepted = self.sources[row], self.replays[row], self.accepted[row]
        end = self.start_positions[row] + count
        if count == self.block:
            torch._foreach_copy_(replay.states, [verified.layers[i].recurrent_states for i in self.layers])
            torch._foreach_copy_(replay.histories, [verified.layers[i].conv_states for i in self.layers])
        else:
            states = (self.initial_states[row] if self.sequences > 1 else
                      [source.layers[i].recurrent_states for i in self.layers])
            histories = (self.initial_histories[row] if self.sequences > 1 else
                         [source.layers[i].conv_states for i in self.layers])
            # These cache tensors are retained by this verifier and keep their
            # allocation/layout throughout the session; only their values change.
            # A different source still goes through full binding validation.
            if (self.sequences == 1
                    and all(a is b for a, b in zip(states, replay.source_owners[0], strict=True))
                    and all(a is b for a, b in zip(histories, replay.source_owners[1], strict=True))):
                replay.launch_bound(count)
            else:
                replay.launch(states, histories, count)
        accepted.seq_length = end
        session.cache, session.features, session.bonus = accepted, features[:, :count], expected[count - 1]
        self.pending_rows.remove(row)
        self.pending = bool(self.pending_rows)


def verification_weight_loading_callbacks(runtime):
    """Load eligible large verifiers directly into their final weight slabs."""
    verification_path = runtime._cfg.verification_path
    if verification_path == "auto":
        verification_path = runtime.decode_path
    automatic = verification_path == "auto"
    if (verification_path not in ("generated", "auto")
            or torch.cuda.get_device_capability(runtime.device) != (9, 0)):
        return None, None
    if automatic and (not runtime._cfg.enable_cuda_graphs or runtime._cfg.page_size != 1
                      or runtime.max_batch_size != 1):
        return None, None
    from kestrel_kernels.generated_decode import (
        allocate_weight_storage_for_loading,
        finalize_weight_storage_after_loading,
    )
    from .dflash import DFlashConfig

    draft = DFlashConfig.from_dict(json.loads(
        (Path(runtime._draft_model_path) / "config.json").read_text()))
    if automatic and draft.selector_rank:
        return None, None
    program = None
    prefix = "model.language_model.layers"

    def prepare(model):
        nonlocal program
        try:
            _validate_hopper_contract(runtime, model.model.language_model.config, draft, 1)
        except ValueError:
            if automatic:
                return
            raise
        program = resolve_program(
            registration="qwen35_moe_verification", program="qwen35_35b_a3b_bf16_dflash_c1_t16",
            arch="sm90", device_sms=torch.cuda.get_device_properties(runtime.device).multi_processor_count,
            verification=True)
        if program is None:
            raise RuntimeError("missing AOT Qwen generated verification program")
        runtime._verification_weight_storage = allocate_weight_storage_for_loading(
            model, program.descriptor, device=runtime.device, layer_prefix=prefix)

    def finalize(model):
        if program is not None:
            runtime._verification_weight_storage = finalize_weight_storage_after_loading(
                model, program.descriptor, runtime._verification_weight_storage, layer_prefix=prefix)

    return prepare, finalize
