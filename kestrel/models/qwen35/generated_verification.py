"""Prepared AOT causal verification for independent Qwen DFlash sessions."""

import torch

from kestrel_kernels.generated_decode import assemble_bindings, materialize_weights, resolve_program
from kestrel_kernels.generated_verification import GeneratedGdnPrefixReplay

from .cache import Qwen35InferenceCache


_PROGRAMS = {
    (1, 16, (1, 10, 18, 27, 35, 44, 52, 61)): "qwen35_27b_fp8_dflash_c1_t16",
    (1, 8, (5, 19, 33, 47, 61)): "qwen35_27b_fp8_dflash2_c1_t8",
    (2, 8, (5, 19, 33, 47, 61)): "qwen35_27b_fp8_dflash2_c2_t8",
}


class Qwen35GeneratedVerification:
    @staticmethod
    def supports(runtime, draft):
        if runtime.max_batch_size not in (1, 2) or runtime.page_size != 1:
            return False
        target = runtime.model.model.language_model.config
        if (target.hidden_size != 5120 or target.num_hidden_layers != 64
                or target.intermediate_size != 17408 or target.vocab_size != 248320
                or target.dense_weight_format != "fp8_e4m3"):
            return False
        if torch.cuda.get_device_capability(runtime.device) != (10, 0):
            return False
        return all((sequences, draft.config.block_size, tuple(draft.config.target_layer_ids))
                   in _PROGRAMS for sequences in range(1, runtime.max_batch_size + 1))

    def __init__(self, runtime, draft, *, sequences=1, weights=None):
        if runtime.max_batch_size not in (1, 2) or not 1 <= sequences <= runtime.max_batch_size:
            raise ValueError("generated DFlash verification supports one or two sequences")
        program = _PROGRAMS.get((sequences, draft.config.block_size, tuple(draft.config.target_layer_ids)))
        if (runtime.page_size != 1
                or torch.cuda.get_device_capability(runtime.device) != (10, 0)
                or program is None):
            raise ValueError("generated DFlash verification requires B200, page size 1, "
                             "and a supported Qwen 27B draft geometry")
        self.runtime = runtime
        self.text = runtime.model.model.language_model
        self.block = draft.config.block_size
        self.sequences = sequences
        self.rows = self.block * sequences
        self.program = resolve_program(
            registration="qwen35_verification", program=program,
            arch="sm100", device_sms=torch.cuda.get_device_properties(
                runtime.device).multi_processor_count, verification=True)
        if self.program is None:
            raise RuntimeError("missing AOT Qwen generated verification program")
        if self.program.descriptor.get("verification") != {
                "sequence_length": self.block, "sequence_capacity": sequences}:
            raise ValueError("generated verification artifact has incompatible sequence geometry")
        # This runs before target/prefill graph capture: native parameters may
        # become views into the generated weight slabs.
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
        inputs = dict(
            batch_idx=torch.arange(self.sequences, dtype=torch.int64, device=device).repeat_interleave(self.block),
            page_table=runtime.page_table.page_table[:self.sequences].clone(),
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
            kv_len=runtime.page_table.page_table.shape[1],
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
        return self.target_many([tokens], [committed], [slot])[0]

    def target_many(self, candidates, committed, slots):
        if self.pending:
            raise RuntimeError("generated verification still has an uncommitted block")
        if (len(candidates) != self.sequences or len(committed) != self.sequences
                or len(slots) != self.sequences or len(set(slots)) != self.sequences
                or any(len(tokens) != self.block for tokens in candidates)):
            raise ValueError("generated verification requires complete independent candidate blocks")
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
        self.inputs["input_ids"].copy_(torch.tensor(candidates, dtype=torch.int32,
                                                  device=self.runtime.device).flatten())
        for row, (cache, slot) in enumerate(zip(committed, slots, strict=True)):
            span = slice(row * self.block, (row + 1) * self.block)
            torch.add(self.positions, cache.seq_length, out=self.inputs["input_pos"][span])
            self.inputs["page_table"][row:row + 1].copy_(self.runtime.page_table.page_table[slot:slot + 1])
            self.inputs["rope_delta_table"][row:row + 1].copy_(self.runtime._decode_rope_deltas[slot:slot + 1])
            self.verified[row].seq_length = cache.seq_length + self.block
        self.pending = True
        self.pending_rows = set(range(self.sequences))
        self.launch()
        predictions = self.inputs["input_ids"].reshape(self.sequences, self.block).tolist()
        features = self.inputs["verification_features"].reshape(self.sequences, self.block, -1)
        return [(predictions[row], features[row:row + 1], self.verified[row])
                for row in range(self.sequences)]

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
            replay.launch(states, histories, count)
        accepted.seq_length = end
        session.cache, session.features, session.bonus = accepted, features[:, :count], expected[count - 1]
        self.pending_rows.remove(row)
        self.pending = bool(self.pending_rows)
