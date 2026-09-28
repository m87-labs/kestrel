"""Prepared AOT causal verification for a single Qwen DFlash session."""

import torch

from kestrel_kernels.generated_decode import assemble_bindings, materialize_weights, resolve_program
from kestrel_kernels.generated_verification import GeneratedGdnPrefixReplay

from .cache import Qwen35InferenceCache


class Qwen35GeneratedVerification:
    def __init__(self, runtime, draft):
        if (runtime.max_batch_size != 1 or runtime.page_size != 1
                or torch.cuda.get_device_capability(runtime.device) != (10, 0)
                or draft.config.block_size != 16
                or tuple(draft.config.target_layer_ids) != (1, 10, 18, 27, 35, 44, 52, 61)):
            raise ValueError("generated DFlash verification requires B200 C1, page size 1, "
                             "and the qualified Qwen 27B block16 draft")
        self.runtime = runtime
        self.text = runtime.model.model.language_model
        self.block = draft.config.block_size
        self.program = resolve_program(
            registration="qwen35_verification", program="qwen35_27b_fp8_dflash_c1_t16",
            arch="sm100", device_sms=torch.cuda.get_device_properties(
                runtime.device).multi_processor_count, verification=True)
        if self.program is None:
            raise RuntimeError("missing AOT Qwen generated verification program")
        if self.program.descriptor.get("verification") != {
                "sequence_length": self.block, "sequence_capacity": 1}:
            raise ValueError("generated verification artifact has incompatible sequence geometry")
        # This runs before target/prefill graph capture: native parameters may
        # become views into the generated weight slabs.
        self.weights = materialize_weights(
            runtime.model, self.program.descriptor, layer_prefix="model.language_model.layers")
        torch.cuda.current_stream(runtime.device).synchronize()
        self.layers = tuple(i for i, kind in enumerate(self.text.config.layer_types)
                            if kind == "linear_attention")
        self.invocation = None
        self.pending = False

    def _prepare(self, committed, slot):
        runtime, device = self.runtime, self.runtime.device
        self.stream = torch.cuda.current_stream(device)
        inputs = dict(
            batch_idx=torch.zeros(self.block, dtype=torch.int64, device=device),
            page_table=runtime.page_table.page_table[slot:slot + 1].clone(),
            rope_delta_table=runtime._decode_rope_deltas[slot:slot + 1].clone(),
            rope_inv_freq=self.text.rotary_emb.inv_freq,
            gdn_conv_state=[None if i not in self.layers else torch.empty_like(layer.conv_states)
                            for i, layer in enumerate(committed.layers)],
            gdn_recurrent_state=[None if i not in self.layers else torch.empty_like(layer.recurrent_states)
                                 for i, layer in enumerate(committed.layers)],
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
            runtime_extents=dict(active_batch=self.block, state_rows=1, n_pages=n_pages,
                                 page_table_capacity=inputs["page_table"].shape[1]),
            stream=self.stream, device=device)
        self.invocation = self.program.bind(self.bindings)
        # Tried host live-kv overrides: 395.62 vs 395.77 tok/s at C1;
        # keep the fixed launcher (device work already follows input_pos).
        self.launch = self.invocation.prepare_repeated_launch(active_batch=self.block)
        self.inputs = inputs
        self.positions = torch.arange(self.block, dtype=torch.int32, device=device)
        self.replay = GeneratedGdnPrefixReplay(
            raw=inputs["gdn_projection_snapshots"], convolved=inputs["gdn_convolution_snapshots"],
            layers=self.layers,
            decay=torch.stack([self.text.layers[i].linear_attn.A_log for i in self.layers]).float(),
            bias=torch.stack([self.text.layers[i].linear_attn.dt_bias for i in self.layers]).float(),
            states=[committed.layers[i].recurrent_states for i in self.layers],
            histories=[committed.layers[i].conv_states for i in self.layers])
        self.verified = Qwen35InferenceCache(config=self.text.config, paged_kv=runtime._paged_kv)
        self.accepted = Qwen35InferenceCache(config=self.text.config, paged_kv=runtime._paged_kv)
        for row, index in enumerate(self.layers):
            for cache, state, history in (
                (self.verified, inputs["gdn_recurrent_state"][index], inputs["gdn_conv_state"][index]),
                (self.accepted, self.replay.states[row], self.replay.histories[row]),
            ):
                layer = cache.layers[index]
                layer.recurrent_states, layer.conv_states = state, history
                layer.has_previous_state = True

    def target(self, tokens, committed, slot):
        if self.pending:
            raise RuntimeError("generated verification still has an uncommitted block")
        if len(tokens) != self.block:
            raise ValueError("generated verification requires a complete candidate block")
        if self.invocation is None:
            self._prepare(committed, slot)
        if torch.cuda.current_stream(self.runtime.device) != self.stream:
            raise RuntimeError("generated verification must use its prepared CUDA stream")
        self.source = committed
        for field, name in (("conv_states", "gdn_conv_state"),
                            ("recurrent_states", "gdn_recurrent_state")):
            torch._foreach_copy_([self.inputs[name][i] for i in self.layers],
                                [getattr(committed.layers[i], field) for i in self.layers])
        self.inputs["input_ids"].copy_(torch.tensor(tokens, dtype=torch.int32, device=self.runtime.device))
        torch.add(self.positions, committed.seq_length, out=self.inputs["input_pos"])
        self.inputs["page_table"].copy_(self.runtime.page_table.page_table[slot:slot + 1])
        self.inputs["rope_delta_table"].copy_(self.runtime._decode_rope_deltas[slot:slot + 1])
        self.pending = True
        self.launch()
        self.verified.seq_length = committed.seq_length + self.block
        return (self.inputs["input_ids"].tolist(),
                self.inputs["verification_features"].reshape(1, self.block, -1), self.verified)

    def commit(self, context):
        session, verified, features, expected, count = context
        if (not self.pending or verified is not self.verified
                or type(count) is not int or not 1 <= count <= self.block):
            raise ValueError("accepted prefix does not belong to the outstanding verification")
        if torch.cuda.current_stream(self.runtime.device) != self.stream:
            raise RuntimeError("generated verification must use its prepared CUDA stream")
        end = self.source.seq_length + count
        if count == self.block:
            torch._foreach_copy_(self.replay.states,
                                 [self.inputs["gdn_recurrent_state"][i] for i in self.layers])
            torch._foreach_copy_(self.replay.histories,
                                 [self.inputs["gdn_conv_state"][i] for i in self.layers])
        else:
            self.replay.launch([self.source.layers[i].recurrent_states for i in self.layers],
                               [self.source.layers[i].conv_states for i in self.layers], count)
        self.accepted.seq_length = end
        session.cache, session.features, session.bonus = self.accepted, features[:, :count], expected[count - 1]
        self.pending = False
