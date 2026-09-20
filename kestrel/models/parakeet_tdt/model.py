"""Inference-only Parakeet FastConformer + token-duration transducer."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kestrel_kernels import get_runtime

from .config import ParakeetEncoderConfig, ParakeetTdtConfig
from .vad import VadHead


def _norm_args(norm: nn.LayerNorm) -> tuple[Tensor, Tensor, float]:
    """A LayerNorm module as the ``conformer`` domain takes it."""
    return norm.weight, norm.bias, norm.eps


class FeedForward(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.linear2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden: Tensor) -> Tensor:
        # The activation is a runtime op so a backend can do it in one pass over the [B, T, 4C] projection
        # instead of the read-write-read torch needs; the reference is F.silu.
        return self.linear2(get_runtime().conformer.silu(self.linear1(hidden)))


class Convolution(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        channels = config.hidden_size
        self.pointwise_conv1 = nn.Linear(channels, 2 * channels, bias=False)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            config.conv_kernel_size,
            padding=(config.conv_kernel_size - 1) // 2,
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Linear(channels, channels, bias=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # The checkpoint stores the 1x1 convolutions as ``Conv1d`` weights ``[C_out, C_in, 1]``. Running them
        # as linears measured 42.87 ms vs 45.37 ms encoder wall on L4 at batch 1, and it is the layout the
        # ternary export already packs, so the trailing kernel axis is dropped on the way in.
        for name in ("pointwise_conv1", "pointwise_conv2"):
            key = f"{prefix}{name}.weight"
            weight = state_dict.get(key)
            if weight is not None and weight.dim() == 3:
                state_dict[key] = weight[..., 0]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, hidden: Tensor, valid: Tensor | None) -> Tensor:
        hidden = get_runtime().conformer.glu(self.pointwise_conv1(hidden))
        norm = self.norm
        # The depthwise convolution, the eval-mode BatchNorm and the SiLU are one runtime op: it masks the
        # invalid rows, and each backend picks its own implementation (see kestrel_kernels.conformer_ops).
        hidden = get_runtime().conformer.depthwise_conv_bn_silu(
            hidden,
            self.depthwise_conv.weight[:, 0, :],  # the op takes the depthwise weight as [C, k]
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            norm.eps,
            valid,
        )
        return self.pointwise_conv2(hidden)


class RelativeAttention(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=False
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # ``relative_k_proj`` is not here: it does not depend on the activations, so the encoder runs all
        # 24 layers' copies as one projection and hands each block its slice (see ``Encoder``).
        self.bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # One QKV GEMM measured 43.35 ms vs 45.37 ms encoder wall on L4 at
        # batch 1. Fuse the checkpoint's separate tensors without retaining a
        # duplicate 144 MiB BF16 copy across the 24 encoder layers.
        # A ternary checkpoint carries packed codes and scales instead of a weight; ternary rows are
        # independent, so those concatenate by row just as the dense weight does.
        for suffix in ("weight", "qweight", "scales"):
            fused_key = f"{prefix}qkv_proj.{suffix}"
            source_keys = tuple(f"{prefix}{name}_proj.{suffix}" for name in "qkv")
            if fused_key not in state_dict and all(
                key in state_dict for key in source_keys
            ):
                state_dict[fused_key] = torch.cat(
                    tuple(state_dict.pop(key) for key in source_keys)
                ).contiguous()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, hidden: Tensor, rel_k: Tensor, mask: Tensor | None) -> Tensor:
        # Everything between the projections and o_proj is one runtime op: the relative shift, the two score
        # products, the mask and the softmax. It takes the projections fused, which is what lets a backend
        # read q, k and v in place instead of materializing the chunk/view/transpose chain.
        attended = get_runtime().conformer.rel_attention(
            self.qkv_proj(hidden),
            rel_k,
            self.bias_u,
            self.bias_v,
            mask,
            self.scale,
        )
        return self.o_proj(attended)


class EncoderBlock(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        self.feed_forward1 = FeedForward(config)
        self.self_attn = RelativeAttention(config)
        self.conv = Convolution(config)
        self.feed_forward2 = FeedForward(config)
        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden: Tensor,
        rel_k: Tensor,
        pair_mask: Tensor | None,
        valid: Tensor | None,
    ) -> Tensor:
        # Every residual add feeds straight into the next LayerNorm, so the two go through the runtime as one
        # op returning both: the sum the next residual needs and the normalized input the next sublayer takes.
        # A backend that fuses them reads the activation once instead of three times; the reference is the
        # ``hidden + alpha * y`` then ``LayerNorm`` this replaces, and alpha is a power of two, so the bits
        # are the ones the model had.
        conformer = get_runtime().conformer
        normed = conformer.layer_norm(hidden, *_norm_args(self.norm_feed_forward1))
        hidden, normed = conformer.add_scaled_layer_norm(
            hidden, self.feed_forward1(normed), 0.5, *_norm_args(self.norm_self_att)
        )
        hidden, normed = conformer.add_scaled_layer_norm(
            hidden,
            self.self_attn(normed, rel_k, pair_mask),
            1.0,
            *_norm_args(self.norm_conv),
        )
        hidden, normed = conformer.add_scaled_layer_norm(
            hidden, self.conv(normed, valid), 1.0, *_norm_args(self.norm_feed_forward2)
        )
        _, out = conformer.add_scaled_layer_norm(
            hidden, self.feed_forward2(normed), 0.5, *_norm_args(self.norm_out)
        )
        return out


class Subsampling(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        # Tried exact unfold/linear replacements on H100: 2.011 ms vs
        # 0.321 ms for these convolutions (6.26x slower); keeping Conv2d.
        layers: list[nn.Module] = [
            nn.Conv2d(1, config.subsampling_conv_channels, 3, 2, padding=1),
            nn.ReLU(),
        ]
        for _ in range(int(math.log2(config.subsampling_factor)) - 1):
            layers.extend(
                (
                    nn.Conv2d(
                        config.subsampling_conv_channels,
                        config.subsampling_conv_channels,
                        3,
                        2,
                        padding=1,
                        groups=config.subsampling_conv_channels,
                    ),
                    nn.Conv2d(
                        config.subsampling_conv_channels,
                        config.subsampling_conv_channels,
                        1,
                    ),
                    nn.ReLU(),
                )
            )
        self.layers = nn.ModuleList(layers)
        frequency = config.num_mel_bins // config.subsampling_factor
        self.linear = nn.Linear(
            config.subsampling_conv_channels * frequency, config.hidden_size
        )

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        hidden = features.unsqueeze(1)
        lengths = mask.sum(-1)
        conformer = get_runtime().conformer
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels > 1:
                # The two depthwise convolutions go through the runtime: the reference is this same
                # ``F.conv2d``, and a backend whose torch has no depthwise 2-D kernel (Apple silicon runs it
                # one channel at a time -- 14 of the front end's 22 ms per utterance) replaces it.
                hidden = conformer.depthwise_conv2d(
                    hidden, layer.weight[:, 0], layer.bias, layer.stride[0], layer.padding[0]
                )
            else:
                hidden = layer(hidden)
            if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
                lengths = (
                    lengths + 2 * layer.padding[0] - layer.kernel_size[0]
                ) // layer.stride[0] + 1
                valid = (
                    torch.arange(hidden.shape[2], device=hidden.device)[None]
                    < lengths[:, None]
                )
                hidden = hidden * valid[:, None, :, None]
        hidden = hidden.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[2], -1)
        valid = (
            torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None]
        )
        return self.linear(hidden), valid


class Encoder(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.subsampling = Subsampling(config)
        self.layers = nn.ModuleList(
            EncoderBlock(config) for _ in range(config.num_hidden_layers)
        )
        # Every block projects the same relative-position table with its own weight, and the table does not
        # depend on the activations -- so all 24 projections are one [2L-1, C] x [24C, C] matrix multiply
        # instead of 24 narrow ones. Same arithmetic, 8.6 % of the encoder's projection work at 24x the
        # output width, and one activation quantization on the packed CPU path rather than 24.
        self.relative_k_proj = nn.Linear(
            config.hidden_size, config.num_hidden_layers * config.hidden_size, bias=False
        )
        inverse = 1 / (
            10_000
            ** (
                torch.arange(0, config.hidden_size, 2, dtype=torch.float32)
                / config.hidden_size
            )
        )
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # The checkpoint keeps one relative-position projection per block; they concatenate by row into the
        # encoder's single one, exactly as a block's q/k/v triple does. Ternary rows are independent, so the
        # packed codes and their scales concatenate the same way.
        for suffix in ("weight", "qweight", "scales"):
            fused_key = f"{prefix}relative_k_proj.{suffix}"
            source_keys = tuple(
                f"{prefix}layers.{index}.self_attn.relative_k_proj.{suffix}"
                for index in range(len(self.layers))
            )
            if fused_key not in state_dict and all(key in state_dict for key in source_keys):
                state_dict[fused_key] = torch.cat(
                    tuple(state_dict.pop(key) for key in source_keys)
                ).contiguous()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def reset_nonpersistent_buffers(self) -> None:
        config = self.config
        self.inverse_frequency = (
            1
            / (
                10_000
                ** (
                    torch.arange(0, config.hidden_size, 2, dtype=torch.float32)
                    / config.hidden_size
                )
            )
        ).to(self.subsampling.linear.weight.device)

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        hidden, valid = self.subsampling(features, mask)
        return self.forward_subsampled(hidden, valid)

    def forward_subsampled(self, hidden: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        length = hidden.shape[1]
        relative_positions = torch.arange(length - 1, -length, -1, device=hidden.device)
        phase = torch.outer(relative_positions.float(), self.inverse_frequency)
        positions = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        positions = positions[None].expand(hidden.shape[0], -1, -1).to(hidden.dtype)
        # One projection for all the blocks, then a contiguous slice each: the layer axis moves to the front
        # so a block's slice is the ordinary ``[B, 2L-1, C]`` the attention op takes on any backend.
        relative = self.relative_k_proj(positions)
        relative = relative.view(*relative.shape[:2], len(self.layers), -1)
        relative = relative.permute(2, 0, 1, 3).contiguous()
        pair_mask = valid[:, :, None] & valid[:, None, :]
        for layer, rel_k in zip(self.layers, relative):
            hidden = layer(hidden, rel_k, pair_mask, valid)
        return hidden, valid


class Decoder(nn.Module):
    def __init__(self, config: ParakeetTdtConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.decoder_hidden_size)
        self.lstm = nn.LSTM(
            config.decoder_hidden_size,
            config.decoder_hidden_size,
            config.num_decoder_layers,
            batch_first=True,
        )
        self.decoder_projector = nn.Linear(
            config.decoder_hidden_size, config.decoder_hidden_size
        )
        for layer in range(config.num_decoder_layers):
            self.register_buffer(
                f"_cell_weight_{layer}", torch.empty(0), persistent=False
            )
            self.register_buffer(
                f"_cell_bias_{layer}", torch.empty(0), persistent=False
            )

    def prepare_inference(self) -> None:
        for layer in range(self.lstm.num_layers):
            weight = torch.cat(
                (
                    getattr(self.lstm, f"weight_ih_l{layer}"),
                    getattr(self.lstm, f"weight_hh_l{layer}"),
                ),
                dim=1,
            ).contiguous()
            bias = (
                getattr(self.lstm, f"bias_ih_l{layer}")
                + getattr(self.lstm, f"bias_hh_l{layer}")
            ).contiguous()
            setattr(self, f"_cell_weight_{layer}", weight)
            setattr(self, f"_cell_bias_{layer}", bias)

    def forward(
        self,
        token: Tensor,
        state: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # CuDNN repacks this BF16 LSTM for every one-token call. Explicit
        # inference cells measured 39.0 ms vs 60.3 ms end-to-end on H100.
        value = self.embedding(token)[:, 0]
        if state is None:
            shape = (self.lstm.num_layers, value.shape[0], self.lstm.hidden_size)
            old_hidden = value.new_zeros(shape)
            old_cell = value.new_zeros(shape)
        else:
            old_hidden, old_cell = state
        new_hidden = []
        new_cell = []
        for layer in range(self.lstm.num_layers):
            gates = F.linear(
                torch.cat((value, old_hidden[layer]), dim=-1),
                getattr(self, f"_cell_weight_{layer}"),
                getattr(self, f"_cell_bias_{layer}"),
            )
            # One sigmoid over the whole gate row rather than three over its quarters: elementwise, so the
            # bits are the same, and a single-row decode step costs what it dispatches. Fusing the rest into
            # a Metal kernel was tried and is not shippable -- see kestrel_kernels.tdt_ops.
            gated = gates.sigmoid()
            width = old_cell.shape[-1]
            candidate = gates[..., 2 * width : 3 * width].tanh()
            cell = (
                gated[..., width : 2 * width] * old_cell[layer]
                + gated[..., :width] * candidate
            )
            value = gated[..., 3 * width :] * cell.tanh()
            new_hidden.append(value)
            new_cell.append(cell)
        state = torch.stack(new_hidden), torch.stack(new_cell)
        return self.decoder_projector(value[:, None]), state


class Joint(nn.Module):
    def __init__(self, config: ParakeetTdtConfig) -> None:
        super().__init__()
        self.head = nn.Linear(
            config.decoder_hidden_size, config.vocab_size + len(config.durations)
        )

    def forward(self, encoder: Tensor, decoder: Tensor) -> Tensor:
        return self.head(F.relu(encoder + decoder))


@dataclass(frozen=True, slots=True)
class TdtState:
    decoder_hidden: Tensor
    hidden: Tensor
    cell: Tensor
    carry: int = 0


@dataclass(frozen=True, slots=True)
class TdtOutput:
    sequences: Tensor
    durations: Tensor
    lengths: Tensor
    state: TdtState | None = None
    encoder_frame_seconds: float = 0.08


@dataclass(slots=True)
class _TdtBatchDecodeState:
    config: ParakeetTdtConfig
    valid_lengths: list[int]
    frames: list[int]
    steps_remaining: list[int]
    tokens_remaining: list[int | None]
    sequences: list[list[int]]
    durations: list[list[int]]

    @classmethod
    def create(
        cls,
        config: ParakeetTdtConfig,
        valid_lengths: list[int],
        max_tokens: int | None,
    ) -> _TdtBatchDecodeState:
        batch = len(valid_lengths)
        return cls(
            config=config,
            valid_lengths=valid_lengths,
            frames=[0] * batch,
            steps_remaining=[
                config.max_symbols_per_step * length for length in valid_lengths
            ],
            tokens_remaining=[max_tokens] * batch,
            sequences=[[config.blank_token_id] for _ in range(batch)],
            durations=[[0] for _ in range(batch)],
        )

    def active(self) -> list[bool]:
        return [
            frame < valid_length
            and steps > 0
            and (tokens is None or tokens > 0)
            for frame, valid_length, steps, tokens in zip(
                self.frames,
                self.valid_lengths,
                self.steps_remaining,
                self.tokens_remaining,
                strict=True,
            )
        ]

    def commit(
        self,
        decisions: list[list[int]],
        active: list[bool],
    ) -> list[int]:
        """Commit decisions and return rows stopped by host-only budgets."""
        newly_policy_stopped: list[int] = []
        for index, ((token_id, duration_index), is_active) in enumerate(
            zip(decisions, active, strict=True)
        ):
            if not is_active:
                continue
            duration = self.config.durations[duration_index]
            if token_id == self.config.blank_token_id and duration == 0:
                duration = 1
            self.sequences[index].append(token_id)
            self.durations[index].append(duration)
            self.frames[index] += duration
            self.steps_remaining[index] -= 1
            remaining_tokens = self.tokens_remaining[index]
            if (
                token_id != self.config.blank_token_id
                and remaining_tokens is not None
            ):
                self.tokens_remaining[index] = remaining_tokens - 1
            tokens = self.tokens_remaining[index]
            if (
                self.frames[index] < self.valid_lengths[index]
                and (
                    self.steps_remaining[index] <= 0
                    or (tokens is not None and tokens <= 0)
                )
            ):
                newly_policy_stopped.append(index)
        return newly_policy_stopped

    def output(self, device: torch.device) -> TdtOutput:
        length_values = [len(sequence) for sequence in self.sequences]
        lengths = torch.tensor(length_values, device=device)
        width = max(length_values)
        sequence_tensor = torch.tensor(
            [
                sequence
                + [self.config.blank_token_id] * (width - len(sequence))
                for sequence in self.sequences
            ],
            device=device,
        )
        duration_tensor = torch.tensor(
            [
                duration + [0] * (width - len(duration))
                for duration in self.durations
            ],
            device=device,
        )
        return TdtOutput(sequence_tensor, duration_tensor, lengths)


def _decode_batch(
    config: ParakeetTdtConfig,
    valid_lengths: list[int],
    max_tokens: int | None,
    device: torch.device,
    step: Callable[[list[int], list[bool]], list[list[int]]],
) -> TdtOutput:
    state = _TdtBatchDecodeState.create(config, valid_lengths, max_tokens)

    while True:
        active = state.active()
        if not any(active):
            break
        state.commit(step(state.frames, active), active)

    return state.output(device)


class ParakeetTdt(nn.Module):
    vad_head: VadHead | None

    def __init__(self, config: ParakeetTdtConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.encoder)
        self.encoder_projector = nn.Linear(
            config.encoder.hidden_size, config.decoder_hidden_size
        )
        self.decoder = Decoder(config)
        self.joint = Joint(config)
        # Only checkpoints that ship `vad_head.*` get one; nothing is bundled.
        self.vad_head = None

    def attach_vad_head(self) -> None:
        """Make room for a checkpoint's speech head before its tensors load."""

        self.vad_head = VadHead(self.config.encoder.hidden_size)

    @property
    def encoder_frame_seconds(self) -> float:
        # Mel frames advance one hop of 160 samples at 16 kHz, and the
        # subsampler folds `subsampling_factor` of them into one encoder frame.
        return 160 / 16_000 * self.config.encoder.subsampling_factor

    def reset_nonpersistent_buffers(self) -> None:
        self.encoder.reset_nonpersistent_buffers()
        self.decoder.prepare_inference()

    def speech_probabilities(
        self, features: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Per-frame speech probability from this checkpoint's own head.

        Only the subsampler runs, not the conformer layers. It is purely local,
        so a block of a long recording gives exactly the frames it would have
        given inside the whole file -- which is what lets the head scan an hour
        of audio for a fraction of one encoder layer.
        """

        if self.vad_head is None:
            raise ValueError("this Parakeet checkpoint carries no VAD head")
        hidden, valid = self.encoder.subsampling(features, attention_mask)
        return torch.sigmoid(self.vad_head(hidden).float()), valid

    def encode(self, features: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        encoded, valid = self.encoder(features, attention_mask)
        return self.encoder_projector(encoded), valid

    def encode_subsampled(self, hidden: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        encoded, valid = self.encoder.forward_subsampled(hidden, valid)
        return self.encoder_projector(encoded), valid

    def generate(
        self,
        features: Tensor,
        attention_mask: Tensor,
        *,
        max_tokens: int | None = None,
    ) -> TdtOutput:
        encoded, valid = self.encode(features, attention_mask)
        if features.shape[0] > 1:
            return self._generate_batch(encoded, valid, max_tokens=max_tokens)

        return self.generate_encoded(encoded, valid, max_tokens=max_tokens)

    def generate_encoded(
        self,
        encoded: Tensor,
        valid: Tensor,
        *,
        max_tokens: int | None = None,
        start_frame: int = 0,
        frame_count: int | None = None,
        state: TdtState | None = None,
    ) -> TdtOutput:
        if encoded.shape[0] != 1 or valid.shape[0] != 1:
            raise ValueError("stateful TDT decoding expects one encoded row")

        valid_length = int(valid.sum())
        end_frame = (
            valid_length
            if frame_count is None
            else min(valid_length, start_frame + frame_count)
        )
        if not 0 <= start_frame <= end_frame:
            raise ValueError("TDT decode frames are outside the encoded audio")
        token = torch.tensor([[self.config.blank_token_id]], device=encoded.device)
        if state is None:
            decoder_hidden, decoder_state = self.decoder(token, None)
            carry = 0
        else:
            decoder_hidden = state.decoder_hidden
            decoder_state = (state.hidden, state.cell)
            carry = state.carry
        frame = start_frame + carry
        sequences = [self.config.blank_token_id]
        durations = [min(carry, end_frame - start_frame)]
        steps_remaining = self.config.max_symbols_per_step * (end_frame - start_frame)
        tokens_remaining = max_tokens
        greedy_step = get_runtime().tdt.greedy_step
        while (
            frame < end_frame
            and steps_remaining > 0
            and (tokens_remaining is None or tokens_remaining > 0)
        ):
            logits = self.joint(encoded[:, frame : frame + 1], decoder_hidden)
            # Both argmaxes are one runtime op, because how many times the host waits for the GPU here is a
            # backend question: separate reads measured faster on an H100 (78.2 ms against 90.4 ms end to
            # end), one fused read is faster on MPS.
            token_id, duration_index = greedy_step(logits, self.config.vocab_size)
            duration = self.config.durations[duration_index]
            if token_id == self.config.blank_token_id and duration == 0:
                duration = 1
            sequences.append(token_id)
            durations.append(duration)
            frame += duration
            if token_id != self.config.blank_token_id:
                token.fill_(token_id)
                decoder_hidden, decoder_state = self.decoder(token, decoder_state)
                if tokens_remaining is not None:
                    tokens_remaining -= 1
            steps_remaining -= 1
        hidden, cell = decoder_state
        return TdtOutput(
            sequences=torch.tensor(
                [sequences], dtype=torch.long, device=encoded.device
            ),
            durations=torch.tensor(
                [durations], dtype=torch.long, device=encoded.device
            ),
            lengths=torch.tensor([len(sequences)], device=encoded.device),
            state=TdtState(
                decoder_hidden,
                hidden,
                cell,
                max(0, frame - end_frame),
            ),
        )

    def _generate_batch(
        self,
        encoded: Tensor,
        valid: Tensor,
        *,
        max_tokens: int | None,
    ) -> TdtOutput:
        batch = encoded.shape[0]
        valid_lengths = valid.sum(-1).tolist()
        token = torch.full(
            (batch, 1),
            self.config.blank_token_id,
            dtype=torch.long,
            device=encoded.device,
        )
        decoder_hidden, state = self.decoder(token, None)
        batch_indices = torch.arange(batch, device=encoded.device)

        def step(frames: list[int], active: list[bool]) -> list[list[int]]:
            nonlocal decoder_hidden, state
            frame_indices = torch.tensor(frames, device=encoded.device)
            logits = self.joint(
                encoded[batch_indices, frame_indices.clamp_max(encoded.shape[1] - 1)][
                    :, None
                ],
                decoder_hidden,
            )
            token_ids = logits[..., : self.config.vocab_size].argmax(-1).flatten()
            duration_indices = (
                logits[..., self.config.vocab_size :].argmax(-1).flatten()
            )
            decisions = torch.stack((token_ids, duration_indices), dim=1).tolist()

            emitted = [
                is_active and token_id != self.config.blank_token_id
                for (token_id, _duration_index), is_active in zip(
                    decisions, active, strict=True
                )
            ]
            if any(emitted):
                candidate_hidden, candidate_state = self.decoder(
                    token_ids[:, None], state
                )
                hidden_mask = torch.tensor(emitted, device=encoded.device)[
                    :, None, None
                ]
                state_mask = hidden_mask.transpose(0, 1)
                decoder_hidden = torch.where(
                    hidden_mask, candidate_hidden, decoder_hidden
                )
                state = tuple(
                    torch.where(state_mask, candidate, current)
                    for candidate, current in zip(candidate_state, state, strict=True)
                )
            return decisions

        return _decode_batch(
            self.config,
            valid_lengths,
            max_tokens,
            encoded.device,
            step,
        )


__all__ = ["ParakeetTdt", "TdtOutput", "TdtState"]
