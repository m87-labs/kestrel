"""Inference-only Parakeet FastConformer + token-duration transducer."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ParakeetEncoderConfig, ParakeetTdtConfig


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
        return self.linear2(F.silu(self.linear1(hidden)))


class Convolution(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        channels = config.hidden_size
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            config.conv_kernel_size,
            padding=(config.conv_kernel_size - 1) // 2,
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1, bias=False)

    def forward(self, hidden: Tensor, valid: Tensor | None) -> Tensor:
        hidden = F.glu(self.pointwise_conv1(hidden.transpose(1, 2)), dim=1)
        if valid is not None:
            hidden = hidden.masked_fill(~valid[:, None], 0)
        hidden = self.depthwise_conv(hidden)
        hidden = self.pointwise_conv2(F.silu(self.norm(hidden)))
        return hidden.transpose(1, 2)


class RelativeAttention(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.relative_k_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

    @staticmethod
    def _relative_shift(scores: Tensor) -> Tensor:
        batch, heads, query, positions = scores.shape
        scores = F.pad(scores, (1, 0)).view(batch, heads, -1, query)
        return scores[:, :, 1:].view(batch, heads, query, positions)

    def forward(self, hidden: Tensor, positions: Tensor, mask: Tensor | None) -> Tensor:
        batch, length, _ = hidden.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        q = self.q_proj(hidden).view(shape).transpose(1, 2)
        k = self.k_proj(hidden).view(shape).transpose(1, 2)
        v = self.v_proj(hidden).view(shape).transpose(1, 2)
        if mask is not None:
            key_valid = mask[:, :1, :].transpose(1, 2)
            k = k.masked_fill(~key_valid[:, None], 0)
            v = v.masked_fill(~key_valid[:, None], 0)
        relative_k = self.relative_k_proj(positions).view(
            batch, -1, self.num_heads, self.head_dim
        )
        relative = (q + self.bias_v[None, :, None]) @ relative_k.permute(0, 2, 3, 1)
        relative = self._relative_shift(relative)[..., :length] * self.scale
        content = (q + self.bias_u[None, :, None]) @ k.transpose(-1, -2) * self.scale
        scores = content + relative
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :, :], float("-inf"))
        probabilities = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
        attended = (probabilities @ v).transpose(1, 2).reshape(batch, length, -1)
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
        positions: Tensor,
        pair_mask: Tensor | None,
        valid: Tensor | None,
    ) -> Tensor:
        hidden = hidden + 0.5 * self.feed_forward1(self.norm_feed_forward1(hidden))
        hidden = hidden + self.self_attn(
            self.norm_self_att(hidden), positions, pair_mask
        )
        hidden = hidden + self.conv(self.norm_conv(hidden), valid)
        hidden = hidden + 0.5 * self.feed_forward2(self.norm_feed_forward2(hidden))
        return self.norm_out(hidden)


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
        for layer in self.layers:
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
        inverse = 1 / (
            10_000
            ** (
                torch.arange(0, config.hidden_size, 2, dtype=torch.float32)
                / config.hidden_size
            )
        )
        self.register_buffer("inverse_frequency", inverse, persistent=False)

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
        length = hidden.shape[1]
        relative_positions = torch.arange(length - 1, -length, -1, device=hidden.device)
        phase = torch.outer(relative_positions.float(), self.inverse_frequency)
        positions = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        positions = positions[None].expand(hidden.shape[0], -1, -1).to(hidden.dtype)
        pair_mask = valid[:, :, None] & valid[:, None, :]
        for layer in self.layers:
            hidden = layer(hidden, positions, pair_mask, valid)
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
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
            cell = (
                forget_gate.sigmoid() * old_cell[layer]
                + input_gate.sigmoid() * candidate.tanh()
            )
            value = output_gate.sigmoid() * cell.tanh()
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


def _decode_batch(
    config: ParakeetTdtConfig,
    valid_lengths: list[int],
    max_tokens: int | None,
    device: torch.device,
    step: Callable[[list[int], list[bool]], list[list[int]]],
) -> TdtOutput:
    batch = len(valid_lengths)
    frames = [0] * batch
    steps_remaining = [
        config.max_symbols_per_step * length for length in valid_lengths
    ]
    tokens_remaining = [max_tokens] * batch
    sequences = [[config.blank_token_id] for _ in range(batch)]
    durations = [[0] for _ in range(batch)]

    while True:
        active = [
            frame < valid_length
            and steps > 0
            and (tokens is None or tokens > 0)
            for frame, valid_length, steps, tokens in zip(
                frames,
                valid_lengths,
                steps_remaining,
                tokens_remaining,
                strict=True,
            )
        ]
        if not any(active):
            break
        decisions = step(frames, active)
        for index, ((token_id, duration_index), is_active) in enumerate(
            zip(decisions, active, strict=True)
        ):
            if not is_active:
                continue
            duration = config.durations[duration_index]
            if token_id == config.blank_token_id and duration == 0:
                duration = 1
            sequences[index].append(token_id)
            durations[index].append(duration)
            frames[index] += duration
            steps_remaining[index] -= 1
            remaining_tokens = tokens_remaining[index]
            if token_id != config.blank_token_id and remaining_tokens is not None:
                tokens_remaining[index] = remaining_tokens - 1

    lengths = torch.tensor([len(sequence) for sequence in sequences], device=device)
    width = int(lengths.max())
    sequence_tensor = torch.tensor(
        [
            sequence + [config.blank_token_id] * (width - len(sequence))
            for sequence in sequences
        ],
        device=device,
    )
    duration_tensor = torch.tensor(
        [duration + [0] * (width - len(duration)) for duration in durations],
        device=device,
    )
    return TdtOutput(sequence_tensor, duration_tensor, lengths)


class ParakeetTdt(nn.Module):
    def __init__(self, config: ParakeetTdtConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.encoder)
        self.encoder_projector = nn.Linear(
            config.encoder.hidden_size, config.decoder_hidden_size
        )
        self.decoder = Decoder(config)
        self.joint = Joint(config)

    def reset_nonpersistent_buffers(self) -> None:
        self.encoder.reset_nonpersistent_buffers()
        self.decoder.prepare_inference()

    def encode(self, features: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        encoded, valid = self.encoder(features, attention_mask)
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
        while (
            frame < end_frame
            and steps_remaining > 0
            and (tokens_remaining is None or tokens_remaining > 0)
        ):
            logits = self.joint(encoded[:, frame : frame + 1], decoder_hidden)
            # Tried stacking both argmax results into one host read: 90.4 ms
            # vs 78.2 ms end-to-end on H100; keeping the separate reads.
            token_id = int(logits[..., : self.config.vocab_size].argmax())
            duration_index = int(logits[..., self.config.vocab_size :].argmax())
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
