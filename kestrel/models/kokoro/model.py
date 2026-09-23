"""Minimal checkpoint-compatible Kokoro-82M inference graph."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .albert import KokoroAlbert
from .config import KokoroConfig
from .vocoder import AdaptiveResidualBlock, Decoder


class LinearNorm(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transposed = x.transpose(1, -1)
        normalized = F.layer_norm(
            transposed, (self.channels,), self.gamma, self.beta, self.eps
        )
        return normalized.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels: int, kernel_size: int, depth: int, n_symbols: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    ChannelLayerNorm(channels),
                    nn.LeakyReLU(0.2),
                )
                for _ in range(depth)
            ]
        )
        self.lstm = nn.LSTM(
            channels,
            channels // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids).transpose(1, 2)
        for convolution in self.cnn:
            hidden = convolution(hidden)
        hidden, _ = self.lstm(hidden.transpose(1, 2))
        return hidden.transpose(1, 2)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(style).chunk(2, dim=1)
        normalized = F.layer_norm(x, (self.channels,), eps=self.eps)
        return (1.0 + gamma.unsqueeze(1)) * normalized + beta.unsqueeze(1)


class DurationEncoder(nn.Module):
    def __init__(self, style_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(layers):
            self.lstms.append(
                nn.LSTM(
                    hidden_dim + style_dim,
                    hidden_dim // 2,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.lstms.append(AdaLayerNorm(style_dim, hidden_dim))

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        expanded_style = style.unsqueeze(1).expand(
            sequence.shape[0], sequence.shape[1], -1
        )
        sequence = torch.cat((sequence, expanded_style), dim=-1)
        for block in self.lstms:
            if isinstance(block, nn.LSTM):
                sequence, _ = block(sequence)
            else:
                sequence = block(sequence, style)
                sequence = torch.cat((sequence, expanded_style), dim=-1)
        return sequence


class ProsodyPredictor(nn.Module):
    def __init__(
        self,
        style_dim: int,
        hidden_dim: int,
        layers: int,
        max_duration: int,
    ) -> None:
        super().__init__()
        self.text_encoder = DurationEncoder(style_dim, hidden_dim, layers)
        self.lstm = nn.LSTM(
            hidden_dim + style_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.duration_proj = LinearNorm(hidden_dim, max_duration)
        self.shared = nn.LSTM(
            hidden_dim + style_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.F0 = nn.ModuleList(
            [
                AdaptiveResidualBlock(hidden_dim, hidden_dim, style_dim),
                AdaptiveResidualBlock(
                    hidden_dim, hidden_dim // 2, style_dim, upsample=True
                ),
                AdaptiveResidualBlock(hidden_dim // 2, hidden_dim // 2, style_dim),
            ]
        )
        self.N = nn.ModuleList(
            [
                AdaptiveResidualBlock(hidden_dim, hidden_dim, style_dim),
                AdaptiveResidualBlock(
                    hidden_dim, hidden_dim // 2, style_dim, upsample=True
                ),
                AdaptiveResidualBlock(hidden_dim // 2, hidden_dim // 2, style_dim),
            ]
        )
        self.F0_proj = nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)
        self.N_proj = nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)

    def f0_and_noise(
        self, encoded: torch.Tensor, style: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared, _ = self.shared(encoded.transpose(1, 2))
        shared = shared.transpose(1, 2)
        f0 = shared
        for block in self.F0:
            f0 = block(f0, style)
        noise = shared
        for block in self.N:
            noise = block(noise, style)
        return self.F0_proj(f0).squeeze(1), self.N_proj(noise).squeeze(1)


@dataclass(frozen=True, slots=True)
class KokoroOutput:
    audio: torch.Tensor
    durations: torch.Tensor


class KokoroModel(nn.Module):
    """Batch-one Kokoro V1 synthesis model."""

    def __init__(self, config: KokoroConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab = dict(config.vocab)
        self.bert = KokoroAlbert(config.albert)
        self.bert_encoder = nn.Linear(config.albert.hidden_size, config.hidden_dim)
        self.predictor = ProsodyPredictor(
            config.style_dim,
            config.hidden_dim,
            config.n_layer,
            config.max_dur,
        )
        self.decoder = Decoder(config.hidden_dim, config.style_dim, config.istftnet)
        self.text_encoder = TextEncoder(
            config.hidden_dim,
            config.text_encoder_kernel_size,
            config.n_layer,
            config.n_token,
        )

    @property
    def device(self) -> torch.device:
        return self.bert.device

    @property
    def context_length(self) -> int:
        return self.config.albert.max_position_embeddings

    def encode_phonemes(self, phonemes: str) -> torch.Tensor:
        if not isinstance(phonemes, str) or not phonemes:
            raise ValueError("phonemes must be a non-empty string")
        token_ids = [self.vocab[symbol] for symbol in phonemes if symbol in self.vocab]
        if not token_ids:
            raise ValueError("phonemes contain no symbols in the Kokoro vocabulary")
        if len(token_ids) + 2 > self.context_length:
            raise ValueError(
                f"phonemes encode to {len(token_ids)} tokens; maximum is "
                f"{self.context_length - 2}"
            )
        return torch.tensor(
            [[0, *token_ids, 0]], dtype=torch.long, device=self.device
        )

    @torch.inference_mode()
    def forward_tokens(
        self,
        input_ids: torch.Tensor,
        reference_style: torch.Tensor,
        speed: float = 1.0,
    ) -> KokoroOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Kokoro input_ids must have shape [1, sequence]")
        if reference_style.shape != (1, 256):
            raise ValueError("Kokoro reference style must have shape [1, 256]")
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not math.isfinite(speed)
            or speed <= 0.0
        ):
            raise ValueError("speed must be a finite positive number")

        duration_hidden = self.bert_encoder(self.bert(input_ids)).transpose(1, 2)
        prosody_style = reference_style[:, 128:]
        duration_features = self.predictor.text_encoder(
            duration_hidden, prosody_style
        )
        duration_lstm, _ = self.predictor.lstm(duration_features)
        duration_logits = self.predictor.duration_proj(duration_lstm)
        durations = (
            torch.sigmoid(duration_logits).sum(dim=-1) / float(speed)
        ).round().clamp(min=1).long().squeeze(0)

        token_indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=self.device), durations
        )
        prosody = duration_features.transpose(1, 2).index_select(
            -1,
            token_indices,
        )
        f0, noise = self.predictor.f0_and_noise(prosody, prosody_style)
        text = self.text_encoder(input_ids).index_select(-1, token_indices)
        audio = self.decoder(text, f0, noise, reference_style[:, :128]).reshape(-1)
        return KokoroOutput(audio=audio, durations=durations)

    def forward(
        self,
        phonemes: str,
        reference_style: torch.Tensor,
        speed: float = 1.0,
    ) -> KokoroOutput:
        return self.forward_tokens(
            self.encode_phonemes(phonemes),
            reference_style.to(device=self.device, dtype=self.bert_encoder.weight.dtype),
            speed,
        )
__all__ = ["KokoroModel", "KokoroOutput"]
