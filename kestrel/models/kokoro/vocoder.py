"""Inference-only Kokoro ISTFTNet decoder.

The module hierarchy intentionally matches the published V1 checkpoint. The
network is adapted from the Apache-licensed ``hexgrad/kokoro`` inference code;
training initializers, export wrappers, and unused discriminator paths are not
included.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .config import IstftNetConfig


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class AdaIN1d(nn.Module):
    def __init__(self, style_dim: int, channels: int) -> None:
        super().__init__()
        # The checkpoint was trained without affine InstanceNorm parameters.
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(style).unsqueeze(-1).chunk(2, dim=1)
        return (1.0 + gamma) * self.norm(x) + beta


class AdaINResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: Sequence[int],
        style_dim: int,
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=value,
                    padding=_padding(kernel_size, value),
                )
                for value in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=_padding(kernel_size),
                )
                for _ in dilation
            ]
        )
        self.adain1 = nn.ModuleList(
            [AdaIN1d(style_dim, channels) for _ in dilation]
        )
        self.adain2 = nn.ModuleList(
            [AdaIN1d(style_dim, channels) for _ in dilation]
        )
        self.alpha1 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation]
        )
        self.alpha2 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation]
        )

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, norm1, norm2, alpha1, alpha2 in zip(
            self.convs1,
            self.convs2,
            self.adain1,
            self.adain2,
            self.alpha1,
            self.alpha2,
        ):
            residual = norm1(x, style)
            residual = residual + torch.sin(alpha1 * residual).square() / alpha1
            residual = conv1(residual)
            residual = norm2(residual, style)
            residual = residual + torch.sin(alpha2 * residual).square() / alpha2
            x = x + conv2(residual)
        return x


class TorchSTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _window(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.hann_window(
            self.n_fft,
            periodic=True,
            dtype=reference.dtype,
            device=reference.device,
        )

    def transform(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=self._window(waveform),
            return_complex=True,
        )
        return transformed.abs(), transformed.angle()

    def inverse(
        self, magnitude: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        transformed = magnitude * torch.exp(phase * 1j)
        waveform = torch.istft(
            transformed,
            self.n_fft,
            self.hop_length,
            self.n_fft,
            window=self._window(magnitude),
        )
        return waveform.unsqueeze(1)


class SineGenerator(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        upsample_scale: int,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 10.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.upsample_scale = upsample_scale
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def _sines(self, frequencies: torch.Tensor) -> torch.Tensor:
        radians = (frequencies / self.sample_rate).remainder(1.0)
        initial = torch.rand(
            frequencies.shape[0], frequencies.shape[2], device=frequencies.device
        )
        initial[:, 0] = 0.0
        radians[:, 0] = radians[:, 0] + initial
        downsampled = F.interpolate(
            radians.transpose(1, 2),
            scale_factor=1.0 / self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(downsampled, dim=1) * (2.0 * math.pi)
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        harmonics = torch.arange(
            1,
            self.harmonic_num + 2,
            dtype=f0.dtype,
            device=f0.device,
        ).view(1, 1, -1)
        frequencies = f0 * harmonics
        voiced = (f0 > self.voiced_threshold).to(torch.float32)
        waves = self._sines(frequencies) * self.sine_amp
        noise_amplitude = (
            voiced * self.noise_std + (1.0 - voiced) * self.sine_amp / 3.0
        )
        return waves * voiced + noise_amplitude * torch.randn_like(waves)


class SourceModule(nn.Module):
    def __init__(self, upsample_scale: int) -> None:
        super().__init__()
        self.l_sin_gen = SineGenerator(24000, upsample_scale)
        self.l_linear = nn.Linear(9, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        return self.l_tanh(self.l_linear(self.l_sin_gen(f0)))


class Generator(nn.Module):
    def __init__(self, style_dim: int, config: IstftNetConfig) -> None:
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        scale = math.prod(config.upsample_rates) * config.gen_istft_hop_size
        self.m_source = SourceModule(scale)
        self.f0_upsamp = nn.Upsample(scale_factor=scale)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            input_channels = config.upsample_initial_channel // (2**index)
            output_channels = input_channels // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    kernel,
                    rate,
                    padding=(kernel - rate) // 2,
                )
            )
            if index + 1 < len(config.upsample_rates):
                stride = math.prod(config.upsample_rates[index + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        config.gen_istft_n_fft + 2,
                        output_channels,
                        kernel_size=stride * 2,
                        stride=stride,
                        padding=(stride + 1) // 2,
                    )
                )
                noise_kernel = 7
            else:
                self.noise_convs.append(
                    nn.Conv1d(
                        config.gen_istft_n_fft + 2, output_channels, kernel_size=1
                    )
                )
                noise_kernel = 11
            self.noise_res.append(
                AdaINResBlock(
                    output_channels, noise_kernel, (1, 3, 5), style_dim
                )
            )

        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = config.upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilations in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes
            ):
                self.resblocks.append(
                    AdaINResBlock(channels, kernel, dilations, style_dim)
                )

        self.post_n_fft = config.gen_istft_n_fft
        final_channels = config.upsample_initial_channel // (
            2 ** len(config.upsample_rates)
        )
        self.conv_post = nn.Conv1d(
            final_channels, self.post_n_fft + 2, 7, padding=3
        )
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(
            config.gen_istft_n_fft, config.gen_istft_hop_size
        )

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, f0: torch.Tensor
    ) -> torch.Tensor:
        upsampled_f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        harmonic_source = self.m_source(upsampled_f0).transpose(1, 2).squeeze(1)
        magnitude, phase = self.stft.transform(harmonic_source)
        harmonic = torch.cat((magnitude, phase), dim=1)

        for index in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            source = self.noise_res[index](
                self.noise_convs[index](harmonic), style
            )
            x = self.ups[index](x)
            if index == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + source
            residual_sum = self.resblocks[index * self.num_kernels](x, style)
            for offset in range(1, self.num_kernels):
                residual_sum = residual_sum + self.resblocks[
                    index * self.num_kernels + offset
                ](x, style)
            x = residual_sum / self.num_kernels

        x = self.conv_post(F.leaky_relu(x))
        split = self.post_n_fft // 2 + 1
        magnitude = torch.exp(x[:, :split])
        phase = torch.sin(x[:, split:])
        return self.stft.inverse(magnitude, phase)


class AdaptiveResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        style_dim: int,
        *,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.upsample_type = upsample
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="nearest")
            if upsample
            else nn.Identity()
        )
        self.learned_sc = input_dim != output_dim
        self.conv1 = nn.Conv1d(input_dim, output_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(output_dim, output_dim, 3, padding=1)
        self.norm1 = AdaIN1d(style_dim, input_dim)
        self.norm2 = AdaIN1d(style_dim, output_dim)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(input_dim, output_dim, 1, bias=False)
        self.pool = (
            nn.ConvTranspose1d(
                input_dim,
                input_dim,
                kernel_size=3,
                stride=2,
                groups=input_dim,
                padding=1,
                output_padding=1,
            )
            if upsample
            else nn.Identity()
        )

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.conv1x1(x) if self.learned_sc else x

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        residual = self.norm1(x, style)
        residual = self.pool(F.leaky_relu(residual, negative_slope=0.2))
        residual = self.conv1(residual)
        residual = self.norm2(residual, style)
        residual = self.conv2(F.leaky_relu(residual, negative_slope=0.2))
        return (residual + self._shortcut(x)) / math.sqrt(2.0)


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        style_dim: int,
        config: IstftNetConfig,
    ) -> None:
        super().__init__()
        self.encode = AdaptiveResidualBlock(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList(
            [
                AdaptiveResidualBlock(1090, 1024, style_dim),
                AdaptiveResidualBlock(1090, 1024, style_dim),
                AdaptiveResidualBlock(1090, 1024, style_dim),
                AdaptiveResidualBlock(1090, 512, style_dim, upsample=True),
            ]
        )
        self.F0_conv = nn.Conv1d(
            1, 1, kernel_size=3, stride=2, groups=1, padding=1
        )
        self.N_conv = nn.Conv1d(
            1, 1, kernel_size=3, stride=2, groups=1, padding=1
        )
        self.asr_res = nn.Sequential(nn.Conv1d(512, 64, kernel_size=1))
        self.generator = Generator(style_dim, config)

    def forward(
        self,
        asr: torch.Tensor,
        f0_curve: torch.Tensor,
        noise: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        f0 = self.F0_conv(f0_curve.unsqueeze(1))
        noise = self.N_conv(noise.unsqueeze(1))
        x = self.encode(torch.cat((asr, f0, noise), dim=1), style)
        asr_residual = self.asr_res(asr)
        concatenate_residual = True
        for block in self.decode:
            if concatenate_residual:
                x = torch.cat((x, asr_residual, f0, noise), dim=1)
            x = block(x, style)
            if block.upsample_type:
                concatenate_residual = False
        return self.generator(x, style, f0_curve)


__all__ = ["Decoder"]
