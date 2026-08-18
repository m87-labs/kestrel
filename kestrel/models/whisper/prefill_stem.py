"""Optimized fixed-shape Whisper large-v3-turbo audio stem."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kestrel_kernels import get_runtime

_GELU = get_runtime().gelu

MEL_BINS = 128
INPUT_FRAMES = 3000
HIDDEN_SIZE = 1280
ENCODER_FRAMES = 1500


@dataclass(frozen=True)
class WhisperAudioStemWeights:
    """Inference-ready audio-stem weights, packed once at model load."""

    conv1_weight: torch.Tensor
    conv1_bias: torch.Tensor
    conv2_weight: torch.Tensor
    conv2_bias: torch.Tensor
    position_embedding: torch.Tensor


def _check_weight_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"{name} must be bfloat16 or float32, got {tensor.dtype}")


def prepare_whisper_audio_stem_weights(
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    position_embedding: torch.Tensor,
) -> WhisperAudioStemWeights:
    """Validate and prepack standard Conv1d checkpoint tensors.

    The checkpoint-facing convolution weights are ``[out, in, 3]``.  This
    converts them once to channels-last ``[out, in, 3, 1]`` tensors for the
    selected cuDNN Conv2d implementation. Checkpoint key lookup remains in the
    model's public weight loader rather than the kernel-facing stem.
    """

    device = conv1_weight.device
    _check_weight_tensor(
        "conv1_weight", conv1_weight, (HIDDEN_SIZE, MEL_BINS, 3), device
    )
    _check_weight_tensor("conv1_bias", conv1_bias, (HIDDEN_SIZE,), device)
    _check_weight_tensor(
        "conv2_weight", conv2_weight, (HIDDEN_SIZE, HIDDEN_SIZE, 3), device
    )
    _check_weight_tensor("conv2_bias", conv2_bias, (HIDDEN_SIZE,), device)
    _check_weight_tensor(
        "position_embedding",
        position_embedding,
        (ENCODER_FRAMES, HIDDEN_SIZE),
        device,
    )

    def pack_conv(weight: torch.Tensor) -> torch.Tensor:
        return (
            weight.detach()
            .to(dtype=torch.bfloat16)
            .unsqueeze(-1)
            .contiguous(memory_format=torch.channels_last)
        )

    return WhisperAudioStemWeights(
        conv1_weight=pack_conv(conv1_weight),
        conv1_bias=conv1_bias.detach().to(dtype=torch.bfloat16).contiguous(),
        conv2_weight=pack_conv(conv2_weight),
        conv2_bias=conv2_bias.detach().to(dtype=torch.bfloat16).contiguous(),
        position_embedding=(
            position_embedding.detach().to(dtype=torch.bfloat16).contiguous()
        ),
    )


def _validate_prepared_weights(weights: WhisperAudioStemWeights) -> None:
    device = weights.conv1_weight.device
    expected = (
        ("conv1_weight", weights.conv1_weight, (HIDDEN_SIZE, MEL_BINS, 3, 1)),
        ("conv1_bias", weights.conv1_bias, (HIDDEN_SIZE,)),
        (
            "conv2_weight",
            weights.conv2_weight,
            (HIDDEN_SIZE, HIDDEN_SIZE, 3, 1),
        ),
        ("conv2_bias", weights.conv2_bias, (HIDDEN_SIZE,)),
        (
            "position_embedding",
            weights.position_embedding,
            (ENCODER_FRAMES, HIDDEN_SIZE),
        ),
    )
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"prepared {name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.device != device:
            raise ValueError(
                f"prepared {name} must be on {device}, got {tensor.device}"
            )
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"prepared {name} must be bfloat16, got {tensor.dtype}")
    if not weights.conv1_weight.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("prepared conv1_weight must be channels-last contiguous")
    if not weights.conv2_weight.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("prepared conv2_weight must be channels-last contiguous")


def _postprocess(
    inp: torch.Tensor,
    *,
    frames: int,
    position: torch.Tensor | None,
    out: torch.Tensor | None = None,
    require_native: bool = False,
    require_packed: bool = False,
) -> torch.Tensor:
    if tuple(inp.shape[-2:]) != (frames, HIDDEN_SIZE):
        raise ValueError(
            f"postprocess input must end in {(frames, HIDDEN_SIZE)}, "
            f"got {tuple(inp.shape)}"
        )
    if not inp.is_contiguous():
        raise ValueError("postprocess input must be contiguous [batch, frames, hidden]")
    if inp.dtype != torch.bfloat16:
        raise ValueError(f"postprocess input must be bfloat16, got {inp.dtype}")
    if position is not None:
        if tuple(position.shape) != (frames, HIDDEN_SIZE):
            raise ValueError(
                f"position must have shape {(frames, HIDDEN_SIZE)}, "
                f"got {tuple(position.shape)}"
            )
        if position.device != inp.device or position.dtype != inp.dtype:
            raise ValueError(
                "position must match the postprocess input device and dtype"
            )
        if not position.is_contiguous():
            raise ValueError("position must be contiguous")
    if out is None:
        out = torch.empty_like(inp)
    elif out.shape != inp.shape or out.device != inp.device or out.dtype != inp.dtype:
        raise ValueError(
            "out must match the postprocess input shape, device, and dtype"
        )
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")

    # Tried leaving both exact GELUs and the position add as torch ops. At B1
    # the custom path's Python dispatch was slower eagerly (H100 158.1 vs
    # 142.0 us; B200 93.5 vs 86.0 us), but the production CUDA graph was
    # faster (101.5 vs 104.8 us; 55.46 vs 55.74 us). At B8 it won both eager
    # and graphed: H100 626.9/596.1 vs 665.4/642.8 us, B200 304.4/276.6 vs
    # 321.6/294.2 us. Keeping the generic GELU/add epilogue for graph serving.
    return _GELU.gelu_cute(
        inp,
        out=out,
        add=position,
        require_native=require_native,
        require_packed=require_packed,
    )


def whisper_audio_stem(
    input_features: torch.Tensor,
    weights: WhisperAudioStemWeights,
    *,
    out: torch.Tensor | None = None,
    require_native: bool = False,
    require_packed: bool = False,
) -> torch.Tensor:
    """Run the fixed Whisper large-v3-turbo convolutional audio stem.

    ``input_features`` is BF16 or FP32 ``[B, 128, 3000]``.  The result is
    BF16 contiguous ``[B, 1500, 1280]`` with the learned positions added.
    ``out`` can be supplied to keep the final pointer stable for CUDA graphs.
    """

    if input_features.ndim != 3 or tuple(input_features.shape[1:]) != (
        MEL_BINS,
        INPUT_FRAMES,
    ):
        raise ValueError(
            "input_features must have shape [batch, 128, 3000], got "
            f"{tuple(input_features.shape)}"
        )
    if input_features.shape[0] <= 0:
        raise ValueError("input_features batch must be positive")
    if input_features.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"input_features must be bfloat16 or float32, got {input_features.dtype}"
        )
    _validate_prepared_weights(weights)
    if weights.conv1_weight.device != input_features.device:
        raise ValueError(
            "input_features and prepared weights must be on the same device, got "
            f"{input_features.device} and {weights.conv1_weight.device}"
        )

    batch = int(input_features.shape[0])
    if out is not None:
        expected_out = (batch, ENCODER_FRAMES, HIDDEN_SIZE)
        if (
            tuple(out.shape) != expected_out
            or out.device != input_features.device
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous bfloat16 with shape "
                f"{expected_out} on {input_features.device}"
            )

    # Physical NHWC for W=1 is exactly the contiguous [B, T, C] layout that
    # the post kernels and encoder consume.  Tried NCL Conv1d and explicit
    # im2col+linear under flock on H100/B200 (2026-08-09): at B1 graphed they
    # took 179.9/152.6 us and 79.0/60.1 us, versus 69.4/56.3 us here; at B8
    # they took 813.0/547.5 us and 584.0/415.3 us, versus 470.4/296.0 us.
    # Keeping channels-last k3x1 cuDNN Conv2d.
    packed_input = (
        input_features.to(dtype=torch.bfloat16)
        .unsqueeze(-1)
        .contiguous(memory_format=torch.channels_last)
    )
    conv1 = F.conv2d(
        packed_input,
        weights.conv1_weight,
        weights.conv1_bias,
        padding=(1, 0),
    )
    conv1_ntc = conv1.squeeze(-1).transpose(1, 2)
    activated1 = _postprocess(
        conv1_ntc,
        frames=INPUT_FRAMES,
        position=None,
        require_native=require_native,
        require_packed=require_packed,
    )

    conv2_input = activated1.transpose(1, 2).unsqueeze(-1)
    conv2 = F.conv2d(
        conv2_input,
        weights.conv2_weight,
        weights.conv2_bias,
        stride=(2, 1),
        padding=(1, 0),
    )
    conv2_ntc = conv2.squeeze(-1).transpose(1, 2)
    return _postprocess(
        conv2_ntc,
        frames=ENCODER_FRAMES,
        position=weights.position_embedding,
        out=out,
        require_native=require_native,
        require_packed=require_packed,
    )


__all__ = [
    "ENCODER_FRAMES",
    "HIDDEN_SIZE",
    "INPUT_FRAMES",
    "MEL_BINS",
    "WhisperAudioStemWeights",
    "prepare_whisper_audio_stem_weights",
    "whisper_audio_stem",
]
