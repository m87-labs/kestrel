"""Direct loader for pinned Qwen3-TTS CustomVoice checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .codec import Qwen3TTSCodecDecoder, _SnakeBeta
from .config import Qwen3TTSConfig
from .model import Qwen3TTSCodePredictor, Qwen3TTSTalkerModel
from .text import Qwen3TTSTextProcessor
from .weights import load_module_from_safetensors, resolve_checkpoint_files


@dataclass(frozen=True, slots=True)
class LoadedQwen3TTS:
    config: Qwen3TTSConfig
    talker: Qwen3TTSTalkerModel
    code_predictor: Qwen3TTSCodePredictor
    codec: Qwen3TTSCodecDecoder
    text_processor: Qwen3TTSTextProcessor


def load_qwen3_tts(
    checkpoint: str | Path,
    *,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
) -> LoadedQwen3TTS:
    """Load the model in ``dtype``, retaining SnakeBeta parameters in float32."""

    resolved_device = torch.device(device)
    files = resolve_checkpoint_files(
        checkpoint,
        revision=revision,
        local_files_only=local_files_only,
    )
    config = Qwen3TTSConfig.from_directory(files.root)
    text_processor = Qwen3TTSTextProcessor(files.root)

    # Construct on meta, establish the serving dtype there, then allocate each
    # destination tensor once on its final device. The safetensor copier streams
    # one source tensor at a time, avoiding a second model-sized state dict.
    with torch.device("meta"):
        talker = Qwen3TTSTalkerModel(config).to(dtype=dtype)
        code_predictor = Qwen3TTSCodePredictor(config).to(dtype=dtype)
        codec = Qwen3TTSCodecDecoder(config).to(dtype=dtype)
        for module in codec.modules():
            if isinstance(module, _SnakeBeta):
                module.float()

    talker.to_empty(device=resolved_device)
    load_module_from_safetensors(
        talker,
        files.model_weights,
        prefix="talker.",
        excluded_prefixes=("talker.code_predictor.",),
    )
    code_predictor.to_empty(device=resolved_device)
    load_module_from_safetensors(
        code_predictor,
        files.model_weights,
        prefix="talker.code_predictor.",
    )
    codec.to_empty(device=resolved_device)
    load_module_from_safetensors(
        codec,
        files.codec_weights,
        prefix="decoder.",
    )
    codec.prepare_for_inference()

    talker.eval()
    code_predictor.eval()
    codec.eval()
    return LoadedQwen3TTS(
        config=config,
        talker=talker,
        code_predictor=code_predictor,
        codec=codec,
        text_processor=text_processor,
    )


__all__ = ["LoadedQwen3TTS", "load_qwen3_tts"]
