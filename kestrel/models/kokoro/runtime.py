"""Kestrel single-pass runtime for Kokoro-82M speech synthesis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from kestrel.device import empty_cache, resolve_device
from kestrel.runtime import ExecutionShape

from .contract import KokoroSynthesisRequest
from .model import KokoroModel, KokoroOutput
from .weights import (
    DEFAULT_KOKORO_MODEL,
    DEFAULT_KOKORO_REPO_ID,
    DEFAULT_KOKORO_REVISION,
    VoiceStore,
    load_kokoro,
)


SAMPLE_RATE = 24000


def _prepared_synthesis(
    value: object,
) -> KokoroSynthesisRequest:
    """Validate the model-owned payload prepared outside the scheduler."""

    if not isinstance(value, Mapping) or set(value) != {"_prepared"}:
        raise ValueError(
            "Kokoro leaves require model-prepared synthesis inputs; use "
            "ModelHandle.synthesize()"
        )
    prepared = value["_prepared"]
    if not isinstance(prepared, KokoroSynthesisRequest):
        raise TypeError("Kokoro leaf payload has an invalid prepared value")
    return prepared


class KokoroRuntime:
    """Batch-one exact eager baseline serving ``synthesize``."""

    execution_shape = ExecutionShape.SINGLE_PASS
    batch_capacity = 1

    def __init__(
        self,
        cfg: Any,
        *,
        compute_stream: Any = None,
        kv_pool: Any = None,
        max_lora_rank: int | None = None,
        model: KokoroModel,
        voices: VoiceStore,
    ) -> None:
        del kv_pool, max_lora_rank
        self._model_name = getattr(cfg, "model", DEFAULT_KOKORO_MODEL)
        self.device = resolve_device(
            cfg.resolved_device()
            if hasattr(cfg, "resolved_device")
            else getattr(cfg, "device", "cpu")
        )
        # V1's published checkpoint and reference path are FP32. The direct
        # baseline keeps that contract; compiled lower precision is a later path.
        self.dtype = torch.float32
        self.compute_stream = compute_stream
        self.model = model.eval()
        self.voices = voices
        self._shutdown = False

    @property
    def model_name(self) -> str:
        return self._model_name

    def tasks(self) -> tuple[str, ...]:
        return ("synthesize",)

    @torch.inference_mode()
    def forward(
        self, task: str, inputs: Sequence[Any]
    ) -> tuple[dict[str, object], ...]:
        if self._shutdown:
            raise RuntimeError("KokoroRuntime is shut down")
        if task != "synthesize":
            raise ValueError(f"KokoroRuntime does not support task {task!r}")
        if len(inputs) != 1:
            raise ValueError("KokoroRuntime serves one request per forward")
        prepared = _prepared_synthesis(inputs[0])
        reference = self.voices.style(prepared.voice, len(prepared.phonemes))
        output = self.model(prepared.phonemes, reference, prepared.speed)
        if not isinstance(output, KokoroOutput):
            raise TypeError(
                f"Kokoro model returned {type(output).__name__}, expected KokoroOutput"
            )
        output_audio = output.audio.detach().reshape(-1)
        if output_audio.device.type == "cpu":
            waveform = output_audio.to(dtype=torch.float32)
        elif output_audio.device.type == "cuda":
            # Keep the terminal transfer asynchronous; the executor's
            # completion event covers this stream-ordered copy.
            waveform = torch.empty(
                output_audio.shape,
                device="cpu",
                dtype=torch.float32,
                pin_memory=True,
            )
            waveform.copy_(output_audio, non_blocking=True)
        else:
            waveform = output_audio.to(dtype=torch.float32)
        return (
            {
                "audio": waveform,
                "sample_rate": SAMPLE_RATE,
                "duration_seconds": waveform.numel() / SAMPLE_RATE,
                "voice": prepared.voice,
            },
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        empty_cache(self.device)


def create_kokoro_runtime(
    cfg: Any,
    *,
    compute_stream: Any = None,
    kv_pool: Any = None,
    max_lora_rank: int | None = None,
) -> KokoroRuntime:
    checkpoint = getattr(cfg, "model_path", None)
    device = resolve_device(
        cfg.resolved_device()
        if hasattr(cfg, "resolved_device")
        else getattr(cfg, "device", "cpu")
    )
    loaded = load_kokoro(
        checkpoint or DEFAULT_KOKORO_REPO_ID,
        revision=DEFAULT_KOKORO_REVISION,
        device=device,
    )
    return KokoroRuntime(
        cfg,
        compute_stream=compute_stream,
        kv_pool=kv_pool,
        max_lora_rank=max_lora_rank,
        model=loaded.model,
        voices=loaded.voices,
    )


__all__ = [
    "KokoroRuntime",
    "SAMPLE_RATE",
    "create_kokoro_runtime",
]
