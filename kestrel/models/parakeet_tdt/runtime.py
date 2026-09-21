"""Kestrel runtime for Parakeet TDT transcription."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from functools import partial
import ctypes
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from kestrel.device import empty_cache, make_stream, resolve_device, stream_context
from kestrel.runtime import ExecutionShape

from kestrel.models.asr.audio import AudioChunks, DecodedAudio
from kestrel.models.asr.contract import (
    Character,
    DecodeSettings,
    Segment,
    TranscriptionRequest,
    TranscriptionResult,
    Word,
)

from .contract import parse_request
from .decode_graph import _TdtBatchGraphDecoder
from .encoder_graph import ParakeetEncoderGraph
from .generated_decode import _TdtBatchGeneratedDecoder
from .features import parakeet_features
from .model import ParakeetTdt, TdtState
from .segment import SpeechRegions, energy_speech, pause_segments
from .tokenizer import ParakeetTokenizer
from .vad import head_speech
from .weights import MODEL_ID, load_parakeet_tdt


# One live-PCM window: the block the orchestrator commits exactly, and the unit
# the streaming decoder carries its state across.
STREAM_WINDOW_SECONDS = 180
_CPU_THREAD_CAP = 8
_NATIVE_GEMM_THREAD_CAP = 4


def _cpu_dtype(dtype: torch.dtype) -> torch.dtype:
    """Avoid emulated BF16 GEMMs on CPUs without native BF16 support."""

    if dtype != torch.bfloat16:
        return dtype
    for probe in ("_is_avx512_bf16_supported", "_is_amx_tile_supported"):
        fn = getattr(torch.cpu, probe, None)
        try:
            if fn is not None and fn():
                return torch.bfloat16
        except Exception:  # noqa: BLE001 — private torch capability probes
            continue
    return torch.float32


def _physical_cpu_count() -> int:
    """Physical cores available to this process (P-cores on Apple silicon)."""

    if platform.system() == "Darwin":
        for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", key], timeout=2, stderr=subprocess.DEVNULL
                )
                return int(out.strip())
            except Exception:  # noqa: BLE001 — probe, never fatal
                continue
        return os.cpu_count() or 1
    try:
        usable = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1
    cores: set[tuple[str, str]] = set()
    for cpu in usable:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            return len(usable)
        cores.add((package, core))
    return len(cores) or len(usable)


def _default_cpu_threads(cap: int = _CPU_THREAD_CAP) -> int:
    return max(1, min(_physical_cpu_count(), cap))


def _timed_segments(
    words: tuple[Word, ...],
    *,
    include_words: bool,
    max_duration_seconds: float = 30,
) -> tuple[Segment, ...]:
    segments: list[Segment] = []
    pending: list[Word] = []
    for word in words:
        pending.append(word)
        if (
            word.text.rstrip().endswith((".", "?", "!"))
            or word.end - pending[0].start >= max_duration_seconds
        ):
            grouped = tuple(pending)
            segments.append(
                Segment(
                    " ".join(item.text for item in grouped),
                    grouped[0].start,
                    grouped[-1].end,
                    grouped if include_words else (),
                )
            )
            pending.clear()
    if pending:
        grouped = tuple(pending)
        segments.append(
            Segment(
                " ".join(item.text for item in grouped),
                grouped[0].start,
                grouped[-1].end,
                grouped if include_words else (),
            )
        )
    return tuple(segments)


def _set_kernel_worker_threads(threads: int | None) -> None:
    """Hand the kernels their pool size, or ``None`` to leave them their cache-domain policy."""
    try:
        from kestrel_kernels.ternary import set_worker_threads
    except ImportError:
        return
    set_worker_threads(threads)


def _confine_submitter_to_cache_domain() -> None:
    """Pin the calling thread to the cores chosen by the native kernel pool.

    The submitting thread participates in each parallel region, so letting it
    roam outside the pool's cache domain defeats the workers' placement. Linux
    affinity is per-thread; other application threads are left alone.

    A no-op where the topology cannot be read (macOS has no affinity interface), and where the current mask
    is already inside the chosen group.
    """
    try:
        from kestrel_kernels import _cpu
    except ImportError:
        return
    cpus = set(getattr(_cpu, "pool_cpus", tuple)())
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return
    try:
        current = os.sched_getaffinity(0)
        if current <= cpus:
            return
        os.sched_setaffinity(0, cpus)
    except OSError:  # a container that forbids it; the workers are still placed
        pass


def _configure_cpu_threads(threads: int | None, *, native_gemm: bool) -> int:
    """Apply the shared CPU policy to torch and the native kernel pool.

    With no explicit count, the pool chooses one cache domain. Torch uses the
    smaller cap only when the ternary model's GEMMs run in that pool. An
    explicit count sizes both pools and leaves affinity to the caller.
    """

    if threads is None:
        # Reset the pool first: ``pool_cpus`` must describe the placement this
        # runtime will actually use, not a previous explicit configuration.
        _set_kernel_worker_threads(None)
        threads = (
            _default_cpu_threads(_NATIVE_GEMM_THREAD_CAP)
            if native_gemm
            else _default_cpu_threads()
        )
    else:
        _set_kernel_worker_threads(int(threads))
    torch.set_num_threads(int(threads))
    return torch.get_num_threads()


def _encoder_frames(samples: int, factor: int) -> int:
    frames = samples // 160
    while factor > 1:
        frames = (frames + 1) // 2
        factor //= 2
    return frames


@dataclass(slots=True)
class _StagingSlot:
    host: torch.Tensor | None = None
    copied: torch.cuda.Event | None = None


class _WaveformStaging:
    """Pinned host staging for one cohort's waveforms: one packed async copy.

    ``torch.from_numpy(...).to(device)`` copies from pageable memory, which
    Torch completes with a stream synchronize -- so one such copy per distinct
    waveform length did not just cost a transfer, it drained every kernel the
    compute stream still held. Measured on a B200 behind 145 ms of queued GEMMs,
    128 pageable copies returned after 155 ms; the same rows packed into one
    pinned buffer and copied once returned in 3.4 ms. That is what lets a batch
    be enqueued while its predecessor is still running.

    ``slots`` buffers rotate so the copy out of one buffer may still be in
    flight while the next cohort is packed into another.
    """

    def __init__(self, device: torch.device, *, slots: int = 2) -> None:
        if slots < 1:
            raise ValueError("waveform staging needs at least one slot")
        self._device = device
        self._slots = [_StagingSlot() for _ in range(slots)]
        self._next = 0

    def stage(self, blocks: Sequence[np.ndarray]) -> torch.Tensor:
        """Pack ``blocks`` end to end on the device, ordered as given."""

        total = sum(int(block.size) for block in blocks)
        slot = self._slots[self._next]
        self._next = (self._next + 1) % len(self._slots)
        if slot.copied is not None:
            # The previous copy out of this buffer must have read it before we
            # overwrite it. In steady state this event is long past.
            slot.copied.synchronize()
        if slot.host is None or slot.host.numel() < total:
            slot.host = torch.empty(total, dtype=torch.float32, pin_memory=True)
        host = slot.host[:total]
        view = host.numpy()
        at = 0
        for block in blocks:
            size = int(block.size)
            view[at : at + size] = block
            at += size
        device = torch.empty(total, dtype=torch.float32, device=self._device)
        device.copy_(host, non_blocking=True)
        if slot.copied is None:
            slot.copied = torch.cuda.Event()
        slot.copied.record()
        return device


@dataclass(frozen=True, slots=True)
class _StreamState:
    decoder: TdtState
    token_ids: tuple[int, ...]
    durations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _StreamWindow:
    state: _StreamState | None
    start_sample: int
    sample_count: int | None
    duration_seconds: float


# One request as ``launch`` read it: the transcription request, its decode
# settings, and the live stream window it belongs to (``None`` for a clip).
_Parsed = tuple[TranscriptionRequest, DecodeSettings, _StreamWindow | None]


@dataclass(slots=True)
class _ParakeetBatch:
    """A cohort between ``launch`` and ``collect``.

    ``encoded``/``valid`` hold device tensors only when the cohort stopped
    after the encoder -- one segment per request, a batch the decoder takes
    encoded. Every other cohort arrives with ``results`` already filled in and
    nothing left on the device; ``collect`` then just hands them back.
    """

    parsed: list[_Parsed | None]
    results: list[dict[str, object] | Exception | None]
    totals: list[tuple[float, float, float] | None] = field(default_factory=list)
    rows: tuple[tuple[int, DecodedAudio], ...] = ()
    encoded: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    max_tokens: int | None = None


# Requests per forward. On CUDA the encoder runs eagerly above the graph threshold, so a large batch is pure
# throughput: measured on a B200 with parakeet-tdt-0.6b-v3 on LibriSpeech test-clean (longest rows 35 s), real
# time factor and peak allocated memory -- capacity 8 2,324x / 2.0 GiB, 16 2,530x / 2.7 GiB, 64 3,800x / 7.0 GiB,
# 128 4,051x / 12.8 GiB, 256 4,516x / 24.4 GiB. CPU and MPS keep 8: there a batch costs latency and memory and
# buys little. ``RuntimeConfig.single_pass_batch_capacity`` overrides the choice.
_BATCH_CAPACITY = 8
_CUDA_BATCH_CAPACITY = 128
_CUDA_BATCH_CAPACITY_SMALL = 64  # devices under 40 GiB


_libc: Any = None


def _trim_heap() -> None:
    """Return freed heap pages to the OS on Linux (glibc ``malloc_trim``); a no-op elsewhere.

    Loading decodes the packed weights through transient tensors and every batch frees its activations, and
    glibc keeps those pages resident: on the ternary model at 8 threads the process held about 200 MB of such
    slack after the load and about 300 MB more in steady state at batch 1 (2026-09-21, LibriSpeech test-clean),
    a third of its resident memory. One syscall per call.
    """
    global _libc
    if sys.platform != "linux":
        return
    try:
        if _libc is None:
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except (OSError, AttributeError):
        return


def _batch_capacity(cfg: Any, device: torch.device) -> int:
    configured = getattr(cfg, "single_pass_batch_capacity", None)
    if configured is not None:
        if type(configured) is not int or configured <= 0:
            raise ValueError("single_pass_batch_capacity must be a positive integer")
        return configured
    if device.type != "cuda" or not torch.cuda.is_available():
        return _BATCH_CAPACITY
    total = torch.cuda.get_device_properties(device).total_memory
    return _CUDA_BATCH_CAPACITY if total >= 40 * 2**30 else _CUDA_BATCH_CAPACITY_SMALL


class ParakeetTdtRuntime:
    execution_shape = ExecutionShape.SINGLE_PASS
    batch_capacity = _BATCH_CAPACITY  # resolved per instance in __init__
    # Transducer decoding keeps its own small decoder state; there is no paged
    # KV cache here, so the engine skips building (and importing) one.
    needs_kv_pool = False
    # Pinned staging for waveform uploads; CUDA only, resolved per instance.
    _staging: "_WaveformStaging | None" = None

    def __init__(
        self,
        cfg: Any,
        *,
        compute_stream: Any = None,
        kv_pool: Any = None,
        max_lora_rank: int | None = None,
        model: ParakeetTdt | None = None,
        tokenizer: ParakeetTokenizer | None = None,
    ) -> None:
        del kv_pool, max_lora_rank
        self._model_name = getattr(cfg, "model", MODEL_ID)
        self.device = resolve_device(
            cfg.resolved_device()
            if hasattr(cfg, "resolved_device")
            else getattr(cfg, "device", "cuda")
        )
        self.dtype = (
            cfg.resolved_dtype()
            if hasattr(cfg, "resolved_dtype")
            else getattr(cfg, "dtype", torch.float32)
        )
        if self.device.type == "cpu":
            self.dtype = _cpu_dtype(self.dtype)
        self.batch_capacity = _batch_capacity(cfg, self.device)
        self.compute_stream = (
            compute_stream
            if compute_stream is not None
            else make_stream(self.device)
        )
        if model is None or tokenizer is None:
            loaded = load_parakeet_tdt(
                getattr(cfg, "model_path", None) or self._model_name,
                device=self.device,
                dtype=self.dtype,
            )
            model, tokenizer = loaded.model, loaded.tokenizer
        self.model = model.eval()
        self.tokenizer = tokenizer
        configured_cpu_threads = getattr(cfg, "cpu_threads", None)
        self._confine_cpu_submitter = (
            self.device.type == "cpu" and configured_cpu_threads is None
        )
        self.cpu_threads = (
            _configure_cpu_threads(
                configured_cpu_threads,
                native_gemm=self.model.is_ternary,
            )
            if self.device.type == "cpu"
            else None
        )
        self.decode_path = getattr(cfg, "decode_path", "auto")
        if self.decode_path not in {"auto", "native", "generated"}:
            raise ValueError("decode_path must be 'auto', 'native', or 'generated'")
        self._batch_decoder = None
        generated_supported = (
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and self.dtype == torch.bfloat16
        )
        if self.decode_path == "generated" and not generated_supported:
            raise RuntimeError(
                "Parakeet generated decode requires CUDA with BF16 weights"
            )
        if self.decode_path != "native" and generated_supported:
            stream = self.compute_stream
            assert stream is not None
            self._batch_decoder = _TdtBatchGeneratedDecoder.create(
                self.model,
                max_batch=self.batch_capacity,
                compute_stream=stream,
                required=self.decode_path == "generated",
            )
        if (
            self._batch_decoder is None
            and bool(getattr(cfg, "enable_cuda_graphs", True))
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            stream = self.compute_stream
            assert stream is not None
            self._batch_decoder = _TdtBatchGraphDecoder(
                self.model,
                max_batch=self.batch_capacity,
                compute_stream=stream,
            )
        self._staging = (
            _WaveformStaging(self.device)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        self._encoder_graph = ParakeetEncoderGraph(
            self.model,
            max_batch=self.batch_capacity,
            enabled=(
                bool(getattr(cfg, "enable_cuda_graphs", True))
                and self.device.type == "cuda"
                and torch.cuda.is_available()
            ),
            device=self.device,
            stream=self.compute_stream,
        )
        if self.device.type == "cpu":
            _trim_heap()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _speech_regions(self) -> SpeechRegions:
        """The pause source, chosen by capability of the loaded weights.

        A checkpoint carrying `vad_head.*` marks speech with its own head off
        the subsampler it already runs. Everything else -- stock NVIDIA
        checkpoints included -- reads frame energy. There is no option and no
        bundled default head: the loaded tensors decide. Nothing is cached on
        the runtime, so a shared one segments concurrent requests safely.
        """

        if getattr(self.model, "vad_head", None) is None:
            return energy_speech
        return partial(head_speech, self.model)

    def tasks(self) -> tuple[str, ...]:
        return ("transcribe",)

    def preprocess_image_async(self, image: object) -> None:
        del image
        raise ValueError("Parakeet does not accept images")

    def _batch_audio_features(
        self,
        rows: Sequence[tuple[int, DecodedAudio]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups: dict[int, list[int]] = {}
        for group_index, (_request_index, audio) in enumerate(rows):
            groups.setdefault(audio.waveform.size, []).append(group_index)

        staged = None
        if self._staging is not None:
            staged = self._staging.stage(
                [
                    rows[index][1].waveform
                    for indices in groups.values()
                    for index in indices
                ]
            )

        feature_rows: dict[int, torch.Tensor] = {}
        mask_rows: dict[int, torch.Tensor] = {}
        at = 0
        for size, indices in groups.items():
            if staged is None:
                waveforms = torch.from_numpy(
                    np.stack([rows[index][1].waveform for index in indices])
                ).to(self.device)
            else:
                span = len(indices) * size
                waveforms = staged[at : at + span].view(len(indices), size)
                at += span
            features, masks = parakeet_features(waveforms)
            for batch_index, row_index in enumerate(indices):
                feature_rows[row_index] = features[batch_index]
                mask_rows[row_index] = masks[batch_index]

        completed_features = [feature_rows[index] for index in range(len(rows))]
        completed_masks = [mask_rows[index] for index in range(len(rows))]
        width = max(row.shape[0] for row in completed_features)
        features = torch.stack(
            [F.pad(row, (0, 0, 0, width - row.shape[0])) for row in completed_features]
        )
        masks = torch.stack(
            [F.pad(row, (0, width - row.shape[0])) for row in completed_masks]
        )
        return features.to(self.dtype), masks

    def _append_chunk_result(
        self,
        request: TranscriptionRequest,
        audio: DecodedAudio,
        token_ids: list[int],
        durations: list[int],
        text_parts: list[str],
        segments: list[Segment],
        *,
        frame_seconds: float,
    ) -> None:
        text = self.tokenizer.decode(token_ids)
        text_parts.append(text.strip())
        if request.timestamps == "none" or not text:
            return
        clip_end = audio.clip_start_seconds + audio.duration_seconds
        if request.timestamps == "character":
            characters = tuple(
                Character(
                    item.text,
                    min(clip_end, item.start + audio.clip_start_seconds),
                    min(clip_end, item.end + audio.clip_start_seconds),
                )
                for item in self.tokenizer.characters(
                    token_ids, durations, frame_seconds
                )
            )
            if characters:
                segments.append(
                    Segment(
                        text,
                        characters[0].start,
                        characters[-1].end,
                        characters=characters,
                    )
                )
            else:
                segments.append(Segment(text, audio.clip_start_seconds, clip_end))
            return
        timed_words = tuple(
            Word(
                word.text,
                min(clip_end, word.start + audio.clip_start_seconds),
                min(clip_end, word.end + audio.clip_start_seconds),
            )
            for word in self.tokenizer.words(token_ids, durations, frame_seconds)
        )
        if timed_words:
            segments.extend(
                _timed_segments(
                    timed_words,
                    include_words=request.timestamps == "word",
                )
            )
        else:
            segments.append(Segment(text, audio.clip_start_seconds, clip_end))

    def _run_stream_group(
        self,
        rows: Sequence[tuple[int, DecodedAudio]],
        windows: Sequence[_StreamWindow],
        requests: Sequence[TranscriptionRequest],
        features: torch.Tensor,
        mask: torch.Tensor,
        *,
        max_tokens: int,
    ) -> tuple[dict[str, object], ...]:
        values = []
        factor = self.model.config.encoder.subsampling_factor
        with self._encoder_graph.launch(features, mask) as (encoded, valid):
            for row, window, request, row_encoded, row_valid in zip(
                rows, windows, requests, encoded, valid, strict=True
            ):
                _index, audio = row
                previous = window.state
                generated = self.model.generate_encoded(
                    row_encoded[None],
                    row_valid[None],
                    max_tokens=max_tokens,
                    start_frame=_encoder_frames(window.start_sample, factor),
                    frame_count=(
                        None
                        if window.sample_count is None
                        else _encoder_frames(window.sample_count, factor)
                    ),
                    state=None if previous is None else previous.decoder,
                )
                length = int(generated.lengths[0])
                token_ids = generated.sequences[0, 1:length].tolist()
                durations = generated.durations[0, 1:length].tolist()
                if previous is None:
                    all_token_ids = (self.tokenizer.blank_token_id, *token_ids)
                    all_durations = (0, *durations)
                else:
                    all_token_ids = (*previous.token_ids, *token_ids)
                    all_durations = (*previous.durations, *durations)
                if generated.state is None:
                    raise RuntimeError("stateful TDT decoding returned no state")
                state = _StreamState(generated.state, all_token_ids, all_durations)
                text_parts: list[str] = []
                segments: list[Segment] = []
                logical_audio = DecodedAudio(
                    audio.waveform,
                    window.duration_seconds,
                    window.duration_seconds,
                    0.0,
                )
                self._append_chunk_result(
                    request,
                    logical_audio,
                    list(all_token_ids),
                    list(all_durations),
                    text_parts,
                    segments,
                    frame_seconds=generated.encoder_frame_seconds,
                )
                value = TranscriptionResult(
                    text=" ".join(text_parts),
                    language=None,
                    duration_seconds=window.duration_seconds,
                    source_duration_seconds=window.duration_seconds,
                    clip_start_seconds=0.0,
                    segments=tuple(segments),
                ).as_dict()
                value["_stream_state"] = state
                values.append(value)
        return tuple(values)

    def _parse_inputs(
        self, task: str, inputs: Sequence[Any]
    ) -> tuple[list[_Parsed | None], list[dict[str, object] | Exception | None]]:
        if task != "transcribe":
            raise ValueError("ParakeetTdtRuntime only accepts transcribe requests")
        if getattr(self, "_confine_cpu_submitter", False):
            _confine_submitter_to_cache_domain()

        parsed: list[_Parsed | None] = [None] * len(inputs)
        results: list[dict[str, object] | Exception | None] = [None] * len(inputs)
        for index, value in enumerate(inputs):
            try:
                if not isinstance(value, Mapping):
                    raise ValueError("Parakeet transcribe inputs must be mappings")
                prompt = dict(value)
                stream_window = prompt.pop("_stream_window", None)
                if stream_window is not None and not isinstance(
                    stream_window, _StreamWindow
                ):
                    raise TypeError("invalid internal Parakeet stream window")
                request, settings = parse_request(prompt, prompt.pop("settings", None))
                if request.stream:
                    raise ValueError(
                        "Parakeet stream=True must run through the model capability handle"
                    )
                parsed[index] = (request, settings, stream_window)
            except Exception as exc:
                results[index] = exc
        return parsed, results

    @staticmethod
    def _next_chunks(
        iterators: list[Iterator[DecodedAudio] | None],
        results: list[dict[str, object] | Exception | None],
    ) -> list[tuple[int, DecodedAudio]]:
        """Pull one segment from every live row, retiring the exhausted ones."""

        chunks: list[tuple[int, DecodedAudio]] = []
        for index, iterator in enumerate(iterators):
            if iterator is None:
                continue
            try:
                chunks.append((index, next(iterator)))
            except StopIteration:
                iterators[index] = None
            except Exception as exc:
                iterators[index] = None
                results[index] = exc
        return chunks

    @staticmethod
    def _grouped(
        parsed: Sequence[_Parsed | None],
        chunks: Sequence[tuple[int, DecodedAudio]],
    ) -> dict[tuple[int, bool], list[tuple[int, DecodedAudio]]]:
        groups: dict[tuple[int, bool], list[tuple[int, DecodedAudio]]] = {}
        for index, audio in chunks:
            item = parsed[index]
            assert item is not None
            groups.setdefault((item[1].max_tokens, item[2] is not None), []).append(
                (index, audio)
            )
        return groups

    @staticmethod
    def _long_enough(
        group: Sequence[tuple[int, DecodedAudio]],
        iterators: list[Iterator[DecodedAudio] | None],
        results: list[dict[str, object] | Exception | None],
    ) -> list[tuple[int, DecodedAudio]]:
        keep: list[tuple[int, DecodedAudio]] = []
        for index, audio in group:
            if audio.waveform.size < 320:
                iterators[index] = None
                results[index] = ValueError("Parakeet audio is too short to normalize")
            else:
                keep.append((index, audio))
        return keep

    def _packed_decode(
        self, encoded: torch.Tensor, valid: torch.Tensor, *, max_tokens: int | None
    ) -> tuple[Any, list[list[int]]]:
        assert self._batch_decoder is not None
        output = self._batch_decoder.generate(encoded, valid, max_tokens=max_tokens)
        packed = torch.cat(
            (output.lengths[:, None], output.sequences, output.durations), dim=1
        ).tolist()
        return output, packed

    def _append_group_results(
        self,
        parsed: Sequence[_Parsed | None],
        group: Sequence[tuple[int, DecodedAudio]],
        output: Any,
        packed: Sequence[Sequence[int]],
        text_parts: list[list[str]],
        segments: list[list[Segment]],
    ) -> None:
        width = output.sequences.shape[1]
        for (index, audio), row in zip(group, packed, strict=True):
            length = row[0]
            token_ids = list(row[1 : 1 + length])
            durations = list(row[1 + width :][:length])
            item = parsed[index]
            assert item is not None
            self._append_chunk_result(
                item[0],
                audio,
                token_ids,
                durations,
                text_parts[index],
                segments[index],
                frame_seconds=output.encoder_frame_seconds,
            )

    def _run_group(
        self,
        parsed: Sequence[_Parsed | None],
        results: list[dict[str, object] | Exception | None],
        iterators: list[Iterator[DecodedAudio] | None],
        group: Sequence[tuple[int, DecodedAudio]],
        *,
        max_tokens: int | None,
        is_stream: bool,
        text_parts: list[list[str]],
        segments: list[list[Segment]],
    ) -> None:
        """Features -> encoder -> decode -> transcript for one segment group."""

        valid_group = self._long_enough(group, iterators, results)
        if not valid_group:
            return
        features, mask = self._batch_audio_features(valid_group)
        if is_stream:
            stream_windows = []
            stream_requests = []
            for index, _audio in valid_group:
                item = parsed[index]
                assert item is not None and item[2] is not None
                stream_requests.append(item[0])
                stream_windows.append(item[2])
            stream_results = self._run_stream_group(
                valid_group,
                stream_windows,
                stream_requests,
                features,
                mask,
                max_tokens=max_tokens,
            )
            for (index, _audio), value in zip(
                valid_group, stream_results, strict=True
            ):
                results[index] = value
            return
        encoder_lease = (
            self._encoder_graph.launch(features, mask)
            if self._splits_decode(len(valid_group))
            else nullcontext(None)
        )
        with encoder_lease as encoded_result:
            if encoded_result is None:
                output = self.model.generate(
                    features,
                    mask,
                    max_tokens=max_tokens,
                )
                packed = torch.cat(
                    (
                        output.lengths[:, None],
                        output.sequences,
                        output.durations,
                    ),
                    dim=1,
                ).tolist()
            else:
                encoded, valid = encoded_result
                output, packed = self._packed_decode(
                    encoded, valid, max_tokens=max_tokens
                )
        self._append_group_results(
            parsed, valid_group, output, packed, text_parts, segments
        )

    def _splits_decode(self, batch: int) -> bool:
        """Whether this cohort size runs the encoder and the decoder apart."""

        return (
            self._batch_decoder is not None
            and batch >= self._batch_decoder.minimum_batch
        )

    @staticmethod
    def _finalize(
        results: list[dict[str, object] | Exception | None],
        totals: Sequence[tuple[float, float, float] | None],
        text_parts: Sequence[list[str]],
        segments: Sequence[list[Segment]],
    ) -> None:
        for index, total in enumerate(totals):
            if total is None or results[index] is not None:
                continue
            duration, source_duration, clip_start = total
            results[index] = TranscriptionResult(
                text=" ".join(part for part in text_parts[index] if part),
                language=None,
                duration_seconds=duration,
                source_duration_seconds=source_duration,
                clip_start_seconds=clip_start,
                segments=tuple(segments[index]),
            ).as_dict()

    @torch.inference_mode()
    def launch(self, task: str, inputs: Sequence[Any]) -> _ParakeetBatch:
        """Enqueue a cohort's device work and return what collecting it needs.

        A cohort whose every request fits in one pause-aligned segment stops
        after the encoder: the encoding, not the transcript, is what ``launch``
        returns, so the caller can enqueue the next cohort while this one's
        encoder still runs. Anything else -- several segments, a live stream
        window, a batch the decoder will not take encoded -- is finished here,
        because its segments have to be decoded before the next can be cut.
        """

        parsed, results = self._parse_inputs(task, inputs)
        batch = _ParakeetBatch(parsed=parsed, results=results)
        stack = ExitStack()
        try:
            sources: list[AudioChunks | None] = [None] * len(inputs)
            for index, item in enumerate(parsed):
                if item is None:
                    continue
                request, _settings, _stream_window = item
                try:
                    sources[index] = stack.enter_context(
                        AudioChunks(
                            request.audio,  # type: ignore[arg-type]
                            sample_rate=request.sample_rate,
                            clip_start_seconds=request.clip_start_seconds,
                            clip_end_seconds=request.clip_end_seconds,
                            target_sample_rate=16_000,
                            max_duration_seconds=24 * 60 * 60,
                        )
                    )
                except Exception as exc:
                    results[index] = exc
            batch.totals = [
                None
                if source is None
                else (
                    source.duration_seconds,
                    source.source_duration_seconds,
                    source.clip_start_seconds,
                )
                for source in sources
            ]
            # Pause-aligned segments of at most 30 s, contiguous and never
            # overlapping: 6.37 WER against 10.82 for the fixed 180 s windows
            # this replaces (six Earnings-22 calls, parakeet-tdt-0.6b-v3).
            # A live stream window arrives already cut, carrying decoder state
            # and sample offsets into itself, so it passes through whole.
            speech = self._speech_regions()
            iterators: list[Iterator[DecodedAudio] | None] = []
            for index, source in enumerate(sources):
                item = parsed[index]
                if source is None or item is None:
                    iterators.append(None)
                elif item[2] is not None:
                    iterators.append(iter(source.chunks(STREAM_WINDOW_SECONDS)))
                else:
                    iterators.append(iter(pause_segments(source, speech)))
            text_parts: list[list[str]] = [[] for _ in inputs]
            segments: list[list[Segment]] = [[] for _ in inputs]

            first = self._next_chunks(iterators, results)
            # One more pull decides the shape of the cohort: empty means every
            # row held a single segment, and nothing here needs the transcript
            # of one segment to cut the next.
            second = self._next_chunks(iterators, results)
            groups = self._grouped(parsed, first)
            if not second and len(groups) == 1:
                (max_tokens, is_stream), group = next(iter(groups.items()))
                rows = self._long_enough(group, iterators, results)
                if not is_stream and rows and self._splits_decode(len(rows)):
                    stack.close()  # the waveforms are decoded; the readers are not needed
                    features, mask = self._batch_audio_features(rows)
                    encoded, valid = self._encoder_graph.encode(features, mask)
                    batch.rows = tuple(rows)
                    batch.encoded = encoded
                    batch.valid = valid
                    batch.max_tokens = max_tokens
                    return batch
                groups = {(max_tokens, is_stream): rows} if rows else {}

            queued = [groups, self._grouped(parsed, second)]
            while queued or any(
                iterator is not None for iterator in iterators
            ):
                pending = (
                    queued.pop(0)
                    if queued
                    else self._grouped(
                        parsed, self._next_chunks(iterators, results)
                    )
                )
                for (max_tokens, is_stream), group in pending.items():
                    self._run_group(
                        parsed,
                        results,
                        iterators,
                        group,
                        max_tokens=max_tokens,
                        is_stream=is_stream,
                        text_parts=text_parts,
                        segments=segments,
                    )
            self._finalize(results, batch.totals, text_parts, segments)
            return batch
        finally:
            stack.close()

    @torch.inference_mode()
    def collect(
        self, batch: _ParakeetBatch
    ) -> tuple[dict[str, object] | Exception, ...]:
        """Read back a launched cohort and assemble one result per request."""

        results = batch.results
        if batch.encoded is not None:
            assert batch.valid is not None
            text_parts: list[list[str]] = [[] for _ in results]
            segments: list[list[Segment]] = [[] for _ in results]
            # The encoding was produced on the runtime's compute stream, and
            # so is everything read back from it here.
            with stream_context(self.compute_stream):
                output, packed = self._packed_decode(
                    batch.encoded, batch.valid, max_tokens=batch.max_tokens
                )
            batch.encoded = None
            batch.valid = None
            self._append_group_results(
                batch.parsed, batch.rows, output, packed, text_parts, segments
            )
            self._finalize(results, batch.totals, text_parts, segments)

        finalized = []
        for result in results:
            assert result is not None
            finalized.append(result)
        if self.device.type == "cpu":
            _trim_heap()
        return tuple(finalized)

    def forward(
        self, task: str, inputs: Sequence[Any]
    ) -> tuple[dict[str, object] | Exception, ...]:
        return self.collect(self.launch(task, inputs))

    def shutdown(self) -> None:
        self._encoder_graph.shutdown()
        empty_cache(self.device)


__all__ = ["ParakeetTdtRuntime"]
