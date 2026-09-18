"""The ternary Parakeet student runs end to end on CPU through the public engine.

Builds a tiny ternary export (small config, random packed 2-bit weights, a
throwaway tokenizer) in ``tmp_path`` and drives ``InferenceEngine`` over it, so
the CPU path — device policy, no KV pool, thread policy, the ASR contract —
is covered without a GPU and without downloading a checkpoint.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from kestrel.config import (
    NATIVE_GEMM_THREAD_CAP,
    RuntimeConfig,
    default_cpu_threads,
    physical_cpu_count,
)
from kestrel.models.parakeet_tdt import TERNARY_MODEL_ID
from kestrel.models.parakeet_tdt.config import ParakeetTdtConfig
from kestrel.models.parakeet_tdt.model import ParakeetTdt
from kestrel.models.parakeet_tdt.weights import ternarize


# The real export's group size. The native dequant/GEMM ops constrain it
# (a multiple of 64 that divides K, and 128 for the VNNI path), so the tiny
# model keeps 128 and sizes its layers around it rather than shrinking it.
GROUP_SIZE = 128
_HIDDEN = 128
_QUANTIZED = (
    "feed_forward1.linear1",
    "feed_forward1.linear2",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.relative_k_proj",
    "conv.pointwise_conv1",
    "conv.pointwise_conv2",
    "feed_forward2.linear1",
    "feed_forward2.linear2",
)

_CONFIG: dict[str, Any] = {
    "architectures": ["ParakeetForTDT"],
    "blank_token_id": 31,
    "decoder_hidden_size": 32,
    "durations": [0, 1, 2],
    "encoder_config": {
        "hidden_size": _HIDDEN,
        "intermediate_size": 2 * _HIDDEN,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        # The mel filterbank is fixed at 128 bins by ``parakeet_features``.
        "num_mel_bins": 128,
        "conv_kernel_size": 9,
        "subsampling_conv_channels": 16,
        "subsampling_conv_kernel_size": 3,
        "subsampling_conv_stride": 2,
        "subsampling_factor": 8,
        "max_position_embeddings": 512,
        "hidden_act": "silu",
        "model_type": "parakeet_encoder",
    },
    "hidden_act": "relu",
    "max_symbols_per_step": 3,
    "model_type": "parakeet_tdt",
    "num_decoder_layers": 1,
    "pad_token_id": 2,
    "vocab_size": 32,
}


def _write_tokenizer(path: Path, vocab_size: int, *, blank: int, pad: int) -> None:
    """A word-level tokenizer with Parakeet's two special tokens."""
    tokenizers = pytest.importorskip("tokenizers")

    vocab = {"<pad>": pad, "<blank>": blank}
    for index in range(vocab_size):
        if index not in (pad, blank):
            vocab[f"▁t{index}"] = index
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token="<pad>")
    )
    tokenizer.decoder = tokenizers.decoders.Metaspace()
    tokenizer.save(str(path))


def _pack_ternary(out_features: int, in_features: int, generator: torch.Generator):
    """Random 2-bit codes in ``{0, 1, 2}``, four to a byte, plus fp16 scales."""
    codes = torch.randint(
        0, 3, (out_features, in_features), generator=generator, dtype=torch.uint8
    )
    packed = torch.zeros(out_features, in_features // 4, dtype=torch.uint8)
    for shift in range(4):
        packed |= codes[:, shift::4] << (2 * shift)
    scales = (
        torch.rand(
            out_features, in_features // GROUP_SIZE, generator=generator
        ).to(torch.float16)
        * 0.1
        + 0.01
    )
    return packed, scales


def build_tiny_ternary_export(root: Path, *, seed: int = 0) -> Path:
    """Write config.json / tokenizer.json / ternary.json / model.safetensors."""
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(_CONFIG))
    _write_tokenizer(
        root / "tokenizer.json",
        _CONFIG["vocab_size"],
        blank=_CONFIG["blank_token_id"],
        pad=_CONFIG["pad_token_id"],
    )

    config = ParakeetTdtConfig.from_json_file(root / "config.json")
    with torch.device("meta"):
        reference = ParakeetTdt(config)
    shapes = {
        name: (module.weight.shape[0], module.weight.shape[1])
        for name, module in reference.named_modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d))
    }

    # The export keeps q/k/v separate (HF naming); ``ternarize`` fuses them.
    quantized = []
    for layer in range(config.encoder.num_hidden_layers):
        for suffix in _QUANTIZED:
            name = f"encoder.layers.{layer}.{suffix}"
            if suffix.startswith("self_attn.") and suffix.endswith(
                ("q_proj", "k_proj", "v_proj")
            ):
                out_features = in_features = config.encoder.hidden_size
            else:
                out_features, in_features = shapes[name]
            quantized.append(
                {
                    "name": name,
                    "out_features": int(out_features),
                    "in_features": int(in_features),
                    "group_size": GROUP_SIZE,
                    "as_conv1d": False,
                    "has_bias": False,
                    "zero_fraction": 0.33,
                }
            )
    (root / "ternary.json").write_text(
        json.dumps(
            {
                "format": "thrush-ternary-v1",
                "names": "hf",
                "packing": {"bits": 2, "code_offset": 1, "elements_per_byte": 4},
                "quant": {"mode": "ternary", "group_size": GROUP_SIZE},
                "source": {"checkpoint": "tests/tiny", "step": 0},
                "quantized_modules": quantized,
                "n_quantized_params": 0,
                "n_dense_params": 0,
                "size_mb": 0.1,
            }
        )
    )

    generator = torch.Generator().manual_seed(seed)
    quantized_names = {entry["name"] for entry in quantized}
    state: dict[str, torch.Tensor] = {}
    for name, tensor in reference.state_dict().items():
        module_name = name.rsplit(".", 1)[0]
        if module_name in quantized_names or (
            module_name.endswith("qkv_proj")
            and f"{module_name[: -len('qkv_proj')]}q_proj" in quantized_names
        ):
            continue
        if tensor.dtype.is_floating_point:
            state[name] = torch.randn(
                tensor.shape, generator=generator, dtype=torch.float32
            ) * 0.05
        else:
            state[name] = torch.zeros(tensor.shape, dtype=tensor.dtype)
    # BatchNorm must not divide by a random (possibly negative) variance.
    for name in list(state):
        if name.endswith("norm.running_var"):
            state[name] = torch.ones_like(state[name])
        elif name.endswith("norm.running_mean"):
            state[name] = torch.zeros_like(state[name])
    for entry in quantized:
        packed, scales = _pack_ternary(
            entry["out_features"], entry["in_features"], generator
        )
        state[f"{entry['name']}.qweight"] = packed
        state[f"{entry['name']}.scales"] = scales
    save_file(state, str(root / "model.safetensors"))
    return root


# ---- device / thread policy (no ternary kernels needed) -------------------


def test_ternary_model_resolves_to_cpu_without_accelerators(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for requested in (None, "cuda", "cpu"):
        cfg = RuntimeConfig(
            model=TERNARY_MODEL_ID, model_path="/nonexistent", device=requested
        )
        assert cfg.device == "cpu"
        assert cfg.resolved_device() == torch.device("cpu")


def test_ternary_model_prefers_mps_when_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    cfg = RuntimeConfig(model=TERNARY_MODEL_ID, model_path="/nonexistent", device=None)
    assert cfg.device == "mps"
    assert cfg.resolved_dtype() == torch.float16


def test_unrestricted_models_keep_the_cuda_default() -> None:
    """A model with no device_types is unaffected: unset still means CUDA."""
    from kestrel.config import resolve_model_device

    # The dataclass default is "you choose", not a hard-coded device.
    assert RuntimeConfig.__dataclass_fields__["device"].default is None
    assert resolve_model_device("moondream3-preview", None) == "cuda"
    assert resolve_model_device("moondream3-preview", "cpu") == "cpu"
    assert resolve_model_device("an-unregistered-model", None) == "cuda"


def test_cpu_thread_policy(monkeypatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    physical = physical_cpu_count()
    assert physical is None or physical >= 1
    assert 1 <= default_cpu_threads() <= 8
    assert 1 <= default_cpu_threads(NATIVE_GEMM_THREAD_CAP) <= NATIVE_GEMM_THREAD_CAP
    # An explicit environment knob is a deliberate choice, cap or no cap.
    monkeypatch.setenv("OMP_NUM_THREADS", "11")
    assert default_cpu_threads() == 11
    assert default_cpu_threads(NATIVE_GEMM_THREAD_CAP) == 11
    monkeypatch.delenv("OMP_NUM_THREADS")
    cfg = RuntimeConfig(
        model=TERNARY_MODEL_ID, model_path="/nonexistent", device="cpu", cpu_threads=3
    )
    assert cfg.resolved_cpu_threads() == 3
    # An explicit count is the caller's decision; a runtime cap cannot lower it.
    assert cfg.resolved_cpu_threads(cap=1) == 3
    with pytest.raises(ValueError):
        RuntimeConfig(
            model=TERNARY_MODEL_ID,
            model_path="/nonexistent",
            device="cpu",
            cpu_threads=0,
        )


# ---- the engine path ------------------------------------------------------


@pytest.fixture
def offline_engine(monkeypatch):
    """``InferenceEngine`` with its two network side-effects stubbed out."""
    pytest.importorskip(
        "kestrel_kernels.ternary",
        reason="the ternary layers ship with kestrel-kernels",
    )
    import kestrel.engine.core as core
    import kestrel.model_download as model_download
    from kestrel.photon import PhotonReporter

    async def no_flush(self) -> str:
        return "inactive"

    monkeypatch.setattr(PhotonReporter, "_flush_window", no_flush)
    monkeypatch.setattr(PhotonReporter, "start", lambda self: None)
    monkeypatch.setattr(
        model_download, "probe_supported_model_configs", lambda *_args: None
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("MOONDREAM_API_KEY", raising=False)
    return core.InferenceEngine


def _noise(seconds: float = 1.0, sample_rate: int = 16_000) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (0.05 * rng.standard_normal(int(seconds * sample_rate))).astype(np.float32)


def test_tiny_ternary_export_loads_and_transcribes_on_cpu(
    tmp_path, offline_engine
) -> None:
    root = build_tiny_ternary_export(tmp_path / "export")

    async def run() -> None:
        cfg = RuntimeConfig(
            model=TERNARY_MODEL_ID,
            model_path=root,
            device="cpu",
            cpu_threads=2,
        )
        assert cfg.device == "cpu"
        engine = await offline_engine.create(cfg)
        try:
            # The engine builds no KV pool for a runtime that stores no KV cache.
            assert engine._kv_pool is None
            runtime = engine._runtimes[TERNARY_MODEL_ID]
            assert runtime.device == torch.device("cpu")
            assert runtime.cpu_threads == 2
            handle = engine.model(TERNARY_MODEL_ID)
            assert handle.tasks == ("transcribe",)

            result = await handle.transcribe(audio=_noise(), sample_rate=16_000)
            output = result.output
            assert isinstance(output["text"], str)
            assert output["task"] == "transcribe"
            assert output["duration_seconds"] == pytest.approx(1.0, abs=0.05)

            batched = await asyncio.gather(
                *(
                    handle.transcribe(audio=_noise(0.6), sample_rate=16_000)
                    for _ in range(3)
                )
            )
            assert all(isinstance(item.output["text"], str) for item in batched)
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_tiny_ternary_export_streams_live_pcm_on_cpu(tmp_path, offline_engine) -> None:
    root = build_tiny_ternary_export(tmp_path / "export")

    async def run() -> None:
        cfg = RuntimeConfig(
            model=TERNARY_MODEL_ID, model_path=root, device="cpu", cpu_threads=2
        )
        engine = await offline_engine.create(cfg)
        try:
            handle = engine.model(TERNARY_MODEL_ID)
            pcm = _noise(1.5)

            async def chunks():
                for start in range(0, pcm.size, 8_000):
                    yield pcm[start : start + 8_000]

            stream = await handle.transcribe(
                audio=chunks(), sample_rate=16_000, stream=True
            )
            updates = 0
            async for _update in stream:
                updates += 1
            result = await stream.result()
            assert updates >= 1
            assert isinstance(result.output["text"], str)
        finally:
            await engine.shutdown()

    asyncio.run(run())


def test_ternarize_fuses_the_exports_separate_qkv(tmp_path) -> None:
    pytest.importorskip("kestrel_kernels.ternary")
    from kestrel_kernels.ternary import TernaryLinear

    from kestrel.models.parakeet_tdt.weights import TernaryManifest

    root = build_tiny_ternary_export(tmp_path / "export")
    config = ParakeetTdtConfig.from_json_file(root / "config.json")
    manifest = TernaryManifest.load(root / "ternary.json")
    with torch.device("meta"):
        model = ternarize(ParakeetTdt(config), manifest)
    attention = model.get_submodule("encoder.layers.0.self_attn")
    assert isinstance(attention.qkv_proj, TernaryLinear)
    assert attention.qkv_proj.out_features == 3 * config.encoder.hidden_size
    assert isinstance(attention.o_proj, TernaryLinear)
