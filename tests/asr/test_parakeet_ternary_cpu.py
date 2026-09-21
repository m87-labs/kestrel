"""The ternary Parakeet student runs end to end on CPU through the public engine.

Builds a tiny ternary export (small config, random packed ternary weights, a
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

from kestrel.config import RuntimeConfig
from kestrel.models.parakeet_tdt import TERNARY_MODEL_ID
import kestrel.models.parakeet_tdt.runtime as runtime
from kestrel.models.parakeet_tdt.config import ParakeetTdtConfig
from kestrel.models.parakeet_tdt.model import ParakeetTdt
from kestrel.models.parakeet_tdt.weights import (
    _quantized_modules,
    load_parakeet_tdt,
    ternarize,
)


# The real export's group size. The native GEMM constrains it (a multiple of 64
# that divides K, and 128 for the int8-activation ``gemm8`` mode), so the tiny
# model keeps 128 and sizes its layers around it rather than shrinking it.
GROUP_SIZE = 128
_HIDDEN = 128
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
    """Random ternary codes in ``{0, 1, 2}``, packed four per byte, plus fp16 scales."""
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

    # Every encoder linear is quantized. The export keeps the attention projections separate under their HF
    # names, the way thrush writes them; ``ternarize`` fuses them back into the model's qkv_proj, and the
    # blocks' relative-position projections into the encoder's single one.
    hidden = config.encoder.hidden_size
    layers = config.encoder.num_hidden_layers
    quantized: list[dict[str, Any]] = []
    fused: set[str] = set()
    for name, module in reference.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name == "encoder.relative_k_proj":
            fused.add(name)
            shapes = [
                (f"encoder.layers.{index}.self_attn.relative_k_proj", hidden, hidden)
                for index in range(layers)
            ]
        elif not name.startswith("encoder.layers."):
            continue
        elif name.endswith("qkv_proj"):
            fused.add(name)
            shapes = [(f"{name[: -len('qkv_proj')]}{attr}", hidden, hidden) for attr in ("q_proj", "k_proj", "v_proj")]
        else:
            shapes = [(name, module.out_features, module.in_features)]
        quantized += [
            {"name": n, "out_features": out, "in_features": inp, "group_size": GROUP_SIZE, "has_bias": False}
            for n, out, inp in shapes
        ]
    (root / "ternary.json").write_text(
        json.dumps(
            {
                "format": "thrush-ternary-v1",
                "names": "hf",
                "quant": {"mode": "ternary", "group_size": GROUP_SIZE},
                "quantized_modules": quantized,
            }
        )
    )

    generator = torch.Generator().manual_seed(seed)
    skip = fused | {entry["name"] for entry in quantized}
    state: dict[str, torch.Tensor] = {}
    for name, tensor in reference.state_dict().items():
        if name.rsplit(".", 1)[0] in skip:
            continue
        if not tensor.dtype.is_floating_point:
            state[name] = torch.zeros(tensor.shape, dtype=tensor.dtype)
        elif name.endswith("norm.running_var"):
            state[name] = torch.ones(tensor.shape)  # BatchNorm must not divide by a random variance
        elif name.endswith("norm.running_mean"):
            state[name] = torch.zeros(tensor.shape)
        else:
            state[name] = 0.05 * torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
    for entry in quantized:
        packed, scales = _pack_ternary(entry["out_features"], entry["in_features"], generator)
        state[f"{entry['name']}.qweight"] = packed
        state[f"{entry['name']}.scales"] = scales
    save_file(state, str(root / "model.safetensors"))
    return root


# ---- device / thread policy ----------------------------------------------


def test_cpu_dtype_uses_bf16_only_when_the_cpu_supports_it(monkeypatch) -> None:
    for probe in ("_is_avx512_bf16_supported", "_is_amx_tile_supported"):
        monkeypatch.setattr(torch.cpu, probe, lambda: False, raising=False)
    assert runtime._cpu_dtype(torch.bfloat16) is torch.float32
    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: True, raising=False)
    assert runtime._cpu_dtype(torch.bfloat16) is torch.bfloat16


def test_ternary_checkpoint_rejects_cuda_at_the_loader_boundary(tmp_path) -> None:
    root = build_tiny_ternary_export(tmp_path / "export")
    with pytest.raises(ValueError, match="support CPU and MPS only"):
        load_parakeet_tdt(root, device="cuda")


def test_cpu_thread_policy() -> None:
    """The counts come from the machine and the caps, and from nowhere else -- there is no environment
    variable that changes them; ``RuntimeConfig.cpu_threads`` is the way to name one."""
    assert 1 <= runtime._default_cpu_threads() <= 8
    assert 1 <= runtime._default_cpu_threads(runtime._NATIVE_GEMM_THREAD_CAP) <= 4
    assert runtime._default_cpu_threads(1) == 1
    with pytest.raises(ValueError):
        RuntimeConfig(
            model=TERNARY_MODEL_ID,
            model_path="/nonexistent",
            device="cpu",
            cpu_threads=0,
        )


def test_cpu_thread_policy_resets_the_pool_before_reading_its_affinity(
    monkeypatch,
) -> None:
    events: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        runtime,
        "_set_kernel_worker_threads",
        lambda threads: events.append(("workers", threads)),
    )
    monkeypatch.setattr(
        runtime,
        "_confine_submitter_to_cache_domain",
        lambda: events.append(("affinity", None)),
    )
    caps: list[int] = []

    def default_threads(cap: int = 8) -> int:
        caps.append(cap)
        return min(cap, 3)

    monkeypatch.setattr(
        runtime,
        "_default_cpu_threads",
        default_threads,
    )
    current = [0]

    def set_torch_threads(threads: int) -> None:
        current[0] = threads
        events.append(("torch", threads))

    monkeypatch.setattr(
        torch,
        "set_num_threads",
        set_torch_threads,
    )
    monkeypatch.setattr(torch, "get_num_threads", lambda: current[0])

    assert runtime._configure_cpu_threads(None, native_gemm=True) == 3
    assert events == [("workers", None), ("affinity", None), ("torch", 3)]
    assert caps == [runtime._NATIVE_GEMM_THREAD_CAP]

    events.clear()
    assert runtime._configure_cpu_threads(2, native_gemm=True) == 2
    assert events == [("workers", 2), ("torch", 2)]

    events.clear()
    assert runtime._configure_cpu_threads(None, native_gemm=False) == 3
    assert caps == [runtime._NATIVE_GEMM_THREAD_CAP, 8]


# ---- the engine path ------------------------------------------------------


@pytest.fixture
def offline_engine(monkeypatch):
    """``InferenceEngine`` with its two network side-effects stubbed out."""
    for module in ("kestrel_kernels.ternary", "kestrel_kernels.conformer_ops"):
        pytest.importorskip(module, reason="the ternary layers and conformer ops ship with kestrel-kernels")
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
            assert runtime.model.is_ternary
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

    root = build_tiny_ternary_export(tmp_path / "export")
    config = ParakeetTdtConfig.from_json_file(root / "config.json")
    with torch.device("meta"):
        model = ParakeetTdt(config)
        ternarize(model, _quantized_modules(root / "ternary.json"))
    attention = model.get_submodule("encoder.layers.0.self_attn")
    assert isinstance(attention.qkv_proj, TernaryLinear)
    assert attention.qkv_proj.out_features == 3 * config.encoder.hidden_size
    assert isinstance(attention.o_proj, TernaryLinear)


@pytest.mark.parametrize("field,value", [("group_size", 64), ("out_features", 64)])
def test_ternarize_rejects_incompatible_fused_projection_metadata(
    tmp_path, field, value
) -> None:
    pytest.importorskip("kestrel_kernels.ternary")

    root = build_tiny_ternary_export(tmp_path / "export")
    config = ParakeetTdtConfig.from_json_file(root / "config.json")
    quantized = [dict(entry) for entry in _quantized_modules(root / "ternary.json")]
    key = "encoder.layers.0.self_attn.k_proj"
    next(entry for entry in quantized if entry["name"] == key)[field] = value

    with torch.device("meta"):
        model = ParakeetTdt(config)
        with pytest.raises(ValueError, match=f"disagree on {field}"):
            ternarize(model, tuple(quantized))


def test_ternarize_fuses_the_blocks_relative_position_projections(tmp_path) -> None:
    """They do not depend on the activations, so all of them are one layer on the encoder -- the export
    still writes one per block, under the HF names."""
    pytest.importorskip("kestrel_kernels.ternary")
    from kestrel_kernels.ternary import TernaryLinear

    root = build_tiny_ternary_export(tmp_path / "export")
    config = ParakeetTdtConfig.from_json_file(root / "config.json")
    names = {entry["name"] for entry in _quantized_modules(root / "ternary.json")}
    assert "encoder.layers.0.self_attn.relative_k_proj" in names
    assert "encoder.relative_k_proj" not in names
    with torch.device("meta"):
        model = ParakeetTdt(config)
        ternarize(model, _quantized_modules(root / "ternary.json"))
    projection = model.encoder.relative_k_proj
    assert isinstance(projection, TernaryLinear)
    assert projection.out_features == config.encoder.num_hidden_layers * config.encoder.hidden_size
    assert not hasattr(model.encoder.layers[0].self_attn, "relative_k_proj")
