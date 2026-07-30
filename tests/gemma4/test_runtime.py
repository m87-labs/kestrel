"""Gemma4Runtime smoke and registration tests."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

import kestrel.models.gemma4  # noqa: F401
from kestrel.config import RuntimeConfig
from kestrel.kv_cache import KVMemoryPool
from kestrel.models import get_spec, known_models
from kestrel.runtime import ExecutionShape
from kestrel.models.gemma4 import model as gemma_model
from kestrel.models.gemma4.decode_slot import GemmaDecodeSlot
from kestrel.models.gemma4.config import Gemma4TextConfig, Gemma4VisionConfig
from kestrel.models.gemma4.model import Gemma4TextModel
from kestrel.models.gemma4.paged_cache import _paged_layer_config, _stores_shared_kv
from kestrel.models.gemma4.prompt_template import (
    END_OF_TURN_ID,
    EOS_ID,
    Gemma4PromptTemplate,
    TOOL_RESPONSE_ID,
)
from kestrel.models.gemma4.runtime import (
    Gemma4Runtime,
    _DirectPagedPrefillCache,
    _decode_slot_rows,
)
from kestrel.models.gemma4.skills import Gemma4QuerySkill, build_skill_registry


_MODEL_ID = "google/gemma-4-E2B-it"
_BASE_MODEL_ID = "google/gemma-4-E2B"


def _fake_capacity_compiled(capacity: int, *, dynamic: bool):
    capacity = int(capacity)
    static = {} if dynamic and capacity > 1 else {"active_batch": capacity}

    def refusal(name, value):
        assert name == "active_batch"
        value = int(value)
        if value < 1 or value > capacity:
            return "outside capacity"
        if static and value != capacity:
            return "does not match static extent"
        return None

    return SimpleNamespace(device_program=SimpleNamespace(
        static_runtime_extents=static,
        runtime_extent_refusal=refusal,
    ))


class _FakePageTable:
    def __init__(self, free_batch_idx: list[int], pages_available: int = 4096) -> None:
        self.free_batch_idx = free_batch_idx
        self.pages_available = pages_available

    def erase(self, batch_idx: int, cached_page_count: int = 0) -> None:
        del cached_page_count
        if batch_idx not in self.free_batch_idx:
            self.free_batch_idx.append(batch_idx)


class _GenerateTokenizer:
    def encode(self, _text: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[7])

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "answer" if list(token_ids) == [42] else repr(list(token_ids))


class _ScriptedGenerateModel:
    def __init__(self, token_ids: list[int]) -> None:
        self._token_ids = iter(token_ids)

    def __call__(self, *, input_ids, use_cache: bool) -> SimpleNamespace:
        assert use_cache is False
        token_id = next(self._token_ids)
        logits = torch.full((1, input_ids.shape[1], 128), -1.0)
        logits[0, -1, token_id] = 1.0
        return SimpleNamespace(logits=logits)


def _generate_runtime(model_name: str, token_ids: list[int]) -> Gemma4Runtime:
    runtime = Gemma4Runtime.__new__(Gemma4Runtime)
    runtime.device = torch.device("cpu")
    runtime.prompt_template = Gemma4PromptTemplate(model_name)
    runtime.tokenizer = _GenerateTokenizer()
    runtime.model = _ScriptedGenerateModel(token_ids)
    return runtime


def test_rmsnorm_uses_dense_runtime_with_uniform_fp32_weight(monkeypatch):
    calls = []

    def fake_rmsnorm(x, weight, eps):
        calls.append((x, weight, eps))
        return x

    monkeypatch.setattr(gemma_model, "_kestrel_rmsnorm", fake_rmsnorm)
    scaled = gemma_model.Gemma4RMSNorm(1536)
    unscaled = gemma_model.Gemma4RMSNorm(512, with_scale=False)
    x_scaled = torch.zeros((2, 1536), dtype=torch.bfloat16)
    x_unscaled = torch.zeros((2, 512), dtype=torch.bfloat16)

    assert scaled(x_scaled) is x_scaled
    assert unscaled(x_unscaled) is x_unscaled
    assert scaled.weight.dtype == torch.float32
    assert unscaled.weight.dtype == torch.float32
    assert "weight" in scaled.state_dict()
    assert "weight" not in unscaled.state_dict()
    assert calls == [
        (x_scaled, scaled.weight, 1.0e-6),
        (x_unscaled, unscaled.weight, 1.0e-6),
    ]


def test_text_rotary_runtime_preserves_existing_cpu_math():
    torch.manual_seed(0)
    query = torch.randn(2, 5, 3, 8)
    key = torch.randn(2, 5, 1, 8)
    angles = torch.randn(2, 5, 4)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    rotary = gemma_model._prepare_neox_rotary(cos, sin)

    actual_query, actual_key = gemma_model._apply_neox_rotary(
        query, key, rotary
    )

    expected_query = gemma_model._apply_rope(
        query, cos, sin, unsqueeze_dim=2
    )
    expected_key = gemma_model._apply_rope(
        key, cos, sin, unsqueeze_dim=2
    )
    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)


def test_direct_paged_prefill_cache_writes_batched_kv_once():
    calls = []

    class FakePagedLayer:
        def update(self, input_pos, k_val, v_val, *, slot_mapping):
            calls.append((input_pos, k_val, v_val, slot_mapping))

    positions = torch.arange(6).reshape(2, 3)
    slot_mapping = positions + 10
    cache = _DirectPagedPrefillCache(
        layers=(FakePagedLayer(),),
        positions=positions,
        slot_mapping=slot_mapping,
    )
    key = torch.randn(2, 1, 3, 4)
    value = torch.randn(2, 1, 3, 4)

    actual_key, actual_value = cache.update(key, value, 0)

    assert actual_key is key
    assert actual_value is value
    assert len(calls) == 1
    input_pos, stored_key, stored_value, stored_slots = calls[0]
    assert input_pos is positions
    assert stored_slots is slot_mapping
    torch.testing.assert_close(stored_key, key.transpose(1, 2))
    torch.testing.assert_close(stored_value, value.transpose(1, 2))
    assert cache._k == []
    assert cache._v == []


def test_text_attention_emits_typed_runtime_attention(monkeypatch):
    calls = []

    class FakeTensor:
        dtype = torch.bfloat16
        device = SimpleNamespace(type="cuda")

        def __init__(self, name):
            self.name = name

        def transpose(self, left, right):
            calls.append(("transpose", self.name, left, right))
            return self

        def contiguous(self):
            return self

    output = FakeTensor("out")

    def fake_flash(q, k, v, **kwargs):
        calls.append(("flash", q, k, v, kwargs))
        return output, None

    monkeypatch.setattr(
        gemma_model,
        "get_runtime",
        lambda: SimpleNamespace(
            attention=SimpleNamespace(flash_attn_fwd=fake_flash)
        ),
    )
    q, k, v = FakeTensor("q"), FakeTensor("k"), FakeTensor("v")

    result = gemma_model._attention_forward(
        q,
        k,
        v,
        num_key_value_groups=8,
        attention_mask=None,
        scaling=1.0,
        causal=False,
        window_size_left=511,
        window_size_right=0,
    )

    assert result is output
    flash = next(call for call in calls if call[0] == "flash")
    assert flash[4] == {
        "causal": False,
        "window_size_left": 511,
        "window_size_right": 0,
        "softmax_scale": 1.0,
    }


def test_text_attention_cpu_causal_gqa_matches_reference():
    torch.manual_seed(0)
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 1, 3, 8)
    value = torch.randn(1, 1, 3, 8)

    actual = gemma_model._attention_forward(
        query,
        key,
        value,
        num_key_value_groups=4,
        attention_mask=None,
        scaling=0.5,
        causal=True,
    )

    key = key.repeat_interleave(4, dim=1)
    value = value.repeat_interleave(4, dim=1)
    scores = torch.matmul(query, key.transpose(2, 3)) * 0.5
    mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask, torch.finfo(scores.dtype).min)
    expected = torch.matmul(scores.softmax(dim=-1), value)
    expected = expected.transpose(1, 2).contiguous()

    torch.testing.assert_close(actual, expected)


def test_text_mlp_uses_generic_gated_activation_provider(monkeypatch):
    cfg = Gemma4TextConfig(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        hidden_activation="gelu_pytorch_tanh",
    )
    mlp = gemma_model.Gemma4TextMLP(cfg, layer_idx=0)
    calls = []

    def fake_gated_activation(out, gate_up, *, activation, layout):
        calls.append((activation, layout, tuple(gate_up.shape)))
        gate, up = gate_up.chunk(2, dim=-1)
        out.copy_(torch.nn.functional.gelu(gate, approximate="tanh") * up)

    monkeypatch.setattr(
        gemma_model,
        "_kestrel_gated_activation_into",
        fake_gated_activation,
    )
    x = torch.randn((2, 4))
    actual = mlp(x)
    gate_up = mlp.gate_up_proj(x)
    gate, up = gate_up.chunk(2, dim=-1)
    expected = mlp.down_proj(
        torch.nn.functional.gelu(gate, approximate="tanh") * up
    )

    torch.testing.assert_close(actual, expected)
    assert calls == [("gelu_tanh", "contiguous", (2, 16))]


def test_vision_mlp_uses_generic_gated_activation_provider(monkeypatch):
    cfg = Gemma4VisionConfig(
        hidden_size=4,
        intermediate_size=8,
        hidden_activation="gelu_pytorch_tanh",
        use_clipped_linears=True,
    )
    mlp = gemma_model.Gemma4VisionMLP(cfg)
    calls = []

    def fake_gated_activation(out, gate_up, *, activation, layout):
        calls.append((activation, layout, tuple(gate_up.shape)))
        gate, up = gate_up.chunk(2, dim=-1)
        out.copy_(torch.nn.functional.gelu(gate, approximate="tanh") * up)

    monkeypatch.setattr(
        gemma_model,
        "_kestrel_gated_activation_into",
        fake_gated_activation,
    )
    x = torch.randn((2, 4))
    actual = mlp(x)
    gate_up = mlp.gate_up_proj(x)
    gate, up = gate_up.chunk(2, dim=-1)
    expected = mlp.down_proj(
        torch.nn.functional.gelu(gate, approximate="tanh") * up
    )

    torch.testing.assert_close(actual, expected)
    assert calls == [("gelu_tanh", "contiguous", (2, 16))]


def test_vision_attention_cpu_matches_additive_padding_mask():
    torch.manual_seed(7)
    query = torch.randn((2, 5, 2, 4))
    key = torch.randn((2, 5, 2, 4))
    value = torch.randn((2, 5, 2, 4))
    valid = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    scale = 0.5

    actual = gemma_model._vision_attention_forward(
        query,
        key,
        value,
        num_key_value_groups=1,
        seqused_k=valid.sum(dim=-1, dtype=torch.int32),
        scaling=scale,
    )

    mask = gemma_model._build_bidirectional_mask(valid, dtype=query.dtype)
    query_bhsd = query.transpose(1, 2)
    key_bhsd = key.transpose(1, 2)
    value_bhsd = value.transpose(1, 2)
    scores = torch.matmul(query_bhsd, key_bhsd.transpose(2, 3)) * scale + mask
    expected = (
        torch.softmax(scores, dim=-1, dtype=torch.float32)
        .to(query.dtype)
        .matmul(value_bhsd)
        .transpose(1, 2)
        .contiguous()
    )
    torch.testing.assert_close(actual, expected)


def test_vision_attention_mps_uses_uniform_runtime(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    calls = []

    def fake_flash_attn_fwd(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return torch.empty_like(query), None

    runtime = SimpleNamespace(
        attention=SimpleNamespace(flash_attn_fwd=fake_flash_attn_fwd),
    )
    monkeypatch.setattr(gemma_model, "get_runtime", lambda: runtime)

    with FakeTensorMode():
        query = torch.empty((2, 5, 2, 4), dtype=torch.float16, device="mps")
        key = torch.empty_like(query)
        value = torch.empty_like(query)
        seqused_k = torch.empty((2,), dtype=torch.int32, device="mps")
        out = gemma_model._vision_attention_forward(
            query,
            key,
            value,
            num_key_value_groups=1,
            seqused_k=seqused_k,
            scaling=0.5,
        )

    assert out.shape == query.shape
    assert calls == [
        (
            query,
            key,
            value,
            {
                "seqused_k": seqused_k,
                "causal": False,
                "softmax_scale": 0.5,
            },
        )
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vision_position_embeddings_match_one_hot_reference(dtype):
    cfg = Gemma4VisionConfig(
        hidden_size=8,
        position_embedding_size=16,
    )
    embedder = gemma_model.Gemma4VisionPatchEmbedder(cfg).to(dtype=dtype)
    pixel_position_ids = torch.tensor(
        [
            [[0, 1], [5, 9], [-1, -1]],
            [[15, 0], [3, 7], [2, 14]],
        ],
        dtype=torch.int64,
    )
    padding_positions = torch.tensor(
        [[False, False, True], [False, True, False]]
    )

    actual = embedder._position_embeddings(
        pixel_position_ids,
        padding_positions,
    )

    clamped_positions = pixel_position_ids.clamp(min=0)
    one_hot = torch.nn.functional.one_hot(
        clamped_positions,
        num_classes=cfg.position_embedding_size,
    )
    one_hot = one_hot.permute(0, 2, 1, 3).to(embedder.position_embedding_table)
    expected = (one_hot @ embedder.position_embedding_table).sum(dim=1)
    expected = torch.where(
        padding_positions.unsqueeze(-1),
        0.0,
        expected,
    )

    assert actual.shape == (2, 3, cfg.hidden_size)
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[padding_positions]) == 0


def test_modelspecs_register_on_import():
    names = set(known_models())
    expected = {
        "google/gemma-4-E2B-it",
        "google/gemma-4-E2B",
        "google/gemma-4-E4B-it",
        "google/gemma-4-E4B",
        "google/gemma-4-31B-it",
        "google/gemma-4-31B",
    }
    assert expected <= names, f"missing variants: {expected - names}"
    spec = get_spec(_MODEL_ID)
    assert spec.runtime is Gemma4Runtime
    assert spec.tokenizer_id == _MODEL_ID
    assert spec.skills is build_skill_registry
    assert spec.skills().names() == ("query",)


def test_decode_slot_implements_constraint_buffer_abi():
    names = {field.name for field in fields(GemmaDecodeSlot)}
    assert {"disallow_mask", "mask_ready_event"} <= names


def test_decode_slot_rows_cover_non_bucket_compiled_capacity():
    assert _decode_slot_rows(1) == 3
    assert _decode_slot_rows(8) == 10
    assert _decode_slot_rows(11) == 16
    assert _decode_slot_rows(16) == 18


def test_decode_with_slot_runs_generated_program_for_b1(monkeypatch):
    rt = Gemma4Runtime.__new__(Gemma4Runtime)
    calls = []
    slot = SimpleNamespace(compute_stream="decode-stream")

    class _StreamContext:
        def __enter__(self):
            calls.append(("enter", slot.compute_stream))

        def __exit__(self, exc_type, exc, traceback):
            calls.append(("exit", slot.compute_stream))

    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _StreamContext())
    rt._decode_megakernel = SimpleNamespace(
        supports=lambda batch_size: batch_size == 1,
        run=lambda bound_slot, batch_size: calls.append(
            ("generated", bound_slot, batch_size)))

    Gemma4Runtime.decode_with_slot(rt, slot, batch_size=1)

    assert calls == [
        ("enter", "decode-stream"),
        ("generated", slot, 1),
        ("exit", "decode-stream"),
    ]


def test_decode_compile_translates_model_config(monkeypatch):
    from kestrel.models.gemma4 import generated_decode
    from mkl.compiler.frontend.models import gemma as gemma_frontend

    calls = []
    monkeypatch.setattr(
        gemma_frontend,
        "compile_gemma4_decode",
        lambda **kwargs: calls.append(kwargs) or "compiled",
    )
    config = SimpleNamespace(
        enable_moe_block=False,
        attention_bias=False,
        attention_k_eq_v=False,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=32,
        vocab_size=48,
        rms_norm_eps=1e-6,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention"] * 2,
        num_kv_shared_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        sliding_window=8,
        use_double_wide_mlp=True,
    )

    assert generated_decode._compile_from_config(
        config, max_kv_len=256, num_ctas=132, gpu="test-gpu"
    ) == "compiled"
    assert calls == [{
        "layer_types": config.layer_types,
        "num_kv_shared_layers": 2,
        "hidden": 64,
        "inter": 128,
        "nh": 4,
        "nkv": 1,
        "global_nkv": 1,
        "local_head_dim": 16,
        "global_head_dim": 32,
        "window": 8,
        "max_kv_len": 256,
        "ple_hidden": 16,
        "ple_vocab": 32,
        "vocab_size": 48,
        "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30.0,
        "tie_word_embeddings": True,
        "double_wide_mlp": True,
        "num_ctas": 132,
        "num_splits": None,
        "batch_tile": 1,
        "static_extent_bindings": {},
        "gpu": "test-gpu",
    }]


def test_decode_factory_fails_closed_on_aot_bundle_miss(monkeypatch):
    from kestrel.models.gemma4 import generated_decode
    from mkl.megakernel.device_runtime import DeviceRuntimeError

    config = SimpleNamespace()
    calls = []
    runtime = SimpleNamespace(
        device=torch.device("cuda:0"),
        max_batch_size=1,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        _shared_paged_layers=[SimpleNamespace(
            k_cache=torch.empty(1, 1, 1, 1),
            v_cache=torch.empty(1, 1, 1, 1),
        )],
        model=SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(config=config))),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: calls.append(("properties", device)) or SimpleNamespace(
            multi_processor_count=132,
            major=9,
            minor=0,
            name="test-gpu",
        ),
    )
    def compile_config(model_config, **kwargs):
        calls.append(("compile", model_config, kwargs))
        return _fake_capacity_compiled(
            kwargs["batch_capacity"], dynamic=True)

    monkeypatch.setattr(
        generated_decode, "_compile_from_config", compile_config)
    monkeypatch.setattr(
        "mkl.compiler.frontend.validate_program.validate_compiled_tape",
        lambda artifact: SimpleNamespace(program=artifact),
    )
    monkeypatch.setattr(
        "mkl.megakernel.device_runtime.resolve_shipped_aot_bundle",
        lambda artifact, *, arch: calls.append(
            ("resolve", artifact.program.device_program.static_runtime_extents,
             arch)
        ) or None,
    )

    with pytest.raises(
        DeviceRuntimeError,
        match=r"missing active extents \[1\].*unresolved artifacts",
    ):
        generated_decode.Gemma4DecodeMegakernel.try_create(runtime)
    assert calls[0] == ("properties", torch.device("cuda:0"))
    assert calls[1:] == [
        call
        for batch_capacity in (1, 2, 4, 8)
        for call in (
            ("compile", config, {
                "batch_capacity": batch_capacity,
                "max_kv_len": 2048,
                "num_ctas": 132,
                "gpu": "test-gpu",
            }),
            (
                "resolve",
                (
                    {"active_batch": 1}
                    if batch_capacity == 1 else {}
                ),
                "sm90",
            ),
        )
    ]


def test_decode_factory_falls_back_above_largest_production_capacity(monkeypatch):
    from kestrel.models.gemma4 import generated_decode

    compiled_capacities = []
    resolved_capacities = []
    config = SimpleNamespace()
    runtime = SimpleNamespace(
        device=torch.device("cuda:0"),
        max_batch_size=11,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        _shared_paged_layers=[SimpleNamespace(
            k_cache=torch.empty(1, 1, 1, 1),
            v_cache=torch.empty(1, 1, 1, 1),
        )],
        model=SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(config=config))),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            multi_processor_count=132,
            major=9,
            minor=0,
            name="test-gpu",
        ),
    )

    def compile_config(model_config, **kwargs):
        capacity = int(kwargs["batch_capacity"])
        compiled_capacities.append(capacity)
        compiled = _fake_capacity_compiled(capacity, dynamic=True)
        compiled.capacity = capacity
        return compiled

    monkeypatch.setattr(
        generated_decode, "_compile_from_config", compile_config)
    monkeypatch.setattr(
        "mkl.compiler.frontend.validate_program.validate_compiled_tape",
        lambda artifact: SimpleNamespace(
            program=artifact, capacity=artifact.capacity),
    )

    def resolve_bundle(artifact, *, arch):
        resolved_capacities.append((artifact.capacity, arch))
        return "bundle" if artifact.capacity == 8 else None

    monkeypatch.setattr(
        "mkl.megakernel.device_runtime.resolve_shipped_aot_bundle",
        resolve_bundle,
    )
    monkeypatch.setattr(
        generated_decode.Gemma4DecodeMegakernel,
        "__init__",
        lambda self, bound_runtime, *, programs: setattr(
            self, "_programs", programs),
    )

    result = generated_decode.Gemma4DecodeMegakernel.try_create(runtime)

    assert result is not None
    assert compiled_capacities == [1, 2, 4, 8]
    assert resolved_capacities == [
        (capacity, "sm90") for capacity in compiled_capacities]
    assert tuple(result._programs) == (8,)
    assert result.supports(8)
    assert not result.supports(11)


@pytest.mark.parametrize("missing_name", ["mkl", "mkl.megakernel"])
def test_decode_factory_uses_native_path_without_compiler(monkeypatch, missing_name):
    import builtins

    from kestrel.models.gemma4 import generated_decode

    runtime = SimpleNamespace(
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        _shared_paged_layers=[SimpleNamespace(
            k_cache=torch.empty(1, 1, 1, 1),
            v_cache=torch.empty(1, 1, 1, 1),
        )],
    )
    real_import = builtins.__import__

    def import_without_mkl(name, *args, **kwargs):
        if name == "mkl.megakernel.device_runtime":
            raise ModuleNotFoundError(
                f"No module named {missing_name!r}", name=missing_name
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mkl)

    assert generated_decode.Gemma4DecodeMegakernel.try_create(runtime) is None


def test_decode_factory_fails_closed_without_device_calibration(monkeypatch):
    from kestrel.models.gemma4 import generated_decode
    from mkl.compiler.frontend.gpu_model import CalibrationUnavailable
    from mkl.megakernel.device_runtime import DeviceRuntimeError

    runtime = SimpleNamespace(
        device=torch.device("cuda:0"),
        max_batch_size=1,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        _shared_paged_layers=[SimpleNamespace(
            k_cache=torch.empty(1, 1, 1, 1),
            v_cache=torch.empty(1, 1, 1, 1),
        )],
        model=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(config=SimpleNamespace())
            )
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            multi_processor_count=132,
            major=9,
            minor=0,
            name="NVIDIA H200",
        ),
    )
    monkeypatch.setattr(
        generated_decode,
        "_compile_from_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CalibrationUnavailable("NVIDIA H200 has no calibration")
        ),
    )

    with pytest.raises(
        DeviceRuntimeError,
        match="no calibration for 'NVIDIA H200'",
    ):
        generated_decode.Gemma4DecodeMegakernel.try_create(runtime)


@pytest.mark.parametrize("batch_size", [11, 16])
def test_decode_megakernel_run_uses_smallest_capacity_and_logical_extent(batch_size):
    from kestrel.models.gemma4.generated_decode import (
        Gemma4DecodeMegakernel,
        _SlotInvocation,
    )

    calls = []
    batch_capacity = 16
    state = _SlotInvocation(
        invocation=SimpleNamespace(
            launch=lambda **kwargs: calls.append(kwargs)),
    )
    megakernel = Gemma4DecodeMegakernel.__new__(Gemma4DecodeMegakernel)
    megakernel._programs = {1: (object(),) * 3, 8: (object(),) * 3,
                            batch_capacity: (object(),) * 3}
    megakernel._slots = {(7, batch_capacity): state}
    slot = SimpleNamespace(
        slot_id=7,
        meta=SimpleNamespace(input_pos=SimpleNamespace(
            cpu=torch.arange(batch_size, dtype=torch.int32))),
    )

    assert megakernel.supports(batch_size)
    assert megakernel._capacity_for(batch_size) == batch_capacity
    megakernel.run(slot, batch_size)

    assert calls == [{"active_batch": batch_size, "kv_len": batch_size}]


def test_decode_megakernel_capacity_selection_rejects_uncovered_extents():
    from kestrel.models.gemma4.generated_decode import Gemma4DecodeMegakernel

    megakernel = Gemma4DecodeMegakernel.__new__(Gemma4DecodeMegakernel)
    megakernel._programs = {
        1: (object(),) * 3,
        8: (object(),) * 3,
        16: (object(),) * 3,
    }

    assert megakernel._capacity_for(1) == 1
    assert megakernel._capacity_for(3) == 8
    assert megakernel._capacity_for(11) == 16
    assert megakernel._capacity_for(16) == 16
    assert megakernel._capacity_for(0) is None
    assert megakernel._capacity_for(17) is None
    assert not megakernel.supports(17)


def test_prefill_batches_uncached_images_and_splits_active_tokens():
    runtime = Gemma4Runtime.__new__(Gemma4Runtime)
    runtime.device = torch.device("cpu")
    runtime.max_batch_size = 4
    runtime._vision_feature_cache = {}
    calls = []
    input_ptrs = []

    def get_image_features(pixel_values, position_ids):
        input_ptrs.append((pixel_values.data_ptr(), position_ids.data_ptr()))
        calls.append((pixel_values.clone(), position_ids.clone()))
        return torch.arange(12, dtype=torch.float32).reshape(3, 4)

    runtime.model = SimpleNamespace(
        model=SimpleNamespace(get_image_features=get_image_features)
    )
    first = SimpleNamespace(
        pixel_values=torch.full((6, 3), 1.0),
        image_position_ids=torch.full((6, 2), 11),
        num_image_tokens=2,
    )
    second = SimpleNamespace(
        pixel_values=torch.full((6, 3), 2.0),
        image_position_ids=torch.full((6, 2), 22),
        num_image_tokens=1,
    )

    features = runtime._image_features_for_batch(
        [first, second, first, None]
    )

    assert len(calls) == 1
    assert calls[0][0].shape == (2, 6, 3)
    assert calls[0][1].shape == (2, 6, 2)
    assert runtime._vision_pixel_staging.cpu.shape == (4, 6, 3)
    assert runtime._vision_position_staging.cpu.shape == (4, 6, 2)
    assert input_ptrs[0] == (
        runtime._vision_pixel_staging.gpu.data_ptr(),
        runtime._vision_position_staging.gpu.data_ptr(),
    )
    assert features[0] is features[2]
    assert features[3] is None
    torch.testing.assert_close(
        features[0],
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
    )
    torch.testing.assert_close(
        features[1],
        torch.arange(8, 12, dtype=torch.float32).reshape(1, 4),
    )

    cached = runtime._image_features_for_batch([second, first])
    assert len(calls) == 1
    assert cached[0] is features[1]
    assert cached[1] is features[0]

    pixel_gpu_ptr = runtime._vision_pixel_staging.gpu.data_ptr()
    position_gpu_ptr = runtime._vision_position_staging.gpu.data_ptr()
    runtime._vision_feature_cache.clear()
    runtime._image_features_for_batch([first, second])
    assert len(calls) == 2
    assert runtime._vision_pixel_staging.gpu.data_ptr() == pixel_gpu_ptr
    assert runtime._vision_position_staging.gpu.data_ptr() == position_gpu_ptr
    assert input_ptrs[1] == (pixel_gpu_ptr, position_gpu_ptr)


def test_decode_state_tables_keep_local_and_global_storage_disjoint():
    from kestrel.models.gemma4.generated_decode import _paged_tensors

    layers = [
        SimpleNamespace(
            k_cache=torch.empty(3, 1, 1, 4),
            v_cache=torch.empty(3, 1, 1, 4),
        ),
        SimpleNamespace(
            k_cache=torch.empty(3, 1, 1, 8),
            v_cache=torch.empty(3, 1, 1, 8),
        ),
        None,
    ]
    kinds = ["sliding_attention", "full_attention", "sliding_attention"]

    local = _paged_tensors(layers, kinds, kind="sliding_attention", field="k_cache")
    global_ = _paged_tensors(layers, kinds, kind="full_attention", field="k_cache")

    assert [None if tensor is None else tuple(tensor.shape) for tensor in local] == [
        (3, 1, 4), None, None]
    assert [None if tensor is None else tuple(tensor.shape) for tensor in global_] == [
        None, (3, 1, 8), None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_runtime_constructs():
    cfg = RuntimeConfig(device="cuda", model=_MODEL_ID, max_batch_size=1)
    rt = Gemma4Runtime(
        cfg,
        kv_pool=KVMemoryPool(device=cfg.resolved_device()),
        compute_stream=torch.cuda.Stream(),
    )

    assert rt.model_name == _MODEL_ID
    assert rt.execution_shape is ExecutionShape.AUTOREGRESSIVE
    assert rt.spec is None
    assert rt.max_batch_size == 1
    assert rt.max_seq_length > 0
    assert rt.vocab_size == rt._config.text_config.vocab_size
    assert rt.tasks() == ("query",)
    from kestrel.models.gemma4.image import MAX_IMAGE_TOKENS
    assert rt.image_prefix_length == MAX_IMAGE_TOKENS + 2

    assert callable(rt.tokenizer.encode)
    assert callable(rt.tokenizer.decode)
    assert rt.prompt_template.bos_id == 2
    assert rt.page_table.page_size == 1
    padding_batch_idx = int(rt._padding_batch_idx)
    assert padding_batch_idx not in rt.page_table.free_batch_idx
    assert rt.page_table.capacity[padding_batch_idx] >= 1
    padding_slot = rt.page_table.build_slot_mapping(
        torch.tensor([padding_batch_idx], dtype=torch.long, device=rt.device),
        torch.zeros((1, 1), dtype=torch.long, device=rt.device),
    )
    assert int(padding_slot.item()) >= 0
    assert callable(rt.prefill_slots[0].step_done_event.record)
    assert callable(rt.prefill_slots[0].commit_done_event.record)
    rt.shutdown()


@pytest.mark.parametrize(
    "stop_id",
    [EOS_ID, END_OF_TURN_ID, TOOL_RESPONSE_ID],
)
def test_runtime_generate_stops_on_instruct_terminators(stop_id: int) -> None:
    runtime = _generate_runtime(_MODEL_ID, [42, stop_id, 43])

    assert runtime.generate("question", max_new_tokens=3) == "answer"


def test_runtime_generate_base_ignores_instruct_terminators() -> None:
    runtime = _generate_runtime(
        _BASE_MODEL_ID,
        [TOOL_RESPONSE_ID, END_OF_TURN_ID, EOS_ID],
    )

    assert runtime.generate("question", max_new_tokens=3) == repr(
        [TOOL_RESPONSE_ID, END_OF_TURN_ID]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_runtime_generate_produces_text():
    cfg = RuntimeConfig(device="cuda", model=_MODEL_ID, max_batch_size=1)
    rt = Gemma4Runtime(cfg, kv_pool=KVMemoryPool(device=cfg.resolved_device()))
    try:
        answer = rt.generate("Why is the sky blue?", max_new_tokens=32)
    finally:
        rt.shutdown()
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0, f"expected non-empty answer; got {answer!r}"


def test_batch_index_allocation_gates_capacity():
    rt = Gemma4Runtime.__new__(Gemma4Runtime)
    rt.max_batch_size = 2
    rt.max_seq_length = 4096
    rt.active_sequences = {}
    rt.page_table = _FakePageTable(list(range(rt.max_batch_size)))

    assert rt.prefill_budget() == (4096, 2)

    first = rt.page_table.free_batch_idx.pop(0)
    second = rt.page_table.free_batch_idx.pop(0)
    assert {first, second} == {0, 1}
    assert rt.prefill_budget()[1] == 0

    rt._release_batch_idx(first)
    assert first in rt.page_table.free_batch_idx
    assert rt.prefill_budget()[1] == 1

    rt._release_batch_idx(first)
    assert rt.page_table.free_batch_idx.count(first) == 1


def test_active_sequence_lifecycle_registers_and_frees_batch_index():
    rt = Gemma4Runtime.__new__(Gemma4Runtime)
    rt.active_sequences = {}
    rt.page_table = _FakePageTable([])

    state = SimpleNamespace(batch_idx=0)
    prepared = SimpleNamespace(state=state)

    rt.finalize_prepared_sequence_after_prefill(prepared)
    assert rt.active_sequences[0] is state

    rt.release_sequence(state)
    assert 0 not in rt.active_sequences
    assert rt.page_table.free_batch_idx == [0]

    rt.release_sequence(state)
    assert rt.page_table.free_batch_idx == [0]


def test_gemma_paged_cache_includes_sliding_shared_kv_producer():
    cfg = Gemma4TextConfig(
        num_hidden_layers=2,
        num_kv_shared_layers=1,
        layer_types=["sliding_attention", "sliding_attention"],
        num_key_value_heads=1,
        head_dim=256,
    )

    assert _stores_shared_kv(cfg, 0)
    assert _paged_layer_config(cfg, 0) == (1, 256)
    assert _paged_layer_config(cfg, 1) is None


def test_gemma_paged_cache_includes_global_shared_kv_producer():
    cfg = Gemma4TextConfig(
        num_hidden_layers=2,
        num_kv_shared_layers=1,
        layer_types=["full_attention", "full_attention"],
        num_key_value_heads=1,
        num_global_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
    )

    assert _stores_shared_kv(cfg, 0)
    assert _paged_layer_config(cfg, 0) == (1, 512)
    assert _paged_layer_config(cfg, 1) is None

    cfg.attention_k_eq_v = True
    assert _paged_layer_config(cfg, 0) == (2, 512)


def test_gemma_shared_sliding_layers_reuse_paged_kv(monkeypatch):
    cfg = Gemma4TextConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        global_head_dim=4,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=1,
        layer_types=["sliding_attention", "sliding_attention"],
    )
    model = Gemma4TextModel(cfg)

    class FakePagedLayer:
        def __init__(self) -> None:
            self.updates = 0

        def update(self, **_kwargs) -> None:
            self.updates += 1

    fake_paged_layer = FakePagedLayer()

    class FakeCache:
        def get_seq_length(self, layer_idx: int = 0) -> int:
            del layer_idx
            return 0

        def get_paged_layer(self, layer_idx: int):
            return fake_paged_layer if layer_idx == 0 else None

    paged_calls = []

    def fake_paged_attention_forward(query, **kwargs):
        paged_calls.append(kwargs["paged_kv_layer"])
        b, h, s, d = query.shape
        return torch.zeros(b, s, h, d, dtype=query.dtype, device=query.device)

    monkeypatch.setattr(
        gemma_model,
        "_paged_attention_forward",
        fake_paged_attention_forward,
    )

    model(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        past_key_values=FakeCache(),
        use_cache=True,
        cache_position_ids=torch.zeros((1, 1), dtype=torch.long),
        slot_mapping=torch.zeros((1, 1), dtype=torch.long),
        page_table=torch.zeros((1, 1), dtype=torch.int32),
        paged_kv_seqlens_k=torch.ones((1,), dtype=torch.int32),
    )

    assert fake_paged_layer.updates == 1
    assert paged_calls == [fake_paged_layer, fake_paged_layer]


def test_query_skill_defaults_to_direct_answer_mode():
    skill = Gemma4QuerySkill()
    built = skill.build_request(
        image=None,
        prompt={"question": "What is 2+2?"},
        settings={"max_tokens": 10},
    )

    assert built.temperature == 0.0
    assert built.top_p == 1.0
    assert built.max_new_tokens == 10
    assert built.request_context.reasoning is False

    built = skill.build_request(
        image=None,
        prompt={"question": "What is 2+2?", "reasoning": True},
        settings={"max_tokens": 10},
    )
    assert built.request_context.reasoning is False

    built = skill.build_request(
        image=None,
        prompt={"question": "What is 2+2?"},
        settings={"max_tokens": 10, "reasoning": True},
    )
    assert built.request_context.reasoning is True


def test_float_image_arrays_are_scaled_before_uint8_conversion():
    import numpy as np

    from kestrel.models.gemma4.image import preprocess_image

    image = np.ones((32, 32, 3), dtype=np.float32)
    inputs = preprocess_image(image)
    valid = inputs.pixel_values[: inputs.num_image_tokens * 9]

    assert inputs.pixel_values.dtype is torch.bfloat16
    assert float(valid.max()) > 0.9


@pytest.mark.parametrize("pixel", [0, 127, 255])
def test_image_preprocessing_matches_official_rescale_domain(pixel):
    import numpy as np

    from kestrel.models.gemma4.image import preprocess_image

    image = np.full((32, 32, 3), pixel, dtype=np.uint8)
    inputs = preprocess_image(image)
    expected = torch.tensor(
        float(pixel) / 255.0,
        dtype=torch.bfloat16,
    )
    valid = inputs.pixel_values[: inputs.num_image_tokens * 9]

    assert torch.equal(valid, torch.full_like(valid, expected))


def test_image_preprocessing_uses_consumer_dtype():
    import numpy as np

    from kestrel.models.gemma4.image import preprocess_image

    inputs = preprocess_image(
        np.zeros((32, 32, 3), dtype=np.uint8),
        dtype=torch.float16,
    )

    assert inputs.pixel_values.dtype is torch.float16


def test_image_preprocessing_preserves_patch_order_and_padding():
    import numpy as np

    from kestrel.models.gemma4.image import (
        Gemma4ImageProcessorConfig,
        preprocess_image,
    )

    config = Gemma4ImageProcessorConfig(
        max_patches=6,
        patch_size=2,
        pooling_kernel_size=1,
    )
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    inputs = preprocess_image(image, config=config, dtype=torch.float32)

    expected_patches = (
        image.reshape(2, 2, 2, 2, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(4, 12)
    )
    expected_pixels = torch.zeros((6, 12))
    expected_pixels[:4] = torch.from_numpy(expected_patches).float().mul_(1.0 / 255.0)

    assert torch.equal(inputs.pixel_values, expected_pixels)
    assert torch.equal(
        inputs.image_position_ids,
        torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [-1, -1],
                [-1, -1],
            ]
        ),
    )


def test_image_preprocessing_preserves_grid_larger_than_configured_max():
    import numpy as np

    from kestrel.models.gemma4.image import (
        Gemma4ImageProcessorConfig,
        preprocess_image,
    )

    inputs = preprocess_image(
        np.zeros((2, 64, 3), dtype=np.uint8),
        config=Gemma4ImageProcessorConfig(
            max_patches=1,
            patch_size=1,
            pooling_kernel_size=1,
        ),
    )

    assert inputs.pixel_values.shape[0] > 1
    assert inputs.image_position_ids.shape[0] == inputs.pixel_values.shape[0]


@pytest.mark.parametrize("default_dtype", [torch.float16, torch.bfloat16])
def test_image_preprocessing_rescale_ignores_default_dtype(default_dtype):
    import numpy as np

    from kestrel.models.gemma4.image import preprocess_image

    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    expected = preprocess_image(image, dtype=torch.float32)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        actual = preprocess_image(image, dtype=torch.float32)
    finally:
        torch.set_default_dtype(original_dtype)

    assert torch.equal(actual.pixel_values, expected.pixel_values)
    assert torch.equal(actual.image_position_ids, expected.image_position_ids)
