"""Qwen35Runtime smoke + registration tests."""

from __future__ import annotations

import contextlib
import importlib.util
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

import kestrel.models.qwen35  # noqa: F401
import kestrel.models.qwen35.qwen_model as qwen_model
from kestrel.config import RuntimeConfig
from kestrel.kv_cache import KVMemoryPool, LayeredPagedKV, PageTable
from kestrel.models import get_spec, known_models
from kestrel.models.qwen35.decode_slot import Qwen35DecodeSlot
from kestrel.models.qwen35.paged_cache import (
    Qwen35InferenceCache,
    Qwen35LinearStatePool,
    allocate_qwen35_paged_kv,
)
from kestrel.models.qwen35.inference_ops import LinearAttentionLayer
from kestrel.models.qwen35.prompt_template import IMAGE_PAD_ID


def _text_config_data(**overrides):
    hidden = int(overrides.get("hidden_size", 8))
    heads = int(overrides.get("num_attention_heads", 2))
    data = {
        "vocab_size": 32,
        "hidden_size": hidden,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": 1,
        "hidden_act": "silu",
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
        "attention_bias": False,
        "head_dim": hidden // heads,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "layer_types": ["full_attention"],
    }
    data.update(overrides)
    moe_fields = {
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts_per_tok",
        "num_experts",
    }
    if moe_fields.intersection(overrides):
        data.setdefault("moe_intermediate_size", 8)
        data.setdefault("shared_expert_intermediate_size", 8)
        data.setdefault("num_experts_per_tok", 1)
        data.setdefault("num_experts", 1)
    return data


def _text_config(**overrides):
    data = _text_config_data(**overrides)
    rope = data.pop("rope_parameters")
    data.pop("hidden_act")
    data.pop("mamba_ssm_dtype")
    data.pop("attention_bias")
    data["tie_word_embeddings"] = False
    data["rope_theta"] = rope["rope_theta"]
    data["partial_rotary_factor"] = rope["partial_rotary_factor"]
    data["mrope_section"] = tuple(rope["mrope_section"])
    return Qwen3_5TextConfig(**data)


def _qwen_config_data(*, text=None, **overrides):
    text_data = _text_config_data(**(text or {}))
    data = {
        "model_type": "qwen3_5",
        "text_config": text_data,
        "vision_config": {
            "depth": 1,
            "hidden_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 16,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": int(text_data["hidden_size"]),
            "num_position_embeddings": 16,
        },
        "image_token_id": 30,
        "tie_word_embeddings": False,
    }
    data.update(overrides)
    return data
from kestrel.models.qwen35.qwen_config import Qwen3_5Config, Qwen3_5TextConfig
from kestrel.models.qwen35.qwen_model import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNormGated,
    Qwen3_5SparseMoeBlock,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5TopKRouter,
)
from kestrel.models.qwen35.runtime import (
    QwenImageInputs,
    Qwen35Runtime,
    _QwenForwardCache,
)
from kestrel.models.qwen35.skills import Qwen35QuerySkill


requires_mkl = pytest.mark.skipif(
    importlib.util.find_spec("mkl") is None,
    reason="requires the optional generated-decode compiler",
)


_MODEL_ID = "Qwen/Qwen3.5-2B"


def _make_qwen_cache(
    *,
    config,
    page_table,
    pool,
    dtype,
    shared_paged_layers: LayeredPagedKV | None = None,
    replay_capacity: int = 16,
) -> Qwen35InferenceCache:
    paged_kv = shared_paged_layers or allocate_qwen35_paged_kv(
        config=config,
        page_table=page_table,
        pool=pool,
        dtype=dtype,
    )
    return Qwen35InferenceCache(
        config=config,
        paged_kv=paged_kv,
        replay_capacity=int(replay_capacity),
    )


def _fake_capacity_compiled(capacity: int):
    capacity = int(capacity)

    def refusal(name, value):
        assert name == "active_batch"
        return (
            None
            if 1 <= int(value) <= capacity
            else "exceeds physical capacity"
        )

    return SimpleNamespace(
        capacity=capacity,
        device_program=SimpleNamespace(
            static_runtime_extents={},
            runtime_extent_refusal=refusal,
        ),
    )


def test_modelspecs_register_on_import():
    names = set(known_models())
    expected = {
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-397B-A17B",
    }
    assert expected <= names, f"missing variants: {expected - names}"
    spec = get_spec(_MODEL_ID)
    assert spec.runtime is Qwen35Runtime
    assert spec.tokenizer_id == _MODEL_ID


def test_decode_slot_implements_constraint_buffer_abi():
    names = {field.name for field in fields(Qwen35DecodeSlot)}
    assert {"disallow_mask", "mask_ready_event"} <= names


def test_fused_qkv_attention_handles_multitoken_prefill_views():
    cfg = _text_config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        mrope_section=(1, 1, 0),
    )
    attn = Qwen3_5Attention(cfg, layer_idx=0)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    hidden_states = torch.randn(2, 3, cfg.hidden_size)
    position_ids = torch.arange(3, dtype=torch.long).unsqueeze(0).expand(2, -1)
    cos, sin = rotary(hidden_states, position_ids)

    out, _ = attn(
        hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
    )

    assert out.shape == hidden_states.shape


def test_attention_updates_paged_cache_with_fused_value_view(monkeypatch):
    cfg = _text_config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        mrope_section=(1, 1, 0),
    )
    attn = Qwen3_5Attention(cfg, layer_idx=0)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    hidden_states = torch.randn(2, 3, cfg.hidden_size)
    position_ids = torch.arange(3, dtype=torch.long).unsqueeze(0).expand(2, -1)
    cos, sin = rotary(hidden_states, position_ids)
    captured: dict[str, torch.Tensor] = {}

    class FakePagedLayer:
        def update(self, *, input_pos, k_val, v_val, slot_mapping):
            captured["k_val"] = k_val
            captured["v_val"] = v_val

    class FakeCache:
        def __init__(self):
            self.layer = FakePagedLayer()

        def get_paged_layer(self, layer_idx):
            assert layer_idx == 0
            return self.layer

    def fake_paged_attention_forward(query, **kwargs):
        return query.transpose(1, 2), None

    monkeypatch.setattr(
        qwen_model,
        "paged_attention_forward",
        fake_paged_attention_forward,
    )

    out, _ = attn(
        hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=FakeCache(),
        cache_position_ids=torch.arange(3, dtype=torch.long).expand(2, -1),
        slot_mapping=torch.arange(6, dtype=torch.long).reshape(2, 3),
    )

    assert out.shape == hidden_states.shape
    assert captured["v_val"].shape == (2, 3, 1, 4)
    assert captured["v_val"].stride(1) > (
        captured["v_val"].shape[2] * captured["v_val"].shape[3]
    )
    assert captured["v_val"].stride(2) == captured["v_val"].shape[3]
    assert captured["v_val"].stride(3) == 1


def test_paged_attention_delegates_device_support_to_runtime(monkeypatch):
    query = torch.randn(1, 2, 3, 4, dtype=torch.float16)
    paged_kv_layer = SimpleNamespace(
        k_cache=torch.randn(5, 1, 1, 4, dtype=torch.float16),
        v_cache=torch.randn(5, 1, 1, 4, dtype=torch.float16),
        k_scale=1.0,
        v_scale=1.0,
    )
    page_table = torch.tensor([[0, 1, 2, -1, -1]], dtype=torch.int32)
    seqused_k = torch.tensor([3], dtype=torch.int32)
    captured: dict[str, object] = {}

    def fake_flash_attn_fwd(q, k, v, **kwargs):
        captured["q"] = q
        captured["k"] = k
        captured["v"] = v
        captured["page_table"] = kwargs["page_table"]
        captured["seqused_k"] = kwargs["seqused_k"]
        captured["causal"] = kwargs["causal"]
        return torch.zeros_like(q), None

    monkeypatch.setattr(qwen_model, "_flash_attn_fwd", fake_flash_attn_fwd)

    out, _ = qwen_model.paged_attention_forward(
        query,
        paged_kv_layer=paged_kv_layer,
        page_table=page_table,
        paged_kv_seqlens_k=seqused_k,
    )

    assert out.shape == (1, 3, 2, 4)
    q = captured["q"]
    k = captured["k"]
    v = captured["v"]
    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert q.device.type == "cpu"
    assert q.dtype == torch.float16
    assert k.shape == (5, 1, 1, 4)
    assert v.shape == (5, 1, 1, 4)
    assert captured["page_table"] is page_table
    assert captured["seqused_k"] is seqused_k
    assert captured["causal"] is True


def test_moe_config_builds_sparse_mlp_with_checkpoint_keys():
    cfg = Qwen3_5Config.from_dict(
        _qwen_config_data(
            text={
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "shared_expert_intermediate_size": 8,
            },
            model_type="qwen3_5_moe",
        )
    )

    assert cfg.text_config.is_moe
    model = Qwen3_5TextModel(cfg.text_config)
    mlp = model.layers[0].mlp
    assert isinstance(mlp, Qwen3_5SparseMoeBlock)

    state_keys = set(model.state_dict())
    assert "layers.0.mlp.experts.gate_up_proj" in state_keys
    assert "layers.0.mlp.experts.down_proj" in state_keys
    assert "layers.0.mlp.gate.weight" in state_keys
    assert "layers.0.mlp.shared_expert.gate_up_proj.weight" in state_keys
    assert "layers.0.mlp.shared_expert.down_proj.weight" in state_keys
    assert "layers.0.mlp.shared_expert_gate.weight" in state_keys
    assert "layers.0.mlp.gate_proj.weight" not in state_keys
    assert "layers.0.mlp.up_proj.weight" not in state_keys
    assert "layers.0.mlp.shared_expert.gate_proj.weight" not in state_keys
    assert "layers.0.mlp.shared_expert.up_proj.weight" not in state_keys
    assert "layers.0.self_attn.qkv_proj.weight" in state_keys
    assert "layers.0.self_attn.q_proj.weight" not in state_keys
    assert "layers.0.self_attn.k_proj.weight" not in state_keys
    assert "layers.0.self_attn.v_proj.weight" not in state_keys

    with torch.no_grad():
        for param in mlp.parameters():
            param.zero_()
    output = mlp(torch.ones(2, 3, cfg.text_config.hidden_size))
    assert output.shape == (2, 3, cfg.text_config.hidden_size)


def test_text_model_fused_layer_boundaries_match_layer_loop():
    cfg = _text_config(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        layer_types=["full_attention", "full_attention"],
        rope_theta=10000,
        partial_rotary_factor=1.0,
        mrope_section=(1, 1, 0),
    )
    torch.manual_seed(0)
    model = Qwen3_5TextModel(cfg).eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    with torch.no_grad():
        actual = model(input_ids=input_ids).last_hidden_state

        inputs_embeds = model.embed_tokens(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        text_position_ids = position_ids[0]
        position_embeddings = model.rotary_emb(inputs_embeds, position_ids[1:])
        causal_mask = qwen_model.create_causal_mask(
            config=model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=text_position_ids,
        )

        expected = inputs_embeds
        for decoder_layer in model.layers:
            expected, _ = decoder_layer._forward_from_normalized(
                expected,
                decoder_layer.input_layernorm(expected),
                position_embeddings=position_embeddings,
                output_layernorm=None,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
            )
        expected = model.norm(expected)

    torch.testing.assert_close(actual, expected)


def test_topk_router_matches_full_softmax_renormalization():
    cfg = _text_config(
        hidden_size=4,
        num_experts=5,
        num_experts_per_tok=2,
        moe_intermediate_size=3,
    )
    router = Qwen3_5TopKRouter(cfg)
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [0.25, -0.5, 0.75, 1.0],
                    [-0.75, 0.5, 0.25, -0.25],
                    [1.25, 0.75, -0.5, 0.5],
                    [-0.25, 1.0, 0.5, -0.75],
                    [0.5, -1.25, 1.0, 0.25],
                ]
            )
        )
    hidden_states = torch.tensor(
        [
            [[0.5, -0.25, 1.0, 0.75], [1.25, 0.5, -0.5, 0.25]],
            [[-1.0, 0.75, 0.25, 0.5], [0.25, 1.5, -0.75, -0.5]],
        ]
    )

    reference_logits = torch.nn.functional.linear(
        hidden_states.reshape(-1, cfg.hidden_size),
        router.weight,
    )
    router_scores, router_indices = router(hidden_states)

    reference_probs = torch.softmax(reference_logits, dtype=torch.float, dim=-1)
    reference_scores, reference_indices = torch.topk(
        reference_probs,
        cfg.num_experts_per_tok,
        dim=-1,
    )
    reference_scores = reference_scores / reference_scores.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(router_scores, reference_scores)
    assert torch.equal(router_indices, reference_indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_runtime_constructs():
    from tokenizers import Tokenizer

    cfg = RuntimeConfig(device="cuda", model=_MODEL_ID, max_batch_size=1)
    rt = Qwen35Runtime(cfg, kv_pool=KVMemoryPool(device=cfg.resolved_device()))

    assert rt.model_name == _MODEL_ID
    assert rt.max_batch_size == 1
    assert rt.max_seq_length > 0
    assert rt.image_prefix_length > 0
    assert callable(rt.tokenizer.encode)
    assert callable(rt.tokenizer.decode)
    assert rt.tokenizer.post_processor is None
    assert rt.vocab_size == rt.architecture.text_config.vocab_size
    text = "No hidden special tokens."
    reference = Tokenizer.from_pretrained(_MODEL_ID)
    assert rt.tokenizer.encode(text).ids == reference.encode(
        text,
        add_special_tokens=False,
    ).ids
    assert rt.prompt_template.bos_id == 248045
    assert not hasattr(rt, "region")
    assert len(rt.prefill_slots) == 2
    assert rt.page_table.page_size == 1
    padding_batch_idx = int(rt._padding_batch_idx)
    assert padding_batch_idx not in rt.page_table.free_batch_idx
    assert rt.page_table.capacity[padding_batch_idx] >= 1
    padding_slot = rt.page_table.build_slot_mapping(
        torch.tensor([padding_batch_idx], dtype=torch.long, device=rt.device),
        torch.zeros((1, 1), dtype=torch.long, device=rt.device),
    )
    assert int(padding_slot.item()) >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_pool_resized_before_decode_graph_capture(monkeypatch):
    """The spec GDN pool must be sized to ``flush_cap`` BEFORE graph capture.

    With ``enable_cuda_graphs`` + ``spec_decode``, ``__init__`` captures the
    normal ``decode_with_slot`` graph, which binds the linear-state pool's replay
    tensors (``_prepare_decode_slot`` -> ``bind_to_cache``). Spec decode then
    reallocates those replay tensors to ``flush_cap`` (the default
    ``linear_replay_capacity`` is small), CHANGING their device addresses + shapes.
    If that resize ran AFTER capture, a later fallback to ``decode_with_slot``
    would replay kernels bound to the freed/stale buffers.

    The fix resizes the pool first. This test records, at the moment the decode
    graph is captured, the pool's ``replay_capacity`` and asserts it already
    equals ``flush_cap`` (not the default) -- i.e. the resize precedes the
    capture. ``ensure_ready`` is stubbed to a no-op recorder so the heavy real
    capture is skipped; only the ordering contract is under test.
    """
    from kestrel.runtime.decode_graph import DecodeGraphManager
    from kestrel.models.qwen35.dflash.model import DFlashConfig, DFlashDraftModel
    from kestrel.models.qwen35.runtime import Qwen35Runtime as _RT

    flush_cap = 48  # non-default; the pool default linear_replay_capacity is 16

    # Build a dimensionally-matched random drafter so ``_maybe_init_spec_decode``
    # constructs a real ``SpecStepRunner`` (no gated checkpoint needed for the
    # ordering contract). The drafter must match the target dims, so read them
    # from a plain (graphs-off) runtime first -- BEFORE any monkeypatch is
    # installed, so this probe captures/builds entirely normally.
    probe = _RT(
        RuntimeConfig(device="cuda", model=_MODEL_ID, max_batch_size=1),
        kv_pool=KVMemoryPool(
            device=RuntimeConfig(device="cuda", model=_MODEL_ID).resolved_device()
        ),
    )
    tc = probe.architecture.text_config
    n_layers = int(tc.num_hidden_layers)
    step = max(1, n_layers // 8)
    target_layer_ids = tuple(range(1, n_layers, step))[:8]
    head_dim = int(getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads))
    torch.manual_seed(0)
    dcfg = DFlashConfig(
        hidden_size=int(tc.hidden_size), intermediate_size=4096,
        num_hidden_layers=2, num_attention_heads=int(tc.num_attention_heads),
        num_key_value_heads=int(tc.num_attention_heads), head_dim=head_dim,
        vocab_size=int(tc.vocab_size), rope_theta=1e7, block_size=16,
        mask_token_id=0, target_layer_ids=target_layer_ids,
    )
    drafter = DFlashDraftModel(dcfg).to(torch.device("cuda"), torch.bfloat16).eval()
    del probe

    # NOW install the spies (the probe above is already built, so they only see
    # the spec-enabled runtime constructed below).
    events: list[str] = []
    capture_capacity: dict[str, int] = {}
    real_resize = _RT._resize_linear_pool_for_spec
    real_ensure_ready = DecodeGraphManager.ensure_ready

    def _spy_resize(self, cap):
        events.append(f"resize:{int(cap)}")
        return real_resize(self, cap)

    def _spy_ensure_ready(self, slots, *a, **k):
        # ``_run_forward`` is a bound method of the owning runtime, so its
        # ``__self__`` is the runtime whose pool the captured graph would bind.
        rt = self._run_forward.__self__
        is_spec = bool(getattr(rt._cfg, "spec_decode", None))
        if is_spec:
            capture_capacity["at_capture"] = int(
                rt._linear_state_pool.replay_capacity)
            events.append("capture")
            # No-op: skip the heavy real capture; ordering is the contract.
            return None
        return real_ensure_ready(self, slots, *a, **k)  # pragma: no cover

    monkeypatch.setattr(_RT, "_resize_linear_pool_for_spec", _spy_resize)
    monkeypatch.setattr(DecodeGraphManager, "ensure_ready", _spy_ensure_ready)

    cfg = RuntimeConfig(
        device="cuda", model=_MODEL_ID, max_batch_size=1, enable_cuda_graphs=True
    )
    cfg.spec_decode = {  # type: ignore[attr-defined]
        "drafter": drafter,
        "drafter_config": dcfg,
        "flush_cap": flush_cap,
        "use_graphs": False,
    }

    rt = None
    build_exc = None
    try:
        rt = _RT(cfg, kv_pool=KVMemoryPool(device=cfg.resolved_device()))
    except Exception as exc:  # env dependent (the GDN flush/verify kernel build
        # may be GQA-incomplete on this host so the SpecStepRunner -- built AFTER
        # the graph capture -- can fail). The resize/capture ordering under test
        # happens in __init__ BEFORE that, so the evidence is already recorded;
        # assert it regardless, then skip only if even the ordering never ran.
        build_exc = exc

    if "capture" not in events:
        pytest.skip(
            f"spec decode-graph capture path not exercised on this host: {build_exc}")

    # The pool must already be at flush_cap when the graph is captured (the whole
    # point of the fix: resize precedes capture).
    assert capture_capacity.get("at_capture") == flush_cap, (
        f"decode graph captured against pool replay_capacity="
        f"{capture_capacity.get('at_capture')} but spec uses flush_cap={flush_cap}; "
        "the spec pool resize must run BEFORE ensure_ready captures the graph")
    # And a spec-budget resize must have happened strictly before the capture.
    assert events.index(f"resize:{flush_cap}") < events.index("capture"), (
        f"resize/capture order wrong: {events}")
    # If the full runtime built, its final pool capacity is the spec capacity.
    if rt is not None:
        assert int(rt._linear_state_pool.replay_capacity) == flush_cap


def test_resize_linear_pool_for_spec_is_idempotent(monkeypatch):
    """A second resize at the SAME ``flush_cap`` must NOT drop/realloc tensors.

    ``__init__`` resizes the GDN pool to ``flush_cap`` BEFORE the decode graph is
    captured (so the captured ``decode_with_slot`` graph binds the final replay
    tensors), then ``_maybe_init_spec_decode`` resizes again AFTER capture. If the
    second call freed the replay tensors and reallocated via
    ``initialize_from_config`` (new device addresses), the already-captured graph
    would replay kernels bound to freed/stale buffers on any fallback to the
    normal decode graph. So the second call -- pool already at ``flush_cap`` --
    must early-return without touching the tensors.

    Pure-Python stub (no CUDA/model): a fake pool records every realloc and keeps
    object identity of the replay tensors so we can assert they survive intact.
    """
    from kestrel.models.qwen35.runtime import Qwen35Runtime as _RT

    sentinel_k = object()  # stand-ins for the captured replay buffers

    class _FakeStorage:
        def __init__(self):
            self.replay_checkpoint_states = sentinel_k
            self.replay_k = sentinel_k
            self.replay_u = sentinel_k
            self.replay_g = sentinel_k
            self.replay_lengths = sentinel_k

    class _FakePool:
        def __init__(self, capacity):
            self.replay_capacity = int(capacity)
            self.layers = [_FakeStorage(), None, _FakeStorage()]
            self.init_calls = 0

        def initialize_from_config(self, config, *, dtype):
            self.init_calls += 1

    flush_cap = 48
    # First resize: pool at the small default -> must reallocate to flush_cap.
    pool = _FakePool(capacity=16)
    stub = SimpleNamespace(
        _linear_state_pool=pool,
        architecture=SimpleNamespace(text_config=object()),
        dtype=torch.bfloat16,
    )
    _RT._resize_linear_pool_for_spec(stub, flush_cap)
    assert pool.replay_capacity == flush_cap
    assert pool.init_calls == 1, "initial resize (16 -> flush_cap) must reallocate"
    # The drop happened on this first (real) resize.
    assert pool.layers[0].replay_k is None

    # Re-stamp the buffers to mimic the post-capture state, then resize AGAIN at
    # the same flush_cap: this must be a pure no-op (no drop, no realloc).
    for st in pool.layers:
        if st is not None:
            st.replay_checkpoint_states = sentinel_k
            st.replay_k = st.replay_u = st.replay_g = st.replay_lengths = sentinel_k
    _RT._resize_linear_pool_for_spec(stub, flush_cap)
    assert pool.init_calls == 1, (
        "second resize at the same flush_cap must NOT reallocate (idempotent); "
        "reallocating moves the buffers the captured decode graph bound")
    for st in pool.layers:
        if st is not None:
            assert st.replay_k is sentinel_k, (
                "idempotent resize must leave replay tensors untouched")
            assert st.replay_checkpoint_states is sentinel_k


def _capture_spec_runner_kwargs(monkeypatch, spec_cfg):
    """Run ``_maybe_init_spec_decode`` on a CPU stub, returning the kwargs the
    production code passes to ``SpecStepRunner``.

    Patches ``SpecStepRunner`` / ``DFlashDraftModel`` in ``kestrel.models.qwen35.dflash``
    (the names ``_maybe_init_spec_decode`` imports) so no GDN kernels are built,
    and binds the real (unbound) ``_maybe_init_spec_decode`` to a minimal carrier
    exposing only what it reads before constructing the runner."""
    import kestrel.models.qwen35.dflash as _dflash
    from kestrel.models.qwen35.runtime import Qwen35Runtime as _RT

    captured: dict = {}

    class _FakeRunner:
        def __init__(self, rt, drafter, dcfg, **kwargs):
            captured["kwargs"] = kwargs
            self.proposer = object()

    class _FakeDrafter:
        """Stand-in accepted by the ``isinstance(drafter, DFlashDraftModel)``
        guard; the patched class IS this type, so a plain instance passes."""

    monkeypatch.setattr(_dflash, "SpecStepRunner", _FakeRunner)
    monkeypatch.setattr(_dflash, "DFlashDraftModel", _FakeDrafter)

    dcfg = SimpleNamespace(target_layer_ids=(1, 2, 3))
    cfg = SimpleNamespace(spec_decode={"drafter": _FakeDrafter(),
                                       "drafter_config": dcfg, **spec_cfg})
    stub = SimpleNamespace(
        _cfg=cfg,
        max_batch_size=2,
        # Pass an explicit ``max_seq_len`` so ``_default_spec_max_seq_len`` (which
        # needs a real KV pool) is bypassed; the runner-construction kwargs under
        # test are independent of it.
        _spec_flush_cap=_RT._spec_flush_cap,
        _resize_linear_pool_for_spec=lambda cap: None,
    )
    _RT._maybe_init_spec_decode.__get__(stub)()
    return captured["kwargs"]


def test_spec_runner_built_with_sampling_enabled(monkeypatch):
    """``_maybe_init_spec_decode`` must build the production ``SpecStepRunner``
    with ``sampling=True`` by default.

    Regression: without it the runner keeps its default ``_sampling=False`` and
    captures the draft graph as greedy *tokens*, so a ``temperature > 0`` request
    routed through the runtime spec path hits ``step()``'s ``NotImplementedError``
    -- non-greedy spec decode is unusable via the runtime integration. The
    ``sampling=True`` path is a superset that also serves greedy, so wiring it on
    keeps greedy working while unlocking sampled spec decode end-to-end."""
    kwargs = _capture_spec_runner_kwargs(monkeypatch, {"max_seq_len": 256})
    assert kwargs.get("sampling") is True, (
        "SpecStepRunner must be constructed with sampling=True so the runtime "
        "spec path serves temperature>0 requests; got sampling="
        f"{kwargs.get('sampling')!r}")


def test_spec_runner_sampling_overridable_via_config(monkeypatch):
    """The spec config may pin ``sampling`` explicitly (e.g. greedy-only deploy)."""
    kwargs = _capture_spec_runner_kwargs(
        monkeypatch, {"max_seq_len": 256, "sampling": False})
    assert kwargs.get("sampling") is False, (
        "explicit spec_decode['sampling']=False must be honored")


def test_spec_decode_flush_cap_defaults_to_32(monkeypatch):
    """Runtime and direct DFlash entry points should agree on the safe default."""
    import inspect

    from kestrel.models.qwen35.dflash.spec_decoder import SpecDecoder, SpecRunner
    from kestrel.models.qwen35.runtime import Qwen35Runtime as _RT

    assert _RT._spec_flush_cap({}) == 32
    kwargs = _capture_spec_runner_kwargs(monkeypatch, {"max_seq_len": 256})
    assert kwargs.get("flush_cap") == 32
    assert inspect.signature(SpecDecoder).parameters["flush_cap"].default == 32
    assert inspect.signature(SpecRunner).parameters["flush_cap"].default == 32


def test_validate_flush_cap_accepts_only_shipped_caps(monkeypatch):
    """``flush_cap`` must be one of the capacities for which the precompiled
    replay verify/materialize cubins ship.

    Regression (gated_delta registry): ``SpecRunner`` accepts any
    ``flush_cap >= block_size``, but the precompiled ``gated_delta`` bundle ships
    replay verify/materialize cubins only for caps 32 and 64. A valid but
    unshipped cap (e.g. 16) with ``KESTREL_CUTE_JIT`` unset has no verify cubin,
    so the spec verify path falls through to the uncapturable torch fallback and
    the spec runner cannot capture its verify graph. Reject such caps up front
    with a clear error instead of failing opaquely during CUDA-graph capture."""
    from kestrel.models.qwen35.dflash.spec_decoder import (
        _SHIPPED_FLUSH_CAPS,
        _validate_flush_cap,
    )

    monkeypatch.delenv("KESTREL_CUTE_JIT", raising=False)
    block_size = 16

    # Every shipped cap validates (and they all satisfy the ring invariant).
    for cap in _SHIPPED_FLUSH_CAPS:
        assert cap >= block_size
        _validate_flush_cap(cap, block_size)  # no raise

    # A valid-but-unshipped cap (>= block_size, not in the shipped set) is
    # rejected because no verify cubin ships for it.
    unshipped = 16
    assert unshipped not in _SHIPPED_FLUSH_CAPS and unshipped >= block_size
    with pytest.raises(ValueError, match="no shipped replay verify"):
        _validate_flush_cap(unshipped, block_size)

    # The pre-existing ring invariant (cap >= block_size) still fires first.
    with pytest.raises(ValueError, match="must be >= block_size"):
        _validate_flush_cap(8, block_size)


def test_validate_flush_cap_allows_unshipped_under_dev_jit(monkeypatch):
    """With dev JIT on (``KESTREL_CUTE_JIT=1``) an unshipped cap can compile a
    verify cubin on demand, so the shipped-cap restriction is relaxed."""
    from kestrel.models.qwen35.dflash.spec_decoder import _validate_flush_cap

    monkeypatch.setenv("KESTREL_CUTE_JIT", "1")
    _validate_flush_cap(16, 16)  # no raise: JIT can compile the c16 verify cubin
    # The ring invariant is independent of JIT and still enforced.
    with pytest.raises(ValueError, match="must be >= block_size"):
        _validate_flush_cap(8, 16)


def test_validate_flush_cap_rejects_unshipped_block_size(monkeypatch):
    """The verify cubin variant key includes the block size, not just the cap.

    Regression (gated_delta registry): the verify variant key is
    ``replay_verify_indexed_..._c{cap}_t{block_size}`` and the precompiled bundle
    ships verify cubins only at ``t16`` (both the cap-32 and cap-64 entries are
    ``_t16``). A drafter with a shipped ``flush_cap`` but a non-16 ``block_size``
    passes the cap check yet resolves no verify cubin, so the spec verify path
    falls through to the uncapturable torch fallback at CUDA-graph capture --
    exactly like an unshipped cap. Reject the non-shipped ``block_size`` up front
    with a clear error (unless dev JIT can compile it on demand)."""
    from kestrel.models.qwen35.dflash.spec_decoder import (
        _SHIPPED_FLUSH_CAPS,
        _SHIPPED_VERIFY_BLOCK_SIZES,
        _validate_flush_cap,
    )

    monkeypatch.delenv("KESTREL_CUTE_JIT", raising=False)

    # Sanity: the only shipped verify block size is 16, and it validates against
    # every shipped cap (the t16 cubins ship for both cap 32 and cap 64).
    assert _SHIPPED_VERIFY_BLOCK_SIZES == (16,)
    for cap in _SHIPPED_FLUSH_CAPS:
        _validate_flush_cap(cap, 16)  # no raise

    # A shipped cap paired with a non-shipped block size (the verify key's ``_t``
    # field) is rejected: no ``_t{block_size}`` verify cubin ships for it. Use a
    # block_size below the cap so the ring invariant passes and the block-size
    # check is what fires.
    bad_block = 24
    assert bad_block not in _SHIPPED_VERIFY_BLOCK_SIZES
    for cap in _SHIPPED_FLUSH_CAPS:
        assert cap >= bad_block  # ring invariant satisfied; block-size check fires
        with pytest.raises(ValueError, match="no shipped replay verify cubin"):
            _validate_flush_cap(cap, bad_block)

    # With dev JIT on, the non-shipped block size compiles on demand: no raise.
    monkeypatch.setenv("KESTREL_CUTE_JIT", "1")
    _validate_flush_cap(_SHIPPED_FLUSH_CAPS[0], bad_block)


def _runtime_admission_stub(*, model_max: int, spec_max=None, spec_block_size=16):
    """Minimal ``Qwen35Runtime`` carrier exercising only the admission-cap path.

    Builds the runtime via ``__new__`` and pins just the attributes
    ``_admission_max_seq_len`` / ``can_reserve`` / ``prepare_sequence`` read, so
    the cap logic is testable without a GPU or a real KV pool. When a spec runner
    is present it exposes both ``max_seq_len`` and ``block_size`` (the verify
    block width), which the cap subtracts as ``block_size + 4`` headroom."""
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    rt.max_seq_length = int(model_max)
    rt._spec_runner = (
        None
        if spec_max is None
        else SimpleNamespace(
            max_seq_len=int(spec_max), block_size=int(spec_block_size)
        )
    )
    return rt


def test_admission_max_seq_len_caps_to_spec_row_reservation():
    """When a spec runner is active, admission is capped to its fixed per-row
    ``max_seq_len`` (which may be set below the model context), not the model
    maximum.

    Regression (runtime admission): ``SpecRunner`` reserves only
    ``max_seq_len`` KV pages for each fixed spec row, but ``can_reserve`` /
    ``prepare_sequence`` validated against the full model context. A request
    longer than the spec row but within the model context would be admitted and
    then re-pointed onto a spec row whose pages cannot hold it, so ``step()``
    builds slot mappings past the reservation and corrupts KV. Cap admission to
    the spec row length."""
    # No spec runner -> model context governs.
    rt = _runtime_admission_stub(model_max=4096, spec_max=None)
    assert rt._admission_max_seq_len() == 4096

    # Spec runner with a SMALLER per-row reservation -> that governs, minus the
    # ``block_size + 4`` verify headroom the final macro-step needs.
    rt = _runtime_admission_stub(model_max=4096, spec_max=512, spec_block_size=16)
    assert rt._admission_max_seq_len() == 512 - (16 + 4)  # == 492
    assert rt.can_reserve.__func__  # bound method exists

    # Spec row larger than the model context never raises the cap above it (the
    # headroom is subtracted from the row length, which still exceeds the model
    # context, so the model context wins).
    rt = _runtime_admission_stub(model_max=4096, spec_max=8192, spec_block_size=16)
    assert rt._admission_max_seq_len() == 4096


def test_can_reserve_rejects_over_spec_row_length():
    """``can_reserve`` rejects a total length over the spec row reservation even
    when it would fit the model context."""
    rt = _runtime_admission_stub(model_max=4096, spec_max=512, spec_block_size=16)
    # Stub the remaining gates so only the length cap is under test.
    rt.page_table = SimpleNamespace(can_reserve_with_eviction=lambda n: True)
    rt._available_batch_slots = lambda: 4

    cap = 512 - (16 + 4)                        # spec row minus verify headroom (492)
    assert rt.can_reserve(cap) is True          # exactly at the admission cap
    assert rt.can_reserve(cap + 1) is False     # one past the cap -> rejected
    assert rt.can_reserve(512) is False         # the bare spec row has no verify room
    assert rt.can_reserve(4096) is False        # within model context, past spec row


def test_prepare_sequence_rejects_over_spec_row_length(monkeypatch):
    """``prepare_sequence`` rejects a prompt+generation over the spec row length
    before it reserves pages / runs the model forward.

    The error must reference the (capped) admission limit so the caller sees the
    spec row reservation, not the larger model context."""
    rt = _runtime_admission_stub(model_max=4096, spec_max=512, spec_block_size=16)
    rt.image_prefix_length = 0
    # A prompt that fits the model context (4096) but, with generation, exceeds
    # the spec row reservation (512) less its verify headroom must be rejected at
    # admit. The error references the capped admission limit (492), not 512.
    prompt = [SimpleNamespace() for _ in range(500)]
    # Avoid the image branch / token-splice path: patch the image-detect to None.
    with pytest.raises(ValueError, match="exceeds max_seq_length=492"):
        rt.prepare_sequence(prompt, max_new_tokens=128)


def test_admission_reserves_verify_block_headroom():
    """The admission cap leaves room for the final macro-step's ``block_size``
    verify so no ``step()`` writes KV past the fixed spec-row reservation.

    Regression (runtime.py admission headroom): ``step()`` verifies a
    full ``block_size`` from the current KV cursor, so a request admitted with
    ``target_length == SpecRunner.max_seq_len`` reaches a final macro-step whose
    pending token sits at ``ctx_len == max_seq_len - 1`` and builds slot mappings
    through ``ctx_len + block_size - 1`` -- past the reserved pages. The cap must
    subtract ``block_size + 4`` (the same headroom ``decode_batch`` /
    ``_prefill_row`` reserve on the prefill path) so the final block fits."""
    max_seq_len, block_size = 512, 16
    headroom = block_size + 4  # 20
    cap = max_seq_len - headroom  # 492

    rt = _runtime_admission_stub(
        model_max=4096, spec_max=max_seq_len, spec_block_size=block_size
    )
    rt.page_table = SimpleNamespace(can_reserve_with_eviction=lambda n: True)
    rt._available_batch_slots = lambda: 4

    assert rt._admission_max_seq_len() == cap

    # A target_length that exactly fills the row (no headroom) would put the final
    # macro-step's verify block past the reservation -> rejected at admit.
    assert rt.can_reserve(max_seq_len) is False
    # One token short of the bare row still leaves < block_size of room -> rejected.
    assert rt.can_reserve(max_seq_len - 1) is False
    # Exactly the cap: the final block_size verify lands on the last reserved
    # pages (ctx through max_seq_len - 1 stays inside) -> admitted.
    assert rt.can_reserve(cap) is True
    # One past the cap erodes the headroom by a token -> rejected.
    assert rt.can_reserve(cap + 1) is False

    # Sanity: the cap matches the prefill-path budget (prompt + gen + block + 4).
    # A prompt+gen summing to ``cap`` is admitted; ``cap + 1`` is not -- the same
    # boundary ``decode_batch`` enforces as ``+ block_size + 4 <= max_seq_len``.
    assert (cap + headroom) == max_seq_len

    # A larger block size widens the reserved headroom proportionally.
    rt32 = _runtime_admission_stub(
        model_max=4096, spec_max=max_seq_len, spec_block_size=32
    )
    assert rt32._admission_max_seq_len() == max_seq_len - (32 + 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_runtime_generate_produces_text():
    cfg = RuntimeConfig(device="cuda", model=_MODEL_ID, max_batch_size=1)
    rt = Qwen35Runtime(cfg, kv_pool=KVMemoryPool(device=cfg.resolved_device()))
    answer = rt.generate("What is 2+2?", max_new_tokens=32)
    assert isinstance(answer, str)
    assert answer.strip(), f"expected non-empty answer; got {answer!r}"
    assert "4" in answer, f"expected a numeric answer about 2+2; got {answer!r}"


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        input_ids = kwargs["input_ids"]
        return SimpleNamespace(
            last_hidden_state=torch.ones(
                input_ids.shape[0], input_ids.shape[1], 2
            ),
            past_key_values=f"language-cache-{len(self.calls)}",
        )


class _FakeQwenVLModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.language_model = _FakeLanguageModel()

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        input_ids = kwargs["input_ids"]
        rope_deltas = (
            torch.tensor([[7]], dtype=torch.long)
            if kwargs.get("pixel_values") is not None
            else kwargs.get("rope_deltas")
        )
        return SimpleNamespace(
            last_hidden_state=torch.ones(
                input_ids.shape[0], input_ids.shape[1], 2
            ),
            past_key_values=f"vision-cache-{len(self.calls)}",
            rope_deltas=rope_deltas,
        )


def test_image_forward_cache_routes_decode_through_multimodal_wrapper():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    qwen_model = _FakeQwenVLModel()
    rt.model = SimpleNamespace(model=qwen_model)

    _, cache = rt._forward_base(
        input_ids=torch.tensor([[1, 2]]),
        past_key_values=None,
        pixel_values=torch.ones(1),
        image_grid_thw=torch.ones(1, 3, dtype=torch.long),
        mm_token_type_ids=torch.tensor([[0, 1]], dtype=torch.int32),
    )

    _, next_cache = rt._forward_base(
        input_ids=torch.tensor([[3]]),
        past_key_values=cache,
    )

    assert len(qwen_model.calls) == 2
    assert qwen_model.language_model.calls == []
    assert qwen_model.calls[1]["past_key_values"] == "vision-cache-1"
    assert qwen_model.calls[1]["rope_deltas"] is cache.rope_deltas
    assert next_cache.rope_deltas is cache.rope_deltas


def test_multimodal_forward_uses_explicit_position_ids():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    qwen_model = _FakeQwenVLModel()
    rt.model = SimpleNamespace(model=qwen_model)

    position_ids = torch.arange(4, dtype=torch.long).view(4, 1, 1)
    cache = _QwenForwardCache(
        past_key_values=object(),
        rope_deltas=torch.tensor([[7]], dtype=torch.long),
    )

    _, next_cache = rt._forward_base(
        input_ids=torch.tensor([[3]]),
        past_key_values=cache,
        position_ids=position_ids,
    )

    assert len(qwen_model.calls) == 1
    assert qwen_model.calls[0]["position_ids"] is position_ids
    assert qwen_model.calls[0]["rope_deltas"] is cache.rope_deltas
    assert next_cache.rope_deltas is cache.rope_deltas


def test_text_forward_cache_keeps_language_model_shortcut():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    qwen_model = _FakeQwenVLModel()
    rt.model = SimpleNamespace(model=qwen_model)

    _, cache = rt._forward_base(
        input_ids=torch.tensor([[1, 2]]),
        past_key_values=None,
    )
    _, next_cache = rt._forward_base(
        input_ids=torch.tensor([[3]]),
        past_key_values=cache,
    )

    assert qwen_model.calls == []
    assert len(qwen_model.language_model.calls) == 2
    assert next_cache.rope_deltas is None


def test_text_forward_uses_batch_idx_for_gdn_state_indices():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    qwen_model = _FakeQwenVLModel()
    rt.model = SimpleNamespace(model=qwen_model)
    rt.device = torch.device("cpu")
    rt.page_table = SimpleNamespace(
        page_table=torch.arange(8, dtype=torch.int32).view(8, 1),
        build_slot_mapping=lambda *, batch_idx, positions: batch_idx.view(-1, 1),
    )
    cache = object()

    _, next_cache = rt._forward_base(
        input_ids=torch.tensor([[3]]),
        past_key_values=cache,
        batch_idx=5,
        cache_position_ids=torch.tensor([[2]], dtype=torch.long),
    )

    call = qwen_model.language_model.calls[0]
    assert next_cache.past_key_values == "language-cache-1"
    assert torch.equal(call["gdn_state_indices"], torch.tensor([5]))
    assert torch.equal(call["page_table"], torch.tensor([[5]], dtype=torch.int32))


def test_text_forward_uses_compact_gdn_state_indices_for_compact_cache():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    qwen_model = _FakeQwenVLModel()
    rt.model = SimpleNamespace(model=qwen_model)
    rt.device = torch.device("cpu")
    cfg = SimpleNamespace(
        layer_types=["linear_attention"],
        num_hidden_layers=1,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
    )
    rt.page_table = SimpleNamespace(
        page_table=torch.arange(8, dtype=torch.int32).view(8, 1),
        build_slot_mapping=lambda *, batch_idx, positions: batch_idx.view(-1, 1),
    )
    cache = _make_qwen_cache(
        config=cfg,
        page_table=PageTable(n_pages=4, page_size=1, max_batch_size=4, device="cpu"),
        pool=KVMemoryPool(device=torch.device("cpu")),
        dtype=torch.float32,
    )

    rt._forward_base(
        input_ids=torch.tensor([[3]]),
        past_key_values=cache,
        batch_idx=1,
        cache_position_ids=torch.tensor([[2]], dtype=torch.long),
    )

    call = qwen_model.language_model.calls[0]
    assert torch.equal(call["gdn_state_indices"], torch.tensor([0]))
    assert torch.equal(call["page_table"], torch.tensor([[1]], dtype=torch.int32))


def test_text_forward_preserves_packed_gdn_row_mapping():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    qwen_model = _FakeQwenVLModel()
    rt.model = SimpleNamespace(model=qwen_model)
    rt.device = torch.device("cpu")
    cfg = SimpleNamespace(
        layer_types=["linear_attention"],
        num_hidden_layers=1,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
    )
    rt.page_table = SimpleNamespace(
        page_table=torch.arange(8, dtype=torch.int32).view(8, 1),
        build_slot_mapping=lambda *, batch_idx, positions: batch_idx.view(-1, 1),
    )
    cache = _make_qwen_cache(
        config=cfg,
        page_table=PageTable(n_pages=4, page_size=1, max_batch_size=4, device="cpu"),
        pool=KVMemoryPool(device=torch.device("cpu")),
        dtype=torch.float32,
    )

    rt._forward_base(
        input_ids=torch.tensor([[3]]),
        past_key_values=_QwenForwardCache(
            cache,
            linear_state_row_indices=torch.tensor([1], dtype=torch.long),
        ),
        batch_idx=5,
        cache_position_ids=torch.tensor([[2]], dtype=torch.long),
    )

    call = qwen_model.language_model.calls[0]
    assert torch.equal(call["gdn_state_indices"], torch.tensor([1]))
    assert torch.equal(call["page_table"], torch.tensor([[5]], dtype=torch.int32))


def test_decode_with_slot_runs_one_prepared_batch():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)

    calls = []

    graph_manager = SimpleNamespace(
        run=lambda slot, batch_size: calls.append(("graph", slot, batch_size)),
    )

    slot = SimpleNamespace(
        meta=SimpleNamespace(batch_idx=SimpleNamespace(np=[3, 5])),
    )

    rt._decode_graphs = graph_manager

    Qwen35Runtime.decode_with_slot(rt, slot, batch_size=2)

    assert calls == [("graph", slot, 2)]


def test_decode_with_slot_runs_bound_megakernel_for_b1(monkeypatch):
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    calls = []
    slot = SimpleNamespace(compute_stream="decode-stream")

    class _StreamContext:
        def __enter__(self):
            calls.append(("enter-stream", slot.compute_stream))

        def __exit__(self, exc_type, exc, traceback):
            calls.append(("exit-stream", slot.compute_stream))

    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _StreamContext())
    rt._decode_graphs = SimpleNamespace(
        run=lambda bound_slot, batch_size: calls.append(
            ("graph", bound_slot, batch_size)),
    )
    rt._decode_megakernel = SimpleNamespace(
        supports=lambda batch_size: batch_size == 1,
        run=lambda bound_slot, batch_size: calls.append(
            ("megakernel", bound_slot, batch_size)),
    )
    rt._decode_state_coordinator = None

    Qwen35Runtime.decode_with_slot(rt, slot, batch_size=1)

    assert calls == [
        ("enter-stream", "decode-stream"),
        ("megakernel", slot, 1),
        ("exit-stream", "decode-stream"),
    ]


@requires_mkl
def test_decode_path_switch_prepares_compiler_declared_state(monkeypatch):
    from mkl.megakernel.state_runtime import StateRepresentationRequirement

    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    calls = []
    generated_c1 = (StateRepresentationRequirement(
        "gdn_recurrent_state",
        "materialized",
        ("state_row", "value_head", "key", "value"),
        "fp32",
    ),)
    generated_c8 = (StateRepresentationRequirement(
        "gdn_recurrent_state",
        "materialized",
        ("state_row", "value_head", "value", "key"),
        "fp32",
    ),)
    native = (StateRepresentationRequirement(
        "gdn_recurrent_state",
        "replay",
        ("state_row", "value_head", "value", "key"),
        "fp32",
    ),)
    megakernel = SimpleNamespace(
        supports=lambda batch_size: batch_size <= 2,
        state_requirements_for=(
            lambda batch_size: generated_c1 if batch_size == 1 else generated_c8
        ),
        run=lambda slot, batch_size: calls.append(
            ("generated", slot, batch_size)),
    )
    coordinator = SimpleNamespace(
        prepare=lambda requirements, rows: calls.append(
            ("prepare", requirements, rows)),
    )
    slot = SimpleNamespace(
        compute_stream=None,
        meta=SimpleNamespace(batch_idx=SimpleNamespace(
            cpu=torch.tensor([3, 5, 7], dtype=torch.int64))),
    )
    rt._decode_megakernel = megakernel
    rt._decode_state_coordinator = coordinator
    rt._native_decode_state_requirements = native
    rt._decode_graphs = SimpleNamespace(
        run=lambda bound_slot, batch_size: calls.append(
            ("native", bound_slot, batch_size)))

    Qwen35Runtime.decode_with_slot(rt, slot, batch_size=2)
    Qwen35Runtime.decode_with_slot(rt, slot, batch_size=1)
    Qwen35Runtime.decode_with_slot(rt, slot, batch_size=3)

    assert calls == [
        ("prepare", generated_c8, (3, 5)),
        ("generated", slot, 2),
        ("prepare", generated_c1, (3,)),
        ("generated", slot, 1),
        ("prepare", native, (3, 5, 7)),
        ("native", slot, 3),
    ]


@requires_mkl
def test_native_state_requirement_uses_pool_owned_replay_form():
    from kestrel.models.qwen35.runtime import _native_decode_state_requirements
    from mkl.megakernel.state_runtime import (
        StatePhysicalForm,
        StateRepresentationRequirement,
    )

    generated = (
        StateRepresentationRequirement(
            "gdn_recurrent_state",
            "materialized",
            ("state_row", "value_head", "key", "value"),
            "fp32",
        ),
        StateRepresentationRequirement("kv_cache", "paged", (), "bf16"),
    )
    replay = StatePhysicalForm(
        "replay",
        ("state_row", "value_head", "value", "key"),
        "fp32",
    )

    assert _native_decode_state_requirements(
        generated,
        SimpleNamespace(replay_recurrent_form=replay),
    ) == (
        StateRepresentationRequirement(
            "gdn_recurrent_state",
            "replay",
            ("state_row", "value_head", "value", "key"),
            "fp32",
        ),
        generated[1],
    )


def test_prefill_state_coherence_bookkeeping_is_optional():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    calls = []
    rt._decode_state_coordinator = SimpleNamespace(
        mark_coherent=lambda rows: calls.append(tuple(rows)))

    rt._mark_decode_state_coherent((3, 5))
    rt._decode_state_coordinator = None
    rt._mark_decode_state_coherent((7,))

    assert calls == [(3, 5)]


def test_recurrent_checkpoint_reset_retires_replay_by_cursor_only():
    cfg = SimpleNamespace(
        linear_replay_capacity=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    layer = LinearAttentionLayer(cfg, replay_capacity=cfg.linear_replay_capacity)
    first = torch.randn((2, 2, 3, 3), dtype=torch.float32)
    second = torch.randn_like(first)

    layer.update_recurrent_state(first)
    layer.replay_k.fill_(1)
    layer.replay_u.fill_(2)
    layer.replay_g.fill_(3)
    layer.replay_lengths.fill_(4)
    layer.update_recurrent_state(second)

    torch.testing.assert_close(layer.recurrent_states, second)
    torch.testing.assert_close(
        layer.replay_checkpoint_states,
        second.transpose(-1, -2),
    )
    assert torch.all(layer.replay_k == 1)
    assert torch.all(layer.replay_u == 2)
    assert torch.all(layer.replay_g == 3)
    assert layer.replay_lengths.tolist() == [0, 0]


def test_zero_replay_cursor_ignores_stale_payload_when_materializing():
    cfg = SimpleNamespace(
        linear_replay_capacity=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    layer = LinearAttentionLayer(cfg, replay_capacity=cfg.linear_replay_capacity)
    initial = torch.randn((2, 2, 3, 3), dtype=torch.float32)
    layer.update_recurrent_state(initial)
    checkpoint = torch.randn_like(layer.replay_checkpoint_states)
    layer.replay_checkpoint_states.copy_(checkpoint)
    layer.recurrent_states.fill_(-99)
    layer.replay_k.normal_()
    layer.replay_u.normal_()
    layer.replay_g.normal_()
    layer.replay_lengths.zero_()

    layer.materialize_recurrent_from_replay()

    torch.testing.assert_close(
        layer.recurrent_states,
        checkpoint.transpose(-1, -2),
    )
    torch.testing.assert_close(layer.replay_checkpoint_states, checkpoint)


@requires_mkl
def test_linear_state_pool_seeds_selected_replay_rows_from_materialized():
    from kestrel.models.qwen35.paged_cache import Qwen35LinearStatePool
    from mkl.megakernel.state_runtime import StatePhysicalForm

    recurrent = torch.arange(3 * 2 * 2 * 3, dtype=torch.float32).reshape(
        3, 2, 2, 3)
    storage = SimpleNamespace(
        recurrent_states=recurrent.clone(),
        replay_checkpoint_states=torch.full((3, 2, 3, 2), -1.0),
        replay_k=torch.ones(3, 4, 2, 2),
        replay_u=torch.ones(3, 4, 2, 3),
        replay_g=torch.ones(3, 4, 2),
        replay_lengths=torch.full((3,), 4, dtype=torch.int32),
    )
    pool = Qwen35LinearStatePool.__new__(Qwen35LinearStatePool)
    pool.device = torch.device("cpu")
    pool.layers = [storage]
    replay_before = (
        storage.replay_k.clone(),
        storage.replay_u.clone(),
        storage.replay_g.clone(),
    )

    key_major_axes = ("state_row", "value_head", "key", "value")
    value_major_axes = ("state_row", "value_head", "value", "key")
    pool.transition_recurrent_form(
        StatePhysicalForm("materialized", key_major_axes, "fp32"),
        StatePhysicalForm("replay", value_major_axes, "fp32"),
        (0, 2),
    )

    expected = recurrent.index_select(
        0, torch.tensor([0, 2])).transpose(-1, -2)
    assert torch.equal(
        storage.replay_checkpoint_states.index_select(
            0, torch.tensor([0, 2])),
        expected,
    )
    assert torch.equal(
        storage.replay_checkpoint_states[1],
        torch.full((2, 3, 2), -1.0),
    )
    assert torch.equal(storage.replay_k, replay_before[0])
    assert torch.equal(storage.replay_u, replay_before[1])
    assert torch.equal(storage.replay_g, replay_before[2])
    assert storage.replay_lengths.tolist() == [0, 4, 0]


@requires_mkl
def test_linear_state_pool_preserves_value_major_checkpoint_on_replay_switch():
    from kestrel.models.qwen35.paged_cache import Qwen35LinearStatePool
    from mkl.megakernel.state_runtime import StatePhysicalForm

    checkpoint = torch.arange(
        3 * 2 * 3 * 2, dtype=torch.float32).reshape(3, 2, 3, 2)
    stale_recurrent = torch.full((3, 2, 2, 3), -99.0)
    storage = SimpleNamespace(
        recurrent_states=stale_recurrent.clone(),
        replay_checkpoint_states=checkpoint.clone(),
        replay_k=torch.ones(3, 4, 2, 2),
        replay_u=torch.ones(3, 4, 2, 3),
        replay_g=torch.ones(3, 4, 2),
        replay_lengths=torch.full((3,), 4, dtype=torch.int32),
    )
    pool = Qwen35LinearStatePool.__new__(Qwen35LinearStatePool)
    pool.device = torch.device("cpu")
    pool.layers = [storage]
    replay_before = (
        storage.replay_k.clone(),
        storage.replay_u.clone(),
        storage.replay_g.clone(),
    )
    value_major_axes = ("state_row", "value_head", "value", "key")

    pool.transition_recurrent_form(
        StatePhysicalForm("materialized", value_major_axes, "fp32"),
        StatePhysicalForm("replay", value_major_axes, "fp32"),
        (0, 2),
    )

    assert torch.equal(storage.replay_checkpoint_states, checkpoint)
    assert torch.equal(storage.recurrent_states, stale_recurrent)
    assert torch.equal(storage.replay_k, replay_before[0])
    assert torch.equal(storage.replay_u, replay_before[1])
    assert torch.equal(storage.replay_g, replay_before[2])
    assert storage.replay_lengths.tolist() == [0, 4, 0]


@requires_mkl
def test_linear_state_pool_selects_compiler_required_recurrent_storage():
    from kestrel.models.qwen35.paged_cache import Qwen35LinearStatePool
    from mkl.megakernel.state_runtime import StatePhysicalForm

    key_major = torch.empty(3, 2, 2, 3)
    value_major = torch.empty(3, 2, 3, 2)
    storage = SimpleNamespace(
        recurrent_states=key_major,
        replay_checkpoint_states=value_major,
    )
    pool = Qwen35LinearStatePool.__new__(Qwen35LinearStatePool)
    pool.layers = [storage, None]

    selected_key_major = pool.recurrent_tensors_for_form(StatePhysicalForm(
        "materialized",
        ("state_row", "value_head", "key", "value"),
        "fp32",
    ))
    selected_value_major = pool.recurrent_tensors_for_form(StatePhysicalForm(
        "materialized",
        ("state_row", "value_head", "value", "key"),
        "fp32",
    ))

    assert selected_key_major[0] is key_major
    assert selected_key_major[1] is None
    assert selected_value_major[0] is value_major
    assert selected_value_major[1] is None


@requires_mkl
def test_linear_state_pool_refuses_unsupported_in_place_layout_change():
    from kestrel.models.qwen35.paged_cache import Qwen35LinearStatePool
    from mkl.megakernel.state_runtime import StatePhysicalForm

    pool = Qwen35LinearStatePool.__new__(Qwen35LinearStatePool)
    pool.device = torch.device("cpu")
    pool.layers = []
    key_major = StatePhysicalForm(
        "materialized",
        ("state_row", "head", "key", "value"),
        "float32",
    )
    value_major = StatePhysicalForm(
        "materialized",
        ("state_row", "head", "value", "key"),
        "float32",
    )

    with pytest.raises(
        ValueError,
        match="does not support changing physical form",
    ):
        pool.transition_recurrent_form(
            key_major, value_major, (0,))


@requires_mkl
def test_decode_megakernel_binds_each_invocation_to_its_slot_stream(monkeypatch):
    from kestrel.models.qwen35 import megakernel_decode
    from mkl.compiler import frontend
    from mkl.megakernel import device_runtime
    from mkl.megakernel.state_runtime import StateRepresentationRequirement

    original_empty = torch.empty
    streams = []
    bind_calls = []
    ownership_calls = []
    ordering = []
    selected_state_forms = []
    bound_runtime_inputs = []

    class _FakeEvent:
        def __init__(self):
            self.index = len([item for item in ordering if item[0] == "new-event"])
            ordering.append(("new-event", self.index))

        def record(self, stream):
            ordering.append(("record", self.index, stream.name))

    class _FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            ordering.append(("wait", self.name, event.index))

    class _StreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            ordering.append(("enter", self.stream.name))

        def __exit__(self, exc_type, exc, traceback):
            ordering.append(("exit", self.stream.name))

    primary_stream = _FakeStream("primary-stream")
    ambient_stream = _FakeStream("ambient-stream")
    key_major_graph = object()
    value_major_graph = object()

    def compiled_for(graph):
        runtime_input_names = (
            "input_ids",
            "final_norm",
            "logits",
            "page_table",
            "batch_idx",
            "input_pos",
            "position_ids",
            "rope_inv_freq",
            "gdn_conv_state",
            "gdn_recurrent_state",
            "mK",
            "mV",
        )
        external = tuple(
            SimpleNamespace(name=f"{name}_abi")
            for name in runtime_input_names
        )
        return SimpleNamespace(
            graph=graph,
            device_program=SimpleNamespace(
                validate_argument_plan=lambda: None,
                argument_plan=SimpleNamespace(
                    arguments=[],
                    by_source=lambda source: (
                        external if source == "external" else ()
                    ),
                ),
                physical_abi=SimpleNamespace(operands=tuple(
                    SimpleNamespace(
                        abi_name=f"{name}_abi",
                        logical_name=name,
                        owner="engine",
                    )
                    for name in runtime_input_names
                )),
            ),
        )

    compiled_c1 = compiled_for(key_major_graph)
    compiled_c8 = compiled_for(value_major_graph)
    config = SimpleNamespace(hidden_size=4)
    text_model = SimpleNamespace(
        config=config,
        named_parameters=lambda: [],
        rotary_emb=SimpleNamespace(inv_freq=torch.empty(0)),
    )
    slots = [
        SimpleNamespace(
            slot_id=index,
            compute_stream=f"slot-stream-{index}",
            decode_token_ids=torch.zeros(8, dtype=torch.int64),
            hidden_last=torch.zeros(8, config.hidden_size),
            logits=torch.zeros(8, 16),
            meta=SimpleNamespace(
                batch_idx=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
                input_pos=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
            ),
            position_ids=torch.zeros(4, 8, 1, dtype=torch.int32),
        )
        for index in range(2)
    ]
    runtime = SimpleNamespace(
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        primary_stream=primary_stream,
        model=SimpleNamespace(
            model=SimpleNamespace(
                language_model=text_model
            ),
            lm_head=object(),
        ),
        _linear_state_pool=SimpleNamespace(
            layers=[],
            recurrent_tensors_for_form=lambda form: (
                selected_state_forms.append(form)
                or (
                    ["key-major-state"]
                    if form.storage_axis_order[-2:] == ("key", "value")
                    else ["value-major-state"]
                )
            ),
        ),
        _paged_kv=SimpleNamespace(layers=()),
        page_table=SimpleNamespace(
            n_pages=8,
            page_table=torch.zeros(1, 8, dtype=torch.int32),
        ),
        decode_slots=slots,
        _decode_rope_deltas=torch.zeros(1, 1, dtype=torch.int32),
    )

    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: original_empty(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        ),
    )
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: ambient_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _StreamContext(stream))
    monkeypatch.setattr(
        frontend,
        "bind_owned_weight_storage",
        lambda *args, **kwargs: ownership_calls.append((args, kwargs))
        or SimpleNamespace(buffers={}),
    )
    monkeypatch.setattr(
        frontend,
        "derive_device_carried_state_contracts",
        lambda graph, _program: (
            SimpleNamespace(
                buffer="gdn_recurrent_state",
                representation="materialized",
                storage_axis_order=(
                    "state_row",
                    "value_head",
                    *(("key", "value") if graph is key_major_graph
                      else ("value", "key")),
                ),
                storage_dtype="fp32",
            ),
        ),
    )

    def assemble(*args, stream, **kwargs):
        streams.append(stream)
        bound_runtime_inputs.append(kwargs["runtime_inputs"])
        return SimpleNamespace(values={})

    monkeypatch.setattr(device_runtime, "assemble_torch_device_bindings", assemble)
    monkeypatch.setattr(
        device_runtime,
        "bind_aot_device_program",
        lambda *args, **kwargs: bind_calls.append(args)
        or SimpleNamespace(),
    )

    bundle_c1 = object()
    bundle_c8 = object()
    validated_c1 = SimpleNamespace(program=compiled_c1)
    validated_c8 = SimpleNamespace(program=compiled_c8)
    generated = megakernel_decode.Qwen35DecodeMegakernel(
        runtime,
        programs={
            1: (compiled_c1, validated_c1, bundle_c1),
            8: (compiled_c8, validated_c8, bundle_c8),
        },
    )

    assert streams == [
        "slot-stream-0",
        "slot-stream-0",
        "slot-stream-1",
        "slot-stream-1",
    ]
    assert selected_state_forms == [
        StateRepresentationRequirement(
            "gdn_recurrent_state",
            "materialized",
            ("state_row", "value_head", "key", "value"),
            "fp32",
        ).physical_form,
        StateRepresentationRequirement(
            "gdn_recurrent_state",
            "materialized",
            ("state_row", "value_head", "value", "key"),
            "fp32",
        ).physical_form,
    ]
    assert [
        inputs["gdn_recurrent_state"] for inputs in bound_runtime_inputs
    ] == [
        ["key-major-state"],
        ["value-major-state"],
        ["key-major-state"],
        ["value-major-state"],
    ]
    assert all(
        set(("input_ids", "final_norm", "logits")) <= set(inputs)
        for inputs in bound_runtime_inputs
    )
    assert all("x" not in inputs for inputs in bound_runtime_inputs)
    assert generated.state_requirements_for(1)[0].storage_axis_order[-2:] == (
        "key", "value")
    assert generated.state_requirements_for(4)[0].storage_axis_order[-2:] == (
        "value", "key")
    assert ownership_calls == [
        (
            (compiled_c1, runtime.model),
            {
                "device": runtime.device,
                "layer_prefix": "model.language_model.layers",
            },
        )
    ]
    assert ordering == [
        ("new-event", 0),
        ("record", 0, "ambient-stream"),
        ("wait", "primary-stream", 0),
        ("new-event", 1),
        ("enter", "primary-stream"),
        ("record", 1, "primary-stream"),
        ("exit", "primary-stream"),
        ("wait", "ambient-stream", 1),
    ]
    assert bind_calls == [
        (validated_c1, bundle_c1, {}),
        (validated_c8, bundle_c8, {}),
        (validated_c1, bundle_c1, {}),
        (validated_c8, bundle_c8, {}),
    ]


@requires_mkl
def test_decode_megakernel_factory_fails_closed_on_bundle_miss(monkeypatch):
    from kestrel.models.qwen35 import megakernel_decode
    from mkl.megakernel.device_runtime import DeviceRuntimeError

    calls = []
    runtime = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(config="text-config")
            )
        ),
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        max_batch_size=1,
        _paged_kv=SimpleNamespace(layers=()),
        _linear_state_pool=SimpleNamespace(
            initialize_from_config=lambda *args, **kwargs: calls.append(
                ("initialize", args, kwargs)
            )
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: calls.append(("properties", device)) or SimpleNamespace(
            multi_processor_count=132,
            major=9,
            minor=0,
            name="NVIDIA H100 80GB HBM3",
        ),
    )
    def compile_config(config, *, batch_capacity, num_ctas, gpu):
        calls.append(("compile", config, batch_capacity, num_ctas, gpu))
        return _fake_capacity_compiled(batch_capacity)

    monkeypatch.setattr(
        megakernel_decode, "_compile_from_config", compile_config)
    # The AOT boundary requires a validation certificate -- stub the mint so bundle resolution
    # receives the ValidatedProgram rather than the raw artifact.
    monkeypatch.setattr(
        "mkl.compiler.frontend.validate_program.validate_compiled_tape",
        lambda art: SimpleNamespace(program=art),
    )
    monkeypatch.setattr(
        "mkl.megakernel.device_runtime.resolve_shipped_aot_bundle",
        lambda artifact, *, arch: calls.append(
            ("resolve", artifact, arch)
        ) or None,
    )

    with pytest.raises(
        DeviceRuntimeError,
        match=r"missing active extents \[1\].*unresolved artifacts",
    ):
        megakernel_decode.Qwen35DecodeMegakernel.try_create(runtime)
    assert calls[0] == ("properties", torch.device("cuda:0"))
    assert [
        call for call in calls if call[0] == "compile"
    ] == [
        ("compile", "text-config", capacity, 132, "NVIDIA H100 80GB HBM3")
        for capacity in (1, 2, 4, 8)
    ]
    resolve_calls = [call for call in calls if call[0] == "resolve"]
    assert [
        (call[1].program.capacity, call[2])
        for call in resolve_calls
    ] == [
        (capacity, "sm90") for capacity in (1, 2, 4, 8)
    ]


@requires_mkl
def test_decode_compile_passes_model_and_gpu_config_to_compiler(monkeypatch):
    from kestrel.models.qwen35 import megakernel_decode
    from mkl.compiler.frontend.models import qwen as qwen_frontend

    calls = []
    monkeypatch.setattr(
        qwen_frontend,
        "compile_qwen35",
        lambda **kwargs: calls.append(kwargs) or "compiled",
    )
    config = SimpleNamespace(
        is_moe=False,
        max_position_embeddings=65536,
        num_hidden_layers=64,
        hidden_size=2560,
        intermediate_size=9216,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        partial_rotary_factor=0.25,
        mrope_section=(24, 20, 20),
        layer_types=["linear_attention", "full_attention"] * 32,
    )

    assert megakernel_decode._compile_from_config(
        config,
        num_ctas=132,
        gpu="NVIDIA H100 80GB HBM3",
    ) == "compiled"
    assert calls == [{
        "n_layers": 64,
        "hidden": 2560,
        "inter": 9216,
        "nh": 16,
        "nkv": 4,
        "head_dim": 256,
        "num_k_heads": 16,
        "num_v_heads": 32,
        "key_head_dim": 128,
        "value_head_dim": 128,
        "conv_kernel": 4,
        "partial_rotary": 0.25,
        "rope_sections": [24, 20, 20],
        "vocab_size": 248320,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "num_ctas": 132,
        "num_splits": None,
        "max_kv_len": 65536,
        "batch_tile": 1,
        "static_extent_bindings": {},
        "gpu": "NVIDIA H100 80GB HBM3",
        "layer_types": [0, 1] * 32,
    }]


@requires_mkl
def test_decode_megakernel_factory_considers_wider_capacity_domains(monkeypatch):
    from kestrel.models.qwen35 import megakernel_decode

    calls = []
    runtime = SimpleNamespace(
        max_batch_size=2,
        max_batch_slots=2,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        _paged_kv=SimpleNamespace(layers=()),
        _linear_state_pool=SimpleNamespace(
            initialize_from_config=lambda *_args, **_kwargs: None),
        model=SimpleNamespace(model=SimpleNamespace(
            language_model=SimpleNamespace(config="config"))),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            multi_processor_count=132, major=9, minor=0, name="test-gpu"),
    )
    monkeypatch.setattr(
        megakernel_decode,
        "_compile_from_config",
        lambda _config, *, batch_capacity, **_kwargs: calls.append(
            batch_capacity) or _fake_capacity_compiled(batch_capacity),
    )
    monkeypatch.setattr(
        "mkl.compiler.frontend.validate_program.validate_compiled_tape",
        lambda compiled: SimpleNamespace(program=compiled),
    )
    monkeypatch.setattr(
        "mkl.megakernel.device_runtime.resolve_shipped_aot_bundle",
        lambda validated, *, arch: SimpleNamespace(validated=validated, arch=arch),
    )
    monkeypatch.setattr(
        megakernel_decode.Qwen35DecodeMegakernel,
        "__init__",
        lambda self, _runtime, *, programs: setattr(self, "programs", programs),
    )

    result = megakernel_decode.Qwen35DecodeMegakernel.try_create(runtime)
    assert calls == [1, 2]
    assert set(result.programs) == {1, 2}


@requires_mkl
def test_decode_megakernel_routes_intermediate_batch_to_covering_capacity(
    monkeypatch,
):
    from kestrel.models.qwen35 import megakernel_decode

    launches = []
    preparation_calls = []

    def launch(**kwargs):
        launches.append(kwargs)
        preparation_calls.append(("launch", kwargs))

    invocation = SimpleNamespace(launch=launch)
    megakernel = megakernel_decode.Qwen35DecodeMegakernel.__new__(
        megakernel_decode.Qwen35DecodeMegakernel
    )
    megakernel._programs = {
        capacity: object() for capacity in (1, 2, 4, 8, 16)
    }
    megakernel._slots = {
        (7, 16): SimpleNamespace(
            invocation=invocation,
            uses_kv_len=True,
        )
    }
    megakernel._input_preparation_plan = (
        SimpleNamespace(name="first"),
        SimpleNamespace(name="second"),
    )
    megakernel._input_preparation_callbacks = {
        name: (
            lambda slot, batch_size, name=name:
            preparation_calls.append((name, slot.slot_id, batch_size))
        )
        for name in ("first", "second")
    }
    slot = SimpleNamespace(
        slot_id=7,
        compute_stream="stream",
        decode_token_ids=torch.arange(16),
        meta=SimpleNamespace(
            input_pos=SimpleNamespace(cpu=torch.arange(16))
        ),
        hidden_last=torch.empty((16, 3)),
        logits=torch.empty((16, 3)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "stream",
        lambda _stream: contextlib.nullcontext(),
    )

    assert megakernel.supports(11)
    megakernel.run(slot, batch_size=11)

    assert launches == [{"kv_len": 11, "active_batch": 11}]
    assert preparation_calls == [
        ("first", 7, 11),
        ("second", 7, 11),
        ("launch", {"kv_len": 11, "active_batch": 11}),
    ]


def test_decode_megakernel_factory_rejects_non_bf16_runtime(monkeypatch):
    from kestrel.models.qwen35 import megakernel_decode

    runtime = SimpleNamespace(max_batch_size=1, dtype=torch.float16)
    monkeypatch.setattr(
        megakernel_decode,
        "_compile_from_config",
        lambda *_args, **_kwargs: pytest.fail(
            "an ineligible runtime must not compile or resolve a B1 artifact"
        ),
    )

    assert megakernel_decode.Qwen35DecodeMegakernel.try_create(runtime) is None


def test_decode_megakernel_factory_rejects_non_unit_kv_pages(monkeypatch):
    from kestrel.models.qwen35 import megakernel_decode

    runtime = SimpleNamespace(
        max_batch_size=1,
        _paged_kv=SimpleNamespace(
            layers=(
                SimpleNamespace(
                    k_cache=torch.empty(1, 1, 2, 1),
                    v_cache=torch.empty(1, 1, 2, 1),
                ),
            )
        ),
    )
    monkeypatch.setattr(
        megakernel_decode,
        "_compile_from_config",
        lambda *_args, **_kwargs: pytest.fail(
            "an ineligible runtime must not compile or resolve a B1 artifact"
        ),
    )

    assert megakernel_decode.Qwen35DecodeMegakernel.try_create(runtime) is None


def test_decode_megakernel_factory_uses_native_without_compiler(monkeypatch):
    import builtins

    from kestrel.models.qwen35 import megakernel_decode

    runtime = SimpleNamespace(device=torch.device("cuda:0"))
    real_import = builtins.__import__

    def import_without_mkl(name, *args, **kwargs):
        if name == "mkl" or name.startswith("mkl."):
            raise ModuleNotFoundError(
                "No module named 'mkl'", name="mkl"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mkl)

    assert megakernel_decode.Qwen35DecodeMegakernel.try_create(runtime) is None


def test_decode_megakernel_factory_rejects_incomplete_compiler(monkeypatch):
    import builtins

    from kestrel.models.qwen35 import megakernel_decode

    runtime = SimpleNamespace(device=torch.device("cuda:0"))
    real_import = builtins.__import__

    def import_with_incomplete_mkl(name, *args, **kwargs):
        if name == "mkl.compiler.frontend.gpu_model":
            raise ModuleNotFoundError(
                "No module named 'mkl.compiler.frontend.gpu_model'",
                name="mkl.compiler.frontend.gpu_model",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_incomplete_mkl)

    with pytest.raises(
        ModuleNotFoundError,
        match="mkl.compiler.frontend.gpu_model",
    ):
        megakernel_decode.Qwen35DecodeMegakernel.try_create(runtime)


@requires_mkl
def test_decode_megakernel_factory_falls_back_for_unsupported_config(monkeypatch):
    from kestrel.models.qwen35 import megakernel_decode

    config = SimpleNamespace()
    runtime = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(config=config)
            )
        ),
        device=torch.device("cuda:0"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            multi_processor_count=132,
            major=9,
            minor=0,
            name="NVIDIA H100 80GB HBM3",
        ),
    )

    def reject_config(*_args, **_kwargs):
        raise megakernel_decode._UnsupportedDecodeConfig("unsupported")

    monkeypatch.setattr(
        megakernel_decode,
        "_compile_from_config",
        reject_config,
    )

    assert megakernel_decode.Qwen35DecodeMegakernel.try_create(runtime) is None


@requires_mkl
def test_decode_megakernel_factory_fails_closed_without_device_calibration(
    monkeypatch,
):
    from kestrel.models.qwen35 import megakernel_decode
    from mkl.compiler.frontend.gpu_model import CalibrationUnavailable
    from mkl.megakernel.device_runtime import DeviceRuntimeError

    runtime = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(config=SimpleNamespace())
            )
        ),
        device=torch.device("cuda:0"),
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

    def reject_uncalibrated_device(*_args, **_kwargs):
        raise CalibrationUnavailable("NVIDIA H200 has no calibration")

    monkeypatch.setattr(
        megakernel_decode,
        "_compile_from_config",
        reject_uncalibrated_device,
    )
    monkeypatch.setattr(
        "mkl.megakernel.device_runtime.resolve_shipped_aot_bundle",
        lambda *_args, **_kwargs: pytest.fail(
            "an uncalibrated device must fall back before bundle resolution"
        ),
    )

    with pytest.raises(
        DeviceRuntimeError,
        match="no calibration for 'NVIDIA H200'",
    ):
        megakernel_decode.Qwen35DecodeMegakernel.try_create(runtime)


class _FakeEvent:
    def __init__(self) -> None:
        self.records = 0
        self.synchronizes = 0

    def record(self) -> None:
        self.records += 1

    def synchronize(self) -> None:
        self.synchronizes += 1


def test_acquire_prefill_slot_waits_for_requested_pending_event():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    event = _FakeEvent()
    slot = SimpleNamespace(
        slot_id=0,
        step_done_event=event,
        step_done_event_pending=True,
    )
    rt._prefill_slots = (slot,)
    rt._prefill_slot_free = [slot]

    actual = Qwen35Runtime.acquire_prefill_slot(rt, slot_id=0)

    assert actual is slot
    assert event.synchronizes == 1
    assert slot.step_done_event_pending is False
    assert rt._prefill_slot_free == []


def test_acquire_prefill_slot_matches_requested_slot_id():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    slot0 = SimpleNamespace(
        slot_id=0,
        step_done_event=_FakeEvent(),
        step_done_event_pending=False,
    )
    slot1 = SimpleNamespace(
        slot_id=1,
        step_done_event=_FakeEvent(),
        step_done_event_pending=False,
    )
    rt._prefill_slots = (slot0, slot1)
    rt._prefill_slot_free = [slot1, slot0]

    actual = Qwen35Runtime.acquire_prefill_slot(rt, slot_id=1)

    assert actual is slot1
    assert rt._prefill_slot_free == [slot0]


def test_release_prefill_slot_rejects_duplicate_release():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    slot = SimpleNamespace(slot_id=0)
    rt._prefill_slot_free = [slot]

    with pytest.raises(RuntimeError, match="already free"):
        Qwen35Runtime.release_prefill_slot(rt, slot)


def test_launch_prepared_batch_batches_prefill_logits_once():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    rt.max_batch_size = 4
    rt.device = torch.device("cpu")
    committed = []
    stored = []
    packed = SimpleNamespace(
        batch_indices=[3, 5],
        last_token_offsets=torch.tensor([1, 3], dtype=torch.long),
    )
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
    )
    cache = object()

    class FakeLmHead:
        def __init__(self):
            self.calls = []

        def __call__(self, rows):
            self.calls.append(rows.clone())
            return rows + 100.0

    lm_head = FakeLmHead()
    rt.model = SimpleNamespace(lm_head=lm_head)
    rt.page_table = SimpleNamespace(
        commit_block_table=lambda batch_indices: committed.append(list(batch_indices))
    )
    def build_packed_prefill_batch(*args, **kwargs):
        kwargs["prefill_slot"].batch_idx[:2].copy_(torch.tensor([3, 5]))
        return packed

    rt._build_packed_prefill_batch = build_packed_prefill_batch
    rt._forward_packed_prefill = lambda packed_batch: (hidden, cache)
    rt._store_packed_sequence_caches = (
        lambda batch_indices, cache_value, *, host_batch_indices: stored.append(
            (list(batch_indices), cache_value, list(host_batch_indices))
        )
    )
    prepared_sequences = [
        SimpleNamespace(state=SimpleNamespace(batch_idx=3, last_hidden=None)),
        SimpleNamespace(state=SimpleNamespace(batch_idx=5, last_hidden=None)),
    ]
    event = _FakeEvent()
    prefill_slot = SimpleNamespace(
        batch_idx=torch.empty(4, dtype=torch.int64),
        step_done_event=event,
        step_done_event_pending=False,
    )

    logits = Qwen35Runtime.launch_prepared_batch(
        rt,
        prepared_sequences,
        prefill_slot,
    )

    expected_hidden_rows = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
    assert committed == [[3, 5]]
    assert stored == [([3, 5], cache, [3, 5])]
    assert torch.equal(prefill_slot.batch_idx[:2], torch.tensor([3, 5]))
    assert len(lm_head.calls) == 1
    assert torch.equal(lm_head.calls[0], expected_hidden_rows)
    assert torch.equal(logits, expected_hidden_rows + 100.0)
    assert torch.equal(prepared_sequences[0].state.last_hidden, expected_hidden_rows[0])
    assert torch.equal(prepared_sequences[1].state.last_hidden, expected_hidden_rows[1])
    assert event.records == 1
    assert prefill_slot.step_done_event_pending is True


class _FakeBuffer:
    def __init__(self, values: torch.Tensor) -> None:
        self.gpu = values.clone()
        self.cpu = values.clone()


def test_zero_decode_graph_padding_routes_padded_rows_to_reserved_batch():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    rt.max_batch_size = 4
    rt._padding_batch_idx = 5
    slot = SimpleNamespace(
        decode_token_ids=torch.arange(5, dtype=torch.long),
        meta=SimpleNamespace(
            batch_idx=_FakeBuffer(torch.arange(5, dtype=torch.int64)),
            input_pos=_FakeBuffer(torch.arange(5, dtype=torch.int32)),
            lora_slot_ids=_FakeBuffer(torch.arange(5, dtype=torch.int32)),
        ),
    )

    Qwen35Runtime._zero_decode_graph_padding(rt, slot, 2, 4)

    assert torch.equal(slot.decode_token_ids, torch.tensor([0, 1, 0, 0, 4]))
    assert torch.equal(slot.meta.batch_idx.gpu, torch.tensor([0, 1, 5, 5, 4]))
    assert torch.equal(slot.meta.batch_idx.cpu, torch.tensor([0, 1, 5, 5, 4]))
    assert torch.equal(
        slot.meta.input_pos.gpu,
        torch.tensor([0, 1, 0, 0, 4], dtype=torch.int32),
    )
    assert torch.equal(
        slot.meta.input_pos.cpu,
        torch.tensor([0, 1, 0, 0, 4], dtype=torch.int32),
    )
    assert torch.equal(
        slot.meta.lora_slot_ids.gpu,
        torch.tensor([0, 1, 0, 0, 4], dtype=torch.int32),
    )
    assert torch.equal(
        slot.meta.lora_slot_ids.cpu,
        torch.tensor([0, 1, 0, 0, 4], dtype=torch.int32),
    )


def test_zero_decode_graph_capture_buffers_initializes_and_clears_gdn_state():
    cfg = SimpleNamespace(
        layer_types=["linear_attention"],
        num_hidden_layers=1,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=3,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=5,
    )
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    rt.architecture = SimpleNamespace(text_config=cfg)
    rt.dtype = torch.bfloat16
    rt._linear_state_pool = Qwen35LinearStatePool(
        config=cfg,
        max_batch_slots=3,
        device=torch.device("cpu"),
        replay_capacity=int(getattr(cfg, "linear_replay_capacity", 16)),
    )
    rt._decode_rope_deltas = torch.ones((3, 1), dtype=torch.long)
    slot = SimpleNamespace(
        decode_token_ids=torch.ones((3,), dtype=torch.long),
        meta=SimpleNamespace(
            batch_idx=_FakeBuffer(torch.ones((3,), dtype=torch.int64)),
            input_pos=_FakeBuffer(torch.ones((3,), dtype=torch.int32)),
            lora_slot_ids=_FakeBuffer(torch.ones((3,), dtype=torch.int32)),
        ),
        paged_kv_page_table=torch.ones((3, 4), dtype=torch.int32),
        paged_kv_seqlens_k=torch.ones((3,), dtype=torch.int32),
        slot_mapping=torch.ones((3, 1), dtype=torch.long),
        cache_position_ids=torch.ones((3, 1), dtype=torch.long),
        position_ids=torch.ones((4, 3, 1), dtype=torch.long),
        rope_deltas=torch.ones((3, 1), dtype=torch.long),
        sampled_ids=torch.ones((3,), dtype=torch.long),
        sampled_logprobs=torch.ones((3,), dtype=torch.float32),
        logits=torch.ones((3, 8), dtype=torch.bfloat16),
        hidden_last=torch.ones((3, 4), dtype=torch.bfloat16),
    )

    Qwen35Runtime._zero_decode_graph_capture_buffers(rt, slot)

    storage = rt._linear_state_pool.layers[0]
    assert storage.conv_states.shape == (3, 2 * 1 * 3 + 2 * 4, 5)
    assert storage.recurrent_states.shape == (3, 2, 3, 4)
    assert torch.count_nonzero(storage.conv_states) == 0
    assert torch.count_nonzero(storage.recurrent_states) == 0
    assert torch.count_nonzero(rt._decode_rope_deltas) == 0
    assert torch.count_nonzero(slot.logits) == 0
    assert torch.count_nonzero(slot.paged_kv_page_table) == 0
    assert torch.count_nonzero(slot.slot_mapping) == 0


def test_linear_state_pool_captures_gqa_value_head_replay_ring_cpu():
    """The persistent ``Qwen35LinearStatePool`` must size its replay key ring by
    VALUE head, matching ``LinearAttentionLayer._ensure_replay_state``.

    Regression guard for the prefill-cache-capture (SpecRunner / continuous-
    batching) path: on the default GQA k16/v32 Qwen3.5-4B the per-layer ring is
    [*, cap, v_heads, D]; if the pool sized its ring by key_heads,
    ``_copy_rows_from_layer`` would try to copy a [*, cap, v, D] source into a
    [*, cap, k, D] pool row and raise before the runner could admit the
    sequence. Here v_heads (2) != k_heads (1) so a key-head ring would mismatch.
    """
    num_k_heads, num_v_heads, key_dim, value_dim = 1, 2, 3, 4
    cap = 8
    cfg = SimpleNamespace(
        layer_types=["linear_attention"],
        num_hidden_layers=1,
        linear_num_key_heads=num_k_heads,
        linear_num_value_heads=num_v_heads,
        linear_key_head_dim=key_dim,
        linear_value_head_dim=value_dim,
        linear_conv_kernel_dim=5,
        linear_replay_capacity=cap,
    )
    device = torch.device("cpu")

    # Build a hybrid cache and seed its single GDN layer's recurrent + ring state
    # (mirrors what a prefill does: update_recurrent_state -> _ensure_replay_state
    # allocates the value-head ring; we then fill the ring with non-zero rows).
    cache = Qwen35InferenceCache.__new__(Qwen35InferenceCache)
    layer = LinearAttentionLayer(cfg, replay_capacity=cfg.linear_replay_capacity)
    cache.layers = [layer]
    conv_dim = 2 * num_k_heads * key_dim + num_v_heads * value_dim
    layer.conv_states = torch.randn((1, conv_dim, 5), dtype=torch.bfloat16)
    layer.is_conv_states_initialized = True
    recurrent = torch.randn((1, num_v_heads, key_dim, value_dim))
    layer.update_recurrent_state(recurrent)
    layer.has_previous_state = True
    # Confirm the per-layer ring is value-head shaped.
    assert tuple(layer.replay_k.shape) == (1, cap, num_v_heads, key_dim)
    # Put a recognizable ring window in place.
    layer.replay_lengths.fill_(3)
    layer.replay_k.copy_(torch.randn_like(layer.replay_k.float()).to(layer.replay_k.dtype))
    layer.replay_u.copy_(torch.randn_like(layer.replay_u))
    layer.replay_g.copy_(torch.randn_like(layer.replay_g))
    pool = Qwen35LinearStatePool(
        config=cfg,
        max_batch_slots=4,
        device=device,
        replay_capacity=int(getattr(cfg, "linear_replay_capacity", 16)),
    )
    pool.initialize_from_config(cfg, dtype=torch.bfloat16)
    storage = pool.layers[0]
    # The pool ring must be value-head shaped (the bug allocated it key-head sized).
    assert tuple(storage.replay_k.shape) == (4, cap, num_v_heads, key_dim)

    # Capture slot 2 from the layer -- this is the call that raised pre-fix.
    pool.capture_from_cache(2, cache)

    # Captured ring rows match the layer's, at the captured slot.
    torch.testing.assert_close(storage.replay_k[2], layer.replay_k[0])
    torch.testing.assert_close(storage.replay_u[2], layer.replay_u[0])
    torch.testing.assert_close(storage.replay_g[2], layer.replay_g[0])
    assert int(storage.replay_lengths[2]) == 3
    torch.testing.assert_close(storage.recurrent_states[2], layer.recurrent_states[0])

    # Bind back: the layer must point at the pool's (value-head) ring.
    pool.bind_to_cache(cache)
    assert layer.replay_k.shape[2] == num_v_heads
    assert layer.replay_k is storage.replay_k


def test_linear_state_pool_fresh_prefill_capture_skips_unreachable_replay_payload():
    cfg = SimpleNamespace(
        layer_types=["linear_attention"],
        num_hidden_layers=1,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=3,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=5,
        linear_replay_capacity=8,
    )
    cache = Qwen35InferenceCache.__new__(Qwen35InferenceCache)
    layer = LinearAttentionLayer(cfg, replay_capacity=cfg.linear_replay_capacity)
    cache.layers = [layer]
    conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + (
        cfg.linear_num_value_heads * cfg.linear_value_head_dim
    )
    layer.conv_states = torch.randn((1, conv_dim, 5), dtype=torch.bfloat16)
    layer.is_conv_states_initialized = True
    layer.update_recurrent_state(
        torch.randn(
            (
                1,
                cfg.linear_num_value_heads,
                cfg.linear_key_head_dim,
                cfg.linear_value_head_dim,
            )
        )
    )
    layer.has_previous_state = True
    layer.replay_k.fill_(11)
    layer.replay_u.fill_(12)
    layer.replay_g.fill_(13)
    layer.replay_lengths.zero_()

    pool = Qwen35LinearStatePool(
        config=cfg,
        max_batch_slots=3,
        device=torch.device("cpu"),
        replay_capacity=int(getattr(cfg, "linear_replay_capacity", 16)),
    )
    pool.initialize_from_config(cfg, dtype=torch.bfloat16)
    storage = pool.layers[0]
    storage.replay_k.fill_(21)
    storage.replay_u.fill_(22)
    storage.replay_g.fill_(23)
    storage.replay_lengths.fill_(4)

    pool.capture_batch_from_cache(
        torch.tensor([2], dtype=torch.long),
        cache,
        batch_size=1,
        copy_replay_payload=False,
    )

    torch.testing.assert_close(storage.conv_states[2], layer.conv_states[0])
    torch.testing.assert_close(
        storage.recurrent_states[2],
        layer.recurrent_states[0],
    )
    torch.testing.assert_close(
        storage.replay_checkpoint_states[2],
        layer.replay_checkpoint_states[0],
    )
    assert torch.all(storage.replay_k[2] == 21)
    assert torch.all(storage.replay_u[2] == 22)
    assert torch.all(storage.replay_g[2] == 23)
    assert int(storage.replay_lengths[2]) == 0


def test_prepare_decode_position_ids_uses_per_row_rope_deltas():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    slot = SimpleNamespace(
        cache_position_ids=torch.tensor([[10], [20]], dtype=torch.long),
        rope_deltas=torch.tensor([[5], [7]], dtype=torch.long),
        position_ids=torch.empty((4, 2, 1), dtype=torch.long),
    )

    Qwen35Runtime._prepare_decode_position_ids(rt, slot, batch_size=2)

    assert torch.equal(slot.position_ids[0, :, 0], torch.tensor([10, 20]))
    assert torch.equal(slot.position_ids[1, :, 0], torch.tensor([15, 27]))
    assert torch.equal(slot.position_ids[2, :, 0], torch.tensor([15, 27]))
    assert torch.equal(slot.position_ids[3, :, 0], torch.tensor([15, 27]))


def test_packed_prefill_batch_builds_token_level_metadata():
    from kestrel.runtime.tokens import TextToken

    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    rt.device = torch.device("cpu")
    rt.dtype = torch.float32
    rt.page_table = PageTable(
        n_pages=16,
        page_size=1,
        max_batch_size=4,
        device="cpu",
    )
    first = rt.page_table.allocate()
    second = rt.page_table.allocate()
    rt.page_table.reserve(first, 4)
    rt.page_table.reserve(second, 4)
    rt.page_table.commit_block_table([first, second])

    prepared = [
        SimpleNamespace(
            state=SimpleNamespace(batch_idx=first),
            tokens_list=[TextToken(token_id=10), TextToken(token_id=11)],
        ),
        SimpleNamespace(
            state=SimpleNamespace(batch_idx=second),
            tokens_list=[TextToken(token_id=20)],
        ),
    ]

    prefill_slot = SimpleNamespace(
        batch_idx=torch.empty(4, dtype=torch.int64),
        scratch=None,
    )
    packed = Qwen35Runtime._build_packed_prefill_batch(
        rt,
        prepared,
        prefill_slot=prefill_slot,
        image_crops_list=[None, None],
        batch_indices=[first, second],
    )

    assert torch.equal(packed.input_ids, torch.tensor([[10, 11, 20]]))
    assert torch.equal(packed.cache_position_ids, torch.tensor([[0, 1, 0]]))
    assert torch.equal(packed.seq_idx, torch.tensor([[0, 0, 1]], dtype=torch.int32))
    assert torch.equal(packed.cu_seq_lens_q, torch.tensor([0, 2, 3], dtype=torch.int32))
    assert packed.max_length == 2
    assert torch.equal(packed.last_token_offsets, torch.tensor([1, 2]))
    assert packed.position_ids.shape == (3, 1, 3)
    assert torch.equal(packed.position_ids[0, 0], torch.tensor([0, 1, 0]))
    assert packed.paged_kv_page_table.shape[0] == 2
    assert packed.paged_kv_seqlens_k.tolist() == [2, 1]
    assert packed.mm_token_type_ids is None
    assert torch.equal(prefill_slot.batch_idx[:2], torch.tensor([first, second]))


def test_packed_prefill_multimodal_position_ids_match_qwen_mrope_layout():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    rt.architecture = SimpleNamespace(
        vision_config=SimpleNamespace(
            spatial_merge_size=2,
            num_position_embeddings=2304,
        )
    )
    out = torch.empty((3, 1, 9), dtype=torch.long).numpy()
    mm_types = torch.tensor(
        [0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int32
    ).numpy()
    image_grid_thw = torch.tensor([[1, 4, 6]], dtype=torch.long).numpy()

    delta = Qwen35Runtime._fill_multimodal_position_ids(
        rt,
        out,
        start=0,
        end=9,
        mm_token_type_ids=mm_types,
        image_grid_thw=image_grid_thw,
    )

    expected = torch.tensor(
        [
            [[0, 1, 1, 1, 1, 1, 1, 4, 5]],
            [[0, 1, 1, 1, 2, 2, 2, 4, 5]],
            [[0, 1, 2, 3, 1, 2, 3, 4, 5]],
        ],
        dtype=torch.long,
    ).numpy()
    assert (out == expected).all()
    assert delta == -3


def test_packed_prefill_batch_stages_image_metadata_in_slot_buffers():
    from kestrel.runtime.tokens import TextToken

    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    rt.device = torch.device("cpu")
    rt.dtype = torch.float32
    rt.architecture = SimpleNamespace(
        vision_config=SimpleNamespace(
            spatial_merge_size=2,
            num_position_embeddings=2304,
        )
    )
    rt.page_table = PageTable(
        n_pages=16,
        page_size=1,
        max_batch_size=4,
        device="cpu",
    )
    batch_idx = rt.page_table.allocate()
    rt.page_table.reserve(batch_idx, 8)
    rt.page_table.commit_block_table([batch_idx])
    prepared = [
        SimpleNamespace(
            state=SimpleNamespace(batch_idx=batch_idx),
            tokens_list=[
                TextToken(token_id=10),
                TextToken(token_id=IMAGE_PAD_ID),
                TextToken(token_id=IMAGE_PAD_ID),
                TextToken(token_id=IMAGE_PAD_ID),
                TextToken(token_id=IMAGE_PAD_ID),
                TextToken(token_id=11),
            ],
        )
    ]
    crops = QwenImageInputs(
        pixel_values=torch.arange(48, dtype=torch.float32).view(16, 3),
        image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.long),
        num_image_tokens=4,
    )
    prefill_slot = SimpleNamespace(
        batch_idx=torch.empty(4, dtype=torch.int64),
        scratch=None,
    )

    packed = Qwen35Runtime._build_packed_prefill_batch(
        rt,
        prepared,
        prefill_slot=prefill_slot,
        image_crops_list=[crops],
        batch_indices=[batch_idx],
    )

    assert torch.equal(
        packed.mm_token_type_ids,
        torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.int32),
    )
    assert torch.equal(packed.image_grid_thw, crops.image_grid_thw)
    assert torch.equal(packed.pixel_values, crops.pixel_values)
    assert prefill_slot.scratch.pixel_values is not None
    assert torch.equal(
        prefill_slot.scratch.pixel_values.cpu[: crops.pixel_values.shape[0]],
        crops.pixel_values,
    )
    assert torch.equal(
        packed.position_ids,
        torch.tensor(
            [
                [[0, 1, 1, 1, 1, 3]],
                [[0, 1, 1, 2, 2, 3]],
                [[0, 1, 2, 1, 2, 3]],
            ],
            dtype=torch.long,
        ),
    )
    assert torch.equal(packed.rope_deltas, torch.tensor([[-2]], dtype=torch.long))
    assert packed.vision_bilinear_indices is not None
    assert packed.vision_bilinear_weights is not None
    assert packed.vision_position_ids is not None
    assert packed.vision_cu_seqlens is not None
    assert packed.vision_bilinear_indices.shape == (4, 16)
    assert packed.vision_bilinear_weights.shape == (4, 16)
    assert packed.vision_position_ids.shape == (16, 2)
    assert torch.equal(
        packed.vision_cu_seqlens,
        torch.tensor([0, 16], dtype=torch.int32),
    )


def test_qwen_linear_state_pool_binds_decode_cache_to_persistent_rows():
    cfg = SimpleNamespace(
        layer_types=["linear_attention", "full_attention"],
        num_hidden_layers=2,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
    )
    page_table = PageTable(
        n_pages=4,
        page_size=1,
        max_batch_size=3,
        device="cpu",
    )
    pool = KVMemoryPool(device=torch.device("cpu"))
    shared_layers = allocate_qwen35_paged_kv(
        config=cfg,
        page_table=page_table,
        pool=pool,
        dtype=torch.float32,
    )
    allocated_bytes = pool.allocated_bytes

    first = _make_qwen_cache(
        config=cfg,
        page_table=page_table,
        pool=pool,
        dtype=torch.float32,
        shared_paged_layers=shared_layers,
    )
    second = _make_qwen_cache(
        config=cfg,
        page_table=page_table,
        pool=pool,
        dtype=torch.float32,
        shared_paged_layers=shared_layers,
    )
    decode_cache = _make_qwen_cache(
        config=cfg,
        page_table=page_table,
        pool=pool,
        dtype=torch.float32,
        shared_paged_layers=shared_layers,
    )

    assert pool.allocated_bytes == allocated_bytes
    assert first.layers[1] is shared_layers.producer(1)
    assert second.layers[1] is shared_layers.producer(1)
    assert decode_cache.layers[1] is shared_layers.producer(1)

    first.layers[0].update_conv_state(torch.full((1, 2, 3), 1.0))
    first.layers[0].update_recurrent_state(torch.full((1, 2, 2), 11.0))
    first.advance_to(6)
    second.layers[0].update_conv_state(torch.full((1, 2, 3), 2.0))
    second.layers[0].update_recurrent_state(torch.full((1, 2, 2), 12.0))
    second.advance_to(7)

    state_pool = Qwen35LinearStatePool(
        config=cfg,
        max_batch_slots=3,
        device=torch.device("cpu"),
        replay_capacity=int(getattr(cfg, "linear_replay_capacity", 16)),
    )
    state_pool.capture_from_cache(1, first)
    state_pool.capture_from_cache(2, second)

    state_pool.bind_to_cache(decode_cache)
    storage = state_pool.layers[0]
    assert torch.equal(
        decode_cache.layers[0].conv_states[[1, 2], 0, 0],
        torch.tensor([1.0, 2.0]),
    )
    assert torch.equal(
        decode_cache.layers[0].recurrent_states[[1, 2], 0, 0],
        torch.tensor([11.0, 12.0]),
    )

    decode_cache.layers[0].conv_states[1].add_(10.0)
    decode_cache.layers[0].recurrent_states[2].add_(20.0)

    assert torch.equal(
        storage.conv_states[[1, 2]],
        torch.tensor(
            [
                [[11.0, 11.0, 11.0], [11.0, 11.0, 11.0]],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            ]
        ),
    )
    assert torch.equal(
        storage.recurrent_states[[1, 2]],
        torch.tensor(
            [
                [[11.0, 11.0], [11.0, 11.0]],
                [[32.0, 32.0], [32.0, 32.0]],
            ]
        ),
    )


def test_qwen_linear_state_pool_captures_batched_gdn_rows():
    cfg = SimpleNamespace(
        layer_types=["linear_attention"],
        num_hidden_layers=1,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
    )
    page_table = PageTable(
        n_pages=4,
        page_size=1,
        max_batch_size=3,
        device="cpu",
    )
    cache = _make_qwen_cache(
        config=cfg,
        page_table=page_table,
        pool=KVMemoryPool(device=torch.device("cpu")),
        dtype=torch.float32,
    )
    cache.layers[0].update_conv_state(
        torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
    )
    cache.layers[0].update_recurrent_state(
        torch.tensor(
            [
                [[13.0, 14.0], [15.0, 16.0]],
                [[17.0, 18.0], [19.0, 20.0]],
            ]
        )
    )
    state_pool = Qwen35LinearStatePool(
        config=cfg,
        max_batch_slots=3,
        device=torch.device("cpu"),
        replay_capacity=int(getattr(cfg, "linear_replay_capacity", 16)),
    )

    state_pool.capture_batch_from_cache(
        torch.tensor([2, 1], dtype=torch.int64),
        cache,
        batch_size=2,
    )

    storage = state_pool.layers[0]
    assert torch.equal(storage.conv_states[2], cache.layers[0].conv_states[0])
    assert torch.equal(storage.conv_states[1], cache.layers[0].conv_states[1])
    assert torch.equal(
        storage.recurrent_states[2],
        cache.layers[0].recurrent_states[0],
    )
    assert torch.equal(
        storage.recurrent_states[1],
        cache.layers[0].recurrent_states[1],
    )


def test_qwen_linear_state_pool_initializes_from_config_for_graph_capture():
    cfg = SimpleNamespace(
        layer_types=["linear_attention", "full_attention"],
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=3,
        linear_key_head_dim=4,
        linear_value_head_dim=5,
        linear_conv_kernel_dim=6,
    )
    page_table = PageTable(
        n_pages=4,
        page_size=1,
        max_batch_size=3,
        device="cpu",
    )
    pool = KVMemoryPool(device=torch.device("cpu"))
    shared_layers = allocate_qwen35_paged_kv(
        config=cfg,
        page_table=page_table,
        pool=pool,
        dtype=torch.bfloat16,
    )
    decode_cache = _make_qwen_cache(
        config=cfg,
        page_table=page_table,
        pool=pool,
        dtype=torch.bfloat16,
        shared_paged_layers=shared_layers,
    )
    state_pool = Qwen35LinearStatePool(
        config=cfg,
        max_batch_slots=3,
        device=torch.device("cpu"),
        replay_capacity=int(getattr(cfg, "linear_replay_capacity", 16)),
    )

    state_pool.initialize_from_config(cfg, dtype=torch.bfloat16)
    state_pool.bind_to_cache(decode_cache)

    layer = decode_cache.layers[0]
    assert layer.conv_states.shape == (3, 2 * 2 * 4 + 3 * 5, 6)
    assert layer.conv_states.dtype == torch.bfloat16
    assert torch.count_nonzero(layer.conv_states) == 0
    assert layer.recurrent_states.shape == (3, 3, 4, 5)
    assert layer.recurrent_states.dtype == torch.float32
    assert torch.count_nonzero(layer.recurrent_states) == 0


def test_batch_index_allocation_gates_capacity():
    rt = Qwen35Runtime.__new__(Qwen35Runtime)
    rt.max_batch_size = 2
    rt.max_seq_length = 4096
    rt.active_sequences = {}
    rt._caches = {}
    # __init__ is bypassed here; mirror its ``self._spec_runner = None`` default
    # so the admission gate (_admission_max_seq_len, reached via can_reserve)
    # sees "no spec runner" instead of raising AttributeError.
    rt._spec_runner = None
    rt.page_table = PageTable(
        n_pages=16,
        page_size=1,
        max_batch_size=rt.max_batch_size + 1,
        device="cpu",
    )

    assert rt.prefill_budget() == (15, 2)
    first = rt.page_table.allocate()
    second = rt.page_table.allocate()
    assert {first, second} == {1, 2}
    assert rt.prefill_budget()[1] == 0
    assert not rt.can_reserve(1)
    with pytest.raises(IndexError):
        rt.page_table.allocate()

    rt._caches[first] = object()
    rt._release_batch_idx(first)
    assert first in rt.page_table.free_batch_idx
    assert first not in rt._caches
    assert rt.prefill_budget()[1] == 1

    rt._release_batch_idx(first)
    assert rt.page_table.free_batch_idx.count(first) == 1


def test_gated_delta_net_keeps_ssm_params_in_configured_dtype():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        config = _text_config(
            hidden_size=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
        )
        module = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    finally:
        torch.set_default_dtype(old_dtype)

    assert module.A_log.dtype == torch.float32
    assert module.dt_bias.dtype == torch.float32
    assert module.in_proj.weight.dtype == torch.bfloat16


def test_gated_delta_net_uses_kestrel_gdn_kernels():
    from kestrel_kernels import get_runtime

    config = _text_config(
        hidden_size=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
    )
    module = Qwen3_5GatedDeltaNet(config, layer_idx=0)

    assert (
        module.causal_conv1d_update_indexed
        is get_runtime().gated_delta.causal_conv1d_update_indexed
    )
    assert (
        module.packed_prefill_prepare
        is get_runtime().gated_delta.packed_prefill_prepare
    )
    assert (
        module.packed_recurrent_decode_replay_indexed
        is get_runtime().gated_delta.packed_recurrent_gated_delta_rule_decode_replay_indexed
    )
    assert (
        module.packed_recurrent_prefill
        is get_runtime().gated_delta.packed_recurrent_gated_delta_rule_prefill
    )
    assert module.supports_packed_gdn is get_runtime().gated_delta.supports_packed_gdn
    assert module.norm.gated_rmsnorm is get_runtime().gated_delta.gated_rmsnorm


def test_gated_delta_net_packed_prefill_derives_seq_idx_from_cu_seqlens():
    torch.manual_seed(0)
    config = _text_config(
        hidden_size=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
        num_hidden_layers=1,
    )
    module = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    module.norm = Qwen3_5RMSNormGated(
        config.linear_value_head_dim,
        eps=config.rms_norm_eps,
    )

    hidden_a = torch.randn(1, 2, config.hidden_size)
    hidden_b = torch.randn(1, 3, config.hidden_size)
    hidden_packed = torch.cat([hidden_a, hidden_b], dim=1)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    def make_cache() -> Qwen35InferenceCache:
        return _make_qwen_cache(
            config=config,
            page_table=PageTable(
                n_pages=1,
                page_size=1,
                max_batch_size=1,
                device="cpu",
            ),
            pool=KVMemoryPool(device=torch.device("cpu")),
            dtype=torch.float32,
        )

    packed_cache = make_cache()
    with torch.inference_mode():
        packed_out = module(
            hidden_packed,
            cache_params=packed_cache,
            cu_seq_lens_q=cu_seqlens,
        )

        cache_a = make_cache()
        out_a = module(hidden_a, cache_params=cache_a)
        cache_b = make_cache()
        out_b = module(hidden_b, cache_params=cache_b)

    assert torch.allclose(packed_out[:, :2], out_a, atol=1e-5, rtol=1e-5)
    assert torch.allclose(packed_out[:, 2:], out_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        packed_cache.layers[0].conv_states,
        torch.cat([cache_a.layers[0].conv_states, cache_b.layers[0].conv_states], dim=0),
        atol=1e-5,
        rtol=1e-5,
    )
    assert packed_cache.layers[0].conv_kernel_size == config.linear_conv_kernel_dim
    assert packed_cache.layers[0].max_batch_size == 2
    assert packed_cache.layers[0].dtype == hidden_packed.dtype
    assert packed_cache.layers[0].device == hidden_packed.device
    assert torch.allclose(
        packed_cache.layers[0].recurrent_states,
        torch.cat(
            [
                cache_a.layers[0].recurrent_states,
                cache_b.layers[0].recurrent_states,
            ],
            dim=0,
        ),
        atol=1e-5,
        rtol=1e-5,
    )


def test_gated_delta_net_packed_prefill_unequal_head_dims_matches_serial_sequences():
    torch.manual_seed(0)
    config = _text_config(
        hidden_size=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
        num_hidden_layers=1,
    )
    module = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    module.norm = Qwen3_5RMSNormGated(
        config.linear_value_head_dim,
        eps=config.rms_norm_eps,
    )

    hidden_a = torch.randn(1, 2, config.hidden_size)
    hidden_b = torch.randn(1, 3, config.hidden_size)
    hidden_packed = torch.cat([hidden_a, hidden_b], dim=1)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    def make_cache() -> Qwen35InferenceCache:
        return _make_qwen_cache(
            config=config,
            page_table=PageTable(
                n_pages=1,
                page_size=1,
                max_batch_size=1,
                device="cpu",
            ),
            pool=KVMemoryPool(device=torch.device("cpu")),
            dtype=torch.float32,
        )

    packed_cache = make_cache()
    with torch.inference_mode():
        packed_out = module(
            hidden_packed,
            cache_params=packed_cache,
            cu_seq_lens_q=cu_seqlens,
            seq_idx=torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
        )

        cache_a = make_cache()
        out_a = module(hidden_a, cache_params=cache_a)
        cache_b = make_cache()
        out_b = module(hidden_b, cache_params=cache_b)

    assert torch.allclose(packed_out[:, :2], out_a, atol=1e-5, rtol=1e-5)
    assert torch.allclose(packed_out[:, 2:], out_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        packed_cache.layers[0].conv_states,
        torch.cat([cache_a.layers[0].conv_states, cache_b.layers[0].conv_states], dim=0),
        atol=1e-5,
        rtol=1e-5,
    )
    assert packed_cache.layers[0].conv_kernel_size == config.linear_conv_kernel_dim
    assert packed_cache.layers[0].max_batch_size == 2
    assert packed_cache.layers[0].dtype == hidden_packed.dtype
    assert packed_cache.layers[0].device == hidden_packed.device
    assert torch.allclose(
        packed_cache.layers[0].recurrent_states,
        torch.cat(
            [
                cache_a.layers[0].recurrent_states,
                cache_b.layers[0].recurrent_states,
            ],
            dim=0,
        ),
        atol=1e-5,
        rtol=1e-5,
    )


def test_gated_delta_net_batched_prefill_matches_serial_sequences():
    torch.manual_seed(0)
    config = _text_config(
        hidden_size=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
        num_hidden_layers=1,
    )
    module = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    module.norm = Qwen3_5RMSNormGated(
        config.linear_value_head_dim,
        eps=config.rms_norm_eps,
    )

    hidden = torch.randn(2, 3, config.hidden_size)

    def make_cache(max_batch_size: int) -> Qwen35InferenceCache:
        return _make_qwen_cache(
            config=config,
            page_table=PageTable(
                n_pages=max_batch_size,
                page_size=1,
                max_batch_size=max_batch_size,
                device="cpu",
            ),
            pool=KVMemoryPool(device=torch.device("cpu")),
            dtype=torch.float32,
        )

    batched_cache = make_cache(2)
    with torch.inference_mode():
        batched_out = module(hidden, cache_params=batched_cache)

        cache_a = make_cache(1)
        out_a = module(hidden[:1], cache_params=cache_a)
        cache_b = make_cache(1)
        out_b = module(hidden[1:], cache_params=cache_b)

    assert torch.allclose(batched_out[:1], out_a, atol=1e-5, rtol=1e-5)
    assert torch.allclose(batched_out[1:], out_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        batched_cache.layers[0].conv_states,
        torch.cat([cache_a.layers[0].conv_states, cache_b.layers[0].conv_states], dim=0),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        batched_cache.layers[0].recurrent_states,
        torch.cat(
            [
                cache_a.layers[0].recurrent_states,
                cache_b.layers[0].recurrent_states,
            ],
            dim=0,
        ),
        atol=1e-5,
        rtol=1e-5,
    )


def test_gated_delta_net_grouped_head_decode_matches_full_sequence():
    torch.manual_seed(0)
    config = _text_config(
        hidden_size=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
        num_hidden_layers=1,
    )
    module = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    reference = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    reference.load_state_dict(module.state_dict())

    def make_cache() -> Qwen35InferenceCache:
        return _make_qwen_cache(
            config=config,
            page_table=PageTable(
                n_pages=1,
                page_size=1,
                max_batch_size=1,
                device="cpu",
            ),
            pool=KVMemoryPool(device=torch.device("cpu")),
            dtype=torch.float32,
        )

    hidden_prefix = torch.randn(1, 2, config.hidden_size)
    hidden_decode = torch.randn(1, 1, config.hidden_size)

    cache = make_cache()
    with torch.inference_mode():
        module(hidden_prefix, cache_params=cache)
        decoded = module(
            hidden_decode,
            cache_params=cache,
            gdn_state_indices=torch.tensor([0], dtype=torch.long),
        )
        full = reference(torch.cat([hidden_prefix, hidden_decode], dim=1))

    assert torch.allclose(decoded, full[:, -1:], atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gated_delta_net_replay_decode_uses_replay_state():
    # The ReplaySSM single-token decode ring-append path runs only on the native
    # CuTe GDN kernels (gated by ``supports_native_packed_gdn``, which requires
    # ``hidden_states.is_cuda``); on CPU the eager forward falls through without
    # populating the ring. Production decode is always on CUDA, so exercise the
    # ring path on CUDA -- matching how the runtime actually decodes.
    device = torch.device("cuda")
    torch.manual_seed(0)
    config = _text_config(
        hidden_size=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
        num_hidden_layers=1,
    )
    module = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval().to(device)
    reference = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval().to(device)
    reference.load_state_dict(module.state_dict())

    def make_cache() -> Qwen35InferenceCache:
        return _make_qwen_cache(
            config=config,
            page_table=PageTable(
                n_pages=1,
                page_size=1,
                max_batch_size=1,
                device=str(device),
            ),
            pool=KVMemoryPool(device=device),
            dtype=torch.float32,
            replay_capacity=8,
        )

    hidden_prefix = torch.randn(1, 2, config.hidden_size, device=device)
    hidden_decode = torch.randn(1, 2, config.hidden_size, device=device)
    idx = torch.tensor([0], dtype=torch.long, device=device)
    replay_cache = make_cache()
    reference_cache = make_cache()

    with torch.inference_mode():
        # ReplaySSM is the only single-token decode path now, so validate it
        # against the recurrence directly: a token's replay-decode output must
        # equal its output when prefix+token are prefilled in one shot.
        module(hidden_prefix, cache_params=replay_cache)
        first = module(hidden_decode[:, :1], cache_params=replay_cache, gdn_state_indices=idx)
        module(hidden_decode[:, 1:], cache_params=replay_cache, gdn_state_indices=idx)

        reference_out = reference(
            torch.cat([hidden_prefix, hidden_decode[:, :1]], dim=1),
            cache_params=reference_cache,
        )
        first_expected = reference_out[:, -1:]

    replay_lengths = replay_cache.layers[0].replay_lengths
    assert replay_lengths is not None
    torch.testing.assert_close(
        replay_lengths, torch.tensor([2], dtype=torch.int32, device=device)
    )
    torch.testing.assert_close(first, first_expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gated_delta_net_chunk_decode_after_replay_matches_full_prefill():
    """A seq_len>1 chunk continuation after single-token ReplaySSM decode must
    start from the materialized replay state, not a stale ``recurrent_states``.

    Transition run: prefill, then per-token replay decode (advances the ring
    buffer but not ``recurrent_states``), then a chunk continuation. Reference
    run: prefill the prefix+decode tokens in one shot (so ``replay_lengths``==0
    and the chunk reads ``recurrent_states`` directly), then the same chunk. The
    GDN recurrence is linear, so both reach the same pre-chunk state and the
    chunk outputs must match. Without the materialize this reads the stale
    post-prefix state and the outputs diverge grossly.

    Runs on CUDA: the per-token replay-decode ring append only executes on the
    native CuTe GDN kernels (gated by ``supports_native_packed_gdn`` ->
    ``is_cuda``); on CPU the ring stays empty and there is nothing to
    materialize. Production decode is CUDA-only, so this mirrors the real path.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    config = _text_config(
        hidden_size=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
        num_hidden_layers=1,
    )
    module = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval().to(device)

    def make_cache() -> Qwen35InferenceCache:
        return _make_qwen_cache(
            config=config,
            page_table=PageTable(
                n_pages=1, page_size=1, max_batch_size=1, device=str(device)
            ),
            pool=KVMemoryPool(device=device),
            dtype=torch.float32,
            replay_capacity=8,
        )

    hidden_prefix = torch.randn(1, 2, config.hidden_size, device=device)
    # Several sizable single-token decodes so the buffered state differs clearly
    # from the post-prefix state (otherwise stale vs. correct is within fp noise).
    hidden_decode = torch.randn(1, 8, config.hidden_size, device=device) * 3.0
    # seq_len>1 continuation
    hidden_chunk = torch.randn(1, 2, config.hidden_size, device=device)
    idx = torch.tensor([0], dtype=torch.long, device=device)

    state_cache = make_cache()
    chunk_cache = make_cache()
    reference_cache = make_cache()
    with torch.inference_mode():
        # Reference: prefill prefix+decode tokens in one shot -> true pre-chunk state.
        module(torch.cat([hidden_prefix, hidden_decode], dim=1), cache_params=reference_cache)
        assert int(reference_cache.layers[0].replay_lengths[0]) == 0
        reference_state = reference_cache.layers[0].recurrent_states.clone()
        out_reference = module(
            hidden_chunk, cache_params=reference_cache, gdn_state_indices=idx
        )

        # (a) Direct: prefill + per-token replay decode leaves recurrent_states
        # stale; the materialize must reconstruct the true state. Comparing the
        # full state tensor is the most sensitive signal.
        module(hidden_prefix, cache_params=state_cache)
        for t in range(hidden_decode.shape[1]):
            module(
                hidden_decode[:, t : t + 1],
                cache_params=state_cache,
                gdn_state_indices=idx,
            )
        assert int(state_cache.layers[0].replay_lengths[0]) > 0
        stale_state = state_cache.layers[0].recurrent_states.clone()
        state_cache.layers[0].materialize_recurrent_from_replay()
        materialized_state = state_cache.layers[0].recurrent_states
        # Sanity: the stale state really is far from the true state (so the test
        # has teeth), and the materialized state matches it.
        assert (stale_state - reference_state).abs().max() > 1e-2
        # The replay key ring is stored in bf16, so the materialize reconstructs
        # the fp32 reference state to the bf16 noise floor (~1.2e-4 over these 8
        # scaled decode steps) rather than exactly. The stale state diverges by
        # >1e-2 above, so this tolerance still cleanly separates correct from
        # stale.
        torch.testing.assert_close(materialized_state, reference_state, atol=2e-3, rtol=2e-3)

        # (b) Wiring: the seq_len>1 chunk path must trigger the materialize, so a
        # chunk continuation after replay decode matches the one-shot reference.
        module(hidden_prefix, cache_params=chunk_cache)
        for t in range(hidden_decode.shape[1]):
            module(
                hidden_decode[:, t : t + 1],
                cache_params=chunk_cache,
                gdn_state_indices=idx,
            )
        out_transition = module(
            hidden_chunk, cache_params=chunk_cache, gdn_state_indices=idx
        )

    # The chunk continuation reads the bf16-ring materialized state (see above),
    # so it tracks the one-shot fp32 reference to the same bf16 floor.
    torch.testing.assert_close(out_transition, out_reference, atol=2e-3, rtol=2e-3)


def test_gated_delta_net_chunked_decode_selects_indexed_state_rows():
    torch.manual_seed(0)
    config = _text_config(
        hidden_size=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention"],
        num_hidden_layers=1,
    )
    module = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    reference = Qwen3_5GatedDeltaNet(config, layer_idx=0).eval()
    reference.load_state_dict(module.state_dict())

    def make_cache(max_batch_size: int) -> Qwen35InferenceCache:
        return _make_qwen_cache(
            config=config,
            page_table=PageTable(
                n_pages=max_batch_size,
                page_size=1,
                max_batch_size=max_batch_size,
                device="cpu",
            ),
            pool=KVMemoryPool(device=torch.device("cpu")),
            dtype=torch.float32,
        )

    state_indices = torch.tensor([0, 2], dtype=torch.long)
    hidden_prefix = torch.randn(3, 2, config.hidden_size)
    hidden_chunk = torch.randn(2, 2, config.hidden_size)
    persistent_cache = make_cache(3)
    compact_cache = make_cache(2)

    with torch.inference_mode():
        module(hidden_prefix, cache_params=persistent_cache)
        reference(
            hidden_prefix.index_select(0, state_indices),
            cache_params=compact_cache,
        )
        persistent_layer = persistent_cache.layers[0]
        untouched_conv = persistent_layer.conv_states[1].clone()
        untouched_recurrent = persistent_layer.recurrent_states[1].clone()

        decoded = module(
            hidden_chunk,
            cache_params=persistent_cache,
            gdn_state_indices=state_indices,
        )
        expected = reference(hidden_chunk, cache_params=compact_cache)

    torch.testing.assert_close(decoded, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        persistent_cache.layers[0].conv_states.index_select(0, state_indices),
        compact_cache.layers[0].conv_states,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        persistent_cache.layers[0].recurrent_states.index_select(0, state_indices),
        compact_cache.layers[0].recurrent_states,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        persistent_cache.layers[0].conv_states[1],
        untouched_conv,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        persistent_cache.layers[0].recurrent_states[1],
        untouched_recurrent,
        atol=0.0,
        rtol=0.0,
    )


def test_query_skill_defaults_to_non_reasoning():
    skill = Qwen35QuerySkill()
    built = skill.build_request(
        image=None,
        prompt={"question": "What is 2+2?"},
        settings={"max_tokens": 8},
    )
    assert built.request_context.reasoning is False
    assert built.temperature == 0.0
    assert built.top_p == 1.0

    built = skill.build_request(
        image=None,
        prompt={"question": "Warmup prompt.", "reasoning": False},
        settings={"max_tokens": 1, "temperature": 0.2, "top_p": 0.9},
    )
    assert built.temperature == 0.0
    assert built.top_p == 1.0

    built = skill.build_request(
        image=None,
        prompt={"question": "What is 2+2?"},
        settings={"max_tokens": 8, "temperature": 0.7, "top_p": 0.95},
    )
    assert built.temperature == 0.7
    assert built.top_p == 0.95

    built = skill.build_request(
        image=None,
        prompt={"question": "What is 2+2?", "reasoning": True},
        settings={"max_tokens": 8},
    )
    assert built.request_context.reasoning is True

    built = skill.build_request(
        image=None,
        prompt={"question": "What is 2+2?"},
        settings={"max_tokens": 8, "reasoning": True},
    )
    assert built.request_context.reasoning is False


def _spec_budget(*, kv_cache_pages, page_size, max_batch_size, max_seq_length, flush_cap=32):
    """Call ``Qwen35Runtime._default_spec_max_seq_len`` on a minimal stub.

    The method only reads these four ints, so we avoid building a full runtime
    (model + GPU) and exercise just the reservation-budget arithmetic.
    """
    stub = SimpleNamespace(
        _kv_cache_pages=kv_cache_pages,
        page_size=page_size,
        max_batch_size=max_batch_size,
        max_seq_length=max_seq_length,
    )
    return Qwen35Runtime._default_spec_max_seq_len(stub, flush_cap)


def test_default_spec_max_seq_len_fits_serving_kv_budget():
    """Default spec reservation must derive from the KV page budget, not ctx.

    Guards regression (runtime.py): ``SpecRunner`` reserves ``max_seq_len`` KV
    pages for each of ``max_batch_size`` fixed rows up front. Defaulting that to
    the model context (``max_position_embeddings``) reserves
    ``max_batch_size * ctx`` pages, which on large-context models exceeds the
    serving KV pool and makes ``page_table.reserve`` raise / starve the pool even
    for short prompts. The default must instead fit the per-row share of the
    pool.
    """
    page_size = 1
    # Long model context, comparatively small KV pool: model max would massively
    # over-reserve (8 * 262144 >> 4096 pages). The derived budget must keep the
    # total within the pool.
    kv_pages = 4096
    batch = 8
    ctx = 262144
    budget = _spec_budget(
        kv_cache_pages=kv_pages, page_size=page_size,
        max_batch_size=batch, max_seq_length=ctx,
    )
    # Crucially NOT the model context (that is the over-reservation bug).
    assert budget < ctx
    # B rows each reserving `budget` tokens must fit the page pool.
    pages_per_row = -(-budget // page_size)  # ceil
    assert batch * pages_per_row <= kv_pages, (
        f"budget={budget} -> {batch * pages_per_row} pages > pool {kv_pages}"
    )
    # And it should use most of the available per-row share. The budget reserves
    # one extra transient-prefill slot, so the share is taken over ``batch + 1``
    # (== (4096 - 2) // 9 == 454 tokens here), not ``batch``.
    assert budget >= (kv_pages - 2) // (batch + 1)

    # When the pool is huge relative to ctx, clamp to the model context (never
    # reserve beyond what a sequence could ever use).
    big = _spec_budget(
        kv_cache_pages=10_000_000, page_size=1, max_batch_size=4,
        max_seq_length=8192,
    )
    assert big == 8192

    # Honor page_size > 1 in the per-row token budget.
    paged = _spec_budget(
        kv_cache_pages=1000, page_size=16, max_batch_size=4, max_seq_length=10**9,
    )
    # Share over batch+1: (1000-2)//5 = 199 pages/row * 16 = 3184 tokens/row;
    # 4*199=796 <= 1000 (with room for the transient prefill slot).
    assert paged == 199 * 16
    assert 4 * (paged // 16) <= 1000

    # A tiny pool still yields at least one flush window (so a row can hold a
    # block), even if that technically over-subscribes -- the real shortage then
    # surfaces from page_table.reserve, not a silently-broken zero budget.
    tiny = _spec_budget(
        kv_cache_pages=4, page_size=1, max_batch_size=8, max_seq_length=10**9,
        flush_cap=64,
    )
    assert tiny == 64
