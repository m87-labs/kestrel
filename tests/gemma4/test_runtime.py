"""Gemma 4 runtime smoke and registration tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import kestrel.models.gemma4  # noqa: F401
from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.kv_cache import KVMemoryPool, LayeredPagedKV, PagedKVLayerSpec
from kestrel.models import get_spec, known_models
from kestrel.ops import attention as attention_ops
from kestrel.runtime import ExecutionShape, TextToken
from kestrel.models.gemma4 import model as gemma_model
from kestrel.models.gemma4.runtime import Gemma4Runtime
from kestrel.runtime.staging import BatchedTensorStager
from kestrel.models.gemma4.config import (
    Gemma4TextConfig,
    Gemma4VisionConfig,
    RopeSpec,
)
from kestrel.models.gemma4.image import preprocess_image
from kestrel.models.gemma4.model import Gemma4TextModel
from kestrel.models.gemma4.paged_cache import kv_source_layers, paged_kv_layout
from kestrel.models.gemma4.skills import build_skill_registry


_MODEL_ID = "google/gemma-4-E2B-it"


def _text_config(**overrides) -> Gemma4TextConfig:
    layer_types = tuple(overrides.pop("layer_types", ("sliding_attention",)))
    values = {
        "vocab_size": 16,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": len(layer_types),
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "rope": {
            "sliding_attention": RopeSpec("default", 10_000.0),
            "full_attention": RopeSpec("proportional", 1_000_000.0, 0.25),
        },
        "sliding_window": 512,
        "layer_types": layer_types,
        "final_logit_softcapping": 30.0,
        "vocab_size_per_layer_input": 0,
        "hidden_size_per_layer_input": 0,
        "num_global_key_value_heads": 1,
        "global_head_dim": 4,
        "attention_k_eq_v": False,
        "num_kv_shared_layers": 0,
        "use_double_wide_mlp": False,
    }
    values.update(overrides)
    return Gemma4TextConfig(**values)


def _vision_config(**overrides) -> Gemma4VisionConfig:
    values = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "rope": RopeSpec("default", 100.0),
        "pooling_kernel_size": 3,
        "patch_size": 14,
        "position_embedding_size": 16,
        "use_clipped_linears": False,
        "standardize": False,
    }
    values.update(overrides)
    return Gemma4VisionConfig(**values)


class _FakePageTable:
    def __init__(self, free_batch_idx: list[int], pages_available: int = 4096) -> None:
        self.free_batch_idx = free_batch_idx
        self.pages_available = pages_available

    def erase(self, batch_idx: int, cached_page_count: int = 0) -> None:
        del cached_page_count
        if batch_idx not in self.free_batch_idx:
            self.free_batch_idx.append(batch_idx)


def test_embedding_scales_live_with_their_generated_weight_sources():
    model = Gemma4TextModel(
        _text_config(
            vocab_size_per_layer_input=8,
            hidden_size_per_layer_input=2,
        )
    )

    assert model.embed_tokens.embed_scale.item() == pytest.approx(2.0)
    assert model.embed_tokens_per_layer.embed_scale.item() == pytest.approx(2**0.5)
    assert "embed_tokens.embed_scale" not in model.state_dict()
    assert "embed_tokens_per_layer.embed_scale" not in model.state_dict()


def test_packed_text_prefill_matches_individual_rows():
    torch.manual_seed(11)
    model = Gemma4TextModel(
        _text_config(layer_types=["sliding_attention", "full_attention"])
    )
    packed = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
    position_ids = torch.tensor([[0, 1, 0, 1, 2, 3]])

    actual = model(
        input_ids=packed,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
    )
    short = model(input_ids=packed[:, :2])
    long = model(input_ids=packed[:, 2:])

    torch.testing.assert_close(actual[0, :2], short[0])
    torch.testing.assert_close(actual[0, 2:], long[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vision_position_embeddings_match_one_hot_reference(dtype):
    cfg = _vision_config(
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

    actual = embedder(
        torch.zeros(
            (*pixel_position_ids.shape[:-1], embedder.input_proj.in_features),
            dtype=dtype,
        ),
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
    assert spec.filename is None
    assert spec.runtime is Gemma4Runtime
    assert spec.tokenizer_id == _MODEL_ID
    assert spec.skills is build_skill_registry
    assert spec.skills().names() == ("query",)


def test_prefill_slot_release_rejects_foreign_and_duplicate_slots():
    rt = Gemma4Runtime.__new__(Gemma4Runtime)
    rt._prefill_slot = object()
    rt._prefill_slot_in_use = True

    with pytest.raises(ValueError, match="foreign"):
        rt.release_prefill_slot(object())

    rt.release_prefill_slot(rt._prefill_slot)
    with pytest.raises(RuntimeError, match="not acquired"):
        rt.release_prefill_slot(rt._prefill_slot)


def test_launch_prepared_batch_rejects_misaligned_image_rows():
    rt = Gemma4Runtime.__new__(Gemma4Runtime)
    rt.max_batch_size = 2

    with pytest.raises(ValueError, match="must match"):
        rt.launch_prepared_batch(
            [object()],
            object(),
            images=[],
            image_crops_list=[None],
        )


def test_launch_prepared_batch_packs_heterogeneous_rows_without_invalid_slots(
    monkeypatch,
):
    class _PackedPageTable:
        capacity = {1: 24, 2: 528}

        def commit_block_table(self, batch_indices):
            assert batch_indices == [1, 2]

        def build_slot_mapping(self, batch_idx, positions):
            slots = batch_idx * 1_000 + positions
            valid = torch.tensor(
                [
                    position < self.capacity[int(row)]
                    for row, position in zip(
                        batch_idx.flatten().tolist(),
                        positions.flatten().tolist(),
                        strict=True,
                    )
                ],
                dtype=torch.bool,
            ).view_as(positions)
            return torch.where(valid, slots, -1)

    captured = {}

    def prefill(
        _runtime,
        inputs_embeds,
        input_ids,
        position_ids,
        slot_mapping,
        last_token_offsets,
        cu_seqlens,
    ):
        captured.update(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            position_ids=position_ids,
            slot_mapping=slot_mapping,
            last_token_offsets=last_token_offsets,
            cu_seqlens=cu_seqlens,
        )
        return torch.ones((2, 1)), torch.zeros((2, 4))

    rt = Gemma4Runtime.__new__(Gemma4Runtime)
    rt.max_batch_size = 2
    rt.device = torch.device("cpu")
    rt.model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                embed=lambda input_ids: input_ids.unsqueeze(-1).float()
            )
        )
    )
    rt._config = SimpleNamespace(image_token_id=999)
    monkeypatch.setattr(Gemma4Runtime, "_prefill", prefill)
    monkeypatch.setattr(
        Gemma4Runtime,
        "_image_features_for_batch",
        lambda _runtime, _crops: [None, torch.tensor([[-1.0]])],
    )
    rt.page_table = _PackedPageTable()
    rows = [
        SimpleNamespace(
            state=SimpleNamespace(batch_idx=1, last_hidden=None),
            tokens_list=[TextToken(token_id=index) for index in range(8)],
        ),
        SimpleNamespace(
            state=SimpleNamespace(batch_idx=2, last_hidden=None),
            tokens_list=[
                TextToken(token_id=999),
                *[TextToken(token_id=index) for index in range(1, 512)],
            ],
        ),
    ]
    slot = SimpleNamespace(batch_idx=torch.zeros(2, dtype=torch.long))

    logits = rt.launch_prepared_batch(
        rows,
        slot,
        images=[None, object()],
        image_crops_list=[None, object()],
    )

    assert logits.shape == (2, 4)
    assert captured["inputs_embeds"].shape == (1, 520, 1)
    assert captured["input_ids"].shape == (1, 520)
    assert captured["inputs_embeds"][0, 8, 0] == -1
    assert captured["input_ids"][0, 8] == 0
    assert captured["position_ids"][0, :8].tolist() == list(range(8))
    assert captured["position_ids"][0, 8:].tolist() == list(range(512))
    assert captured["cu_seqlens"].tolist() == [0, 8, 520]
    assert captured["last_token_offsets"].tolist() == [7, 519]
    assert int(captured["slot_mapping"].min()) >= 0
    assert captured["slot_mapping"][0, 7].item() == 1_007
    assert captured["slot_mapping"][0, 8].item() == 2_000
    assert rows[0].state.last_hidden.item() == 1
    assert rows[1].state.last_hidden.item() == 1


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

    rt.decode_with_slot(slot, batch_size=1)

    assert calls == [
        ("enter", "decode-stream"),
        ("generated", slot, 1),
        ("exit", "decode-stream"),
    ]


def test_decode_compile_translates_model_config(monkeypatch):
    from kestrel.models.gemma4 import generated_decode
    from mkl.compiler.frontend import DecodeCompileTarget
    from mkl.compiler.frontend.models import gemma as gemma_frontend
    from mkl.compiler.frontend.models.gemma import Gemma4DecodeTraceConfig

    calls = []
    monkeypatch.setattr(
        gemma_frontend,
        "compile_gemma4_decode",
        lambda trace, *, target: calls.append((trace, target)) or "compiled",
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
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        sliding_window=8,
        use_double_wide_mlp=True,
    )

    assert generated_decode._compile_from_config(
        config, max_kv_len=256, num_ctas=132, gpu="test-gpu"
    ) == "compiled"
    trace, target = calls[0]
    assert trace == Gemma4DecodeTraceConfig(
        name="gemma4",
        layer_types=tuple(config.layer_types),
        num_kv_shared_layers=2,
        hidden=64,
        inter=128,
        nh=4,
        nkv=1,
        global_nkv=1,
        local_head_dim=16,
        global_head_dim=32,
        window=8,
        max_kv_len=256,
        ple_hidden=16,
        ple_vocab=32,
        vocab_size=48,
        rms_norm_eps=1e-6,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        double_wide_mlp=True,
    )
    assert target == DecodeCompileTarget(
        batch_capacity=1,
        num_ctas=132,
        gpu="test-gpu",
    )


def test_prefill_deduplicates_images_within_batch_without_persistent_cache():
    runtime = Gemma4Runtime.__new__(Gemma4Runtime)
    runtime.device = torch.device("cpu")
    runtime.max_batch_size = 4
    runtime._vision_stager = BatchedTensorStager(
        capacity=4,
        device=runtime.device,
        with_numpy={"pixel_values": False},
    )
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
    pixel_buffer = runtime._vision_stager.buffers["pixel_values"]
    position_buffer = runtime._vision_stager.buffers["position_ids"]
    assert pixel_buffer.cpu.shape == (4, 6, 3)
    assert position_buffer.cpu.shape == (4, 6, 2)
    assert input_ptrs[0] == (
        pixel_buffer.gpu.data_ptr(),
        position_buffer.gpu.data_ptr(),
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

    pixel_gpu_ptr = pixel_buffer.gpu.data_ptr()
    position_gpu_ptr = position_buffer.gpu.data_ptr()
    repeated = runtime._image_features_for_batch([second, first])
    assert len(calls) == 2
    assert repeated[0] is not features[1]
    assert repeated[1] is not features[0]
    assert pixel_buffer.gpu.data_ptr() == pixel_gpu_ptr
    assert position_buffer.gpu.data_ptr() == position_gpu_ptr
    assert input_ptrs[1] == (pixel_gpu_ptr, position_gpu_ptr)


def test_generated_decode_binds_named_paged_kv_sets():
    from kestrel.runtime.generated_decode import PagedDecodeBindings

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
    runtime = SimpleNamespace(
        page_table=SimpleNamespace(
            n_pages=7,
            page_table=torch.empty(5, 9, dtype=torch.int32),
        )
    )
    inputs = PagedDecodeBindings(
        layers,
        kv_sets=(
            ("local", "sliding_attention"),
            ("global", "full_attention"),
        ),
        layer_kinds=kinds,
    ).runtime_inputs(runtime)

    assert [
        None if tensor is None else tuple(tensor.shape) for tensor in inputs["mK_local"]
    ] == [(3, 1, 4), None, None]
    assert [
        None if tensor is None else tuple(tensor.shape)
        for tensor in inputs["mK_global"]
    ] == [None, (3, 1, 8), None]


def test_engine_adopts_externally_supplied_runtime_kv_pool():
    cfg = RuntimeConfig(device="cpu", model=_MODEL_ID)
    runtime = Gemma4Runtime.__new__(Gemma4Runtime)
    runtime.device = cfg.resolved_device()
    runtime._kv_pool = object()
    runtime._compute_stream = object()

    engine = InferenceEngine(cfg, runtime=runtime)

    assert engine.runtime is runtime
    assert engine._kv_pool is runtime.kv_pool


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


def test_disabling_tokenizer_postprocessor_preserves_no_special_token_ids():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "<bos>": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A",
        special_tokens=[("<bos>", 2)],
    )
    expected = tokenizer.encode("hello", add_special_tokens=False).ids

    tokenizer.post_processor = None

    assert tokenizer.encode("hello").ids == expected == [1]


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


@pytest.mark.parametrize(
    ("layer_type", "heads", "head_dim"),
    [("sliding_attention", 1, 256), ("full_attention", 1, 512)],
)
def test_gemma_paged_cache_includes_shared_kv_producer(
    layer_type, heads, head_dim
):
    cfg = _text_config(
        num_kv_shared_layers=1,
        layer_types=[layer_type, layer_type],
        num_attention_heads=2,
        num_global_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
    )

    specs, sources = paged_kv_layout(cfg)
    assert specs == (PagedKVLayerSpec(heads, head_dim), None)
    assert sources == (0, 0)

    if layer_type == "full_attention":
        specs, _ = paged_kv_layout(replace(cfg, attention_k_eq_v=True))
        assert specs[0] == PagedKVLayerSpec(2, 512)


def test_gemma_shared_kv_sources_preserve_local_global_topology():
    cfg = _text_config(
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_kv_shared_layers=2,
    )

    assert kv_source_layers(cfg) == (0, 1, 2, 3, 2, 3)


def test_gemma_shared_sliding_layers_reuse_paged_kv(monkeypatch):
    cfg = _text_config(
        vocab_size=8,
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

    cache = LayeredPagedKV(
        layers=(fake_paged_layer, None),
        source_layer_idx=(0, 0),
    )

    paged_calls = []

    def fake_paged_attention_forward(query, **kwargs):
        paged_calls.append(kwargs["paged_kv_layer"])
        b, h, s, d = query.shape
        return torch.zeros(b, s, h, d, dtype=query.dtype, device=query.device)

    monkeypatch.setattr(
        attention_ops,
        "paged_attention",
        fake_paged_attention_forward,
    )

    model(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        kv_cache=cache,
        cache_position_ids=torch.zeros((1, 1), dtype=torch.long),
        slot_mapping=torch.zeros((1, 1), dtype=torch.long),
        page_table=torch.zeros((1, 1), dtype=torch.int32),
        paged_kv_seqlens_k=torch.ones((1,), dtype=torch.int32),
    )

    assert fake_paged_layer.updates == 1
    assert paged_calls == [fake_paged_layer, fake_paged_layer]


def test_nonpaged_local_attention_truncates_history_beyond_window():
    query = torch.zeros((1, 1, 1, 2))
    key = torch.zeros((1, 1, 600, 2))
    value = torch.zeros((1, 1, 600, 2))
    value[:, :, 0] = 1.0

    local = attention_ops.dense_attention(
        query,
        key,
        value,
        num_key_value_groups=1,
        attention_mask=None,
        scaling=1.0,
        causal=False,
        window_size_left=511,
        window_size_right=0,
    )
    global_ = attention_ops.dense_attention(
        query,
        key,
        value,
        num_key_value_groups=1,
        attention_mask=None,
        scaling=1.0,
        causal=True,
    )

    assert torch.count_nonzero(local) == 0
    torch.testing.assert_close(global_[0, 0, 0], torch.full((2,), 1 / 600))


def test_query_skill_defaults_to_direct_answer_mode():
    skill = build_skill_registry().resolve("query")
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


@pytest.mark.parametrize("pixel", [0, 127, 255])
def test_image_preprocessing_matches_official_rescale_domain(pixel):
    image = np.full((32, 32, 3), pixel, dtype=np.uint8)
    inputs = preprocess_image(image)
    expected = torch.tensor(
        float(pixel) / 255.0,
        dtype=torch.bfloat16,
    )
    valid = inputs.pixel_values[: inputs.num_image_tokens * 9]

    assert torch.equal(valid, torch.full_like(valid, expected))
