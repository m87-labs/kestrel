"""Resource sizing for Gemma generated decode."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.gemma4.generated_decode import _rope_tables
from kestrel.models.gemma4.image import GemmaImageInputs
from kestrel.models.gemma4.runtime import (
    Gemma4Runtime,
    _PagedRuntimeResources,
    _allocate_decode_page_tables,
    _copy_image_features_into_embeddings,
    _generated_kv_binding_inputs,
    _install_decode_page_tables,
    _materialize_fixed_runtime_resources,
    _maximum_vision_grid,
    _run_image_prefill_transient_probe,
    _vision_probe_inputs,
)
from kestrel.kv_cache import (
    KVMemoryPool,
    PageTable,
    PagedKVLayerSpec,
    allocate_paged_kv_layers,
    allocate_paged_kv_storage,
)
from kestrel.models.gemma4.prompt_template import Gemma4PromptTemplate


class _Rotary:
    def __call__(self, _probe, positions, kind):
        offset = 1 if kind == "sliding_attention" else 10
        values = positions.unsqueeze(-1).expand(-1, -1, 3) + offset
        values = values.to(torch.bfloat16)
        return values, values + 1


def _runtime(max_seq_length: int = 8):
    language_model = SimpleNamespace(rotary_emb=_Rotary())
    return SimpleNamespace(
        max_seq_length=max_seq_length,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        model=SimpleNamespace(
            model=SimpleNamespace(language_model=language_model),
        ),
    )


def test_rope_tables_materialize_only_requested_reachable_positions() -> None:
    tables = _rope_tables(_runtime(), 3)

    assert set(tables) == {
        "rope_cos_local",
        "rope_sin_local",
        "rope_cos_global",
        "rope_sin_global",
    }
    assert all(tuple(table.shape) == (3, 3) for table in tables.values())
    assert all(table.dtype is torch.float32 for table in tables.values())
    assert all(table.is_contiguous() for table in tables.values())


@pytest.mark.parametrize("length", [0, 9])
def test_rope_tables_reject_lengths_outside_model_context(length: int) -> None:
    with pytest.raises(ValueError, match="must lie"):
        _rope_tables(_runtime(), length)


def test_generated_binding_inputs_preserve_sparse_layer_topology() -> None:
    inputs = _generated_kv_binding_inputs(
        (object(), None, object(), object()),
        (
            "sliding_attention",
            "full_attention",
            "full_attention",
            "sliding_attention",
        ),
    )

    assert inputs["mK_local"] is inputs["mV_local"]
    assert inputs["mK_global"] is inputs["mV_global"]
    assert [value is not None for value in inputs["mK_local"]] == [
        True,
        False,
        False,
        True,
    ]
    assert [value is not None for value in inputs["mK_global"]] == [
        False,
        False,
        True,
        False,
    ]


def test_generated_binding_reservation_is_released_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel.models.gemma4 import generated_decode

    released = []

    class _Reservation:
        def __del__(self):
            released.append(True)

    runtime = object.__new__(Gemma4Runtime)
    runtime.decode_path = "auto"
    runtime._generated_weight_storage = object()
    runtime._generated_binding_reservation = _Reservation()
    expected = object()

    def create(_runtime, *, required=False):
        assert released == [True]
        assert required is False
        return expected

    monkeypatch.setattr(generated_decode, "create_generated_decode", create)

    runtime._initialize_generated_decode()

    assert runtime._generated_binding_reservation is None
    assert runtime.generated_decode is expected


def test_decode_page_table_placeholders_are_replaced_before_binding() -> None:
    slots = tuple(
        SimpleNamespace(paged_kv_page_table=torch.empty((3, 1), dtype=torch.int32))
        for _ in range(2)
    )
    old_tables = tuple(weakref.ref(slot.paged_kv_page_table) for slot in slots)
    tables = _allocate_decode_page_tables(
        count=2,
        rows=3,
        pages=11,
        device=torch.device("cpu"),
    )

    _install_decode_page_tables(slots, tables)

    assert all(tuple(slot.paged_kv_page_table.shape) == (3, 11) for slot in slots)
    assert all(slot.paged_kv_page_table is table for slot, table in zip(slots, tables))
    gc.collect()
    assert all(reference() is None for reference in old_tables)
    with pytest.raises(RuntimeError, match="unbound one-column placeholders"):
        _install_decode_page_tables(
            slots,
            _allocate_decode_page_tables(
                count=2,
                rows=3,
                pages=12,
                device=torch.device("cpu"),
            ),
        )


def test_fixed_runtime_resources_use_resident_matmul_and_full_vision_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel.models.gemma4 import runtime as runtime_module

    hidden = torch.randn(4, 3, dtype=torch.bfloat16)
    weight = torch.randn(5, 3, dtype=torch.bfloat16)
    logits = torch.empty(4, 5, dtype=torch.bfloat16)
    compute_stream = object()
    calls = []

    class Stager:
        def stage(self, records):
            calls.append(("stage", len(records)))
            return {
                "pixel_values": torch.stack(
                    [record["pixel_values"] for record in records]
                ),
                "position_ids": torch.stack(
                    [record["position_ids"] for record in records]
                ),
            }

    def get_image_features(pixel_values, position_ids):
        calls.append(("vision", tuple(pixel_values.shape), tuple(position_ids.shape)))
        return torch.ones(pixel_values.shape[0], 2)

    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        _compute_stream=compute_stream,
        max_batch_size=2,
        model=SimpleNamespace(
            lm_head=SimpleNamespace(weight=weight),
            model=SimpleNamespace(get_image_features=get_image_features),
        ),
        decode_slots=(SimpleNamespace(hidden_last=hidden, logits=logits),),
        _vision_stager=Stager(),
    )
    inputs = tuple(
        GemmaImageInputs(
            pixel_values=torch.zeros(3, 4),
            image_position_ids=torch.zeros(3, 2, dtype=torch.long),
            num_image_tokens=2,
        )
        for _row in range(2)
    )

    def materialize(device, stream, operation):
        calls.append((device, stream))
        result = operation()
        calls.append(("result", tuple(result.shape)))

    monkeypatch.setattr(runtime_module, "materialize_blas_runtime", materialize)

    _materialize_fixed_runtime_resources(runtime, inputs)

    assert calls == [
        (torch.device("cpu"), compute_stream),
        ("stage", 2),
        ("vision", (2, 3, 4), (2, 3, 2)),
        ("result", (2, 2)),
    ]
    torch.testing.assert_close(logits[:2], hidden[:2] @ weight.t())


def test_vision_probe_inputs_cover_pooling_aligned_max_patch_grid() -> None:
    runtime = SimpleNamespace(
        max_batch_size=3,
        dtype=torch.bfloat16,
        _config=SimpleNamespace(
            vision_config=SimpleNamespace(
                patch_size=16,
                pooling_kernel_size=3,
                position_embedding_size=64,
            )
        ),
    )

    inputs = _vision_probe_inputs(runtime)

    assert len(inputs) == 3
    assert len({id(item) for item in inputs}) == 3
    assert tuple(inputs[0].pixel_values.shape) == (2520, 768)
    assert tuple(inputs[0].image_position_ids.shape) == (2520, 2)
    assert inputs[0].num_image_tokens == 280
    positions = inputs[0].image_position_ids
    width = int(positions[:, 0].max()) + 1
    height = int(positions[:, 1].max()) + 1
    assert width * height == 2520
    assert width % 3 == height % 3 == 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"patch_size": 14}, "patch_size must be 16"),
        ({"pooling_kernel_size": 2}, "pooling_kernel_size must be 3"),
        ({"position_embedding_size": 59}, "cannot represent"),
    ],
)
def test_maximum_vision_grid_rejects_unsupported_preprocessing_contract(
    overrides: dict[str, int],
    match: str,
) -> None:
    values = {
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "position_embedding_size": 64,
        **overrides,
    }

    with pytest.raises(ValueError, match=match):
        _maximum_vision_grid(SimpleNamespace(**values))


def test_direct_image_feature_copy_matches_masked_scatter_without_packing() -> None:
    image_token_id = 9
    token_rows = ([1, 9, 9, 2], [3, 4], [9, 9, 5])
    shared = torch.arange(6, dtype=torch.bfloat16).view(3, 2).t()
    assert not shared.is_contiguous()
    features = (shared, None, shared)
    embeddings = torch.randn(1, 9, 3, dtype=torch.bfloat16)
    expected = embeddings.clone()
    flat_ids = torch.tensor([[token for row in token_rows for token in row]])
    image_mask = flat_ids == image_token_id
    expected.masked_scatter_(
        image_mask.unsqueeze(-1).expand_as(expected),
        torch.cat([shared, shared]),
    )

    _copy_image_features_into_embeddings(
        embeddings,
        token_rows,
        features,
        image_token_id=image_token_id,
    )

    assert torch.equal(embeddings, expected)


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("dtype", "dtype"),
        ("device", "expected cpu"),
        ("rank", "must be 2D"),
        ("width", "expected \\(2, 3\\)"),
        ("noncontiguous_tokens", "must be contiguous"),
        ("missing", "without features"),
    ],
)
def test_direct_image_feature_copy_validates_all_rows_before_writing(
    failure: str,
    match: str,
) -> None:
    token_rows = ([9, 9], [9, 9])
    valid = torch.ones(2, 3, dtype=torch.bfloat16)
    invalid: torch.Tensor | None = valid
    if failure == "dtype":
        invalid = valid.float()
    elif failure == "device":
        invalid = torch.empty(2, 3, dtype=torch.bfloat16, device="meta")
    elif failure == "rank":
        invalid = torch.ones(6, dtype=torch.bfloat16)
    elif failure == "width":
        invalid = torch.ones(2, 2, dtype=torch.bfloat16)
    elif failure == "noncontiguous_tokens":
        token_rows = ([9, 9], [9, 4, 9])
    elif failure == "missing":
        invalid = None
    embeddings = torch.randn(
        1,
        sum(len(row) for row in token_rows),
        3,
        dtype=torch.bfloat16,
    )
    before = embeddings.clone()

    with pytest.raises(RuntimeError, match=match):
        _copy_image_features_into_embeddings(
            embeddings,
            token_rows,
            (valid, invalid),
            image_token_id=9,
        )

    assert torch.equal(embeddings, before)


class _ProbeStager:
    def stage(self, records):
        return {
            "pixel_values": torch.stack([record["pixel_values"] for record in records]),
            "position_ids": torch.stack([record["position_ids"] for record in records]),
        }


class _ProbeLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.touched_pages: set[int] = set()
        self.raise_oom = False

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        values = input_ids.to(torch.bfloat16).unsqueeze(-1)
        return values.expand(-1, -1, 4).clone()

    def forward(
        self,
        *,
        inputs_embeds,
        position_ids,
        kv_cache,
        per_layer_inputs,
        cache_position_ids,
        slot_mapping,
        cu_seqlens,
    ):
        del position_ids, per_layer_inputs, cache_position_ids, cu_seqlens
        pages = torch.unique(slot_mapping // int(kv_cache[0].page_table.page_size)).to(
            torch.long
        )
        self.touched_pages.update(int(page) for page in pages.tolist())
        for cache in kv_cache:
            if cache is not None:
                cache.k_cache.index_fill_(0, pages, 7)
                cache.v_cache.index_fill_(0, pages, 8)
        if self.raise_oom:
            raise torch.OutOfMemoryError("synthetic language-prefill peak")
        return inputs_embeds


class _ProbeModel(torch.nn.Module):
    def __init__(self, language_model: _ProbeLanguageModel) -> None:
        super().__init__()
        self.language_model = language_model
        self.embed_vision = torch.nn.Identity()

    def get_image_features(self, pixel_values, image_position_ids):
        del image_position_ids
        count = int(pixel_values.shape[0]) * 280
        features = torch.arange(count * 4, dtype=torch.float32).view(count, 4)
        return self.embed_vision(features.to(torch.bfloat16))


class _ProbeHead(torch.nn.Module):
    def forward(self, rows):
        return rows[:, :3]


def _image_prefill_probe_runtime():
    runtime = object.__new__(Gemma4Runtime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.bfloat16
    runtime.max_batch_size = 2
    runtime.max_batch_slots = 4
    runtime.max_seq_length = 1024
    runtime.page_size = 16
    runtime.image_prefix_length = 282
    runtime._padding_batch_idx = 3
    runtime._compute_stream = None
    runtime._prefill_slot = SimpleNamespace(
        batch_idx=torch.full((2,), -1, dtype=torch.long)
    )
    runtime._prefill_slot.batch_idx.zero_()
    runtime._prefill_slot_in_use = False
    runtime.active_sequences = {}
    runtime.prompt_template = Gemma4PromptTemplate("unit-test")
    runtime._config = SimpleNamespace(
        image_token_id=258_880,
        vision_config=SimpleNamespace(
            patch_size=16,
            pooling_kernel_size=3,
            position_embedding_size=64,
        ),
        text_config=SimpleNamespace(
            hidden_size_per_layer_input=False,
            final_logit_softcapping=30.0,
        ),
    )
    runtime._vision_stager = _ProbeStager()
    language_model = _ProbeLanguageModel()
    runtime.model = SimpleNamespace(
        model=_ProbeModel(language_model),
        lm_head=_ProbeHead(),
    )
    pool = KVMemoryPool(device="cpu")
    runtime._kv_pool = pool
    specs = (PagedKVLayerSpec(n_heads=1, head_dim=2),)
    storage = allocate_paged_kv_storage(
        64,
        layer_specs=specs,
        page_size=16,
        dtype=torch.bfloat16,
        pool=pool,
    )
    accepted = PageTable(
        n_pages=64,
        page_size=16,
        max_batch_size=4,
        device="cpu",
    )
    accepted.free_batch_idx.remove(runtime._padding_batch_idx)
    accepted.reserve(runtime._padding_batch_idx, 1)
    accepted.commit_block_table([runtime._padding_batch_idx])
    resources = _PagedRuntimeResources(
        rope_inputs=None,
        decode_page_tables=(),
        page_table=accepted,
    )
    return (
        runtime,
        language_model,
        specs,
        storage,
        resources,
        _vision_probe_inputs(runtime),
    )


def _page_table_state(page_table: PageTable):
    return (
        tuple(page_table.free_pages),
        tuple(page_table.free_batch_idx),
        tuple(tuple(row) for row in page_table.page_table_cpu),
        tuple(page_table.capacity),
        tuple(int(value) for value in page_table.num_blocks_per_row),
        page_table._page_table_cpu_tensor.clone(),
        page_table.page_table.clone(),
    )


def test_image_prefill_probe_restores_accepted_resources_and_runtime() -> None:
    runtime, language, specs, storage, resources, inputs = (
        _image_prefill_probe_runtime()
    )
    accepted_before = _page_table_state(resources.page_table)

    result = _run_image_prefill_transient_probe(
        runtime,
        inputs,
        storage,
        resources,
        specs,
    )

    assert tuple(result.logits.shape) == (2, 3)
    assert len(result.prepared_sequences) == 2
    assert len(result.image_features) == 1
    assert tuple(result.image_features[0].shape) == (560, 4)
    assert not hasattr(runtime, "page_table")
    assert not hasattr(runtime, "_kv_cache")
    assert torch.equal(
        runtime._prefill_slot.batch_idx, torch.zeros(2, dtype=torch.long)
    )
    accepted_after = _page_table_state(resources.page_table)
    assert accepted_after[:-2] == accepted_before[:-2]
    assert torch.equal(accepted_after[-2], accepted_before[-2])
    assert torch.equal(accepted_after[-1], accepted_before[-1])
    assert language.touched_pages and 0 not in language.touched_pages
    layer = storage.layers[0]
    assert layer is not None
    assert torch.count_nonzero(layer.k_cache[0].view(torch.uint8)) == 0
    assert torch.count_nonzero(layer.v_cache[0].view(torch.uint8)) == 0
    assert not runtime.model.model.embed_vision._forward_hooks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_paged_kv_mapping_preserves_raw_zero_page_on_device() -> None:
    device = torch.device("cuda")
    pool = KVMemoryPool(device=device)
    specs = (PagedKVLayerSpec(n_heads=1, head_dim=4),)
    storage = allocate_paged_kv_storage(
        8,
        layer_specs=specs,
        page_size=2,
        dtype=torch.bfloat16,
        pool=pool,
    )
    page_table = PageTable(
        n_pages=8,
        page_size=2,
        max_batch_size=3,
        device=str(device),
    )
    batch_idx = page_table.allocate()
    page_table.reserve(batch_idx, 5)
    page_table.commit_block_table([batch_idx])
    positions = torch.arange(5, dtype=torch.long, device=device).view(1, -1)
    logical_rows = torch.full_like(positions, batch_idx)
    slot_mapping = page_table.build_slot_mapping(
        batch_idx=logical_rows,
        positions=positions,
    )
    mapped_pages = torch.unique(slot_mapping // page_table.page_size)
    assert mapped_pages.numel() and not bool((mapped_pages == 0).any().item())
    cache = allocate_paged_kv_layers(
        layer_specs=specs,
        page_table=page_table,
        pool=pool,
        dtype=torch.bfloat16,
        storage=storage,
    )[0]
    assert cache is not None
    values = torch.ones(
        1,
        5,
        1,
        4,
        dtype=torch.bfloat16,
        device=device,
    )
    cache.update(
        positions,
        values * 7,
        values * 8,
        slot_mapping=slot_mapping,
    )
    layer = storage.layers[0]
    assert layer is not None
    torch.cuda.synchronize(device)

    assert torch.equal(
        layer.k_cache[0].view(torch.uint8).cpu(),
        torch.zeros_like(layer.k_cache[0].view(torch.uint8), device="cpu"),
    )
    assert torch.equal(
        layer.v_cache[0].view(torch.uint8).cpu(),
        torch.zeros_like(layer.v_cache[0].view(torch.uint8), device="cpu"),
    )


def test_image_prefill_probe_preserves_oom_when_cleanup_also_fails() -> None:
    runtime, language, specs, storage, resources, inputs = (
        _image_prefill_probe_runtime()
    )
    accepted_before = _page_table_state(resources.page_table)
    language.raise_oom = True

    def fail_abort(_prepared):
        raise RuntimeError("synthetic abort failure")

    runtime.abort_prepared_sequence = fail_abort

    with pytest.raises(torch.OutOfMemoryError, match="language-prefill peak") as caught:
        _run_image_prefill_transient_probe(
            runtime,
            inputs,
            storage,
            resources,
            specs,
        )

    assert any("synthetic abort failure" in note for note in caught.value.__notes__)
    assert not hasattr(runtime, "page_table")
    assert not hasattr(runtime, "_kv_cache")
    assert torch.equal(
        runtime._prefill_slot.batch_idx, torch.zeros(2, dtype=torch.long)
    )
    accepted_after = _page_table_state(resources.page_table)
    assert accepted_after[:-2] == accepted_before[:-2]
    assert torch.equal(accepted_after[-2], accepted_before[-2])
    assert torch.equal(accepted_after[-1], accepted_before[-1])
    assert not runtime.model.model.embed_vision._forward_hooks
