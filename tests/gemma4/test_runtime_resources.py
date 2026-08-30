"""Resource sizing for Gemma generated decode."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.gemma4.generated_decode import _rope_tables
from kestrel.models.gemma4.runtime import (
    Gemma4Runtime,
    _allocate_decode_page_tables,
    _generated_kv_binding_inputs,
    _install_decode_page_tables,
    _materialize_fixed_runtime_resources,
    _vision_probe_records,
)


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
        calls.append(
            ("vision", tuple(pixel_values.shape), tuple(position_ids.shape))
        )
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
    records = tuple(
        {
            "pixel_values": torch.zeros(3, 4),
            "position_ids": torch.zeros(3, 2, dtype=torch.long),
        }
        for _row in range(2)
    )

    def materialize(device, stream, operation):
        calls.append((device, stream))
        result = operation()
        calls.append(("result", tuple(result.shape)))

    monkeypatch.setattr(runtime_module, "materialize_blas_runtime", materialize)

    _materialize_fixed_runtime_resources(runtime, records)

    assert calls == [
        (torch.device("cpu"), compute_stream),
        ("stage", 2),
        ("vision", (2, 3, 4), (2, 3, 2)),
        ("result", (2, 2)),
    ]
    torch.testing.assert_close(logits[:2], hidden[:2] @ weight.t())


def test_vision_probe_records_cover_pooling_aligned_max_patch_grid() -> None:
    runtime = SimpleNamespace(
        max_batch_size=3,
        dtype=torch.bfloat16,
        _config=SimpleNamespace(
            vision_config=SimpleNamespace(
                patch_size=16,
                pooling_kernel_size=3,
            )
        ),
    )

    records = _vision_probe_records(runtime)

    assert len(records) == 3
    assert tuple(records[0]["pixel_values"].shape) == (2520, 768)
    assert tuple(records[0]["position_ids"].shape) == (2520, 2)
    positions = records[0]["position_ids"]
    width = int(positions[:, 0].max()) + 1
    height = int(positions[:, 1].max()) + 1
    assert width * height == 2520
    assert width % 3 == height % 3 == 0
