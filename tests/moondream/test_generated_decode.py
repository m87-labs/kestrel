from types import SimpleNamespace

import torch

from kestrel.models.moondream.generated_decode import MoondreamDecodeBindings
from kestrel.models.moondream.runtime import _FP8_KV_SUPPORTED_SMS


def _cache(value: float) -> SimpleNamespace:
    return SimpleNamespace(
        quantized=True,
        k_scale_tensor=torch.tensor(value, dtype=torch.float32),
        v_scale_tensor=torch.tensor(value + 1, dtype=torch.float32),
        k_cache=torch.full((3, 1, 1, 2), value, dtype=torch.float32),
        v_cache=torch.full((3, 1, 1, 2), value + 1, dtype=torch.float32),
    )


def test_sm86_uses_checkpoint_fp8_kv_for_generated_decode() -> None:
    assert 86 in _FP8_KV_SUPPORTED_SMS


def test_moondream_bindings_expose_compiler_runtime_resources() -> None:
    first = _cache(1)
    second = _cache(2)
    blocks = [
        SimpleNamespace(
            attn=SimpleNamespace(
                _tau_pos_table=torch.full((5, 2), value, dtype=torch.bfloat16)
            )
        )
        for value in (1, 2)
    ]
    text = SimpleNamespace(
        blocks=blocks,
        cos_sin_cache=torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        ),
    )
    runtime = SimpleNamespace(
        _lora_workspace=None,
        page_size=1,
        model=SimpleNamespace(text=text),
        page_table=SimpleNamespace(page_table=torch.zeros(4, 7)),
    )
    bindings = MoondreamDecodeBindings(
        (SimpleNamespace(cache=first), SimpleNamespace(cache=second))
    )

    assert bindings.is_eligible(runtime)
    inputs = bindings.runtime_inputs(runtime)

    assert set(inputs) == {
        "tau_pos",
        "rope_cos",
        "rope_sin",
        "mK_dequant_scale",
        "mV_dequant_scale",
        "mK",
        "mV",
        "page_table",
    }
    torch.testing.assert_close(
        inputs["rope_cos"],
        torch.tensor([[1.0, 2.0, 1.0, 2.0], [5.0, 6.0, 5.0, 6.0]]),
    )
    torch.testing.assert_close(
        inputs["rope_sin"],
        torch.tensor([[3.0, 4.0, 3.0, 4.0], [7.0, 8.0, 7.0, 8.0]]),
    )
    assert [tensor.shape for tensor in inputs["mK"]] == [
        torch.Size((3, 1, 2)),
        torch.Size((3, 1, 2)),
    ]
    assert all(tensor.dtype is torch.float32 for tensor in inputs["tau_pos"])


def test_moondream_slot_bindings_use_stable_hidden_and_metadata_buffers() -> None:
    bindings = MoondreamDecodeBindings(())
    slot = SimpleNamespace(
        hidden_last=torch.zeros(8, 16),
        meta=SimpleNamespace(
            batch_idx=SimpleNamespace(gpu=torch.arange(8)),
            input_pos=SimpleNamespace(
                gpu=torch.arange(8, dtype=torch.int32),
                cpu=torch.tensor([3, 6, 4, 1, 0, 0, 0, 0], dtype=torch.int32),
            ),
        ),
    )

    inputs = bindings.slot_inputs(slot, 4)

    assert inputs["x"].data_ptr() == slot.hidden_last.data_ptr()
    assert inputs["batch_idx"].tolist() == [0, 1, 2, 3]
    assert inputs["input_pos"].tolist() == [0, 1, 2, 3]
    assert bindings.launch_extents(slot, 4) == {
        "active_batch": 4,
        "kv_len": 7,
    }
