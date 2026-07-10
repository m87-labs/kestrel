from __future__ import annotations

import torch

from kestrel.models.moondream import runtime as runtime_mod


class _FakePageTable:
    def __init__(self) -> None:
        self.committed: list[int] | None = None

    def commit_block_table(self, batch_indices: list[int]) -> None:
        self.committed = list(batch_indices)


def _make_prepared(batch_idx: int, prompt_len: int) -> runtime_mod.PreparedSequence:
    state = runtime_mod.SequenceState(
        batch_idx=batch_idx,
        length=prompt_len,
        max_length=prompt_len + 8,
        prompt_length=prompt_len,
    )
    return runtime_mod.PreparedSequence(
        state=state,
        tokens_list=[runtime_mod.TextToken(i) for i in range(prompt_len)],
        cache_tokens=[],
        cache_result=None,  # type: ignore[arg-type]
        adapter_id=None,
        image_hash=None,
    )


def _make_runtime(seq_lens: dict[int, int]) -> tuple[runtime_mod.MoondreamRuntime, dict]:
    captured: dict = {}
    runtime = runtime_mod.MoondreamRuntime.__new__(runtime_mod.MoondreamRuntime)
    runtime.max_batch_size = 8
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.bos_embed = torch.empty(1, 4)
    runtime._compute_stream = None
    runtime.page_table = _FakePageTable()

    def build_inputs(prepared, *, image, image_crops, staging, row):
        del staging, row
        del image, image_crops
        seq_len = seq_lens[prepared.state.batch_idx]
        hidden = torch.full((1, seq_len, 4), float(prepared.state.batch_idx))
        # (inputs_embeds, position_start, use_prefix_attn, bidirectional_ranges);
        # None = no image blocks for these single-/uniform-token prefill cases.
        return hidden, 0, False, None

    def prefill_impl(
        inputs_embeds,
        attn_mask,
        position_ids,
        batch_idx,
        lora_slot=0,
        *,
        use_prefix_attn,
        paged_kv_seqlens_q,
        paged_kv_seqlens_k,
        last_token_positions,
        input_staging,
        bidirectional_ranges=None,
    ):
        captured["position_ids_contiguous"] = position_ids.is_contiguous()
        captured["batch_idx_contiguous"] = batch_idx.is_contiguous()
        del attn_mask, position_ids, batch_idx, lora_slot, use_prefix_attn
        del input_staging, bidirectional_ranges
        captured["paged_kv_seqlens_q"] = paged_kv_seqlens_q
        captured["paged_kv_seqlens_k"] = paged_kv_seqlens_k
        captured["last_token_positions"] = last_token_positions
        batch_size = inputs_embeds.shape[0]
        return inputs_embeds, torch.zeros((batch_size, 8), dtype=inputs_embeds.dtype)

    runtime._build_prefill_inputs_for_prepared = build_inputs
    runtime._prefill_impl = prefill_impl
    return runtime, captured


def _make_prefill_slot() -> runtime_mod.PrefillSlot:
    return runtime_mod.PrefillSlot(
        slot_id=0,
        batch_idx=torch.zeros(8, dtype=torch.int64),
        step_done_event=None,  # type: ignore[arg-type]
        commit_done_event=None,  # type: ignore[arg-type]
        aux_done_event=None,  # type: ignore[arg-type]
        coord_staging=torch.empty(0),
        size_staging=torch.empty(0),
        coord_cpu=torch.empty(0),
        size_cpu=torch.empty(0),
        input_staging=runtime_mod.PrefillInputStaging(
            max_batch_size=8,
            max_seq_length=16,
            max_lora_slots=0,
            coord_dtype=torch.float32,
            size_dtype=torch.float32,
            device=torch.device("cpu"),
            pin_memory=False,
        ),
    )


def test_launch_prepared_batch_omits_paged_kv_q_lengths_for_uniform_q_lengths() -> None:
    runtime, captured = _make_runtime({1: 3, 2: 3})
    slot = _make_prefill_slot()

    logits = runtime.launch_prepared_batch(
        [_make_prepared(1, 10), _make_prepared(2, 11)],
        slot,
    )

    assert logits.shape == (2, 8)
    assert captured["position_ids_contiguous"]
    assert captured["batch_idx_contiguous"]
    assert captured["paged_kv_seqlens_q"] is None
    torch.testing.assert_close(
        captured["paged_kv_seqlens_k"],
        torch.tensor([10, 11], dtype=torch.int32),
    )
    torch.testing.assert_close(
        captured["last_token_positions"],
        torch.tensor([2, 2], dtype=torch.long),
    )


def test_launch_prepared_batch_omits_paged_kv_q_lengths_for_single_token_append() -> None:
    runtime, captured = _make_runtime({1: 1, 2: 1})
    slot = _make_prefill_slot()

    logits = runtime.launch_prepared_batch(
        [_make_prepared(1, 10), _make_prepared(2, 11)],
        slot,
    )

    assert logits.shape == (2, 8)
    assert captured["paged_kv_seqlens_q"] is None
    torch.testing.assert_close(
        captured["last_token_positions"],
        torch.tensor([0, 0], dtype=torch.long),
    )


def test_launch_prepared_batch_keeps_paged_kv_q_lengths_for_padded_q_lengths() -> None:
    runtime, captured = _make_runtime({1: 1, 2: 3})
    slot = _make_prefill_slot()

    logits = runtime.launch_prepared_batch(
        [_make_prepared(1, 10), _make_prepared(2, 11)],
        slot,
    )

    assert logits.shape == (2, 8)
    torch.testing.assert_close(
        captured["paged_kv_seqlens_q"],
        torch.tensor([1, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        captured["last_token_positions"],
        torch.tensor([0, 2], dtype=torch.long),
    )
