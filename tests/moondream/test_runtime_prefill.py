from __future__ import annotations

import torch

from kestrel.moondream import runtime as runtime_mod


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
    runtime._primary_stream = None
    runtime.page_table = _FakePageTable()

    def build_inputs(prepared, *, image, image_crops):
        del image, image_crops
        seq_len = seq_lens[prepared.state.batch_idx]
        hidden = torch.full((1, seq_len, 4), float(prepared.state.batch_idx))
        positions = torch.arange(seq_len, dtype=torch.long).view(1, seq_len)
        return hidden, positions, False

    def prefill_impl(
        inputs_embeds,
        attn_mask,
        position_ids,
        batch_idx,
        lora_slot=0,
        *,
        use_prefix_attn,
        fa3_seqused_q,
        fa3_seqused_k,
        last_token_positions,
    ):
        del attn_mask, position_ids, batch_idx, lora_slot, use_prefix_attn
        captured["fa3_seqused_q"] = fa3_seqused_q
        captured["fa3_seqused_k"] = fa3_seqused_k
        captured["last_token_positions"] = last_token_positions
        batch_size = inputs_embeds.shape[0]
        return inputs_embeds, torch.zeros((batch_size, 8), dtype=inputs_embeds.dtype)

    runtime._build_prefill_inputs_for_prepared = build_inputs
    runtime._prefill_impl = prefill_impl
    return runtime, captured


def test_launch_prepared_batch_omits_seqused_q_for_uniform_q_lengths() -> None:
    runtime, captured = _make_runtime({1: 3, 2: 3})
    slot = runtime_mod.PrefillSlot(
        slot_id=0,
        batch_idx=torch.zeros(2, dtype=torch.int64),
        step_done_event=None,  # type: ignore[arg-type]
        commit_done_event=None,  # type: ignore[arg-type]
    )

    logits = runtime.launch_prepared_batch(
        [_make_prepared(1, 10), _make_prepared(2, 11)],
        slot,
    )

    assert logits.shape == (2, 8)
    assert captured["fa3_seqused_q"] is None
    torch.testing.assert_close(
        captured["fa3_seqused_k"],
        torch.tensor([10, 11], dtype=torch.int32),
    )
    torch.testing.assert_close(
        captured["last_token_positions"],
        torch.tensor([2, 2], dtype=torch.long),
    )


def test_launch_prepared_batch_omits_seqused_q_for_single_token_append() -> None:
    runtime, captured = _make_runtime({1: 1, 2: 1})
    slot = runtime_mod.PrefillSlot(
        slot_id=0,
        batch_idx=torch.zeros(2, dtype=torch.int64),
        step_done_event=None,  # type: ignore[arg-type]
        commit_done_event=None,  # type: ignore[arg-type]
    )

    logits = runtime.launch_prepared_batch(
        [_make_prepared(1, 10), _make_prepared(2, 11)],
        slot,
    )

    assert logits.shape == (2, 8)
    assert captured["fa3_seqused_q"] is None
    torch.testing.assert_close(
        captured["last_token_positions"],
        torch.tensor([0, 0], dtype=torch.long),
    )


def test_launch_prepared_batch_keeps_seqused_q_for_padded_q_lengths() -> None:
    runtime, captured = _make_runtime({1: 1, 2: 3})
    slot = runtime_mod.PrefillSlot(
        slot_id=0,
        batch_idx=torch.zeros(2, dtype=torch.int64),
        step_done_event=None,  # type: ignore[arg-type]
        commit_done_event=None,  # type: ignore[arg-type]
    )

    logits = runtime.launch_prepared_batch(
        [_make_prepared(1, 10), _make_prepared(2, 11)],
        slot,
    )

    assert logits.shape == (2, 8)
    torch.testing.assert_close(
        captured["fa3_seqused_q"],
        torch.tensor([1, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        captured["last_token_positions"],
        torch.tensor([0, 2], dtype=torch.long),
    )
