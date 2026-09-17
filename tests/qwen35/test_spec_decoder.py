from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.spec_decoder import Qwen35DFlashDecoder, _Session
from kestrel.runtime.spec import DraftResult
from kestrel.runtime.tokens import TextToken


def decoder():
    obj = Qwen35DFlashDecoder.__new__(Qwen35DFlashDecoder)
    state = SimpleNamespace(batch_idx=1, max_length=100, length=10)
    erased = []
    obj.runtime = SimpleNamespace(page_table=SimpleNamespace(
        free_batch_idx=[2], erase=erased.append), max_seq_length=128, vocab_size=256)
    obj.num_lookahead_tokens = 4
    obj.num_speculative_tokens = 3
    obj._session = _Session(state, SimpleNamespace(seq_length=10), None, None, 7)
    obj.propose = lambda ctx: DraftResult(torch.tensor([[8, 9, 10]]))
    commits = []

    class Verified:
        def commit_recurrent_prefix(self, count):
            commits.append(count)
            return SimpleNamespace(seq_length=10+count)

    obj._target = lambda tokens, cache, slot, capture: ([8, 9, 99, 100], torch.ones(1, 4, 2), Verified())
    return obj, state, commits, erased


@pytest.mark.parametrize("cap,expected", [(None, [8, 9, 99]), (1, [8]), (2, [8, 9]), (4, [8, 9, 99])])
def test_bonus_not_duplicated_and_caps_commit_matching_input_prefix(cap, expected):
    obj, state, commits, _ = decoder()
    result = obj.step([state], commit_caps=[cap])
    assert result.tokens == [expected]
    assert result.accept_counts == [len(expected)-1]
    assert commits == [len(expected)]
    assert obj._session.bonus == expected[-1]
    assert obj._session.cache.seq_length == 10+len(expected)
    assert state.length == 10  # Scheduler advances this exactly once.


def test_retire_releases_slot_and_capture_once():
    obj, state, _, erased = decoder()
    obj.retire(state)
    obj.retire(state)
    assert erased == [1]
    assert obj.free_slots == 1 and obj._session is None


def test_base_adapter_slot_retirement_is_valid_without_lora_support():
    from kestrel.runtime.uncached_paged import UncachedPagedRuntime

    runtime = UncachedPagedRuntime()
    runtime.release_adapter_slot(0)
    with pytest.raises(NotImplementedError):
        runtime.release_adapter_slot(1)


@pytest.mark.parametrize("kwargs", [{"temperature": .5}, {"image": object()},
                                    {"allowed_token_ids": [1]}, {"suppress_next_token_ids": [1]},
                                    {"top_p": float("nan")}, {"top_p": float("inf")},
                                    {"temperature": float("nan")}])
def test_unsupported_admit_fails_before_allocation(kwargs):
    obj, state, _, erased = decoder()
    obj._session = None
    with pytest.raises(ValueError):
        obj.admit(state, [TextToken(token_id=1)], **kwargs)
    assert erased == [] and obj._session is None


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_invalid_cap_fails_without_advancing(cap):
    obj, state, commits, _ = decoder()
    with pytest.raises(ValueError, match="commit cap"):
        obj.step([state], commit_caps=[cap])
    assert commits == []


def test_failed_admission_returns_page_slot():
    obj, state, _, erased = decoder()
    obj._session = None
    obj.runtime.page_table.allocate = lambda: 1
    obj.runtime.page_table.reserve = lambda *args: True
    obj.runtime.page_table.commit_block_table = lambda *args: None
    obj.runtime._paged_kv = ()
    obj.runtime._linear_state_pool = SimpleNamespace(bind_prefill_state=lambda cache: None)
    obj.text = SimpleNamespace(config=SimpleNamespace(layer_types=()))

    def fail(*args, **kwargs):
        raise RuntimeError("injected target failure")

    obj._target = fail
    with pytest.raises(RuntimeError, match="injected"):
        obj.admit(state, [TextToken(token_id=1)], top_p=.9)
    assert erased == [1] and state.batch_idx == -1 and obj._session is None


def test_unsupported_runtime_rejects_draft_config_before_loading():
    from kestrel.config import RuntimeConfig

    with pytest.raises(ValueError, match="requires a Qwen"):
        RuntimeConfig(model="moondream3-preview", draft_model_path="unused",
                      model_path="unused", max_batch_size=1)
