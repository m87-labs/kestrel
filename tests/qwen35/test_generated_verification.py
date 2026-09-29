"""AOT verification session coverage; model paths are supplied on GPU hosts."""

import asyncio
import os
from itertools import groupby
from types import SimpleNamespace

import pytest
import torch

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.models.qwen35.generated_verification import Qwen35GeneratedVerification


def test_generated_verification_rejects_unqualified_concurrency():
    with pytest.raises(ValueError, match="one or two sequences"):
        Qwen35GeneratedVerification(SimpleNamespace(max_batch_size=3), None)


@pytest.mark.parametrize("count", [0, 17, True, -1])
def test_generated_verification_rejects_invalid_commit(count):
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    target.pending, target.block, target.verified = True, 16, object()
    with pytest.raises(ValueError, match="outstanding verification"):
        target.commit((None, target.verified, None, None, count))


def test_generated_verification_rejects_overlapping_blocks():
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    target.pending = True
    with pytest.raises(RuntimeError, match="uncommitted block"):
        target.target([], None, 0)


def test_generated_verification_commits_independent_rows_once(monkeypatch):
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    target.pending, target.block, target.pending_rows = True, 8, {0, 1}
    target.verified = [object(), object()]
    target.sources = [SimpleNamespace(seq_length=10, layers=[]), SimpleNamespace(seq_length=20, layers=[])]
    target.start_positions = (10, 20)
    target.sequences = 2
    target.initial_states, target.initial_histories = [[], []], [[], []]
    target.accepted = [SimpleNamespace(), SimpleNamespace()]
    target.layers = ()
    replayed = []
    target.replays = [SimpleNamespace(launch=lambda states, histories, count: replayed.append((0, count))),
                      SimpleNamespace(launch=lambda states, histories, count: replayed.append((1, count)))]
    target.stream = object()
    target.runtime = SimpleNamespace(device="cuda")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: target.stream)
    sessions = [SimpleNamespace(), SimpleNamespace()]
    features = torch.zeros(1, 8, 2)
    contexts = [(sessions[i], target.verified[i], features, list(range(8)), count)
                for i, count in enumerate((2, 4))]
    target.commit(contexts[1])
    assert target.pending and target.pending_rows == {0}
    assert sessions[1].cache.seq_length == 24 and sessions[1].bonus == 3
    with pytest.raises(ValueError, match="outstanding verification"):
        target.commit(contexts[1])
    target.commit(contexts[0])
    assert not target.pending
    assert sessions[0].cache.seq_length == 12 and sessions[0].bonus == 1
    assert replayed == [(1, 4), (0, 2)]


def test_generated_verification_retains_sources_when_rows_swap(monkeypatch):
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    target.pending, target.block, target.sequences = False, 8, 2
    target.invocation, target.stream = object(), object()
    target.layers = (0,)
    target.runtime = SimpleNamespace(
        device="cpu", page_table=SimpleNamespace(page_table=torch.zeros(2, 16)),
        _decode_rope_deltas=torch.zeros(2, 1))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: target.stream)
    target.positions = torch.arange(8, dtype=torch.int32)
    target.inputs = dict(input_ids=torch.zeros(16, dtype=torch.int32),
                         input_pos=torch.zeros(16, dtype=torch.int32),
                         page_table=torch.zeros(2, 16), rope_delta_table=torch.zeros(2, 1),
                         verification_features=torch.zeros(16, 2))
    def cache(value, position):
        return SimpleNamespace(seq_length=position, layers=[SimpleNamespace(
            recurrent_states=torch.tensor([float(value)]), conv_states=torch.tensor([float(value)]))])
    target.accepted = [cache(10, 10), cache(20, 20)]
    target.verified = [cache(0, 0), cache(0, 0)]
    target.initial_states = [[torch.zeros(1)], [torch.zeros(1)]]
    target.initial_histories = [[torch.zeros(1)], [torch.zeros(1)]]
    target.state_destinations = {
        "gdn_recurrent_state": [c.layers[0].recurrent_states for c in target.verified],
        "gdn_conv_state": [c.layers[0].conv_states for c in target.verified]}
    target.launch = lambda: None
    def replay(row, states, histories, count):
        target.accepted[row].layers[0].recurrent_states.copy_(states[0] + count)
        target.accepted[row].layers[0].conv_states.copy_(histories[0] + count)
    target.replays = [SimpleNamespace(launch=lambda s, h, c: replay(0, s, h, c)),
                      SimpleNamespace(launch=lambda s, h, c: replay(1, s, h, c))]
    results = target.target_many([list(range(8))] * 2, list(reversed(target.accepted)), [1, 0])
    sessions = [SimpleNamespace(), SimpleNamespace()]
    for row, (expected, features, verified) in enumerate(results):
        target.commit((sessions[row], verified, features, expected, row + 2))
    assert sessions[0].cache.seq_length == 22
    assert sessions[1].cache.seq_length == 13
    assert sessions[0].cache.layers[0].recurrent_states.item() == 22
    assert sessions[1].cache.layers[0].recurrent_states.item() == 13
    assert not target.pending


def test_generated_verification_repeated_requests():
    target = os.environ.get("QWEN_VERIFICATION_TARGET")
    draft = os.environ.get("QWEN_VERIFICATION_DRAFT")
    if not target or not draft or not torch.cuda.is_available():
        pytest.skip("Qwen target/draft weights and B200 required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 required")

    async def run():
        reference = None
        for path in ("auto", "generated"):
            engine = await InferenceEngine.create(RuntimeConfig(
                model="Qwen/Qwen3.5-27B-FP8", model_path=target, tokenizer_path=target,
                draft_model_path=draft, decode_path=path, max_batch_size=1,
                page_size=1, kv_cache_pages=4096, enable_prefix_cache=False))
            try:
                decoder = engine.runtime.spec.decoder
                if path == "generated":
                    assert decoder._generated_verification is not None
                for _ in range(3):
                    result = await engine.chat(
                        [{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
                        reasoning=False, settings={"temperature": 0, "max_tokens": 32})
                    ids = [token.token_id for token in result.tokens]
                    assert ids
                    if reference is None:
                        reference = ids
                    assert ids == reference
                    if path == "generated":
                        assert decoder._generated_verification.pending is False
                result = await engine.chat(
                    [{"role": "user", "content": "Write a Python function to compute Fibonacci numbers."}],
                    reasoning=False, settings={"temperature": 0, "max_tokens": 32})
                assert result.tokens
            finally:
                await engine.shutdown()

    asyncio.run(run())


def test_generated_verification_long_reasoning_does_not_repeat_zeros():
    target = os.environ.get("QWEN_VERIFICATION_TARGET")
    draft = os.environ.get("QWEN_VERIFICATION_DRAFT")
    if not target or not draft or not torch.cuda.is_available():
        pytest.skip("Qwen target/draft weights and B200 required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 required")

    async def run():
        engine = await InferenceEngine.create(RuntimeConfig(
            model="Qwen/Qwen3.5-27B-FP8", model_path=target, tokenizer_path=target,
            draft_model_path=draft, decode_path="generated", max_batch_size=1,
            page_size=1, kv_cache_pages=32768, enable_prefix_cache=False))
        try:
            for _ in range(2):
                result = await engine.chat(
                    [{"role": "user", "content":
                      "Explain how to derive the quadratic formula by completing the square. "
                      "Work through several examples and discuss numerical stability when implementing it."}],
                    reasoning=True, settings={"temperature": 0, "max_tokens": 16384})
                assert result.output.get("finish_reason") == "stop"
                ids = [token.token_id for token in result.tokens]
                assert ids
                longest_run = max(sum(1 for _ in values) for _, values in groupby(ids))
                assert longest_run < 64, "long reasoning collapsed into repeated tokens"
                assert not engine.runtime.spec.decoder._generated_verification.pending
        finally:
            await engine.shutdown()

    asyncio.run(run())
