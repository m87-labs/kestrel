"""AOT verification session coverage; model paths are supplied on GPU hosts."""

import asyncio
import os
from types import SimpleNamespace

import pytest
import torch

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.models.qwen35.generated_verification import Qwen35GeneratedVerification


def test_generated_verification_rejects_unqualified_concurrency():
    with pytest.raises(ValueError, match="B200 C1"):
        Qwen35GeneratedVerification(SimpleNamespace(max_batch_size=2), None)


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
