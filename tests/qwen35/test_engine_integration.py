"""End-to-end engine query through Qwen35Runtime."""

from __future__ import annotations

import pytest
import torch

import kestrel.models.qwen35  # noqa: F401
from kestrel.config import RuntimeConfig
from kestrel.engine import EngineResult, InferenceEngine


_MODEL_ID = "Qwen/Qwen3.5-2B"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_engine_query_returns_text() -> None:
    import asyncio

    async def run() -> EngineResult:
        cfg = RuntimeConfig(device="cuda", model=_MODEL_ID, max_batch_size=1)
        engine = await InferenceEngine.create(cfg)
        try:
            return await engine.query(
                image=None,
                question="What is 2+2?",
                reasoning=False,
                settings={"max_tokens": 64},
            )
        finally:
            await engine.shutdown()

    result = asyncio.run(run())
    answer = result.output.get("answer")
    assert isinstance(answer, str)
    assert answer.strip(), f"expected non-empty answer; got {answer!r}"
    assert "4" in answer, f"expected a numeric answer about 2+2; got {answer!r}"
    assert result.metrics.output_tokens < 64
    print(f"\n[qwen35 engine answer] {answer!r}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_engine_image_query_returns_text() -> None:
    np = pytest.importorskip("numpy")
    import asyncio

    async def run() -> EngineResult:
        cfg = RuntimeConfig(device="cuda", model=_MODEL_ID, max_batch_size=1)
        engine = await InferenceEngine.create(cfg)
        try:
            return await engine.query(
                image=np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8),
                question="What color is the image?",
                reasoning=False,
                settings={"max_tokens": 64},
            )
        finally:
            await engine.shutdown()

    result = asyncio.run(run())
    answer = result.output.get("answer")
    assert isinstance(answer, str)
    assert answer.strip(), f"expected non-empty answer; got {answer!r}"
    assert "red" in answer.lower(), f"expected an answer about red; got {answer!r}"
    assert result.metrics.output_tokens < 64
    print(f"\n[qwen35 engine image answer] {answer!r}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_engine_batches_distinct_qwen_queries() -> None:
    import asyncio

    async def run() -> tuple[EngineResult, EngineResult]:
        cfg = RuntimeConfig(device="cuda", model=_MODEL_ID, max_batch_size=2)
        engine = await InferenceEngine.create(cfg)
        try:
            first, second = await asyncio.gather(
                engine.query(
                    image=None,
                    question="Answer with just the number. What is 2+2?",
                    reasoning=False,
                    settings={"max_tokens": 16},
                ),
                engine.query(
                    image=None,
                    question="Answer with just the number. What is 3+5?",
                    reasoning=False,
                    settings={"max_tokens": 16},
                ),
            )
            return first, second
        finally:
            await engine.shutdown()

    first, second = asyncio.run(run())
    first_answer = str(first.output.get("answer", "")).strip()
    second_answer = str(second.output.get("answer", "")).strip()
    assert "4" in first_answer, f"expected 2+2 answer; got {first_answer!r}"
    assert "8" in second_answer, f"expected 3+5 answer; got {second_answer!r}"
    assert "user" not in first_answer.lower()
    assert "user" not in second_answer.lower()
