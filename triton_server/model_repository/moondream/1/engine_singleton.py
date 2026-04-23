"""Shared InferenceEngine singleton for Triton Python backend models.

All Triton model instances (one per skill) share a single InferenceEngine and
a dedicated asyncio event loop thread.  The first model to call
``get_or_create_engine()`` creates both; the last to call ``release_engine()``
tears them down.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_engine: Optional["InferenceEngine"] = None  # noqa: F821
_refcount: int = 0
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None


def _start_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Return the shared asyncio event loop (must be called after engine init)."""
    assert _loop is not None, "Engine not initialised"
    return _loop


def get_or_create_engine() -> "InferenceEngine":
    """Return the shared :class:`InferenceEngine`, creating it on first call.

    Thread-safe; subsequent callers simply increment the reference count.
    """
    global _engine, _refcount, _loop, _loop_thread

    with _lock:
        if _engine is not None:
            _refcount += 1
            logger.info("Reusing existing engine (refcount=%d)", _refcount)
            return _engine

        # --- Build RuntimeConfig from env vars ---
        from kestrel.config import RuntimeConfig

        kwargs: dict = {}

        model_path = os.environ.get("KESTREL_MODEL_PATH")
        if model_path:
            kwargs["model_path"] = model_path

        model = os.environ.get("KESTREL_MODEL")
        if model:
            kwargs["model"] = model

        max_batch_size = os.environ.get("KESTREL_MAX_BATCH_SIZE")
        if max_batch_size:
            kwargs["max_batch_size"] = int(max_batch_size)

        runtime_cfg = RuntimeConfig(**kwargs)

        # --- Start asyncio event loop on a background thread ---
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(
            target=_start_event_loop, args=(_loop,), daemon=True, name="kestrel-aio"
        )
        _loop_thread.start()

        # --- Create engine on that loop ---
        from kestrel.engine import InferenceEngine

        future = asyncio.run_coroutine_threadsafe(
            InferenceEngine.create(runtime_cfg), _loop
        )
        _engine = future.result()  # blocks until ready

        _refcount = 1
        logger.info("Engine created (refcount=1)")
        return _engine


def release_engine() -> None:
    """Decrement the reference count; shut down when it reaches zero."""
    global _engine, _refcount, _loop, _loop_thread

    with _lock:
        _refcount -= 1
        logger.info("Engine released (refcount=%d)", _refcount)
        if _refcount > 0:
            return

        engine = _engine
        loop = _loop

        _engine = None
        _loop = None
        _loop_thread = None

    # Shutdown outside the lock to avoid deadlocks.
    if engine is not None and loop is not None:
        future = asyncio.run_coroutine_threadsafe(engine.shutdown(), loop)
        try:
            future.result(timeout=30)
        except Exception:
            logger.exception("Error shutting down engine")
        loop.call_soon_threadsafe(loop.stop)
        logger.info("Engine shut down")
