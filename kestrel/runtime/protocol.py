"""Protocols describing the runtime contract used by engine + scheduler.

The runtime contract is a three-layer stack so the engine can host
runtimes with different *execution shapes* behind one model-agnostic
kernel:

- :class:`Runtime` — the minimal surface every runtime exposes
  (identity, device, and which execution shape it is). The engine uses
  it to route work without caring how a model runs.
- :class:`AutoregressiveRuntime` — runtimes that drive a prefill/decode
  loop (KV cache, slots, prefix cache, per-token decode). This is the
  surface the :class:`~kestrel.scheduler.scheduler.GenerationScheduler`
  consumes; ``MoondreamRuntime`` implements it.
- :class:`SinglePassRuntime` — runtimes that fulfill a request with one
  forward (no decode loop).
- :class:`StreamingRuntime` — runtimes that keep model-owned session
  state across caller-supplied chunks (for example video frames).

Some attributes are typed as ``Any`` because they expose model-specific
state the engine + scheduler currently reach into directly (config,
region, spatial decoding tables, prefix cache, slot containers).
Tightening those — or refactoring callers to stop reaching into them —
is a follow-up; declaring them here keeps the protocol an honest
record of the surface a runtime must satisfy today.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from kestrel.runtime.state import (
    PrefillClassification,
    PreparedSequence,
    SequenceState,
)
from kestrel.runtime.tokens import Token

if TYPE_CHECKING:
    from concurrent.futures import Future
    import numpy as np
    import torch
    from torch import Tensor


class ExecutionShape(Enum):
    """How a runtime turns a request into a result.

    The engine owns one executor per shape and routes by this tag, so a
    new shape is an additive change rather than an ``isinstance`` ladder.
    """

    AUTOREGRESSIVE = "autoregressive"
    SINGLE_PASS = "single_pass"
    STREAMING = "streaming"


class Runtime(Protocol):
    """Minimal surface every runtime exposes, regardless of shape.

    The engine uses this to route work and scope prefix-cache entries
    without committing to a particular execution shape.
    """

    # Identity. Stable across the runtime's lifetime; used to scope
    # prefix-cache entries and to route requests in a multi-runtime
    # engine.
    model_name: str

    # Device the runtime executes on.
    device: torch.device

    # Which executor drives this runtime.
    execution_shape: ExecutionShape

    # Release the runtime's resources. Called once by the engine on
    # shutdown for every registered runtime, regardless of shape, so
    # teardown lives on the universal surface (the AR runtime tears down
    # its image preprocessor here; a single-pass runtime may be a no-op).
    def shutdown(self) -> None: ...


class AutoregressiveRuntime(Runtime, Protocol):
    """Surface the scheduler calls on a prefill/decode runtime.

    This is the full contract the
    :class:`~kestrel.scheduler.scheduler.GenerationScheduler` consumes:
    KV-cache capacity, prefill slots, prefix cache, and the
    plan/prefill/decode/commit calls. ``MoondreamRuntime`` implements it.
    """

    # Capacity / shape
    max_batch_size: int
    max_batch_slots: int
    max_seq_length: int
    image_prefix_length: int

    # Streams
    copy_stream: Any

    # Slot containers + sequence registry
    prefill_slots: Sequence[Any]
    decode_slots: Sequence[Any]
    active_sequences: Mapping[int, SequenceState]

    # Engine + scheduler infrastructure
    prefix_cache: Any  # ``RadixPrefixCache | None`` on MoondreamRuntime
    graph_capture_lock: Any  # context manager serializing CUDA-graph capture
    kv_pool: Any  # Engine-owned KV memory pool shared with co-hosted AR runtimes

    # Model-specific state callers reach into today.
    # Narrowing these is a follow-up.
    config: Any
    region: Any
    spatial_tables: Any
    page_table: Any

    # Capacity queries
    def can_reserve(self, total_length: int) -> bool: ...

    def prefill_budget(self) -> tuple[int, int]: ...

    # Image preprocessing. The engine receives raw images (np.ndarray
    # or bytes) and needs them converted to whatever opaque payload the
    # runtime threads through ``GenerationRequest.image_crops`` and
    # back into ``launch_prepared_batch``. Different runtimes need
    # different preprocessing (Moondream's overlap-crop pipeline vs.
    # Gemma's resize+normalize), so the dispatch lives here. Async so
    # the engine can hide preprocessing latency behind admission /
    # other work.
    def preprocess_image_async(
        self, image: np.ndarray | bytes
    ) -> Future[Any]: ...

    # Slot lifecycle
    def acquire_prefill_slot(self, slot_id: int | None = ...) -> Any: ...

    def release_prefill_slot(self, slot: Any) -> None: ...

    def acquire_adapter_slot(self, adapter_id: str, adapter: Any) -> int: ...

    def release_adapter_slot(self, slot: int) -> None: ...

    # Prefill / decode
    def classify_prefill(
        self,
        prompt_tokens: Sequence[Token],
        *,
        has_image: bool,
        image_hash: bytes | None,
        adapter_id: str | None,
    ) -> PrefillClassification: ...

    def prepare_sequence(
        self,
        prompt_tokens: Sequence[Token],
        *,
        image: np.ndarray | None = ...,
        image_crops: Any | None = ...,
        max_new_tokens: int | None = ...,
        lora_slot: int = ...,
        image_hash: bytes | None = ...,
        adapter_id: str | None = ...,
    ) -> PreparedSequence: ...

    def launch_prepared_batch(
        self,
        prepared_sequences: Sequence[PreparedSequence],
        prefill_slot: Any,
        *,
        images: Sequence[np.ndarray | None] | None = ...,
        image_crops_list: Sequence[Any] | None = ...,
    ) -> Tensor: ...

    def finalize_prepared_sequence_after_prefill(
        self, prepared: PreparedSequence
    ) -> None: ...

    def abort_prepared_sequence(self, prepared: PreparedSequence) -> None: ...

    def retain_sequence_prefix(
        self,
        state: SequenceState,
        generated_tokens: Sequence[Token],
        *,
        adapter_id: str | None,
        image_hash: bytes | None,
    ) -> None: ...

    def release_sequence(self, state: SequenceState) -> None: ...

    def decode_with_slot(self, slot: Any, batch_size: int) -> None: ...


class SinglePassRuntime(Runtime, Protocol):
    """Runtime that fulfills a request with a single forward.

    No KV cache, no decode loop. The driver is pure compute: ``forward``
    enqueues the kernels for one forward and returns the result tensors
    *without* a host sync. The engine supplies the compute stream shared
    with the autoregressive lane when it constructs the runtime. The
    executor owns the completion event, slot, and result delivery (the
    driver↔executor split mirrors the autoregressive path, where the
    model computes and the scheduler owns the pipeline).

    A synchronous ``forward`` (one that blocks on its own result) would
    stall the kernel loop and defeat interleaving — implementations must
    not call ``.item()`` / ``.cpu()`` / ``torch.cuda.synchronize`` before
    returning.
    """

    # Capability names this runtime serves, e.g. ("segment_masks",). The
    # engine advertises these through ModelHandle.tasks / supports() and
    # validates run(model, task, ...) against them, so it's part of the
    # contract — declared here, not discovered via getattr.
    def tasks(self) -> Sequence[str]: ...

    def preprocess_image_async(
        self, image: np.ndarray | bytes
    ) -> Future[Any]: ...

    def forward(self, task: str, inputs: Any) -> Any: ...


class StreamingRuntime(Runtime, Protocol):
    """Runtime that advances model-owned state over caller-supplied chunks.

    Streaming runtimes are for non-token models whose state spans a
    sequence of inputs, such as a video tracker that receives an initial
    prompt and then one frame at a time. The runtime owns the semantic
    session state; the engine/executor owns ordering, backpressure,
    completion events, stream updates, and result delivery.

    Like :class:`SinglePassRuntime`, compute methods must enqueue work and
    return without a host sync so the executor can interleave steps with
    other lanes. Implementations must not call ``.item()`` / ``.cpu()`` /
    ``torch.cuda.synchronize`` before returning from ``start`` or
    ``step``.
    """

    # Stream the executor launches session steps on, so streaming steps
    # share the device's serialize-on-stream invariant with the other
    # engine lanes.
    primary_stream: Any

    # Capability names this runtime serves, e.g. ("point",).
    def tasks(self) -> Sequence[str]: ...

    # Create the model-owned session state from the initial prompt. This
    # may enqueue initial GPU work, so it follows the same no-host-sync
    # rule as ``step``.
    def start(self, task: str, inputs: Any) -> Any: ...

    # Advance an existing session with one caller-supplied chunk/frame and
    # return model-defined output plus the next session state.
    def step(self, session: Any, inputs: Any) -> Any: ...

    # Release model-owned session resources. Called once when the engine
    # closes, cancels, or fails the session.
    def finish(self, session: Any) -> None: ...
