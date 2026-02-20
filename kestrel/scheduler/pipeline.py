"""Pipeline state machine for pipelined async decoding.

PROBLEM
-------
Without pipelining, the decode loop has a GPU bubble between steps:

    CPU:  [plan] [launch] [====== wait =======] [commit] [plan] [launch] ...
    GPU:         [=== forward/sample ===] [D2H]                 [=== forward ===]
                                               ^
                                          GPU IDLE
                                   (waiting for CPU to plan next step)

The GPU finishes a step, then sits idle while the CPU commits results, plans
the next step, and launches it.

SOLUTION
--------
Pipeline the work so CPU and GPU run concurrently:

    CPU:  [plan t] [launch t] [plan t+1] [launch t+1] [commit t] [plan t+2] ...
    GPU:           [==== forward t ====] [==== forward t+1 ====] [==== forward t+2 ====]
                              ^                       ^
                   CPU plans t+1 while          CPU commits t while
                   GPU runs t                   GPU runs t+1

Key techniques:
- Ping-pong slots: Two sets of GPU buffers (DecodeSlot), alternated each step.
  While one is in use by the GPU, the other can be planned/committed.
- Deferred D2H: Don't wait for GPU->CPU copy immediately. Let it run in the
  background; block only when we actually need the results.
- FIFO completion: Always complete the oldest in-flight step first, ensuring
  correct ordering for zombie handling and grammar state updates.

ARCHITECTURE
------------
This module provides PipelineState, a pure state machine with no CUDA
dependencies. It manages:
- A 2-deep queue of in-flight steps (batch_queue)
- A handle for the current forward pass (launch_handle)
- A step currently being completed (committing_step)
- Slot allocation via bitmask

The scheduler is responsible for:
- Incrementing/decrementing inflight_refs on sequences
- Setting the finalized flag on sequences
- Calling wait() on transfer handles
- Actually executing forward/sampling operations

INVARIANTS
----------
I1: Single shared decode_compute_stream across both slots
    Preserves sequential token dependencies (_pending_* write/read, KV ordering)

I2: FIFO completion (oldest step committed first)
    Zombie skipping, mask computation, D2H ordering all depend on commit order

I3: 0 <= inflight_refs <= 2; only scheduler mutates it
    Prevents double-increment/decrement; PipelineState does queue/slot
    bookkeeping only

I6: Slot reusable only after on_step_completed() (not just pop_oldest())
    Pinned host buffers and per-slot state must not be overwritten before
    commit consumes them. The two-phase completion model enforces this:
    pop_oldest() marks the step as "completing" (slot still in use),
    on_step_completed() frees the slot after the scheduler finishes.

CONSTRAINED VS UNCONSTRAINED BATCHES
------------------------------------
Constrained sequences (grammar/structured decoding) require the mask for
sampling, which depends on grammar state after consuming the previous token.
The key insight: forward doesn't need the mask, only sampling does.

- Unconstrained: forward and sampling run immediately, queue depth can be 2
- Constrained: forward runs ahead, but sampling waits for previous commit
  to update grammar state. This naturally limits effective queue depth.

The unified loop handles both cases - constrained batches have delayed
sampling finalization, unconstrained finalize immediately.

ZOMBIE HANDLING
---------------
When a sequence hits EOS at step t but was already included in step t+1's
plan, it becomes a "zombie" - still present in the batch but already finalized.

Zombie behavior emerges from per-sequence state rather than explicit propagation:
- seq.finalized: True once EOS or length cap reached
- seq.inflight_refs: How many in-flight steps include this sequence (0-2)

On completion, if seq.finalized is already True -> skip committing (zombie).
When inflight_refs drops to 0 for a finalized sequence -> release resources.

PREFILL HANDLING
----------------
Prefill can be represented as `kind="prefill"` launches and pending commits in
this same pipeline. The scheduler owns any policy constraints (e.g., whether it
is safe to interleave prefill and decode steps) and is responsible for preserving
correct FIFO commit order across kinds.
"""

from collections import deque
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar


class SequenceLike(Protocol):
    """Protocol for sequence objects tracked in the pipeline.

    Sequences must expose:
    - finalized: True once EOS or length cap reached
    - inflight_refs: Count of in-flight steps referencing this sequence (0-2)
    """

    @property
    def finalized(self) -> bool: ...

    @property
    def inflight_refs(self) -> int: ...


class TransferLike(Protocol):
    """Protocol for transfer handles.

    Transfer handles represent an in-flight D2H copy. The actual wait() call
    is owned by the scheduler's commit_step() method, not by PipelineState.
    """

    pass


Seq = TypeVar("Seq", bound=SequenceLike)
Transfer = TypeVar("Transfer", bound=TransferLike)

LaunchKind = Literal["decode", "prefill"]


@dataclass(slots=True)
class DecodeLaunch:
    """Payload for decode launches."""

    slot_id: int


@dataclass(slots=True)
class PrefillLaunch:
    """Payload for prefill launches."""

    staging: object
    slot_id: int
    logits: object
    prepared_sequences: object
    prefill_slot: object


@dataclass(slots=True)
class DecodePendingCommit:
    """Payload for decode pending commits."""

    slot_id: int


@dataclass(slots=True)
class PrefillPendingCommit:
    """Payload for prefill pending commits."""

    staging: object
    slot_id: int
    prepared_sequences: object
    prefill_slot: object


@dataclass(slots=True)
class LaunchHandle(Generic[Seq]):
    """Handle for an in-flight forward pass (not yet sampled).

    Lifecycle:
    1. Created by scheduler.launch_forward_async(plan, slot_id)
    2. Stored via pipeline.on_launch(handle)
    3. Consumed by scheduler.finalize_sampling(handle, mask) -> PendingCommit
    4. Cleared via pipeline.on_pending_commit(step)

    For decode launches, the payload includes slot_id indicating which
    ping-pong slot's resources (staging buffers, CUDA graphs, etc.) are used.
    """

    kind: LaunchKind
    sequences: list[Seq]
    payload: DecodeLaunch | PrefillLaunch

    @property
    def slot_id(self) -> int:
        """Return the pipeline slot ID for the launch."""
        return self.payload.slot_id


@dataclass(slots=True)
class PendingCommit(Generic[Seq, Transfer]):
    """Record of a fully-sampled step awaiting completion.

    Lifecycle:
    1. Created by scheduler.finalize_sampling(handle, mask)
    2. Added to queue via pipeline.on_pending_commit(step)
    3. Retrieved via pipeline.pop_oldest() (FIFO order) - slot still in use
    4. Consumed by scheduler.commit_step(step) which calls transfer.wait()
    5. Freed via pipeline.on_step_completed() - slot now available

    The sequences list maintains row order: sequences[i] corresponds to row i
    in the staging/render buffers. This ordering is implicit and must be
    preserved by launch_forward_async and finalize_sampling.
    """

    kind: LaunchKind
    sequences: list[Seq]
    transfer: Transfer
    payload: DecodePendingCommit | PrefillPendingCommit

    @property
    def slot_id(self) -> int:
        """Return the pipeline slot ID for the pending commit."""
        return self.payload.slot_id


class PipelineState(Generic[Seq, Transfer]):
    """State machine for 2-deep pipelined decoding.

    This is the CANONICAL state machine - the engine loop delegates all
    pipeline state transitions here rather than manipulating queues directly.
    This ensures the state machine is testable in isolation (no CUDA) and
    prevents drift between test and production code.

    Slot Management
    ---------------
    Two ping-pong slots (0 and 1) alternate each step. Each slot owns:
    - Per-slot decode workspace
    - Staging buffers for sampled outputs (GPU)
    - Pinned host buffers for D2H copies
    - CUDA graph workspace + captures (if enabled)

    A slot is "in use" if referenced by any of:
    - An entry in batch_queue (sampled, awaiting completion)
    - The current launch_handle (forward dispatched, not yet sampled)
    - The committing_step (popped from queue, completion in progress)

    Two-Phase Completion
    --------------------
    To enforce invariant I6 (slot reusable only after commit_step()):
    - pop_oldest() removes from queue but stores in committing_step
    - Slot remains "in use" during the blocking commit_step() call
    - on_step_completed() clears committing_step, freeing the slot

    This prevents the engine loop from accidentally reusing a slot before
    the scheduler has finished reading from its pinned host buffers.

    Queue Ordering
    --------------
    batch_queue uses appendleft/pop for FIFO semantics:
    - appendleft(new_step) adds newest to front
    - pop() removes oldest from back
    - batch_queue[-1] is always the oldest step

    This ordering is critical for correct zombie handling - step t must be
    committed before step t+1 to ensure grammar state is updated before
    computing the mask for t+1.

    Usage Pattern (Unconstrained)
    -----------------------------
        slot_id = pipeline.free_slot_id()
        handle = scheduler.launch_forward_async(plan, slot_id)
        pipeline.on_launch(handle)

        step = scheduler.finalize_sampling(handle, mask=None)
        pipeline.on_pending_commit(step)

        # Later, when ready to commit:
        oldest = pipeline.pop_oldest()
        scheduler.commit_step(oldest)  # calls transfer.wait()
        pipeline.on_step_completed()

    Usage Pattern (Constrained)
    ---------------------------
        slot_id = pipeline.free_slot_id()
        handle = scheduler.launch_forward_async(plan, slot_id)
        pipeline.on_launch(handle)

        # Must commit previous step before finalizing (need updated grammar)
        if oldest := pipeline.pop_oldest():
            scheduler.commit_step(oldest)
            pipeline.on_step_completed()

        mask = scheduler.compute_mask(handle.sequences)
        step = scheduler.finalize_sampling(handle, mask)
        pipeline.on_pending_commit(step)
    """

    __slots__ = ("batch_queue", "launch_handle", "committing_step", "_num_slots")

    def __init__(self, num_slots: int = 2) -> None:
        """Initialize the pipeline state machine.

        Args:
            num_slots: Number of ping-pong slots. Must be 2 - this is a
                fundamental constraint of the ping-pong design where one
                slot runs on GPU while the other is being committed.

        Raises:
            ValueError: If num_slots != 2.
        """
        if num_slots != 2:
            raise ValueError(
                f"PipelineState currently supports exactly 2 slots, got {num_slots}"
            )
        self.batch_queue: deque[PendingCommit[Seq, Transfer]] = deque()
        self.launch_handle: LaunchHandle[Seq] | None = None
        self.committing_step: PendingCommit[Seq, Transfer] | None = None
        self._num_slots = num_slots

    def _used_slot_mask(self) -> int:
        """Return bitmask of slots currently in use.

        Bit 0 = slot 0 in use, Bit 1 = slot 1 in use.
        """
        mask = 0
        for step in self.batch_queue:
            mask |= 1 << step.slot_id
        if self.launch_handle is not None:
            mask |= 1 << self.launch_handle.slot_id
        if self.committing_step is not None:
            mask |= 1 << self.committing_step.slot_id
        return mask

    def free_slot_id(self) -> int | None:
        """Return ID of a free slot, or None if all slots are in use.

        A slot is in use if referenced by batch_queue, launch_handle,
        or committing_step. Returns the lowest-numbered free slot to
        ensure deterministic alternation pattern.

        Returns:
            0 or 1 if a slot is free, None if both slots are in use.
        """
        mask = self._used_slot_mask()

        # mask=0b00 -> both free, return 0
        # mask=0b01 -> slot 0 in use, return 1
        # mask=0b10 -> slot 1 in use, return 0
        # mask=0b11 -> both in use, return None
        if mask == 0b00 or mask == 0b10:
            return 0
        elif mask == 0b01:
            return 1
        else:
            return None

    def is_slot_free(self, slot_id: int) -> bool:
        """Return True if the given slot_id is not currently referenced.

        This is useful for schedulers that want to select a specific slot (e.g.,
        to launch a prepared prefill) rather than always using free_slot_id().
        """
        if slot_id < 0 or slot_id >= self._num_slots:
            raise ValueError(f"Invalid slot_id {slot_id}")
        used_mask = self._used_slot_mask()
        return (used_mask & (1 << slot_id)) == 0

    def can_launch(self) -> bool:
        """Check if we can launch a new forward pass.

        A forward can be launched iff:
        - No forward is currently in-flight (only one forward at a time)
        - At least one slot is available (not all slots busy)

        Returns:
            True if a forward can be launched.
        """
        return self.launch_handle is None and self.free_slot_id() is not None

    def total_in_flight(self) -> int:
        """Return total number of steps in-flight (0, 1, or 2).

        Includes:
        - Sampled steps in batch_queue awaiting completion
        - Forward pass in-flight (if any)

        Note: committing_step is not counted as it's transitioning out.
        """
        return len(self.batch_queue) + (1 if self.launch_handle else 0)

    def queue_depth(self) -> int:
        """Return number of sampled steps awaiting completion."""
        return len(self.batch_queue)

    def has_launch_in_flight(self) -> bool:
        """Check if a forward pass is in-flight (not yet sampled)."""
        return self.launch_handle is not None

    def is_empty(self) -> bool:
        """Check if the pipeline is fully drained.

        True when no forward in-flight, queue is empty, and no step is
        being completed. Required state before running prefill or
        acknowledging pause.
        """
        return (
            self.launch_handle is None
            and len(self.batch_queue) == 0
            and self.committing_step is None
        )

    # ─────────────────────────────────────────────────────────────────────────
    # State transitions
    # ─────────────────────────────────────────────────────────────────────────

    def on_launch(self, handle: LaunchHandle[Seq]) -> None:
        """Record that a forward pass has been launched.

        Called after scheduler.launch_forward_async() returns successfully.
        The handle is stored until sampling completes.

        Args:
            handle: The LaunchHandle returned by the scheduler.

        Raises:
            AssertionError: If a forward is already in-flight, or if the
                slot is not actually free.
        """
        if self.launch_handle is not None:
            raise AssertionError(
                "Cannot launch forward: another forward is already in-flight"
            )

        # Verify slot is actually free (defensive check)
        used_mask = self._used_slot_mask()
        if used_mask & (1 << handle.slot_id):
            raise AssertionError(
                f"Cannot launch forward: slot {handle.slot_id} is in use "
                f"(used_mask=0b{used_mask:02b})"
            )

        self.launch_handle = handle

    def on_pending_commit(self, step: PendingCommit[Seq, Transfer]) -> None:
        """Record that sampling has completed for the in-flight forward.

        Called after scheduler.finalize_sampling() returns successfully.
        The step is added to the front of the queue (newest first) and
        the launch_handle is cleared.

        Args:
            step: The PendingCommit returned by the scheduler.

        Raises:
            AssertionError: If no forward is in-flight, or if the step's
                slot_id doesn't match the launch_handle's slot_id.
        """
        if self.launch_handle is None:
            raise AssertionError(
                "Cannot complete sampling: no forward is in-flight"
            )

        # Verify slot_id matches (defensive check)
        if step.slot_id != self.launch_handle.slot_id:
            raise AssertionError(
                f"Slot mismatch: step has slot_id={step.slot_id}, "
                f"but launch_handle has slot_id={self.launch_handle.slot_id}"
            )

        self.batch_queue.appendleft(step)
        self.launch_handle = None

    def pop_oldest(self) -> PendingCommit[Seq, Transfer] | None:
        """Pop the oldest step from the queue for completion.

        The step is moved to committing_step, keeping the slot marked as
        in-use until on_step_completed() is called. This enforces invariant
        I6: the slot cannot be reused until the scheduler has finished
        reading from its pinned host buffers.

        Steps MUST be completed in FIFO order (oldest first). This is
        required for correct zombie handling - step t must be committed
        before step t+1 so that:
        - Grammar state is updated before computing mask for t+1
        - Finalized sequences are marked before t+1 commits
        - D2H copies complete in order

        Returns:
            The oldest PendingCommit, or None if queue is empty.

        Raises:
            AssertionError: If a step is already being completed (must call
                on_step_completed() first).
        """
        if not self.batch_queue:
            return None

        if self.committing_step is not None:
            raise AssertionError(
                "Cannot pop: previous step still completing "
                f"(slot {self.committing_step.slot_id}). "
                "Call on_step_completed() first."
            )

        self.committing_step = self.batch_queue.pop()
        return self.committing_step

    def on_step_completed(self) -> None:
        """Mark the current completing step as done, freeing its slot.

        Called after scheduler.commit_step() has finished processing
        the step (waited on transfer, committed tokens, updated state).
        This frees the slot for reuse.

        Raises:
            AssertionError: If no step is currently being completed.
        """
        if self.committing_step is None:
            raise AssertionError(
                "Cannot complete: no step is currently being completed. "
                "Call pop_oldest() first."
            )
        self.committing_step = None

    def peek_oldest(self) -> PendingCommit[Seq, Transfer] | None:
        """Peek at the oldest step without removing it.

        Returns:
            The oldest PendingCommit, or None if queue is empty.
        """
        if self.batch_queue:
            return self.batch_queue[-1]
        return None

    def drain_all(self) -> list[PendingCommit[Seq, Transfer]]:
        """Drain all steps from the queue in completion order (oldest first).

        Used for:
        - Prefill handling: pipeline must be drained before prefill
        - Pause/resume: pipeline must be drained before acknowledging pause
        - Error recovery: drain to clear state after CUDA failure

        Does NOT clear launch_handle or committing_step - caller must
        handle those separately.

        Returns:
            List of PendingCommits in completion order (oldest first).

        Raises:
            AssertionError: If a step is currently being completed.
        """
        if self.committing_step is not None:
            raise AssertionError(
                "Cannot drain: a step is currently being completed. "
                "Call on_step_completed() first."
            )

        steps = []
        while self.batch_queue:
            steps.append(self.batch_queue.pop())
        return steps
