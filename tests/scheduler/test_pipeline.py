"""Unit tests for PipelineState state machine.

These tests do not require CUDA - they test pure state machine logic.
"""

import pytest
from dataclasses import dataclass

from kestrel.scheduler.pipeline import (
    PipelineState,
    ForwardHandle,
    InFlightStep,
)


@dataclass
class MockSequence:
    """Mock sequence for testing."""

    seq_id: int = 0
    finalized: bool = False
    inflight_refs: int = 0


@dataclass
class MockTransfer:
    """Mock transfer handle for testing."""

    step_id: int = 0


def make_forward_handle(slot_id: int, num_seqs: int = 1) -> ForwardHandle[MockSequence]:
    """Create a ForwardHandle with mock sequences."""
    return ForwardHandle(
        slot_id=slot_id,
        sequences=[MockSequence(seq_id=i) for i in range(num_seqs)],
    )


def make_step(
    slot_id: int, step_id: int = 0, num_seqs: int = 1
) -> InFlightStep[MockSequence, MockTransfer]:
    """Create an InFlightStep with mock sequences and transfer."""
    return InFlightStep(
        slot_id=slot_id,
        sequences=[MockSequence(seq_id=i) for i in range(num_seqs)],
        transfer=MockTransfer(step_id=step_id),
    )


class TestPipelineStateInit:
    """Tests for PipelineState initialization."""

    def test_init_default_slots(self):
        """Default initialization with 2 slots."""
        state = PipelineState()
        assert state.is_empty()
        assert state.total_in_flight() == 0
        assert state.queue_depth() == 0
        assert state.free_slot_id() == 0
        assert state.completing_step is None

    def test_init_explicit_slots(self):
        """Explicit initialization with 2 slots."""
        state = PipelineState(num_slots=2)
        assert state.is_empty()

    def test_init_invalid_slots_raises(self):
        """Non-2 slot count raises ValueError."""
        with pytest.raises(ValueError, match="exactly 2 slots"):
            PipelineState(num_slots=1)
        with pytest.raises(ValueError, match="exactly 2 slots"):
            PipelineState(num_slots=3)


class TestSlotSelection:
    """Tests for free_slot_id() slot allocation."""

    def test_empty_state_returns_slot_0(self):
        """Empty pipeline returns slot 0."""
        state = PipelineState()
        assert state.free_slot_id() == 0

    def test_forward_on_slot_0_returns_slot_1(self):
        """Forward in-flight on slot 0 returns slot 1."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))
        assert state.free_slot_id() == 1

    def test_forward_on_slot_1_returns_slot_0(self):
        """Forward in-flight on slot 1 returns slot 0."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=1))
        assert state.free_slot_id() == 0

    def test_queued_step_on_slot_0_returns_slot_1(self):
        """Step queued on slot 0 returns slot 1."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))
        assert state.free_slot_id() == 1

    def test_queued_step_on_slot_1_returns_slot_0(self):
        """Step queued on slot 1 returns slot 0."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=1))
        state.on_sampling_complete(make_step(slot_id=1, step_id=0))
        assert state.free_slot_id() == 0

    def test_both_slots_in_use_returns_none(self):
        """Both slots in use returns None."""
        state = PipelineState()

        # Launch and complete sampling on slot 0
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))

        # Launch forward on slot 1
        state.on_forward_launched(make_forward_handle(slot_id=1))

        # Both slots in use: slot 0 in queue, slot 1 in forward
        assert state.free_slot_id() is None

    def test_two_steps_in_queue_returns_none(self):
        """Two steps in queue returns None."""
        state = PipelineState()

        # Step on slot 0
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))

        # Step on slot 1
        state.on_forward_launched(make_forward_handle(slot_id=1))
        state.on_sampling_complete(make_step(slot_id=1, step_id=1))

        assert state.free_slot_id() is None

    def test_completing_step_blocks_slot(self):
        """Slot remains in-use while step is completing."""
        state = PipelineState()

        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))
        assert state.free_slot_id() == 1

        # Pop moves to completing_step, slot still in use
        state.pop_oldest()
        assert state.completing_step is not None
        assert state.free_slot_id() == 1  # slot 0 still blocked

        # Complete frees the slot
        state.on_step_completed()
        assert state.free_slot_id() == 0

    def test_slot_freed_after_full_completion(self):
        """Slot becomes free only after on_step_completed()."""
        state = PipelineState()

        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))

        # Pop but don't complete - slot still blocked
        state.pop_oldest()
        assert state.free_slot_id() == 1

        # Now complete
        state.on_step_completed()
        assert state.free_slot_id() == 0


class TestTwoPhaseCompletion:
    """Tests for the two-phase completion model (pop_oldest + on_step_completed)."""

    def test_pop_oldest_sets_completing_step(self):
        """pop_oldest() moves step to completing_step."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=42))

        step = state.pop_oldest()
        assert step is not None
        assert step.transfer.step_id == 42
        assert state.completing_step is step
        assert state.queue_depth() == 0

    def test_on_step_completed_clears_completing_step(self):
        """on_step_completed() clears completing_step."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0))
        state.pop_oldest()

        assert state.completing_step is not None
        state.on_step_completed()
        assert state.completing_step is None

    def test_pop_oldest_twice_without_complete_raises(self):
        """Cannot pop again while a step is completing."""
        state = PipelineState()

        # Queue two steps
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))
        state.on_forward_launched(make_forward_handle(slot_id=1))
        state.on_sampling_complete(make_step(slot_id=1, step_id=1))

        # Pop first
        state.pop_oldest()

        # Try to pop second without completing first
        with pytest.raises(AssertionError, match="previous step still completing"):
            state.pop_oldest()

    def test_on_step_completed_without_pop_raises(self):
        """Cannot call on_step_completed without pop_oldest."""
        state = PipelineState()

        with pytest.raises(AssertionError, match="no step is currently being completed"):
            state.on_step_completed()

    def test_drain_all_with_completing_step_raises(self):
        """Cannot drain while a step is completing."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0))
        state.pop_oldest()

        with pytest.raises(AssertionError, match="step is currently being completed"):
            state.drain_all()


class TestFIFOCompletion:
    """Tests for FIFO completion order."""

    def test_single_step_pop(self):
        """Single step can be popped."""
        state = PipelineState()

        state.on_forward_launched(make_forward_handle(slot_id=0))
        step = make_step(slot_id=0, step_id=42)
        state.on_sampling_complete(step)

        popped = state.pop_oldest()
        assert popped is not None
        assert popped.transfer.step_id == 42

    def test_fifo_order_two_steps(self):
        """Two steps are completed in FIFO order."""
        state = PipelineState()

        # Launch step 0 on slot 0
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))

        # Launch step 1 on slot 1
        state.on_forward_launched(make_forward_handle(slot_id=1))
        state.on_sampling_complete(make_step(slot_id=1, step_id=1))

        # Pop should return step 0 first (oldest)
        step_0 = state.pop_oldest()
        assert step_0 is not None
        assert step_0.slot_id == 0
        assert step_0.transfer.step_id == 0

        # Complete step 0
        state.on_step_completed()

        # Then step 1
        step_1 = state.pop_oldest()
        assert step_1 is not None
        assert step_1.slot_id == 1
        assert step_1.transfer.step_id == 1

        state.on_step_completed()

        # Queue is now empty
        assert state.pop_oldest() is None

    def test_pop_empty_returns_none(self):
        """Popping empty queue returns None."""
        state = PipelineState()
        assert state.pop_oldest() is None

    def test_peek_does_not_remove(self):
        """Peek returns oldest without removing."""
        state = PipelineState()

        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))

        peeked = state.peek_oldest()
        assert peeked is not None
        assert peeked.transfer.step_id == 0

        # Still in queue
        assert state.queue_depth() == 1
        assert state.completing_step is None

        # Pop returns same step
        popped = state.pop_oldest()
        assert popped is not None
        assert popped.transfer.step_id == 0

    def test_peek_empty_returns_none(self):
        """Peeking empty queue returns None."""
        state = PipelineState()
        assert state.peek_oldest() is None


class TestSlotAssertions:
    """Tests for defensive slot assertions."""

    def test_launch_on_busy_slot_raises(self):
        """Launching forward on an in-use slot raises."""
        state = PipelineState()

        # Put slot 0 in the queue
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0))

        # Try to launch on slot 0 again
        with pytest.raises(AssertionError, match="slot 0 is in use"):
            state.on_forward_launched(make_forward_handle(slot_id=0))

    def test_launch_on_completing_slot_raises(self):
        """Launching forward on a completing slot raises."""
        state = PipelineState()

        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0))
        state.pop_oldest()  # slot 0 now in completing_step

        # Try to launch on slot 0
        with pytest.raises(AssertionError, match="slot 0 is in use"):
            state.on_forward_launched(make_forward_handle(slot_id=0))

    def test_slot_mismatch_on_sampling_complete_raises(self):
        """Mismatched slot_id on sampling complete raises."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))

        # Try to complete with wrong slot_id
        with pytest.raises(AssertionError, match="Slot mismatch"):
            state.on_sampling_complete(make_step(slot_id=1))


class TestStateTransitions:
    """Tests for state transition methods."""

    def test_on_forward_launched_sets_handle(self):
        """on_forward_launched stores the handle."""
        state = PipelineState()
        handle = make_forward_handle(slot_id=0)
        state.on_forward_launched(handle)

        assert state.forward_handle is handle
        assert state.has_forward_in_flight()

    def test_on_forward_launched_twice_raises(self):
        """Launching forward twice without sampling raises."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))

        with pytest.raises(AssertionError, match="already in-flight"):
            state.on_forward_launched(make_forward_handle(slot_id=1))

    def test_on_sampling_complete_clears_handle(self):
        """on_sampling_complete clears forward_handle and adds to queue."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))

        step = make_step(slot_id=0, step_id=0)
        state.on_sampling_complete(step)

        assert state.forward_handle is None
        assert not state.has_forward_in_flight()
        assert state.queue_depth() == 1

    def test_on_sampling_complete_without_forward_raises(self):
        """Completing sampling without forward raises."""
        state = PipelineState()

        with pytest.raises(AssertionError, match="no forward is in-flight"):
            state.on_sampling_complete(make_step(slot_id=0))


class TestQueryMethods:
    """Tests for query/introspection methods."""

    def test_can_launch_forward_empty(self):
        """Can launch forward on empty pipeline."""
        state = PipelineState()
        assert state.can_launch_forward()

    def test_can_launch_forward_with_queued_step(self):
        """Can launch forward when queue has one step."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0))
        assert state.can_launch_forward()

    def test_cannot_launch_forward_with_forward_in_flight(self):
        """Cannot launch forward when forward already in-flight."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))
        assert not state.can_launch_forward()

    def test_cannot_launch_forward_both_slots_busy(self):
        """Cannot launch forward when both slots busy."""
        state = PipelineState()

        # Fill both slots in queue
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0))
        state.on_forward_launched(make_forward_handle(slot_id=1))
        state.on_sampling_complete(make_step(slot_id=1))

        assert not state.can_launch_forward()

    def test_total_in_flight_counts_correctly(self):
        """total_in_flight counts queue + forward (not completing)."""
        state = PipelineState()
        assert state.total_in_flight() == 0

        state.on_forward_launched(make_forward_handle(slot_id=0))
        assert state.total_in_flight() == 1

        state.on_sampling_complete(make_step(slot_id=0))
        assert state.total_in_flight() == 1  # in queue now

        state.on_forward_launched(make_forward_handle(slot_id=1))
        assert state.total_in_flight() == 2

        state.on_sampling_complete(make_step(slot_id=1))
        assert state.total_in_flight() == 2

        state.pop_oldest()
        assert state.total_in_flight() == 1  # completing not counted

        state.on_step_completed()
        assert state.total_in_flight() == 1

        state.pop_oldest()
        state.on_step_completed()
        assert state.total_in_flight() == 0

    def test_is_empty(self):
        """is_empty returns True only when fully drained."""
        state = PipelineState()
        assert state.is_empty()

        state.on_forward_launched(make_forward_handle(slot_id=0))
        assert not state.is_empty()

        state.on_sampling_complete(make_step(slot_id=0))
        assert not state.is_empty()

        state.pop_oldest()
        assert not state.is_empty()  # completing_step is set

        state.on_step_completed()
        assert state.is_empty()


class TestDrainAll:
    """Tests for drain_all() method."""

    def test_drain_empty_returns_empty(self):
        """Draining empty queue returns empty list."""
        state = PipelineState()
        assert state.drain_all() == []

    def test_drain_returns_fifo_order(self):
        """drain_all returns steps in FIFO order (oldest first)."""
        state = PipelineState()

        # Add two steps
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))
        state.on_forward_launched(make_forward_handle(slot_id=1))
        state.on_sampling_complete(make_step(slot_id=1, step_id=1))

        steps = state.drain_all()
        assert len(steps) == 2
        assert steps[0].transfer.step_id == 0  # oldest first
        assert steps[1].transfer.step_id == 1

    def test_drain_clears_queue(self):
        """drain_all clears the queue."""
        state = PipelineState()
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0))

        state.drain_all()
        assert state.queue_depth() == 0
        assert state.pop_oldest() is None

    def test_drain_does_not_affect_forward_handle(self):
        """drain_all only drains queue, not forward_handle."""
        state = PipelineState()

        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0))
        state.on_forward_launched(make_forward_handle(slot_id=1))

        steps = state.drain_all()
        assert len(steps) == 1
        assert state.forward_handle is not None  # still has forward


class TestPingPongScenarios:
    """Integration tests for typical ping-pong scenarios."""

    def test_continuous_ping_pong(self):
        """Continuous ping-pong slot alternation."""
        state = PipelineState()

        for step_num in range(10):
            expected_slot = step_num % 2

            # Complete oldest if queue is full
            if state.queue_depth() == 2:
                state.pop_oldest()
                state.on_step_completed()

            # Launch next step
            slot_id = state.free_slot_id()
            assert slot_id == expected_slot
            state.on_forward_launched(make_forward_handle(slot_id=slot_id))
            state.on_sampling_complete(make_step(slot_id=slot_id, step_id=step_num))

    def test_constrained_path_single_queue_depth(self):
        """Constrained path: commit before finalizing limits queue depth."""
        state = PipelineState()

        for step_num in range(5):
            # Launch forward
            slot_id = state.free_slot_id()
            state.on_forward_launched(make_forward_handle(slot_id=slot_id))

            # Constrained path: commit oldest before finalizing
            if step := state.pop_oldest():
                state.on_step_completed()

            state.on_sampling_complete(make_step(slot_id=slot_id, step_id=step_num))

            # Queue depth should be at most 1 with constrained path
            assert state.queue_depth() <= 1

    def test_unconstrained_path_two_queue_depth(self):
        """Unconstrained path: immediate finalize allows queue depth 2."""
        state = PipelineState()

        # Launch and finalize two steps without committing
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))

        state.on_forward_launched(make_forward_handle(slot_id=1))
        state.on_sampling_complete(make_step(slot_id=1, step_id=1))

        assert state.queue_depth() == 2
        assert state.free_slot_id() is None

        # Must commit to free a slot
        state.pop_oldest()
        state.on_step_completed()
        assert state.free_slot_id() == 0

    def test_prefill_drain_scenario(self):
        """Prefill requires draining entire pipeline."""
        state = PipelineState()

        # Build up some state
        state.on_forward_launched(make_forward_handle(slot_id=0))
        state.on_sampling_complete(make_step(slot_id=0, step_id=0))
        state.on_forward_launched(make_forward_handle(slot_id=1))

        # Prefill wanted: drain queue first
        steps = state.drain_all()
        assert len(steps) == 1

        # Then handle forward (would call finalize_sampling and complete_step)
        assert state.forward_handle is not None

        # Simulate completing the forward
        state.on_sampling_complete(make_step(slot_id=1))
        state.pop_oldest()
        state.on_step_completed()

        assert state.is_empty()
