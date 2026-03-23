"""Unit tests for launch-time prefill batch planning."""

from dataclasses import dataclass

from kestrel.moondream.runtime import PrefillClassification
from kestrel.scheduler.scheduler import _PrefillCandidate, _plan_prefill_launch_batch


@dataclass(slots=True)
class _RequestStub:
    request_id: int
    submitted_at: float
    max_new_tokens: int = 10
    adapter: str | None = None


def _candidate(
    request_id: int,
    *,
    submitted_at: float,
    prompt_length: int,
    skip_positions: int,
    can_reuse: bool,
    use_prefix_attn: bool,
    pages_needed: int,
    cohort_key: tuple[str | None, bytes] | None = None,
    max_new_tokens: int = 10,
    adapter: str | None = None,
) -> _PrefillCandidate:
    return _PrefillCandidate(
        request=_RequestStub(
            request_id=request_id,
            submitted_at=submitted_at,
            max_new_tokens=max_new_tokens,
            adapter=adapter,
        ),
        classification=PrefillClassification(
            prompt_length=prompt_length,
            skip_positions=skip_positions,
            can_reuse=can_reuse,
            use_prefix_attn=use_prefix_attn,
        ),
        reserve_length=prompt_length - skip_positions if can_reuse else prompt_length,
        pages_needed=pages_needed,
        cohort_key=cohort_key,
    )


def test_prefill_planner_seeds_prefix_misses_before_harvesting_hits():
    hit = _candidate(
        1,
        submitted_at=10.0,
        prompt_length=768,
        skip_positions=730,
        can_reuse=True,
        use_prefix_attn=False,
        pages_needed=1,
    )
    text_miss = _candidate(
        2,
        submitted_at=11.0,
        prompt_length=120,
        skip_positions=0,
        can_reuse=False,
        use_prefix_attn=False,
        pages_needed=2,
    )
    image_miss = _candidate(
        3,
        submitted_at=1.0,
        prompt_length=768,
        skip_positions=0,
        can_reuse=False,
        use_prefix_attn=True,
        pages_needed=12,
        cohort_key=(None, b"a"),
    )

    batch = _plan_prefill_launch_batch(
        [image_miss, text_miss, hit],
        capacity_remaining=64,
        slot_budget=64,
        page_budget=64,
        token_floor=2048,
    )

    assert [candidate.request.request_id for candidate in batch] == [3]
    assert all(candidate.use_prefix_attn for candidate in batch)


def test_prefill_planner_harvests_hits_first_when_capacity_is_small():
    hit = _candidate(
        1,
        submitted_at=10.0,
        prompt_length=768,
        skip_positions=730,
        can_reuse=True,
        use_prefix_attn=False,
        pages_needed=1,
    )
    image_miss = _candidate(
        2,
        submitted_at=1.0,
        prompt_length=768,
        skip_positions=0,
        can_reuse=False,
        use_prefix_attn=True,
        pages_needed=12,
        cohort_key=(None, b"a"),
    )

    batch = _plan_prefill_launch_batch(
        [image_miss, hit],
        capacity_remaining=16,
        slot_budget=16,
        page_budget=64,
        token_floor=2048,
    )

    assert [candidate.request.request_id for candidate in batch] == [1]
    assert all(not candidate.use_prefix_attn for candidate in batch)


def test_prefill_planner_harvests_hits_when_no_prefix_misses_are_waiting():
    hit = _candidate(
        1,
        submitted_at=10.0,
        prompt_length=768,
        skip_positions=730,
        can_reuse=True,
        use_prefix_attn=False,
        pages_needed=1,
    )
    text_miss = _candidate(
        2,
        submitted_at=11.0,
        prompt_length=120,
        skip_positions=0,
        can_reuse=False,
        use_prefix_attn=False,
        pages_needed=2,
    )

    batch = _plan_prefill_launch_batch(
        [text_miss, hit],
        capacity_remaining=64,
        slot_budget=64,
        page_budget=64,
        token_floor=2048,
    )

    assert [candidate.request.request_id for candidate in batch][:2] == [1, 2]
    assert all(not candidate.use_prefix_attn for candidate in batch)


def test_prefill_planner_limits_miss_batches_to_one_request_per_cohort():
    cohort_a_old = _candidate(
        1,
        submitted_at=1.0,
        prompt_length=768,
        skip_positions=0,
        can_reuse=False,
        use_prefix_attn=True,
        pages_needed=12,
        cohort_key=(None, b"a"),
    )
    cohort_a_new = _candidate(
        2,
        submitted_at=2.0,
        prompt_length=768,
        skip_positions=0,
        can_reuse=False,
        use_prefix_attn=True,
        pages_needed=12,
        cohort_key=(None, b"a"),
    )
    cohort_b = _candidate(
        3,
        submitted_at=3.0,
        prompt_length=768,
        skip_positions=0,
        can_reuse=False,
        use_prefix_attn=True,
        pages_needed=12,
        cohort_key=(None, b"b"),
    )

    batch = _plan_prefill_launch_batch(
        [cohort_a_old, cohort_a_new, cohort_b],
        capacity_remaining=64,
        slot_budget=64,
        page_budget=64,
        token_floor=4096,
    )

    selected_ids = [candidate.request.request_id for candidate in batch]
    assert 1 in selected_ids
    assert 2 not in selected_ids
    assert 3 in selected_ids


def test_prefill_planner_respects_page_budget():
    smallest_hit = _candidate(
        1,
        submitted_at=1.0,
        prompt_length=80,
        skip_positions=60,
        can_reuse=True,
        use_prefix_attn=False,
        pages_needed=1,
    )
    large_hit = _candidate(
        2,
        submitted_at=2.0,
        prompt_length=512,
        skip_positions=0,
        can_reuse=False,
        use_prefix_attn=False,
        pages_needed=4,
    )
    too_large = _candidate(
        3,
        submitted_at=3.0,
        prompt_length=512,
        skip_positions=0,
        can_reuse=False,
        use_prefix_attn=False,
        pages_needed=4,
    )

    batch = _plan_prefill_launch_batch(
        [too_large, large_hit, smallest_hit],
        capacity_remaining=64,
        slot_budget=64,
        page_budget=5,
        token_floor=2048,
    )

    assert [candidate.request.request_id for candidate in batch] == [1, 2]


def test_prefill_planner_keeps_zero_token_prefill_isolated():
    zero_token = _candidate(
        1,
        submitted_at=1.0,
        prompt_length=64,
        skip_positions=32,
        can_reuse=True,
        use_prefix_attn=False,
        pages_needed=1,
        max_new_tokens=0,
    )
    follower = _candidate(
        2,
        submitted_at=2.0,
        prompt_length=96,
        skip_positions=48,
        can_reuse=True,
        use_prefix_attn=False,
        pages_needed=1,
    )

    batch = _plan_prefill_launch_batch(
        [zero_token, follower],
        capacity_remaining=64,
        slot_budget=64,
        page_budget=64,
        token_floor=2048,
    )

    assert [candidate.request.request_id for candidate in batch] == [1]
