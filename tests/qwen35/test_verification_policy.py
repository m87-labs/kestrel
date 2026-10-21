import pytest
from kestrel.models.qwen35.verification_policy import VerificationWidthPolicy


def policy():
    value = VerificationWidthPolicy()
    value.set_target_samples({width: [cost]*5 for width, cost in zip(value.widths, (2., 3., 4., 4.5, 5., 5.5, 6., 9.))})
    value.record_draft(1.)
    return value


def test_full_proposal_until_costs_are_measured():
    value = VerificationWidthPolicy()
    assert value.choose([0.]*15) == 16
    value.set_target_samples({width: [1.]*5 for width in value.widths})
    assert value.choose([0.]*15) == 16
    value.record_draft(1.)
    assert value.choose([0.]*15) == 2


def test_confidence_survival_limits_wasted_verification():
    value = policy()
    assert value.choose([1.]*15) == 16
    assert value.choose([0.]*15) == 2
    assert value.choose([1.]*5 + [0.]*10) == 6
    assert value.choose([1.]*7 + [0.]*8) == 8
    # One uncertain early proposal controls every later accepted prefix.
    assert value.choose([0.] + [1.]*14) == 2
    assert value.choose([float('nan')]*15) == 2


@pytest.mark.parametrize('bad', ('missing', 'few', 'nan', 'negative', 'bool'))
def test_incomplete_costs_do_not_partially_change_policy(bad):
    value = policy()
    before = value.target_samples.copy()
    samples = {width: [1.]*5 for width in value.widths}
    if bad == 'missing': del samples[6]
    elif bad == 'few': samples[4] = [1.]
    elif bad == 'nan': samples[8][0] = float('nan')
    elif bad == 'negative': samples[16][1] = -1.
    else: samples[6][1] = True
    with pytest.raises(ValueError, match='complete positive'):
        value.set_target_samples(samples)
    assert value.target_samples == before


def test_recent_draft_cost_uses_bounded_history():
    value = policy()
    for _ in range(20): value.record_draft(10.)
    assert len(value.draft_samples) == 16
    assert value.choose([1.]*15) == 16
    with pytest.raises(ValueError, match='complete draft'):
        value.choose([1.]*7)


def test_cycle_cost_includes_host_and_commit_overhead():
    value = policy()
    value.record_cycle(4, 8.)
    assert list(value.cycle_overhead_samples) == [3.]
    value.record_cycle(4, 1.)
    assert list(value.cycle_overhead_samples) == [3., 0.]
    value.reset_request()
    assert not value.cycle_overhead_samples
    assert value.calibrated


def test_shared_cycle_overhead_can_favor_a_longer_prefix():
    value = policy()
    before = value.choose([.6]*15)
    value.record_cycle(4, 100.)
    assert value.choose([.6]*15) > before


@pytest.mark.parametrize('width,elapsed', [(9, 1.), (4, -1.), (4, float('nan'))])
def test_invalid_cycle_measurement_does_not_change_history(width, elapsed):
    value = policy()
    with pytest.raises(ValueError, match='valid width'):
        value.record_cycle(width, elapsed)
    assert not value.cycle_overhead_samples


def test_batched_policy_uses_independent_survival_prefixes_and_joint_cost():
    value = VerificationWidthPolicy((2, 4, 8))
    value.set_target_samples({2: [2.] * 5, 4: [3.] * 5, 8: [6.] * 5})
    value.record_draft(1.)
    assert value.choose([0.] * 30, sequences=2) == 2
    assert value.choose([1.] * 30, sequences=2) == 8
    # Uncertainty in one request cannot zero another request's survival prefix.
    assert value.choose([0.] * 15 + [1.] * 15, sequences=2) == 8
    assert value.choose([1.] * 3 + [0.] * 12 + [1.] * 3 + [0.] * 12, sequences=2) == 4
    with pytest.raises(ValueError, match='complete draft'):
        value.choose([1.] * 15, sequences=2)


@pytest.mark.parametrize('widths', [(), (4, 2), (2, 2), (1, 4), (2, 17), (True, 4)])
def test_batched_policy_rejects_invalid_span_inventory(widths):
    with pytest.raises(ValueError, match='verification widths'):
        VerificationWidthPolicy(widths)
