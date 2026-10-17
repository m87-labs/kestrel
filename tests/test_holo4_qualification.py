import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'holo4_qualification', Path(__file__).parents[1] / 'scripts/holo4/qualify_concurrency.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cohort_decode_counts_only_shared_interval_without_overlapping_time():
    result = module.cohort_decode_window([
        {'decode_events': [(1., 1), (2., 2), (4., 3)]},
        {'decode_events': [(2., 5), (3., 4), (5., 2)]},
    ])
    assert result['elapsed_s'] == 2
    assert result['output_tokens'] == 7
    assert result['output_tokens_s'] == 3.5
    assert result['active_requests'] == 2


@pytest.mark.parametrize('rows', [[], [{'decode_events': []}],
    [{'decode_events': [(1., 1)]}],
    [{'decode_events': [(1., 1), (2., 1)]},
     {'decode_events': [(3., 1), (4., 1)]}]])
def test_cohort_decode_rejects_missing_or_nonoverlapping_output(rows):
    with pytest.raises(ValueError):
        module.cohort_decode_window(rows)
