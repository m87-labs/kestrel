from collections import OrderedDict
from types import SimpleNamespace
import torch
import pytest
from kestrel.models.qwen35.query_metadata import build_query_metadata


def test_workspace_reuses_shape_not_owner_and_bounds_retirement():
    # The same caller uses the explicit CPU implementation for contract tests.
    table = torch.arange(6*24, dtype=torch.int32).reshape(6,24)
    decoder = SimpleNamespace(runtime=SimpleNamespace(device=torch.device('cpu'),
        page_table=SimpleNamespace(page_table=table), page_size=4, max_batch_size=2),
        _query_metadata_workspaces=OrderedDict())
    first = build_query_metadata(decoder, (2,3), (5,1), (3,7))
    positions = first.positions.clone()
    second = build_query_metadata(decoder, (2,3), (1,4), (8,11))
    assert first is second
    assert second.positions.tolist() == [[8,9,11,12,13]]
    assert positions.tolist() == [[3,4,7,8,9]]
    build_query_metadata(decoder, (1,), (5,), (0,))
    build_query_metadata(decoder, (4,), (2,), (1,))
    assert len(decoder._query_metadata_workspaces) == 2
    assert all(key[0] != (2,3) for key in decoder._query_metadata_workspaces)


def test_workspace_rejected_host_rows_do_not_overwrite_live_outputs():
    table = torch.arange(32, dtype=torch.int32).reshape(4,8)
    decoder = SimpleNamespace(runtime=SimpleNamespace(device=torch.device('cpu'),
        page_table=SimpleNamespace(page_table=table), page_size=4, max_batch_size=2),
        _query_metadata_workspaces=OrderedDict())
    plan = build_query_metadata(decoder, (2,), (1,), (0,))
    old = plan.slots.clone()
    with pytest.raises(ValueError):
        build_query_metadata(decoder, (2,), (9,), (0,))
    torch.testing.assert_close(plan.slots, old)
