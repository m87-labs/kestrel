import pytest
import torch

from kestrel.runtime.prefill import (
    gather_padded_last_rows,
    project_padded_last_rows,
)


def test_gather_padded_last_rows_uses_each_segment_extent():
    hidden = torch.arange(3 * 5 * 2).reshape(3, 5, 2)

    actual = gather_padded_last_rows(hidden, (2, 5, 3))

    expected = torch.stack((hidden[0, 1], hidden[1, 4], hidden[2, 2]))
    torch.testing.assert_close(actual, expected)


def test_project_padded_last_rows_issues_one_batched_projection():
    calls = []

    class Projection(torch.nn.Module):
        def forward(self, rows):
            calls.append(tuple(rows.shape))
            return rows + 1

    hidden = torch.arange(3 * 5 * 2).reshape(3, 5, 2)

    rows, projected = project_padded_last_rows(
        hidden,
        (2, 5, 3),
        Projection(),
    )

    assert calls == [(3, 2)]
    torch.testing.assert_close(projected, rows + 1)


@pytest.mark.parametrize(
    ("shape", "lengths", "message"),
    (
        ((2, 3), (1, 1), "shape"),
        ((2, 3, 4), (1,), "match batch"),
        ((2, 3, 4), (0, 2), "lie in"),
        ((2, 3, 4), (1, 4), "lie in"),
    ),
)
def test_gather_padded_last_rows_rejects_invalid_domains(
    shape,
    lengths,
    message,
):
    with pytest.raises(ValueError, match=message):
        gather_padded_last_rows(torch.zeros(shape), lengths)


@pytest.mark.parametrize("lengths", ((True, 1), (1.5, 1)))
def test_gather_padded_last_rows_rejects_non_integer_extents(lengths):
    with pytest.raises(TypeError, match="must be integers"):
        gather_padded_last_rows(torch.zeros(2, 3, 4), lengths)
