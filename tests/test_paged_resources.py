import pytest

from kestrel.runtime.paged_resources import bound_kv_cache_pages


def test_bound_kv_cache_pages_caps_unreachable_capacity() -> None:
    assert bound_kv_cache_pages(
        65_536,
        page_size=1,
        max_batch_size=1,
        max_seq_length=2_048,
    ) == 2_050


def test_bound_kv_cache_pages_accounts_for_batch_and_page_size() -> None:
    assert bound_kv_cache_pages(
        65_536,
        page_size=16,
        max_batch_size=4,
        max_seq_length=2_049,
    ) == 518


def test_bound_kv_cache_pages_preserves_tighter_user_limit() -> None:
    assert bound_kv_cache_pages(
        1_024,
        page_size=1,
        max_batch_size=4,
        max_seq_length=2_048,
    ) == 1_024


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("page_size", 0),
        ("max_batch_size", 0),
        ("max_seq_length", 0),
    ),
)
def test_bound_kv_cache_pages_rejects_non_positive_inputs(
    field: str,
    value: int,
) -> None:
    kwargs = {
        "requested_pages": 64,
        "page_size": 1,
        "max_batch_size": 1,
        "max_seq_length": 32,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="must be positive"):
        bound_kv_cache_pages(**kwargs)


@pytest.mark.parametrize("requested_pages", (0, 1))
def test_bound_kv_cache_pages_requires_both_reserved_pages(
    requested_pages: int,
) -> None:
    with pytest.raises(ValueError, match="reserved and padding pages"):
        bound_kv_cache_pages(
            requested_pages,
            page_size=1,
            max_batch_size=1,
            max_seq_length=32,
        )
