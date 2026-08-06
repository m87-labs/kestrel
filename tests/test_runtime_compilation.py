from unittest.mock import Mock

from kestrel.runtime.compilation import materialize_dynamic_batch_domain


def test_materializes_every_admitted_batch() -> None:
    observed: list[int] = []
    synchronize = Mock()

    def compiled(batch_size: int) -> None:
        observed.append(batch_size)

    materialize_dynamic_batch_domain(
        compiled,
        max_batch_size=4,
        inputs_for_batch=lambda batch_size: (batch_size,),
        synchronize=synchronize,
    )

    assert observed == [1, 4, 2, 3]
    synchronize.assert_called_once_with()
