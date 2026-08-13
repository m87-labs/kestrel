from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from kestrel.runtime.generated_decode import GeneratedDecode


def test_try_create_binds_physical_sm_count_to_program_resolution():
    runtime = SimpleNamespace(
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
    )
    spec = SimpleNamespace(
        bindings=SimpleNamespace(is_eligible=lambda value: value is runtime),
        weight_root=Mock(),
        weight_layer_prefix="model.layers",
    )
    properties = SimpleNamespace(major=10, minor=0, multi_processor_count=148)

    with (
        patch("torch.cuda.get_device_properties", return_value=properties),
        patch(
            "kestrel_kernels.generated_decode.resolve_compatible_programs",
            return_value=(),
        ) as resolve,
    ):
        assert GeneratedDecode.try_create(runtime, spec) is None

    resolve.assert_called_once_with(
        spec.weight_root,
        layer_prefix="model.layers",
        arch="sm100",
        device_sms=148,
    )
