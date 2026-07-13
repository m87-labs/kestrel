import importlib
import inspect

import pytest


def test_pinned_kernels_exposes_megakernel_runtime_api() -> None:
    """The engine integration must not install against a pre-megakernel wheel."""
    try:
        megakernel = importlib.import_module("kestrel_kernels.megakernel")
    except (ImportError, ModuleNotFoundError):
        pytest.fail(
            "the pinned kestrel-kernels wheel has no megakernel runtime API; "
            "publish and pin the wheel paired with the engine integration"
        )

    required = {
        "decode",
        "deploy_target",
        "has_megakernel",
        "launch_count",
    }
    missing = sorted(required - set(dir(megakernel)))
    assert not missing, f"kestrel-kernels megakernel API is missing {missing}"
    decode_parameters = inspect.signature(megakernel.decode).parameters
    assert "batch_idx_cpu" in decode_parameters
    assert "input_pos_cpu" in decode_parameters
