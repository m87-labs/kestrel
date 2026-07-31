"""Qwen registration should remain lightweight until the model is selected."""

from __future__ import annotations

import subprocess
import sys


def test_registration_does_not_import_model_implementation() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import kestrel; "
                "from kestrel.models import get_spec; "
                "spec = get_spec('Qwen/Qwen3.5-4B'); "
                "assert spec.filename is None; "
                "assert 'kestrel.models.qwen35.qwen_model' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
