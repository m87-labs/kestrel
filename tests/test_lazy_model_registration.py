"""Discovering models must not construct their implementation import graphs."""

import subprocess
import sys

import pytest


def _fresh(code):
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path[:] = {sys.path!r}\n" + code],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_registry_discovery_keeps_implementations_unloaded():
    _fresh('''
from kestrel.models import known_models
names = known_models()
assert names == sorted(set(names))
assert "Qwen/Qwen3.5-27B-FP8" in names
assert "moondream3.1-9B-A2B" in names
assert "nvidia/parakeet-tdt-0.6b-v3" in names
for family in ("moondream", "qwen35", "gemma4", "qwen3_asr", "parakeet_tdt", "whisper"):
    assert f"kestrel.models.{family}.runtime" not in sys.modules, family
''')


@pytest.mark.parametrize("name,family,runtime_name", [
    ("Qwen/Qwen3.5-27B-FP8", "qwen35", "Qwen35Runtime"),
    ("google/gemma-4-31B-it", "gemma4", "Gemma4Runtime"),
    ("moondream3.1-9B-A2B", "moondream", "MoondreamRuntime"),
    ("Qwen/Qwen3-ASR-0.6B", "qwen3_asr", "Qwen3AsrRuntime"),
    ("nvidia/parakeet-tdt-0.6b-v3", "parakeet_tdt", "ParakeetTdtRuntime"),
])
def test_lookup_resolves_only_selected_runtime(name, family, runtime_name):
    _fresh(f'''
from importlib import import_module
from kestrel.models import get_spec, known_models
before = known_models()
spec = get_spec({name!r})
assert spec.runtime is getattr(import_module("kestrel.models." + {family!r}), {runtime_name!r})
assert get_spec({name!r}) is spec
assert known_models() == before
for other in ("moondream", "qwen35", "gemma4", "qwen3_asr", "parakeet_tdt", "whisper"):
    if other != {family!r}:
        assert f"kestrel.models.{{other}}.runtime" not in sys.modules, other
''')


def test_custom_registration_can_override_unloaded_builtin():
    _fresh('''
from kestrel.models.registry import ModelSpec, register, get_spec
spec = ModelSpec(name="Qwen/Qwen3.5-27B-FP8", runtime=lambda: None)
register(spec)
assert get_spec(spec.name) is spec
assert "kestrel.models.qwen35.runtime" not in sys.modules
get_spec("Qwen/Qwen3.5-9B")
assert get_spec(spec.name) is spec
''')


def test_unknown_model_reports_lazy_names():
    _fresh('''
from kestrel.models import get_spec
try:
    get_spec("not-a-model")
except ValueError as exc:
    assert "Qwen/Qwen3.5-27B-FP8" in str(exc)
else:
    raise AssertionError("unknown model accepted")
assert "kestrel.models.qwen35.runtime" not in sys.modules
''')


def test_every_advertised_model_resolves_without_changing_inventory():
    _fresh('''
from kestrel.models import get_spec, known_models
names = known_models()
for name in names:
    spec = get_spec(name)
    assert spec.name == name
    assert callable(spec.runtime)
assert known_models() == names
''')
