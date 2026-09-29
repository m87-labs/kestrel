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
assert "Qwen/Qwen3.8-27B" in names
assert "Qwen/Qwen3.8-27B-FP8" in names
assert "moondream3.1-9B-A2B" in names
assert "nvidia/parakeet-tdt-0.6b-v3" in names
assert "hexgrad/Kokoro-82M" in names
assert "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" in names
for family in ("moondream", "qwen35", "gemma4", "qwen3_asr", "parakeet_tdt", "whisper", "kokoro", "qwen3_tts"):
    assert f"kestrel.models.{family}.runtime" not in sys.modules, family
''')


@pytest.mark.parametrize("name,family,runtime_name", [
    ("Qwen/Qwen3.5-27B-FP8", "qwen35", "Qwen35Runtime"),
    ("Qwen/Qwen3.8-27B", "qwen35", "Qwen35Runtime"),
    ("Qwen/Qwen3.8-27B-FP8", "qwen35", "Qwen35Runtime"),
    ("google/gemma-4-31B-it", "gemma4", "Gemma4Runtime"),
    ("moondream3.1-9B-A2B", "moondream", "MoondreamRuntime"),
    ("Qwen/Qwen3-ASR-0.6B", "qwen3_asr", "Qwen3AsrRuntime"),
    ("nvidia/parakeet-tdt-0.6b-v3", "parakeet_tdt", "ParakeetTdtRuntime"),
    ("hexgrad/Kokoro-82M", "kokoro", "create_kokoro_runtime"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "qwen3_tts", "Qwen3TTSRuntime"),
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
for other in ("moondream", "qwen35", "gemma4", "qwen3_asr", "parakeet_tdt", "whisper", "kokoro", "qwen3_tts"):
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


def test_custom_speech_registration_survives_sibling_lookup():
    _fresh('''
from kestrel.models.registry import ModelSpec, register, get_spec
name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
custom = ModelSpec(name=name, runtime=lambda: None)
register(custom)
assert get_spec(name) is custom
assert "kestrel.models.qwen3_tts.runtime" not in sys.modules
get_spec("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
assert get_spec(name) is custom
''')


def test_speech_exports_remain_available_on_demand():
    _fresh('''
from kestrel.models import kokoro, qwen3_tts, get_spec
assert "kestrel.models.kokoro.weights" not in sys.modules
assert kokoro.DEFAULT_KOKORO_MODEL == "hexgrad/Kokoro-82M"
assert kokoro.KokoroRuntime.__name__ == "KokoroRuntime"
spec = get_spec(kokoro.DEFAULT_KOKORO_MODEL)
assert spec.runtime is kokoro.create_kokoro_runtime
assert spec.repo_id == kokoro.DEFAULT_KOKORO_REPO_ID
assert spec.revision == kokoro.DEFAULT_KOKORO_REVISION
for name, revision in qwen3_tts.SUPPORTED_CHECKPOINTS.items():
    spec = get_spec(name)
    assert spec.runtime is qwen3_tts.Qwen3TTSRuntime
    assert spec.repo_id == name and spec.revision == revision
for module in (kokoro, qwen3_tts):
    try:
        module.not_an_export
    except AttributeError:
        pass
    else:
        raise AssertionError("unknown attribute accepted")
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
