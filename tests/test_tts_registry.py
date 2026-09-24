"""Public TTS checkpoint discovery."""

from kestrel.models import get_spec


def test_tts_checkpoints_are_builtin() -> None:
    for model in (
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "hexgrad/Kokoro-82M",
    ):
        assert get_spec(model).repo_id == model
