from __future__ import annotations

import json

import pytest

from kestrel.models.whisper.assets import (
    CHECKPOINT_REVISION,
    REPO_ID,
    WhisperAssets,
)
from kestrel.models.whisper.config import (
    UnsupportedWhisperConfig,
    WhisperPreprocessorConfig,
    WhisperTurboConfig,
)


def test_exact_turbo_config_is_accepted(turbo_config_dict) -> None:
    config = WhisperTurboConfig.from_dict(turbo_config_dict)
    assert config.encoder_layers == 32
    assert config.decoder_layers == 4
    assert config.decoder_head_dim == 64
    assert config.tie_word_embeddings is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "wav2vec2"),
        ("decoder_layers", 32),
        ("num_mel_bins", 80),
        ("vocab_size", 51865),
        ("activation_function", "gelu_new"),
    ],
)
def test_architecture_drift_fails_closed(turbo_config_dict, field, value) -> None:
    turbo_config_dict[field] = value
    with pytest.raises(UnsupportedWhisperConfig, match=field):
        WhisperTurboConfig.from_dict(turbo_config_dict)


def test_preprocessor_geometry_is_exact(preprocessor_config_dict) -> None:
    config = WhisperPreprocessorConfig.from_dict(preprocessor_config_dict)
    assert config.n_samples == 480000
    assert config.nb_max_frames == 3000

    preprocessor_config_dict["hop_length"] = 320
    with pytest.raises(UnsupportedWhisperConfig, match="hop_length"):
        WhisperPreprocessorConfig.from_dict(preprocessor_config_dict)


def test_local_assets_never_fall_back_to_hub(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assets = WhisperAssets(local_dir=tmp_path)
    assert assets.path("config.json") == tmp_path / "config.json"
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        assets.path("tokenizer.json")
    with pytest.raises(ValueError, match="Undeclared"):
        assets.path("../config.json")


def test_remote_assets_use_the_immutable_revision(monkeypatch, tmp_path) -> None:
    downloaded = tmp_path / "generation_config.json"
    downloaded.write_text(json.dumps({"ok": True}), encoding="utf-8")
    calls = []

    def fake_download(repo_id, *, filename, revision):
        calls.append((repo_id, filename, revision))
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assets = WhisperAssets()
    assert assets.load_json("generation_config.json") == {"ok": True}
    assert calls == [(REPO_ID, "generation_config.json", CHECKPOINT_REVISION)]
