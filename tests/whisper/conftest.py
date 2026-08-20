from __future__ import annotations

import pytest
import torch

from kestrel.models.whisper.config import WhisperTurboConfig
from kestrel.models.whisper.tokenizer import SUPPORTED_LANGUAGE_CODES


@pytest.fixture
def turbo_config_dict() -> dict[str, object]:
    return {
        "model_type": "whisper",
        "architectures": ["WhisperForConditionalGeneration"],
        "is_encoder_decoder": True,
        "d_model": 1280,
        "encoder_layers": 32,
        "decoder_layers": 4,
        "encoder_attention_heads": 20,
        "decoder_attention_heads": 20,
        "encoder_ffn_dim": 5120,
        "decoder_ffn_dim": 5120,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": 128,
        "vocab_size": 51866,
        "pad_token_id": 50257,
        "bos_token_id": 50257,
        "eos_token_id": 50257,
        "decoder_start_token_id": 50258,
        "activation_function": "gelu",
        "scale_embedding": False,
        "use_cache": True,
        "num_hidden_layers": 32,
        "tie_word_embeddings": True,
        # Harmless training/reference metadata is intentionally ignored.
        "dropout": 0.0,
        "transformers_version": "test",
    }


@pytest.fixture
def preprocessor_config_dict() -> dict[str, object]:
    return {
        "feature_extractor_type": "WhisperFeatureExtractor",
        "sampling_rate": 16000,
        "chunk_length": 30,
        "n_fft": 400,
        "hop_length": 160,
        "feature_size": 128,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": False,
    }


@pytest.fixture
def generation_config_dict() -> dict[str, object]:
    return {
        "bos_token_id": 50257,
        "decoder_start_token_id": 50258,
        "eos_token_id": 50257,
        "pad_token_id": 50257,
        "prev_sot_token_id": 50362,
        "no_timestamps_token_id": 50364,
        "max_initial_timestamp_index": 50,
        "max_length": 448,
        "is_multilingual": True,
        "return_timestamps": False,
        "alignment_heads": [[2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14]],
        "forced_decoder_ids": [[1, None], [2, 50360]],
        "task_to_id": {"translate": 50359, "transcribe": 50360},
        "lang_to_id": {
            f"<|{code}|>": 50259 + index
            for index, code in enumerate(SUPPORTED_LANGUAGE_CODES)
        },
        "begin_suppress_tokens": [220, 50257],
        "suppress_tokens": [1, 2, 7, 50359, 50360, 50361, 50362, 50363],
    }


@pytest.fixture
def tiny_whisper_config() -> WhisperTurboConfig:
    return WhisperTurboConfig(
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=8,
        max_target_positions=12,
        num_mel_bins=4,
        vocab_size=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=3,
    )


@pytest.fixture
def tiny_transformers_model(tiny_whisper_config):
    transformers = pytest.importorskip("transformers", minversion="4.56.0")
    hf_config = transformers.WhisperConfig(
        vocab_size=tiny_whisper_config.vocab_size,
        num_mel_bins=tiny_whisper_config.num_mel_bins,
        d_model=tiny_whisper_config.d_model,
        encoder_layers=tiny_whisper_config.encoder_layers,
        decoder_layers=tiny_whisper_config.decoder_layers,
        encoder_attention_heads=tiny_whisper_config.encoder_attention_heads,
        decoder_attention_heads=tiny_whisper_config.decoder_attention_heads,
        encoder_ffn_dim=tiny_whisper_config.encoder_ffn_dim,
        decoder_ffn_dim=tiny_whisper_config.decoder_ffn_dim,
        max_source_positions=tiny_whisper_config.max_source_positions,
        max_target_positions=tiny_whisper_config.max_target_positions,
        pad_token_id=tiny_whisper_config.pad_token_id,
        bos_token_id=tiny_whisper_config.bos_token_id,
        eos_token_id=tiny_whisper_config.eos_token_id,
        decoder_start_token_id=tiny_whisper_config.decoder_start_token_id,
        activation_function="gelu",
        scale_embedding=False,
        tie_word_embeddings=True,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
    )
    hf_config._attn_implementation = "eager"
    model = transformers.WhisperForConditionalGeneration(hf_config).eval()
    model.tie_weights()
    generator = torch.Generator().manual_seed(20260809)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
                * 0.05
            )
    return model


@pytest.fixture
def tiny_checkpoint_tensors(tiny_transformers_model):
    return {
        name: tensor.detach().clone().contiguous()
        for name, tensor in tiny_transformers_model.state_dict().items()
        if name != "proj_out.weight"
    }
