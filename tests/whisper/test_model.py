from __future__ import annotations

import torch
from safetensors.torch import save_file

from kestrel.models.whisper.weights import load_whisper_safetensors

from .eager_model import WhisperInferenceModel


def _eager_model(
    tmp_path,
    config,
    checkpoint_tensors,
) -> WhisperInferenceModel:
    path = tmp_path / "tiny-whisper.safetensors"
    save_file(checkpoint_tensors, path, metadata={"format": "pt"})
    weights = load_whisper_safetensors(
        path,
        config,
        checkpoint_dtype=torch.float32,
    )
    return WhisperInferenceModel(config, weights)


def _inputs(config) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.linspace(
        -0.75,
        0.75,
        steps=2 * config.num_mel_bins * config.max_source_positions * 2,
        dtype=torch.float32,
    ).view(2, config.num_mel_bins, config.max_source_positions * 2)
    token_ids = torch.tensor([[3, 4, 5], [3, 6, 7]], dtype=torch.long)
    return features, token_ids


def test_encoder_decoder_and_cross_projection_match_transformers(
    tiny_whisper_config,
    tiny_transformers_model,
    tiny_checkpoint_tensors,
    tmp_path,
) -> None:
    eager = _eager_model(
        tmp_path,
        tiny_whisper_config,
        tiny_checkpoint_tensors,
    )
    features, token_ids = _inputs(tiny_whisper_config)

    with torch.inference_mode():
        oracle_encoder = tiny_transformers_model.model.encoder(
            features,
            return_dict=True,
        ).last_hidden_state
        oracle_decoder = tiny_transformers_model.model.decoder(
            input_ids=token_ids,
            encoder_hidden_states=oracle_encoder,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        oracle_logits = tiny_transformers_model.proj_out(oracle_decoder)

    encoder = eager.encode(features)
    torch.testing.assert_close(encoder, oracle_encoder, rtol=1e-5, atol=1e-6)

    cross_kv = eager.preproject_cross_kv(encoder)
    expected_shape = (
        tiny_whisper_config.decoder_layers,
        features.shape[0],
        tiny_whisper_config.max_source_positions,
        tiny_whisper_config.decoder_attention_heads,
        tiny_whisper_config.decoder_head_dim,
    )
    assert cross_kv.keys.shape == expected_shape
    assert cross_kv.values.shape == expected_shape
    with torch.inference_mode():
        for index, layer in enumerate(tiny_transformers_model.model.decoder.layers):
            expected_key = layer.encoder_attn.k_proj(oracle_encoder).view(
                features.shape[0],
                tiny_whisper_config.max_source_positions,
                tiny_whisper_config.decoder_attention_heads,
                tiny_whisper_config.decoder_head_dim,
            )
            expected_value = layer.encoder_attn.v_proj(oracle_encoder).view_as(
                expected_key
            )
            torch.testing.assert_close(
                cross_kv.keys[index], expected_key, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                cross_kv.values[index], expected_value, rtol=1e-5, atol=1e-6
            )
            assert cross_kv.layer(index)[0].is_contiguous()

    output = eager.decoder_prefix(token_ids, cross_kv)
    torch.testing.assert_close(
        output.hidden_states,
        oracle_decoder,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(output.logits, oracle_logits, rtol=2e-5, atol=2e-6)
    assert output.self_kv.length == token_ids.shape[1]
