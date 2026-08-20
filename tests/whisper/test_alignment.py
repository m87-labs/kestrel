from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from kestrel.models.whisper.alignment import (
    AlignedWord,
    _decoder_alignment_matrix,
    _dtw,
    _median_filter,
    _merge_punctuation,
    _word_grouping_language,
    no_speech_probability,
)
from kestrel.models.whisper.tokenizer import WhisperControlTokens
from kestrel.models.whisper.weights import load_whisper_safetensors


def test_dtw_follows_the_unshifted_low_cost_path() -> None:
    costs = np.full((3, 6), 10.0, dtype=np.float32)
    costs[0, :2] = 0.0
    costs[1, 2:4] = 0.0
    costs[2, 4:] = 0.0

    token_indices, frame_indices = _dtw(costs)

    assert token_indices[0] == 0
    assert token_indices[-1] == 2
    jumps = np.pad(np.diff(token_indices), (1, 0), constant_values=1).astype(bool)
    assert frame_indices[jumps].tolist() == [0, 2, 4]


def test_punctuation_merging_preserves_tokens_times_and_weighted_probability() -> None:
    merged = _merge_punctuation(
        [
            AlignedWord(" hello", (10,), 0.1, 0.3, 0.9),
            AlignedWord("!", (11,), 0.3, 0.4, 0.7),
            AlignedWord(" ¿", (12,), 0.5, 0.6, 0.8),
            AlignedWord("world", (13, 14), 0.6, 0.9, 0.6),
        ]
    )
    assert merged == (
        AlignedWord(" hello!", (10, 11), 0.1, 0.4, 0.8),
        AlignedWord(" ¿world", (12, 13, 14), 0.5, 0.9, 2.0 / 3.0),
    )


def test_translation_groups_emitted_words_as_english() -> None:
    assert _word_grouping_language("ja", "translate") == "en"
    assert _word_grouping_language("ja", "transcribe") == "ja"


def test_alignment_replay_matches_transformers_attention_and_probabilities(
    tiny_whisper_config,
    tiny_transformers_model,
    tiny_checkpoint_tensors,
    tmp_path,
) -> None:
    checkpoint = tmp_path / "tiny-whisper.safetensors"
    save_file(tiny_checkpoint_tensors, checkpoint, metadata={"format": "pt"})
    weights = load_whisper_safetensors(
        checkpoint,
        tiny_whisper_config,
        checkpoint_dtype=torch.float32,
    )
    features = torch.linspace(
        -0.5,
        0.5,
        steps=tiny_whisper_config.num_mel_bins
        * tiny_whisper_config.max_source_positions
        * 2,
        dtype=torch.float32,
    ).view(
        1,
        tiny_whisper_config.num_mel_bins,
        tiny_whisper_config.max_source_positions * 2,
    )
    token_ids = (3, 4, 5, 6, 2)
    text_token_ids = (5, 6)
    selected_heads = ((0, 1), (1, 2))

    with torch.inference_mode():
        encoder = tiny_transformers_model.model.encoder(
            features,
            return_dict=True,
        ).last_hidden_state
        oracle = tiny_transformers_model.model.decoder(
            input_ids=torch.tensor([token_ids], dtype=torch.int64),
            encoder_hidden_states=encoder,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        cross_keys = []
        cross_values = []
        for layer in weights.decoder.layers:
            cross_keys.append(
                F.linear(
                    encoder,
                    layer.cross_attention.key.weight,
                    layer.cross_attention.key.bias,
                ).view(
                    1,
                    tiny_whisper_config.max_source_positions,
                    tiny_whisper_config.decoder_attention_heads,
                    tiny_whisper_config.decoder_head_dim,
                )
            )
            cross_values.append(
                F.linear(
                    encoder,
                    layer.cross_attention.value.weight,
                    layer.cross_attention.value.bias,
                ).view_as(cross_keys[-1])
            )
        matrix, logprobs, actual_no_speech_probability = _decoder_alignment_matrix(
            weights.decoder,
            token_ids,
            (*text_token_ids, tiny_whisper_config.eos_token_id),
            torch.stack(cross_keys),
            torch.stack(cross_values),
            selected_heads,
            prefix_length=2,
            num_frames=tiny_whisper_config.max_source_positions,
            no_speech_position=0,
            no_speech_id=7,
            config=tiny_whisper_config,
        )

    assert matrix is not None
    oracle_cross = torch.stack(
        [oracle.cross_attentions[layer][0, head] for layer, head in selected_heads]
    )
    std, mean = torch.std_mean(
        oracle_cross,
        dim=-2,
        keepdim=True,
        unbiased=False,
    )
    expected_matrix = _median_filter((oracle_cross - mean) / std).mean(dim=0)[1:-1]
    torch.testing.assert_close(
        torch.from_numpy(matrix),
        expected_matrix,
        rtol=1e-5,
        atol=1e-6,
    )
    prediction_hidden = oracle.last_hidden_state[0, 1:4]
    logits = F.linear(prediction_hidden, weights.decoder.token_embedding)
    expected_logprobs = torch.log_softmax(logits, dim=-1).gather(
        1,
        torch.tensor((*text_token_ids, tiny_whisper_config.eos_token_id))[:, None],
    )[:, 0]
    torch.testing.assert_close(
        torch.tensor(logprobs),
        expected_logprobs,
        rtol=1e-5,
        atol=1e-6,
    )
    expected_no_speech = torch.softmax(
        F.linear(oracle.last_hidden_state[0, 0], weights.decoder.token_embedding),
        dim=-1,
    )[7]
    assert actual_no_speech_probability == pytest.approx(
        float(expected_no_speech), rel=1e-5
    )

    probability = no_speech_probability(
        decoder=weights.decoder,
        tokenizer=SimpleNamespace(
            controls=WhisperControlTokens(
                suppress_tokens=(),
                eos_id=tiny_whisper_config.eos_token_id,
                decoder_start_id=tiny_whisper_config.decoder_start_token_id,
                no_speech_id=7,
                vocab_size=tiny_whisper_config.vocab_size,
                max_target_positions=tiny_whisper_config.max_target_positions,
            )
        ),
        prefix_token_ids=token_ids[:2],
        cross_keys=torch.stack(cross_keys),
        cross_values=torch.stack(cross_values),
        config=tiny_whisper_config,
    )
    assert probability == pytest.approx(float(expected_no_speech), rel=1e-5)
