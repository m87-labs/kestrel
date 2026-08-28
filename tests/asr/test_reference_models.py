from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kestrel.models.asr.audio import AudioChunks, DecodedAudio
from kestrel.models.asr.contract import Word
from kestrel.models.parakeet_tdt.config import (
    ParakeetEncoderConfig,
    ParakeetTdtConfig,
)
from kestrel.models.parakeet_tdt.features import parakeet_features
from kestrel.models.parakeet_tdt.model import ParakeetTdt
from kestrel.models.parakeet_tdt.tokenizer import ParakeetTokenizer
from kestrel.models.qwen3_asr.config import (
    AudioEncoderConfig,
    Qwen3AsrConfig,
    TextDecoderConfig,
)
from kestrel.models.qwen3_asr.features import qwen3_asr_features
from kestrel.models.qwen3_asr.alignment import Qwen3ForcedAlignerRuntime
from kestrel.models.qwen3_asr.model import (
    AudioAttention,
    Qwen3AsrForConditionalGeneration,
)
from kestrel.models.qwen3_asr.tokenizer import Qwen3AsrTokenizer


def _transformers():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3ASRForConditionalGeneration"):
        pytest.skip("requires Transformers with Qwen3-ASR")
    return transformers


def test_long_audio_chunks_preserve_every_sample_and_offset(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "kestrel_native",
        SimpleNamespace(
            resample_audio_mono=lambda values, *_args, **_kwargs: values.copy()
        ),
    )
    waveform = np.ones(550, dtype=np.float32)
    waveform[195:205] = 0
    waveform[405:415] = 0
    with AudioChunks(
        waveform,
        sample_rate=100,
        clip_start_seconds=0,
        clip_end_seconds=None,
        target_sample_rate=100,
        max_duration_seconds=10,
    ) as source:
        chunks = tuple(
            source.chunks(
                2,
                boundary_search_seconds=0.2,
                boundary_window_seconds=0.1,
            )
        )
    assert np.array_equal(
        np.concatenate([chunk.waveform for chunk in chunks]), waveform
    )
    assert [chunk.clip_start_seconds for chunk in chunks] == [0, 1.95, 3.95]
    assert all(chunk.duration_seconds <= 2 for chunk in chunks)
    assert sum(chunk.duration_seconds for chunk in chunks) == 5.5


def test_parakeet_character_timestamps_follow_tdt_token_durations() -> None:
    class Backend:
        def decode(self, token_ids, *, skip_special_tokens):
            del skip_special_tokens
            return {10: " hello", 11: ",", 12: "world"}[token_ids[0]]

    tokenizer = object.__new__(ParakeetTokenizer)
    tokenizer.backend = Backend()
    tokenizer.blank_token_id = 20
    tokenizer.pad_token_id = 21

    characters = tokenizer.characters(
        [20, 10, 11, 12],
        [1, 2, 3, 1],
        0.08,
    )
    assert [(item.text, item.start, item.end) for item in characters] == [
        (" hello", 0.08, 0.24),
        (",", 0.24, 0.24),
        ("world", 0.48, 0.56),
    ]


def test_qwen_forced_aligner_runtime_accepts_transcription_language_codes(
    monkeypatch,
) -> None:
    audio = DecodedAudio(np.zeros(16_000, dtype=np.float32), 1.0, 2.0, 0.5)
    monkeypatch.setattr(
        "kestrel.models.qwen3_asr.alignment.decode_audio",
        lambda *_args, **_kwargs: audio,
    )

    def align(_loaded, waveform, text, language, *, offset_seconds):
        assert waveform is audio.waveform
        assert (text, language, offset_seconds) == ("hello", "English", 0.5)
        return (Word("hello", 0.5, 1.0),)

    monkeypatch.setattr("kestrel.models.qwen3_asr.alignment.align_transcript", align)
    cfg = SimpleNamespace(
        model="Qwen/Qwen3-ForcedAligner-0.6B-hf",
        resolved_device=lambda: "cpu",
        resolved_dtype=lambda: torch.float32,
    )
    runtime = Qwen3ForcedAlignerRuntime(cfg, aligner=object())
    (output,) = runtime.forward(
        "align",
        ({"audio": np.zeros(1), "text": " hello ", "language": "en"},),
    )
    assert output == {
        "text": "hello",
        "language": "en",
        "duration_seconds": 1.0,
        "source_duration_seconds": 2.0,
        "clip_start_seconds": 0.5,
        "clip_end_seconds": 1.5,
        "words": [{"word": "hello", "start": 0.5, "end": 1.0}],
    }


def test_qwen_no_language_sentinel_is_not_a_language_name() -> None:
    tokenizer = object.__new__(Qwen3AsrTokenizer)
    tokenizer.backend = SimpleNamespace(
        decode=lambda *_args, **_kwargs: "language None<asr_text>"
    )
    assert tokenizer.decode_result([1], forced_language=None) == ("", None)


def test_audio_frontends_match_transformers() -> None:
    transformers = _transformers()
    librosa = pytest.importorskip("librosa")
    del librosa
    waveform = (0.1 * np.sin(np.arange(16_000, dtype=np.float32) * 0.013)).astype(
        np.float32
    )

    qwen_reference = transformers.Qwen3ASRFeatureExtractor()
    expected = qwen_reference(
        waveform,
        sampling_rate=16_000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    features, mask = qwen3_asr_features(waveform)
    torch.testing.assert_close(features, expected.input_features, rtol=0, atol=0)
    assert torch.equal(mask, expected.attention_mask.bool())
    batched_features, batched_mask = qwen3_asr_features(np.stack((waveform, waveform)))
    torch.testing.assert_close(
        batched_features, features.expand(2, -1, -1), rtol=1e-6, atol=1e-6
    )
    assert torch.equal(batched_mask, mask.expand(2, -1))

    parakeet_reference = transformers.ParakeetFeatureExtractor(feature_size=128)
    expected = parakeet_reference(
        waveform,
        sampling_rate=16_000,
        return_attention_mask=True,
        return_tensors="pt",
    )
    features, mask = parakeet_features(waveform)
    torch.testing.assert_close(features, expected.input_features, rtol=1e-5, atol=1e-5)
    assert torch.equal(mask, expected.attention_mask.bool())
    batched_features, batched_mask = parakeet_features(np.stack((waveform, waveform)))
    torch.testing.assert_close(
        batched_features, features.expand(2, -1, -1), rtol=1e-5, atol=1e-5
    )
    assert torch.equal(batched_mask, mask.expand(2, -1))


def test_qwen_audio_attention_preserves_window_boundaries() -> None:
    torch.manual_seed(8)
    attention = AudioAttention(AudioEncoderConfig(32, 4, 64, 1))
    for lengths in ((13, 13, 13, 13), (13, 7, 13, 7)):
        hidden = torch.randn(sum(lengths), 32)
        expected = []
        for chunk in hidden.split(lengths):
            length = chunk.shape[0]
            shape = (1, length, attention.num_heads, attention.head_dim)
            q = attention.q_proj(chunk).view(shape).transpose(1, 2)
            k = attention.k_proj(chunk).view(shape).transpose(1, 2)
            v = attention.v_proj(chunk).view(shape).transpose(1, 2)
            attended = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=attention.scale
            )
            expected.append(attended.transpose(1, 2).reshape(length, -1))
        expected = attention.out_proj(torch.cat(expected))
        torch.testing.assert_close(attention(hidden, lengths), expected)


def test_qwen_prefill_and_decode_match_transformers() -> None:
    transformers = _transformers()
    torch.manual_seed(7)
    reference_audio = transformers.Qwen3ASREncoderConfig(
        num_mel_bins=16,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        d_model=64,
        downsample_hidden_size=8,
        output_dim=64,
    )
    reference_text = transformers.Qwen3Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1_024,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000},
        tie_word_embeddings=True,
        layer_types=["full_attention"],
    )
    reference = transformers.Qwen3ASRForConditionalGeneration(
        transformers.Qwen3ASRConfig(
            audio_config=reference_audio,
            text_config=reference_text,
            audio_token_id=499,
            timestamp_token_id=500,
            pad_token_id=0,
            eos_token_id=[2, 3],
            tie_word_embeddings=True,
        )
    ).eval()
    with torch.device("meta"):
        model = Qwen3AsrForConditionalGeneration(
            Qwen3AsrConfig(
                AudioEncoderConfig(64, 4, 128, 1, 8, 16, 13, 50, 800, 64),
                TextDecoderConfig(64, 128, 1, 4, 2, 16, 512, 1_024, 1e-6, 1_000_000),
                499,
                (2, 3),
                0,
            )
        )
    model.load_state_dict(reference.state_dict(), assign=True)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    model.reset_nonpersistent_buffers()
    model.eval()

    features = torch.randn(1, 16, 300)
    feature_mask = torch.zeros(1, 300, dtype=torch.bool)
    feature_mask[:, :257] = True
    audio_tower = model.model.audio_tower
    audio_tokens = int(audio_tower._post_tokens(feature_mask))
    for valid_frames in (0, 1, 99, 100, 101, 257, 300):
        layout_mask = torch.arange(300)[None] < valid_frames
        assert audio_tower.feature_layout(valid_frames) == (
            int(audio_tower._post_tokens(layout_mask)),
            audio_tower.window_lengths(layout_mask),
        )
    input_ids = torch.randint(0, 490, (1, audio_tokens + 7))
    input_ids[:, 3 : 3 + audio_tokens] = 499
    with torch.inference_mode():
        expected = reference(
            input_ids=input_ids,
            input_features=features,
            input_features_mask=feature_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        actual = model.prefill(input_ids, features, feature_mask)
        next_token = expected.logits[:, -1].argmax(-1, keepdim=True)
        expected_next = reference(
            input_ids=next_token,
            past_key_values=expected.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        ).logits
        actual_next = model.decode(next_token, actual.cache).logits
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual_next, expected_next)

    second_mask = torch.zeros_like(feature_mask)
    second_mask[:, :157] = True
    second_audio_tokens = int(model.model.audio_tower._post_tokens(second_mask))
    second_ids = torch.randint(0, 490, (1, second_audio_tokens + 5))
    second_ids[:, 2 : 2 + second_audio_tokens] = 499
    lengths = torch.tensor([input_ids.shape[1], second_ids.shape[1]])
    batched_ids = torch.full((2, int(lengths.max())), 0, dtype=torch.long)
    batched_ids[0, : input_ids.shape[1]] = input_ids[0]
    batched_ids[1, : second_ids.shape[1]] = second_ids[0]
    with torch.inference_mode():
        second = model.prefill(second_ids, features, second_mask)
        batched = model.prefill(
            batched_ids,
            features.expand(2, -1, -1),
            torch.cat((feature_mask, second_mask)),
            last_indices=lengths - 1,
        )
    torch.testing.assert_close(batched.logits[0], actual.logits[0])
    torch.testing.assert_close(batched.logits[1], second.logits[0])
    for (batch_key, batch_value), (first_key, first_value), (
        second_key,
        second_value,
    ) in zip(batched.cache, actual.cache, second.cache, strict=True):
        torch.testing.assert_close(batch_key[0, :, : input_ids.shape[1]], first_key[0])
        torch.testing.assert_close(
            batch_value[0, :, : input_ids.shape[1]], first_value[0]
        )
        torch.testing.assert_close(
            batch_key[1, :, : second_ids.shape[1]], second_key[0]
        )
        torch.testing.assert_close(
            batch_value[1, :, : second_ids.shape[1]], second_value[0]
        )


def test_parakeet_encoder_and_tdt_decode_match_transformers() -> None:
    transformers = _transformers()
    torch.manual_seed(4)
    reference_encoder = transformers.ParakeetEncoderConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        attention_bias=False,
        convolution_bias=False,
        conv_kernel_size=9,
        subsampling_factor=4,
        subsampling_conv_channels=4,
        num_mel_bins=16,
        dropout=0,
        dropout_positions=0,
        layerdrop=0,
        activation_dropout=0,
        attention_dropout=0,
        scale_input=False,
    )
    reference = transformers.ParakeetForTDT(
        transformers.ParakeetTDTConfig(
            vocab_size=32,
            decoder_hidden_size=16,
            num_decoder_layers=2,
            encoder_config=reference_encoder,
            pad_token_id=2,
            blank_token_id=31,
            durations=[0, 1, 2, 3, 4],
        )
    ).eval()
    with torch.device("meta"):
        model = ParakeetTdt(
            ParakeetTdtConfig(
                ParakeetEncoderConfig(
                    32, 64, 2, 4, 4, 16, 9, 4, 3, 2, 4, 5_000, "silu"
                ),
                31,
                16,
                (0, 1, 2, 3, 4),
                10,
                2,
                2,
                32,
                "relu",
            )
        )
    model.load_state_dict(reference.state_dict(), assign=True)
    model.reset_nonpersistent_buffers()
    model.eval()

    features = torch.randn(1, 41, 16)
    mask = torch.ones(1, 41, dtype=torch.bool)
    mask[:, -2:] = False
    with torch.inference_mode():
        expected = reference.get_audio_features(features, mask)
        hidden, valid = model.encoder(features, mask)
    selected = valid.unsqueeze(-1).expand_as(hidden)
    torch.testing.assert_close(hidden[selected], expected.last_hidden_state[selected])
    assert torch.equal(valid, expected.attention_mask.bool())

    with torch.no_grad():
        reference.joint.head.weight.zero_()
        reference.joint.head.bias.zero_()
        reference.joint.head.bias[5] = 10
        reference.joint.head.bias[34] = 10
        model.load_state_dict(reference.state_dict(), assign=True)
    generation = transformers.GenerationConfig(
        decoder_start_token_id=31,
        pad_token_id=2,
        suppress_tokens=list(range(32, 37)),
        max_new_tokens=200,
        return_dict_in_generate=True,
    )
    with torch.inference_mode():
        expected_tokens = reference.generate(
            features,
            attention_mask=mask,
            generation_config=generation,
        )
        actual_tokens = model.generate(features, mask)
    assert torch.equal(actual_tokens.sequences, expected_tokens.sequences)
    assert torch.equal(actual_tokens.durations, expected_tokens.durations)
    assert actual_tokens.lengths.tolist() == [actual_tokens.sequences.shape[1]]

    with torch.inference_mode():
        batched = model.generate(features.expand(2, -1, -1), mask.expand(2, -1))
    assert batched.lengths.tolist() == [actual_tokens.sequences.shape[1]] * 2
    assert torch.equal(batched.sequences, actual_tokens.sequences.expand(2, -1))
    assert torch.equal(batched.durations, actual_tokens.durations.expand(2, -1))

    with torch.no_grad():
        model.joint.head.weight.zero_()
        model.joint.head.bias.zero_()
        model.joint.head.bias[31] = 10
        model.joint.head.bias[33] = 10
    blank_only = model.generate(features, mask, max_tokens=1)
    assert blank_only.durations.sum() == actual_tokens.durations.sum()
    blank_batch = model.generate(
        features.expand(2, -1, -1), mask.expand(2, -1), max_tokens=1
    )
    assert torch.equal(
        blank_batch.durations.sum(1), blank_only.durations.sum().expand(2)
    )
