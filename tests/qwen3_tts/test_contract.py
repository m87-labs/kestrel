"""High-value request and tokenizer contracts for Qwen3-TTS."""

from dataclasses import replace
from threading import get_ident
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Regex, pre_tokenizers

from kestrel.runtime.tokens import TextToken
from kestrel.runtime.staging import AsyncPreprocessor
from kestrel.models.qwen3_tts.config import (
    Qwen3TTSCodePredictorConfig,
    Qwen3TTSConfig,
    Qwen3TTSTalkerConfig,
)
from kestrel.models.qwen3_tts.contract import (
    DEFAULT_SAMPLING,
    CustomVoiceRequest,
    EncodedCustomVoiceRequest,
)
from kestrel.models.qwen3_tts.generated_decode import _TalkerPrefillBindings
from kestrel.models.qwen3_tts.input_layout import build_talker_token_layout
from kestrel.models.qwen3_tts.model import Qwen3TTSCodePredictor
from kestrel.models.qwen3_tts.skill import Qwen3TTSSynthesizeSkill
from kestrel.models.qwen3_tts.text import _SPLIT_PATTERN
from kestrel.models.qwen3_tts.runtime import (
    PreparedSynthesis,
    Qwen3TTSRuntime,
    _validate_generation_budget,
    _uses_generated_prefill,
)


def test_short_text_is_inline_but_long_text_and_instructions_use_worker() -> None:
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime._prepare_synthesis = lambda value: get_ident()
    runtime._text_preprocessor = AsyncPreprocessor(
        runtime._prepare_synthesis, workers=1
    )
    try:
        short = runtime.preprocess_encoder_input_async(CustomVoiceRequest("Hello."))
        assert short.result() == get_ident()
        for request in (
            CustomVoiceRequest("hello " * 100),
            CustomVoiceRequest("Hello.", instructions="happy " * 100),
        ):
            assert runtime.preprocess_encoder_input_async(request).result() != get_ident()
    finally:
        runtime._text_preprocessor.shutdown()


def test_default_request_uses_automatic_language_detection() -> None:
    request = CustomVoiceRequest("Bonjour tout le monde")
    assert request.language_key == "auto"
    built = Qwen3TTSSynthesizeSkill().build_request(
        None, {"text": request.text}, None
    )
    assert built.max_new_tokens == 2_048


def test_synthesis_accepts_supported_talker_sampling_settings() -> None:
    built = Qwen3TTSSynthesizeSkill().build_request(
        None,
        {"text": "Hello"},
        {
            "do_sample": False,
            "temperature": 0.7,
            "top_k": 24,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "subtalker_dosample": False,
            "subtalker_temperature": 0.75,
            "subtalker_top_k": 0,
            "subtalker_top_p": 0.85,
            "max_tokens": 64,
        },
    )
    assert built.temperature == 0.0
    assert built.top_p == 0.8
    assert built.max_new_tokens == 64
    sampling = built.request_context.sampling
    assert sampling.top_k == 24
    assert sampling.repetition_penalty == 1.1
    assert sampling.subtalker_parameters == (0.0, 1, 1.0)


def test_synthesis_rejects_ignored_options() -> None:
    skill = Qwen3TTSSynthesizeSkill()
    with pytest.raises(ValueError, match="langauge"):
        skill.build_request(None, {"text": "Hello", "langauge": "English"}, None)


def test_synthesis_rejects_invalid_predictor_sampling_settings() -> None:
    skill = Qwen3TTSSynthesizeSkill()
    with pytest.raises(ValueError, match=r"\[0, 2048\]"):
        skill.build_request(
            None,
            {"text": "Hello"},
            {"subtalker_top_k": 2049},
        )
    with pytest.raises(ValueError, match="top_p"):
        skill.build_request(
            None,
            {"text": "Hello"},
            {"subtalker_top_p": 0.0},
        )
    with pytest.raises(TypeError, match="do_sample"):
        skill.build_request(None, {"text": "Hello"}, {"do_sample": 1})


def test_eos_is_allowed_after_two_generated_frames() -> None:
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime.eos_token_ids = (2150,)
    runtime.vocab_size = 3072
    runtime._allowed_talker_tokens = torch.zeros(3072, dtype=torch.uint8)
    runtime._allowed_talker_tokens[:2048] = 1
    runtime._allowed_talker_tokens[2150] = 1
    runtime._seen_talker_tokens = torch.zeros((1, 3072), dtype=torch.uint8)
    runtime._repetition_penalties_by_batch = SimpleNamespace(gpu=torch.ones(1))

    for generated, suppressed in ((1, True), (2, False)):
        logits = torch.zeros((1, 3072), dtype=torch.bfloat16)
        logits[0, 2150] = 10
        sequence = SimpleNamespace(
            skill_state=SimpleNamespace(
                tokens=[TextToken(1)] * generated,
                synthesis=CustomVoiceRequest("Hello"),
            )
        )
        runtime._process_logits(
            logits,
            sequences=(sequence,),
            batch_idx=torch.tensor([0]),
        )
        assert torch.isneginf(logits[0, 2150]).item() is suppressed


def test_talker_sampling_controls_are_row_local() -> None:
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime.eos_token_ids = (7,)
    runtime.vocab_size = 8
    runtime._allowed_talker_tokens = torch.ones(8, dtype=torch.uint8)
    runtime._seen_talker_tokens = torch.zeros((4, 8), dtype=torch.uint8)
    runtime._seen_talker_tokens[3, [6, 7]] = 1
    runtime._seen_talker_tokens[1, [1, 2]] = 1
    runtime._repetition_penalties_by_batch = SimpleNamespace(
        gpu=torch.tensor([1.0, 1.0, 1.0, 2.0])
    )
    logits = torch.arange(8, dtype=torch.bfloat16).repeat(2, 1)
    sequences = (
        SimpleNamespace(
            skill_state=SimpleNamespace(
                tokens=[TextToken(7), TextToken(6)],
                synthesis=CustomVoiceRequest(
                    "Hello",
                    sampling=replace(
                        DEFAULT_SAMPLING,
                        top_k=1,
                        repetition_penalty=2.0,
                    ),
                ),
            )
        ),
        SimpleNamespace(
            skill_state=SimpleNamespace(
                tokens=[TextToken(1), TextToken(2)],
                synthesis=CustomVoiceRequest(
                    "Hello",
                    sampling=replace(
                        DEFAULT_SAMPLING,
                        do_sample=False,
                        top_k=1,
                        repetition_penalty=1.0,
                    ),
                ),
            )
        ),
    )

    batch_idx = torch.tensor([3, 1])
    runtime._process_logits(logits, sequences=sequences, batch_idx=batch_idx)

    assert torch.isfinite(logits[0]).sum().item() == 1
    assert int(torch.argmax(logits[0])) == 5
    assert torch.isfinite(logits[1]).sum().item() == 8

    runtime._seen_talker_tokens[3].zero_()
    reused_logits = torch.arange(8, dtype=torch.bfloat16).repeat(2, 1)
    runtime._process_logits(
        reused_logits,
        sequences=sequences,
        batch_idx=batch_idx,
    )
    assert int(torch.argmax(reused_logits[0])) == 7


def test_subtalker_sampling_controls_follow_batched_state_rows() -> None:
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime._predictor_temperatures_by_batch = SimpleNamespace(
        gpu=torch.tensor([0.9, 0.0])
    )
    runtime._predictor_top_ks_by_batch = SimpleNamespace(
        gpu=torch.tensor([50, 1], dtype=torch.int32)
    )
    runtime._predictor_top_ps_by_batch = SimpleNamespace(
        gpu=torch.tensor([1.0, 0.8])
    )
    runtime._custom_predictor_slots = set()
    slot = SimpleNamespace(
        scratch={
            "predictor_temperature": torch.empty(2),
            "predictor_top_k": torch.empty(2, dtype=torch.int32),
            "predictor_top_p": torch.empty(2),
        }
    )
    custom = SimpleNamespace(
        skill_state=SimpleNamespace(
            synthesis=CustomVoiceRequest(
                "Hello",
                sampling=replace(DEFAULT_SAMPLING, subtalker_dosample=False),
            )
        )
    )
    default = SimpleNamespace(
        skill_state=SimpleNamespace(synthesis=CustomVoiceRequest("Hello"))
    )

    runtime._stage_predictor_sampling(
        slot,
        (custom, default),
        torch.tensor([1, 0]),
    )

    assert slot.scratch["predictor_temperature"].tolist() == pytest.approx([0.0, 0.9])
    assert slot.scratch["predictor_top_k"].tolist() == [1, 50]
    assert slot.scratch["predictor_top_p"].tolist() == pytest.approx([0.8, 1.0])

    runtime._stage_predictor_sampling(slot, (default,), torch.tensor([0]))
    assert slot.scratch["predictor_temperature"].tolist() == pytest.approx([0.9, 0.9])
    assert slot.scratch["predictor_top_k"].tolist() == [50, 50]
    assert slot.scratch["predictor_top_p"].tolist() == pytest.approx([1.0, 1.0])


def test_tokenizer_splits_mixed_case_and_trailing_slash_like_official() -> None:
    split = pre_tokenizers.Split(Regex(_SPLIT_PATTERN), behavior="isolated")
    pieces = [
        piece
        for piece, _span in split.pre_tokenize_str(
            "iPhone XMLHttpRequest /foo/\nABC def"
        )
    ]
    assert pieces == [
        "i",
        "Phone",
        " XMLHttp",
        "Request",
        " /",
        "foo",
        "/\n",
        "ABC",
        " def",
    ]


def test_talker_uses_short_streaming_text_layout() -> None:
    talker = replace(
        Qwen3TTSTalkerConfig(),
        codec_language_id={"english": 100},
        spk_id={"ryan": 101},
        spk_is_dialect={"ryan": False},
    )
    config = Qwen3TTSConfig(talker=talker)
    encoded = EncodedCustomVoiceRequest(
        CustomVoiceRequest("hello", language="English", stream=True),
        text_token_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        instruction_token_ids=(),
    )

    layout = build_talker_token_layout(config, encoded)

    assert layout.sequence_length == 10
    assert layout.text_token_ids == (
        1,
        2,
        3,
        *(config.tts_pad_token_id for _ in range(5)),
        config.tts_bos_token_id,
        4,
    )
    assert layout.codec_token_ids == (
        talker.codec_pad_id,
        talker.codec_pad_id,
        talker.codec_pad_id,
        talker.codec_think_id,
        talker.codec_think_bos_id,
        100,
        talker.codec_think_eos_id,
        101,
        talker.codec_pad_id,
        talker.codec_bos_id,
    )

    automatic = build_talker_token_layout(
        config,
        EncodedCustomVoiceRequest(
            CustomVoiceRequest("hello", stream=True),
            text_token_ids=encoded.text_token_ids,
            instruction_token_ids=(),
        ),
    )
    assert automatic.sequence_length == 9
    assert layout.continuation_text_token_ids == (5, config.tts_eos_token_id)


def test_streaming_text_requires_enough_generation_budget_for_its_tail() -> None:
    layout = SimpleNamespace(continuation_text_token_ids=(4, 5, 6))

    with pytest.raises(ValueError, match="too small to consume all streaming text"):
        _validate_generation_budget(layout, 3)
    _validate_generation_budget(layout, 4)


def test_06b_custom_voice_rejects_instructions() -> None:
    talker = replace(
        Qwen3TTSTalkerConfig(),
        hidden_size=1024,
        intermediate_size=3072,
        codec_language_id={"english": 100},
        spk_id={"ryan": 101},
        spk_is_dialect={"ryan": False},
    )
    config = Qwen3TTSConfig(talker=talker, model_size="0b6")
    request = CustomVoiceRequest(
        "hello",
        voice="Ryan",
        language="English",
        instructions="Speak happily",
    )
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime.config = config

    with pytest.raises(ValueError, match="0.6B does not support instructions"):
        runtime._prepare_synthesis(request)


def test_talker_uses_full_text_layout_for_non_streaming_output() -> None:
    talker = replace(
        Qwen3TTSTalkerConfig(),
        codec_language_id={"english": 100},
        spk_id={"ryan": 101},
        spk_is_dialect={"ryan": False},
    )
    config = Qwen3TTSConfig(talker=talker)
    encoded = EncodedCustomVoiceRequest(
        CustomVoiceRequest("hello", language="English"),
        text_token_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        instruction_token_ids=(),
    )

    layout = build_talker_token_layout(config, encoded)

    assert layout.text_token_ids == (
        1,
        2,
        3,
        *(config.tts_pad_token_id for _ in range(5)),
        config.tts_bos_token_id,
        4,
        5,
        config.tts_eos_token_id,
        config.tts_pad_token_id,
    )
    assert layout.codec_token_ids == (
        *((talker.codec_pad_id,) * 3),
        talker.codec_think_id,
        talker.codec_think_bos_id,
        100,
        talker.codec_think_eos_id,
        101,
        talker.codec_pad_id,
        talker.codec_pad_id,
        talker.codec_pad_id,
        talker.codec_pad_id,
        talker.codec_bos_id,
    )
    assert layout.codec_embedding_mask == (*((False,) * 3), *((True,) * 10))
    assert layout.continuation_text_token_ids == ()
    assert layout.sequence_length == 13
    assert not _uses_generated_prefill((layout.sequence_length,))
    runtime = object.__new__(Qwen3TTSRuntime)
    runtime.max_seq_length = 20

    with pytest.raises(ValueError, match="context window"):
        runtime.prepare_sequence(
            (TextToken(0),),
            encoder_input=PreparedSynthesis(encoded.request, layout),
            max_new_tokens=8,
        )


def test_equal_width_code_predictor_has_no_projection_parameters() -> None:
    predictor = replace(
        Qwen3TTSCodePredictorConfig(),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=16,
        num_code_groups=2,
    )
    talker = replace(
        Qwen3TTSTalkerConfig(),
        hidden_size=8,
        vocab_size=16,
        num_code_groups=2,
        code_predictor=predictor,
    )
    model = Qwen3TTSCodePredictor(Qwen3TTSConfig(talker=talker))

    assert isinstance(model.small_to_mtp_projection, torch.nn.Identity)
    assert not any(
        name.startswith("small_to_mtp_projection.") for name in model.state_dict()
    )


def test_packed_prefill_binds_the_semantic_input_name() -> None:
    slot = SimpleNamespace(
        scratch={
            "prompt_embeddings": torch.empty(16, 8),
            "prefill_input_pos": torch.empty(16, dtype=torch.int32),
            "prefill_hidden": torch.empty(16, 8),
            "prefill_logits": torch.empty(16, 32),
        },
        packed_batch_indices=SimpleNamespace(gpu=torch.empty(16, dtype=torch.int64)),
    )

    inputs = _TalkerPrefillBindings.slot_inputs(slot, 10)

    assert "input_embedding" in inputs
    assert "prompt_embeddings" not in inputs
