from types import SimpleNamespace

import numpy as np
import pytest

import torch

from kestrel.models.qwen35 import Qwen35Runtime, QwenVideoInputs
from kestrel.models.qwen35.prompt_template import (
    IM_START_ID,
    VIDEO_PAD_ID,
    VISION_END_ID,
    VISION_START_ID,
    _NEWLINE_ID,
    _USER_ID,
)
from kestrel.runtime import TextToken


def _runtime() -> Qwen35Runtime:
    runtime = object.__new__(Qwen35Runtime)
    runtime.architecture = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2)
    )
    return runtime


def test_video_position_ids_expand_temporal_grid_per_timestamp_group() -> None:
    runtime = _runtime()
    out = np.zeros((3, 1, 11), dtype=np.int64)
    mm_types = np.asarray([0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0])

    rope_delta = runtime._fill_multimodal_position_ids(
        out,
        start=0,
        end=11,
        mm_token_type_ids=mm_types,
        vision_grid_thw=np.asarray([[2, 4, 4]], dtype=np.int64),
        vision_modality_type=2,
    )

    np.testing.assert_array_equal(
        out[:, 0],
        np.asarray(
            [
                [0, 1, 1, 1, 1, 3, 4, 4, 4, 4, 6],
                [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6],
                [0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6],
            ],
            dtype=np.int64,
        ),
    )
    assert rope_delta == -4


def test_video_position_ids_require_one_grid_per_pad_group() -> None:
    runtime = _runtime()
    out = np.zeros((3, 1, 5), dtype=np.int64)

    with pytest.raises(RuntimeError, match="unused rows"):
        runtime._fill_multimodal_position_ids(
            out,
            start=0,
            end=5,
            mm_token_type_ids=np.asarray([0, 2, 2, 2, 2]),
            vision_grid_thw=np.asarray([[2, 4, 4]], dtype=np.int64),
            vision_modality_type=2,
        )


class _Tokenizer:
    def encode(self, text: str) -> SimpleNamespace:
        timestamp_ids = {
            "<0.1 seconds>": 901,
            "<0.6 seconds>": 906,
        }
        return SimpleNamespace(ids=[timestamp_ids[text]])


def _video_inputs(**updates: object) -> QwenVideoInputs:
    values: dict[str, object] = {
        "pixel_values": torch.empty((32, 1536)),
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
        "num_video_tokens": 8,
        "timestamps_seconds": (0.125, 0.625),
    }
    values.update(updates)
    return QwenVideoInputs(**values)  # type: ignore[arg-type]


def test_prepare_sequence_inserts_native_timestamped_video_groups() -> None:
    runtime = _runtime()
    runtime.tokenizer = _Tokenizer()
    runtime.prompt_template = SimpleNamespace(
        query=lambda: SimpleNamespace(prefix=[_USER_ID, _NEWLINE_ID])
    )
    runtime._prepare_uncached_sequence = lambda **kwargs: kwargs
    prompt = [
        TextToken(IM_START_ID),
        TextToken(_USER_ID),
        TextToken(_NEWLINE_ID),
        TextToken(777),
        TextToken(778),
    ]
    video = _video_inputs()

    prepared = runtime.prepare_sequence(
        prompt,
        image=object(),
        image_crops=video,
        max_new_tokens=4,
    )

    inserted_ids = [
        901,
        VISION_START_ID,
        *([VIDEO_PAD_ID] * 4),
        VISION_END_ID,
        906,
        VISION_START_ID,
        *([VIDEO_PAD_ID] * 4),
        VISION_END_ID,
    ]
    assert runtime.image_kv_length(prompt, object(), video) == len(inserted_ids)
    assert [token.token_id for token in prepared["tokens"]] == [
        IM_START_ID,
        _USER_ID,
        _NEWLINE_ID,
        *inserted_ids,
        777,
        778,
    ]
    assert prepared["image_length"] == len(inserted_ids)
    assert prepared["target_length"] == len(prompt) + len(inserted_ids) + 4


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"num_video_tokens": 7}, "token count"),
        ({"timestamps_seconds": (0.125,)}, "one value per temporal patch"),
        ({"timestamps_seconds": (0.625, 0.125)}, "must increase"),
    ],
)
def test_prepare_sequence_rejects_inconsistent_video_contract(
    updates: dict[str, object],
    match: str,
) -> None:
    runtime = _runtime()
    runtime.tokenizer = _Tokenizer()
    runtime.prompt_template = SimpleNamespace(
        query=lambda: SimpleNamespace(prefix=[_USER_ID, _NEWLINE_ID])
    )
    runtime._prepare_uncached_sequence = lambda **kwargs: kwargs

    with pytest.raises(ValueError, match=match):
        runtime.prepare_sequence(
            [TextToken(IM_START_ID), TextToken(_USER_ID), TextToken(_NEWLINE_ID)],
            image=object(),
            image_crops=_video_inputs(**updates),
            max_new_tokens=1,
        )
