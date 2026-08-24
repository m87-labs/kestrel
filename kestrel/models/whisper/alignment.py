"""Bounded cross-attention/DTW word alignment for Whisper transcripts."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .config import WhisperTurboConfig
from .tokenizer import WhisperTokenizer
from .weights import (
    AttentionWeights,
    LayerNormWeights,
    LinearWeights,
    WhisperDecoderWeights,
)


_MEDIAN_FILTER_WIDTH = 7
_TOKENS_PER_SECOND = 50.0
_VOCAB_CHUNK = 4096
_PREPENDED_PUNCTUATION = "\"'“¿([{-"
_APPENDED_PUNCTUATION = "\"'.。,，!！?？:：”)]}、"


@dataclass(frozen=True, slots=True)
class AlignedWord:
    """One decoder-attention word interval relative to its audio window."""

    word: str
    token_ids: tuple[int, ...]
    start: float
    end: float
    probability: float


@dataclass(frozen=True, slots=True)
class TranscriptScores:
    """Selected-path confidence and raw no-speech probability."""

    avg_logprob: float
    no_speech_prob: float


@dataclass(frozen=True, slots=True)
class TranscriptAnalysis:
    """One optional alignment pass plus transcript-level confidence."""

    words: tuple[AlignedWord, ...]
    scores: TranscriptScores


def _word_grouping_language(language: str, task: str) -> str:
    """Return the language of emitted text, which translation fixes to English."""

    return "en" if task == "translate" else language


def _merged_probability(left: AlignedWord, right: AlignedWord) -> float:
    left_count = len(left.token_ids)
    right_count = len(right.token_ids)
    return (left.probability * left_count + right.probability * right_count) / (
        left_count + right_count
    )


def _merge_punctuation(words: list[AlignedWord]) -> tuple[AlignedWord, ...]:
    index = len(words) - 2
    following = len(words) - 1
    while index >= 0:
        previous_word = words[index]
        following_word = words[following]
        if (
            previous_word.word.startswith(" ")
            and previous_word.word.strip() in _PREPENDED_PUNCTUATION
        ):
            words[following] = replace(
                following_word,
                word=previous_word.word + following_word.word,
                token_ids=previous_word.token_ids + following_word.token_ids,
                start=previous_word.start,
                probability=_merged_probability(previous_word, following_word),
            )
            words[index] = replace(previous_word, word="", token_ids=())
        else:
            following = index
        index -= 1

    previous = 0
    following = 1
    while following < len(words):
        previous_word = words[previous]
        following_word = words[following]
        if (
            previous_word.word
            and following_word.word
            and not previous_word.word.endswith(" ")
            and following_word.word in _APPENDED_PUNCTUATION
        ):
            words[previous] = replace(
                previous_word,
                word=previous_word.word + following_word.word,
                token_ids=previous_word.token_ids + following_word.token_ids,
                end=following_word.end,
                probability=_merged_probability(previous_word, following_word),
            )
            words[following] = replace(following_word, word="", token_ids=())
        else:
            previous = following
        following += 1
    return tuple(word for word in words if word.word and word.token_ids)


def _linear(value: Tensor, weights: LinearWeights) -> Tensor:
    return F.linear(value, weights.weight, weights.bias)


def _layer_norm(value: Tensor, weights: LayerNormWeights, *, eps: float) -> Tensor:
    return F.layer_norm(
        value,
        (value.shape[-1],),
        weight=weights.weight,
        bias=weights.bias,
        eps=eps,
    )


def _heads(value: Tensor, count: int) -> Tensor:
    batch, tokens, width = value.shape
    return value.view(batch, tokens, count, width // count)


def _query(hidden: Tensor, weights: AttentionWeights, *, heads: int) -> Tensor:
    head_dim = hidden.shape[-1] // heads
    return _heads(_linear(hidden, weights.query) * (head_dim**-0.5), heads)


def _key_value(
    hidden: Tensor,
    weights: AttentionWeights,
    *,
    heads: int,
) -> tuple[Tensor, Tensor]:
    return (
        _heads(_linear(hidden, weights.key), heads),
        _heads(_linear(hidden, weights.value), heads),
    )


def _attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: LinearWeights,
    *,
    causal: bool,
) -> Tensor:
    query_heads = query.transpose(1, 2)
    key_heads = key.transpose(1, 2)
    value_heads = value.transpose(1, 2)
    scores = query_heads @ key_heads.transpose(-2, -1)
    if causal:
        tokens = int(query.shape[1])
        if int(key.shape[1]) != tokens:
            raise ValueError("full causal attention requires equal token counts")
        mask = torch.ones(
            (tokens, tokens),
            dtype=torch.bool,
            device=query.device,
        ).triu(diagonal=1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores.float(), dim=-1).to(value_heads.dtype)
    attended = probabilities @ value_heads
    attended = (
        attended.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], -1)
    )
    return _linear(attended, output)


def _median_filter(values: Tensor, width: int = _MEDIAN_FILTER_WIDTH) -> Tensor:
    if width <= 0 or width % 2 != 1:
        raise ValueError("word-alignment median width must be a positive odd integer")
    padding = width // 2
    if values.shape[-1] <= padding:
        return values
    padded = F.pad(values, (padding, padding), mode="reflect")
    return padded.unfold(-1, width, 1).sort()[0][..., padding]


def _selected_token_logprobs(
    hidden: Tensor,
    token_ids: Sequence[int],
    embedding: Tensor,
) -> tuple[float, ...]:
    if not token_ids:
        return ()
    selected_ids = torch.tensor(token_ids, dtype=torch.int64, device=hidden.device)
    selected = (hidden.float() * embedding.index_select(0, selected_ids).float()).sum(
        dim=-1
    )
    normalizer = torch.full_like(selected, -float("inf"))
    for start in range(0, int(embedding.shape[0]), _VOCAB_CHUNK):
        logits = F.linear(hidden, embedding[start : start + _VOCAB_CHUNK]).float()
        normalizer = torch.logaddexp(normalizer, torch.logsumexp(logits, dim=-1))
    logprobs = selected - normalizer
    if not bool(torch.isfinite(logprobs).all().item()):
        raise RuntimeError("Whisper decoder analysis produced non-finite logprobs")
    return tuple(float(value) for value in logprobs.cpu().tolist())


@torch.inference_mode()
def _decoder_alignment_matrix(
    decoder: WhisperDecoderWeights,
    token_ids: Sequence[int],
    scored_token_ids: Sequence[int],
    cross_keys: Tensor,
    cross_values: Tensor,
    alignment_heads: Sequence[tuple[int, int]],
    *,
    prefix_length: int,
    num_frames: int,
    no_speech_position: int | None,
    no_speech_id: int | None,
    capture_alignment: bool = True,
    config: WhisperTurboConfig,
) -> tuple[np.ndarray | None, tuple[float, ...], float | None]:
    total_tokens = len(token_ids)
    if not 0 < total_tokens <= config.max_target_positions:
        raise ValueError("word-alignment token sequence is outside model capacity")
    expected_cross = (
        config.decoder_layers,
        1,
        config.max_source_positions,
        config.decoder_attention_heads,
        config.decoder_head_dim,
    )
    if (
        tuple(cross_keys.shape) != expected_cross
        or tuple(cross_values.shape) != expected_cross
        or cross_keys.device != decoder.token_embedding.device
        or cross_values.device != decoder.token_embedding.device
        or cross_keys.dtype != decoder.token_embedding.dtype
        or cross_values.dtype != decoder.token_embedding.dtype
    ):
        raise ValueError("word alignment received incompatible cross-attention state")
    if capture_alignment and not 0 < num_frames <= config.max_source_positions:
        raise ValueError("word-alignment frame count is outside model capacity")
    wanted = tuple((int(layer), int(head)) for layer, head in alignment_heads)
    if capture_alignment and (not wanted or len(set(wanted)) != len(wanted)):
        raise ValueError("word alignment requires unique decoder attention heads")
    if any(
        not 0 <= layer < config.decoder_layers
        or not 0 <= head < config.decoder_attention_heads
        for layer, head in wanted
    ):
        raise ValueError("word alignment head index is outside model geometry")

    ids = torch.tensor(
        token_ids, dtype=torch.int64, device=decoder.token_embedding.device
    )
    hidden = F.embedding(ids[None, :], decoder.token_embedding)
    hidden = hidden + decoder.position_embedding[:total_tokens]
    scores_by_head: dict[tuple[int, int], Tensor] = {}
    for layer_index, layer in enumerate(decoder.layers):
        residual = hidden
        normalized = _layer_norm(
            hidden,
            layer.self_attention_layer_norm,
            eps=config.layer_norm_eps,
        )
        query = _query(
            normalized,
            layer.self_attention,
            heads=config.decoder_attention_heads,
        )
        key, value = _key_value(
            normalized,
            layer.self_attention,
            heads=config.decoder_attention_heads,
        )
        hidden = residual + _attention(
            query,
            key,
            value,
            layer.self_attention.output,
            causal=True,
        )

        residual = hidden
        normalized = _layer_norm(
            hidden,
            layer.cross_attention_layer_norm,
            eps=config.layer_norm_eps,
        )
        query = _query(
            normalized,
            layer.cross_attention,
            heads=config.decoder_attention_heads,
        )
        cross_key = cross_keys[layer_index]
        cross_value = cross_values[layer_index]
        for wanted_layer, wanted_head in wanted:
            if capture_alignment and wanted_layer == layer_index:
                scores_by_head[(wanted_layer, wanted_head)] = (
                    query[:, :, wanted_head].float()
                    @ cross_key[:, :, wanted_head].float().transpose(-2, -1)
                )[0]
        hidden = residual + _attention(
            query,
            cross_key,
            cross_value,
            layer.cross_attention.output,
            causal=False,
        )

        residual = hidden
        normalized = _layer_norm(
            hidden,
            layer.final_layer_norm,
            eps=config.layer_norm_eps,
        )
        hidden = residual + _linear(
            F.gelu(_linear(normalized, layer.fc1), approximate="none"),
            layer.fc2,
        )

    hidden = _layer_norm(hidden, decoder.final_layer_norm, eps=config.layer_norm_eps)
    if (no_speech_position is None) != (no_speech_id is None):
        raise ValueError(
            "no-speech score position and token ID must be provided together"
        )
    prediction_start = prefix_length - 1
    if not 0 <= prediction_start < total_tokens:
        raise ValueError("transcript score prefix is outside the decoder sequence")
    prediction_hidden = hidden[
        0, prediction_start : prediction_start + len(scored_token_ids)
    ]
    if int(prediction_hidden.shape[0]) != len(scored_token_ids):
        raise ValueError("transcript score tokens exceed the decoder sequence")
    logprobs = _selected_token_logprobs(
        prediction_hidden,
        scored_token_ids,
        decoder.token_embedding,
    )
    no_speech_probability = None
    if no_speech_position is not None and no_speech_id is not None:
        if not 0 <= no_speech_position < total_tokens:
            raise ValueError("no-speech score position is outside the decoder sequence")
        if not 0 <= no_speech_id < int(decoder.token_embedding.shape[0]):
            raise ValueError("no-speech token ID is outside the decoder vocabulary")
        no_speech_logprob = _selected_token_logprobs(
            hidden[0, no_speech_position : no_speech_position + 1],
            (no_speech_id,),
            decoder.token_embedding,
        )[0]
        no_speech_probability = math.exp(no_speech_logprob)
        if not 0.0 <= no_speech_probability <= 1.0:
            raise RuntimeError(
                "Whisper decoder analysis produced invalid no-speech probability"
            )

    if not capture_alignment:
        return None, logprobs, no_speech_probability

    if set(scores_by_head) != set(wanted):
        raise RuntimeError(
            "word alignment did not capture every declared attention head"
        )
    scores = torch.stack([scores_by_head[index] for index in wanted])
    weights = torch.softmax(scores[..., :num_frames], dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    if not bool(torch.isfinite(std).all().item()) or bool((std <= 0).any().item()):
        raise RuntimeError("Whisper word alignment attention has zero variance")
    weights = _median_filter((weights - mean) / std)
    matrix = weights.mean(dim=0)[prediction_start:-1]
    expected_rows = len(scored_token_ids)
    if tuple(matrix.shape) != (expected_rows, num_frames):
        raise RuntimeError(
            "Whisper word-alignment matrix shape drifted: "
            f"got {tuple(matrix.shape)}, expected {(expected_rows, num_frames)}"
        )
    matrix_cpu = matrix.float().cpu().numpy()
    if not np.isfinite(matrix_cpu).all():
        raise RuntimeError("Whisper word-alignment matrix contains non-finite values")
    return matrix_cpu, logprobs, no_speech_probability


def _dtw(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if costs.ndim != 2 or not costs.shape[0] or not costs.shape[1]:
        raise ValueError("word-alignment DTW costs must be a non-empty matrix")
    token_count, frame_count = map(int, costs.shape)
    cost = np.full((token_count + 1, frame_count + 1), np.inf, dtype=np.float32)
    trace = np.full((token_count + 1, frame_count + 1), -1, dtype=np.int8)
    cost[0, 0] = 0.0
    for frame in range(1, frame_count + 1):
        for token in range(1, token_count + 1):
            diagonal = cost[token - 1, frame - 1]
            vertical = cost[token - 1, frame]
            horizontal = cost[token, frame - 1]
            if diagonal < vertical and diagonal < horizontal:
                previous, direction = diagonal, 0
            elif vertical < diagonal and vertical < horizontal:
                previous, direction = vertical, 1
            else:
                previous, direction = horizontal, 2
            cost[token, frame] = costs[token - 1, frame - 1] + previous
            trace[token, frame] = direction

    token = token_count
    frame = frame_count
    path: list[tuple[int, int]] = []
    while token > 0 or frame > 0:
        path.append((token - 1, frame - 1))
        direction = int(trace[token, frame])
        if direction == 0:
            token -= 1
            frame -= 1
        elif direction == 1 or frame == 0:
            token -= 1
        elif direction == 2 or token == 0:
            frame -= 1
        else:
            raise RuntimeError("word-alignment DTW backtrace is invalid")
    path.reverse()
    result = np.asarray(path, dtype=np.int32)
    return result[:, 0], result[:, 1]


def align_transcript_words(
    *,
    decoder: WhisperDecoderWeights,
    tokenizer: WhisperTokenizer,
    language: str,
    task: str,
    text_token_ids: Sequence[int],
    cross_keys: Tensor,
    cross_values: Tensor,
    num_frames: int,
    config: WhisperTurboConfig,
) -> tuple[AlignedWord, ...]:
    """Align words without shifting, fitting, or an incumbent oracle."""

    text = tuple(int(token_id) for token_id in text_token_ids)
    controls = tokenizer.controls
    prefix = (
        controls.decoder_start_id,
        controls.language_id(language),
        controls.task_id(task),
        controls.no_timestamps_id,
    )
    all_tokens = (*prefix, *text, controls.eos_id)
    matrix, logprobs, _ = _decoder_alignment_matrix(
        decoder,
        all_tokens,
        (*text, controls.eos_id),
        cross_keys,
        cross_values,
        controls.alignment_heads,
        prefix_length=len(prefix),
        num_frames=num_frames,
        no_speech_position=None,
        no_speech_id=None,
        config=config,
    )
    if not text:
        return ()
    if matrix is None:  # pragma: no cover - capture_alignment defaults true
        raise RuntimeError("Whisper word alignment omitted its attention matrix")
    text_indices, time_indices = _dtw(-matrix)
    words, word_tokens = tokenizer.split_to_word_tokens(
        (*text, controls.eos_id),
        language=_word_grouping_language(language, task),
    )
    if len(word_tokens) <= 1:
        return ()
    boundaries = np.pad(
        np.cumsum([len(token_group) for token_group in word_tokens[:-1]]),
        (1, 0),
    )
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / _TOKENS_PER_SECOND
    starts = jump_times[boundaries[:-1]]
    ends = jump_times[boundaries[1:]]
    aligned = []
    for word, token_group, start, end, left, right in zip(
        words,
        word_tokens,
        starts,
        ends,
        boundaries[:-1],
        boundaries[1:],
    ):
        if not word or not token_group or int(token_group[0]) >= controls.eos_id:
            continue
        probability = sum(
            math.exp(value) for value in logprobs[int(left) : int(right)]
        ) / (int(right) - int(left))
        aligned.append(
            AlignedWord(
                word=word,
                token_ids=tuple(token_group),
                start=float(start),
                end=float(max(start, end)),
                probability=float(min(1.0, max(0.0, probability))),
            )
        )
    return _merge_punctuation(aligned)


def no_speech_probability(
    *,
    decoder: WhisperDecoderWeights,
    tokenizer: WhisperTokenizer,
    prefix_token_ids: Sequence[int],
    cross_keys: Tensor,
    cross_values: Tensor,
    config: WhisperTurboConfig,
) -> float:
    """Return the raw no-speech probability at the final decoder-start token."""

    prefix = tuple(int(token_id) for token_id in prefix_token_ids)
    if not prefix:
        raise ValueError("no-speech scoring requires decoder prefix tokens")
    controls = tokenizer.controls
    try:
        no_speech_position = max(
            index
            for index, token_id in enumerate(prefix)
            if token_id == controls.decoder_start_id
        )
    except ValueError as exc:
        raise ValueError("no-speech scoring requires a decoder-start token") from exc
    # Causality makes later prompt controls irrelevant at this position. Keeping
    # only the prefix through SOT avoids replaying the generated transcript.
    scored_prefix = prefix[: no_speech_position + 1]
    _, _, probability = _decoder_alignment_matrix(
        decoder,
        scored_prefix,
        (),
        cross_keys,
        cross_values,
        (),
        prefix_length=len(scored_prefix),
        num_frames=0,
        no_speech_position=no_speech_position,
        no_speech_id=controls.no_speech_id,
        capture_alignment=False,
        config=config,
    )
    if probability is None:  # pragma: no cover - arguments above require a score
        raise RuntimeError("Whisper decoder omitted the no-speech probability")
    return probability


__all__ = [
    "AlignedWord",
    "TranscriptAnalysis",
    "TranscriptScores",
    "align_transcript_words",
    "no_speech_probability",
]
