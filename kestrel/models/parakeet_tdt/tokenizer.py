"""Parakeet TDT text and timestamp decoding."""

from __future__ import annotations

from pathlib import Path
import unicodedata

from tokenizers import Tokenizer

from kestrel.models.asr.contract import Character, Word


def _is_punctuation(piece: str) -> bool:
    stripped = piece.strip()
    return bool(stripped) and all(
        unicodedata.category(character).startswith("P") for character in stripped
    )


class ParakeetTokenizer:
    def __init__(self, path: str | Path) -> None:
        self.backend = Tokenizer.from_file(str(path))
        blank = self.backend.token_to_id("<blank>")
        pad = self.backend.token_to_id("<pad>")
        if blank is None or pad is None:
            raise ValueError("Parakeet tokenizer is missing blank or pad token")
        self.blank_token_id = blank
        self.pad_token_id = pad

    def decode(self, token_ids: list[int]) -> str:
        kept = [
            token
            for token in token_ids
            if token not in (self.blank_token_id, self.pad_token_id)
        ]
        return self.backend.decode(kept, skip_special_tokens=True)

    def words(
        self, token_ids: list[int], durations: list[int], frame_seconds: float
    ) -> tuple[Word, ...]:
        if len(token_ids) != len(durations):
            raise ValueError("Parakeet token and duration counts differ")
        frame = 0
        words: list[Word] = []
        current_ids: list[int] = []
        current_start = 0.0
        current_end = 0.0

        def flush() -> None:
            nonlocal current_ids
            if current_ids:
                text = self.backend.decode(
                    current_ids, skip_special_tokens=True
                ).strip()
                if text:
                    words.append(Word(text, current_start, current_end))
                current_ids = []

        for token_id, duration in zip(token_ids, durations):
            start = frame * frame_seconds
            frame += duration
            if token_id in (self.blank_token_id, self.pad_token_id):
                continue
            raw = self.backend.id_to_token(token_id) or ""
            piece = self.backend.decode([token_id], skip_special_tokens=True)
            if raw.startswith("▁"):
                flush()
            if _is_punctuation(piece):
                if current_ids:
                    current_ids.append(token_id)
                    current_end = frame * frame_seconds
                elif words:
                    previous = words[-1]
                    words[-1] = Word(
                        previous.text + piece,
                        previous.start,
                        frame * frame_seconds,
                    )
                continue
            if not current_ids:
                current_start = start
            current_ids.append(token_id)
            current_end = frame * frame_seconds
        flush()
        return tuple(words)

    def characters(
        self, token_ids: list[int], durations: list[int], frame_seconds: float
    ) -> tuple[Character, ...]:
        """Return NeMo-compatible character/subword token timestamps."""

        if len(token_ids) != len(durations):
            raise ValueError("Parakeet token and duration counts differ")
        frame = 0
        characters: list[Character] = []
        for token_id, duration in zip(token_ids, durations):
            start = frame * frame_seconds
            frame += duration
            if token_id in (self.blank_token_id, self.pad_token_id):
                continue
            text = self.backend.decode([token_id], skip_special_tokens=True)
            if not text:
                continue
            end = frame * frame_seconds
            if _is_punctuation(text) and characters:
                start = end = characters[-1].end
            characters.append(Character(text, start, end))
        return tuple(characters)


__all__ = ["ParakeetTokenizer"]
