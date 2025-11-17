"""Utilities for decoding segmentation SVG token streams."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from tokenizers import Tokenizer

PATH_COMMANDS = {
    "M",
    "m",
    "L",
    "l",
    "H",
    "h",
    "V",
    "v",
    "C",
    "c",
    "S",
    "s",
    "Q",
    "q",
    "T",
    "t",
    "A",
    "a",
    "Z",
    "z",
}


def decode_svg_token_strings(
    tokenizer: Tokenizer, token_ids: Sequence[int]
) -> List[str]:
    """Decode raw token ids into whitespace-trimmed SVG token strings."""

    payload = [[tid] for tid in token_ids if tid > 20]
    if not payload:
        return []
    decoded = tokenizer.decode_batch(payload, skip_special_tokens=True)
    return [item.strip() for item in decoded if item and item.strip()]


def parse_svg_tokens(tokens: Sequence[str]) -> List[str | List[int]]:
    """Group a flat list of SVG token strings into commands and coordinate pairs."""

    result: List[str | List[int]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token not in PATH_COMMANDS:
            raise ValueError(f"Unexpected SVG token '{token}'")
        result.append(token)
        i += 1

        numbers: List[int] = []
        while i < len(tokens) and tokens[i] not in PATH_COMMANDS:
            current = tokens[i]
            if current == "-":
                if i + 1 >= len(tokens):
                    break
                value = -int(tokens[i + 1])
                numbers.append(value)
                i += 2
                continue
            numbers.append(int(current))
            i += 1

        for j in range(0, len(numbers), 2):
            pair = numbers[j : j + 2]
            if len(pair) == 2:
                result.append(pair)
    return result


def svg_tokens_to_path(path: Sequence[str | Sequence[int]]) -> str:
    """Convert grouped SVG tokens into a canonical path string."""

    parts: List[str] = []
    pending_space = False
    for element in path:
        if isinstance(element, str):
            if pending_space:
                parts.append(" ")
            parts.append(element)
            pending_space = True
        else:
            for idx, value in enumerate(element):
                if pending_space:
                    parts.append(" ")
                parts.append(str(value))
                pending_space = True
    return "".join(parts)


def svg_path_from_token_ids(
    tokenizer: Tokenizer, token_ids: Sequence[int]
) -> tuple[str, List[str]]:
    """Decode token ids into a canonical SVG path and decoded token strings."""

    decoded = decode_svg_token_strings(tokenizer, token_ids)
    if not decoded:
        return "", []
    parsed = parse_svg_tokens(decoded)
    path = svg_tokens_to_path(parsed)
    return path, decoded


__all__ = [
    "decode_svg_token_strings",
    "parse_svg_tokens",
    "svg_tokens_to_path",
    "svg_path_from_token_ids",
]
