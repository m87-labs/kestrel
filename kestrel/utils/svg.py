"""Utilities for decoding segmentation SVG token streams."""


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

DEFAULT_VIEWBOX_SIZE = 960.0


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


def _format_number(value: float, decimals: int) -> str:
    """Format a float with up to ``decimals`` decimal places, stripping trailing zeros."""

    formatted = f"{value:.{decimals}f}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def scale_svg_path_tokens(
    parsed: Sequence[str | Sequence[int]],
    *,
    viewbox_size: float = DEFAULT_VIEWBOX_SIZE,
    decimals: int = 3,
) -> List[str]:
    """Scale parsed SVG tokens from a pixel viewbox to unit coords."""

    if viewbox_size == 0:
        raise ValueError("viewbox_size must be non-zero")

    scaled_tokens: List[str] = []
    for element in parsed:
        if isinstance(element, str):
            scaled_tokens.append(element)
            continue
        for value in element:
            scaled_value = float(value) / viewbox_size
            scaled_tokens.append(_format_number(scaled_value, decimals))
    return scaled_tokens


def _tokens_to_path_string(tokens: Sequence[str]) -> str:
    """Join a flat list of tokens into a canonical path string."""

    if not tokens:
        return ""
    return " ".join(tokens)


def svg_path_from_token_ids(
    tokenizer: Tokenizer, token_ids: Sequence[int]
) -> tuple[str, List[str]]:
    """Decode token ids into a scaled SVG path and decoded token strings."""

    decoded = decode_svg_token_strings(tokenizer, token_ids)
    if not decoded:
        return "", []
    parsed = parse_svg_tokens(decoded)
    scaled_tokens = scale_svg_path_tokens(parsed)
    path = _tokens_to_path_string(scaled_tokens)
    return path, decoded


__all__ = [
    "decode_svg_token_strings",
    "parse_svg_tokens",
    "scale_svg_path_tokens",
    "svg_path_from_token_ids",
]
