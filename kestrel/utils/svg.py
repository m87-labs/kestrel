"""Utilities for decoding segmentation SVG token streams."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

import numpy as np
import pyvips

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


def svg_tokens_to_path(path: Sequence[str | Sequence[int | float | str]]) -> str:
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


def tokens_to_path_string(tokens: Sequence[str]) -> str:
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
    path = tokens_to_path_string(scaled_tokens)
    return path, decoded


__all__ = [
    "decode_svg_token_strings",
    "parse_svg_tokens",
    "svg_tokens_to_path",
    "scale_svg_path_tokens",
    "tokens_to_path_string",
    "svg_path_from_token_ids",
    "tokens_to_raw_path",
    "split_path_tokens",
    "svg_from_path",
    "render_svg_to_mask",
]


def tokens_to_raw_path(tokens: Sequence[str]) -> str:
    """Convert decoded SVG tokens into an unscaled path string."""

    if not tokens:
        return ""
    parsed = parse_svg_tokens(tokens)
    return svg_tokens_to_path(parsed)


def split_path_tokens(path: str) -> List[str]:
    """Tokenize a raw SVG path string into command/number tokens."""

    tokens: List[str] = []
    for cmd, rest in re.findall(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)", path):
        tokens.append(cmd)
        if not rest:
            continue
        for num in re.findall(r"-?\d+(?:\.\d+)?", rest):
            if num.startswith("-"):
                tokens.append("-")
                num = num[1:]
            try:
                ival = int(round(float(num)))
            except Exception:
                continue
            tokens.append(str(ival))
    return tokens


def svg_from_path(
    svg_path: str,
    width: float,
    height: float,
    bbox: Sequence[float],
    viewbox: float = DEFAULT_VIEWBOX_SIZE,
) -> str:
    """
    Project a viewbox-space SVG path into image space using a normalized bbox [cx, cy, w, h].
    """

    x0 = (bbox[0] - bbox[2] / 2) * viewbox
    y0 = (bbox[1] - bbox[3] / 2) * viewbox
    sx = bbox[2]
    sy = bbox[3]

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(viewbox)} {int(viewbox)}" '
        f'preserveAspectRatio="none" width="{width}" height="{height}">'
        f'<path d="{svg_path}" fill="white" transform="translate({x0},{y0}) scale({sx},{sy})"/></svg>'
    )


def render_svg_to_mask(svg: str, width: int, height: int) -> np.ndarray:
    """Rasterize an SVG string to a boolean mask using pyvips."""

    width = int(round(width))
    height = int(round(height))

    normalized_svg = (
        svg.replace(",M", " M")
        .replace(",m", " m")
        .replace(",L", " L")
        .replace(",l", " l")
        .replace(",C", " C")
        .replace(",c", " c")
        .replace(",Z", " Z")
        .replace(",z", " z")
    )

    image = pyvips.Image.svgload_buffer(normalized_svg.encode("utf-8"), unlimited=True)

    if image.width != width or image.height != height:
        scale_x = width / image.width if image.width else 1.0
        scale_y = height / image.height if image.height else 1.0
        image = image.resize(scale_x, vscale=scale_y)

    if image.hasalpha():
        alpha = image.extract_band(image.bands - 1)
    else:
        alpha = image.colourspace("b-w")

    buffer = alpha.write_to_memory()
    mask = np.frombuffer(buffer, dtype=np.uint8).reshape(image.height, image.width)
    return mask > 0
