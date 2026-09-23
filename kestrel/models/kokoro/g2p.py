"""Lazy CPU-only grapheme-to-phoneme conversion for Kokoro."""

from __future__ import annotations

from typing import Any


LANGUAGE_ALIASES = {
    "a": "a",
    "en": "a",
    "en-us": "a",
    "b": "b",
    "en-gb": "b",
    "e": "e",
    "es": "e",
    "es-es": "e",
    "f": "f",
    "fr": "f",
    "fr-fr": "f",
    "h": "h",
    "hi": "h",
    "i": "i",
    "it": "i",
    "j": "j",
    "ja": "j",
    "ja-jp": "j",
    "p": "p",
    "pt": "p",
    "pt-br": "p",
    "z": "z",
    "zh": "z",
    "zh-cn": "z",
}

_ESPEAK_LANGUAGES = {
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}
_MAX_PHONEMES = 510
_PHONEME_BREAKS = frozenset(" \t\n.!?…:;,—。！？、，；：")


def _split_phonemes(phonemes: str) -> tuple[str, ...]:
    chunks = []
    remaining = phonemes.strip()
    while len(remaining) > _MAX_PHONEMES:
        window = remaining[:_MAX_PHONEMES]
        boundary = max(
            (window.rfind(char) + 1 for char in _PHONEME_BREAKS),
            default=0,
        )
        if boundary == 0:
            boundary = _MAX_PHONEMES
        chunk = remaining[:boundary].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[boundary:].strip()
    if remaining:
        chunks.append(remaining)
    return tuple(chunks)


def normalize_language(language: str) -> str:
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty string")
    key = language.strip().casefold().replace("_", "-")
    try:
        return LANGUAGE_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(LANGUAGE_ALIASES))
        raise ValueError(
            f"unsupported Kokoro language {language!r} (supported: {supported})"
        ) from exc


class KokoroG2P:
    """Owns lazy Misaki frontends; all calls and returned values stay on CPU."""

    def __init__(self) -> None:
        self._frontends: dict[str, Any] = {}

    def _frontend(self, language: str) -> Any:
        cached = self._frontends.get(language)
        if cached is not None:
            return cached
        if language in {"a", "b"}:
            try:
                from misaki import en
                import spacy.util
            except ImportError as exc:
                raise RuntimeError(
                    "English Kokoro synthesis requires misaki[en]>=0.9.4"
                ) from exc
            model_name = "en_core_web_sm"
            if not spacy.util.is_package(model_name):
                raise RuntimeError(
                    f"English Kokoro synthesis requires {model_name}; "
                    "run `python -m spacy download en_core_web_sm`"
                )
            try:
                from misaki import espeak

                fallback = espeak.EspeakFallback(british=language == "b")
            except Exception as exc:
                raise RuntimeError(
                    "English Kokoro synthesis requires the bundled eSpeak NG "
                    "backend; install `kestrel[kokoro]`"
                ) from exc
            frontend = en.G2P(
                trf=False,
                british=language == "b",
                fallback=fallback,
                unk="",
            )
        elif language == "j":
            try:
                from misaki import ja
            except ImportError as exc:
                raise RuntimeError("Japanese Kokoro synthesis requires misaki[ja]") from exc
            try:
                frontend = ja.JAG2P()
            except RuntimeError as exc:
                raise RuntimeError(
                    "Japanese Kokoro synthesis requires UniDic data; run "
                    "`python -m unidic download` after installing misaki[ja]"
                ) from exc
        elif language == "z":
            try:
                from misaki import zh
            except ImportError as exc:
                raise RuntimeError("Mandarin Kokoro synthesis requires misaki[zh]") from exc
            frontend = zh.ZHG2P(version=None)
        else:
            try:
                from misaki import espeak
                frontend = espeak.EspeakG2P(language=_ESPEAK_LANGUAGES[language])
            except Exception as exc:
                raise RuntimeError(
                    "this Kokoro language requires the bundled eSpeak NG backend; "
                    "install `kestrel[kokoro]`"
                ) from exc
        self._frontends[language] = frontend
        return frontend

    def phonemize(self, text: str, language: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        code = normalize_language(language)
        result = self._frontend(code)(text.strip())
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("Misaki returned an invalid G2P result")
        if code in {"a", "b"}:
            tokens = result[1]
            phonemes = "".join(
                (getattr(token, "phonemes", None) or "")
                + (" " if getattr(token, "whitespace", "") else "")
                for token in tokens
            ).strip()
        else:
            phonemes = result[0]
        if not isinstance(phonemes, str) or not phonemes:
            raise ValueError("G2P produced no Kokoro phonemes")
        return phonemes

    def phonemize_segments(self, text: str, language: str) -> tuple[str, ...]:
        """Phonemize once, then split on model-input rather than raw-text length."""

        return _split_phonemes(self.phonemize(text, language))


__all__ = ["KokoroG2P", "LANGUAGE_ALIASES", "normalize_language"]
