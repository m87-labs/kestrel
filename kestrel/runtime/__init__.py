"""Runtime data types shared across model-specific runtimes.

This package collects the model-agnostic dataclasses, named tuples, and
helper types that the engine and scheduler exchange with a runtime
implementation. Concrete runtimes (e.g. ``kestrel.moondream.runtime``)
build on top of these.
"""

from kestrel.runtime.state import (
    PrefillClassification,
    PreparedSequence,
    RuntimeDecodeResult,
    SequenceState,
    _CacheLookupResult,
)
from kestrel.runtime.tokens import (
    CoordToken,
    SizeToken,
    TextToken,
    Token,
)

__all__ = [
    "CoordToken",
    "PrefillClassification",
    "PreparedSequence",
    "RuntimeDecodeResult",
    "SequenceState",
    "SizeToken",
    "TextToken",
    "Token",
    "_CacheLookupResult",
]
