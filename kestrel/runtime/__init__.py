"""Runtime data types shared across model-specific runtimes.

This package collects the model-agnostic dataclasses, named tuples, and
helper types that the engine and scheduler exchange with a runtime
implementation, plus the :class:`Runtime` protocol describing the
scheduler↔runtime contract. Concrete runtimes (e.g.
``kestrel.models.moondream.runtime``) build on top of these.
"""

from kestrel.runtime.protocol import (
    AutoregressiveRuntime,
    ExecutionShape,
    Runtime,
    SinglePassRuntime,
)
from kestrel.runtime.state import (
    PrefillClassification,
    PreparedSequence,
    RuntimeDecodeResult,
    SequenceState,
)
from kestrel.runtime.tokens import (
    CoordToken,
    SizeToken,
    TextToken,
    Token,
)

__all__ = [
    "AutoregressiveRuntime",
    "CoordToken",
    "ExecutionShape",
    "PrefillClassification",
    "PreparedSequence",
    "Runtime",
    "RuntimeDecodeResult",
    "SequenceState",
    "SinglePassRuntime",
    "SizeToken",
    "TextToken",
    "Token",
]
