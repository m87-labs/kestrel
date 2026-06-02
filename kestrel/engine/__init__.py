"""Engine package: the inference kernel and its executor lanes.

Re-exports the public surface so ``from kestrel.engine import ...``
keeps working after the split from a single module into a package.
"""

from kestrel.engine._types import (
    Completion,
    EngineMetrics,
    EngineResult,
    EngineStream,
    TickResult,
)
from kestrel.engine.executor import (
    AutoregressiveExecutor,
    Executor,
    _AdmissionCoordinator,
)
from kestrel.engine.single_pass import SinglePassExecutor
from kestrel.engine._types import _AutoregressiveRequest
from kestrel.engine.core import InferenceEngine

__all__ = [
    "InferenceEngine",
    "EngineResult",
    "EngineMetrics",
    "EngineStream",
    "Executor",
    "AutoregressiveExecutor",
    "SinglePassExecutor",
    "Completion",
    "TickResult",
]
