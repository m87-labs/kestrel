"""Pin the scheduler↔runtime surface against drift.

If anyone adds a member to the ``Runtime`` protocol without updating
``FakeRuntime``, this test catches it before the next scheduler test
fails with a confusing ``AttributeError``.
"""

from __future__ import annotations

from kestrel.runtime import Runtime

from tests.scheduler._fake_runtime import FakeRuntime


def _expected_members(protocol: type) -> set[str]:
    """Names declared on ``protocol`` — annotations + non-special methods."""

    members = set(getattr(protocol, "__annotations__", {}))
    for name in dir(protocol):
        if name.startswith("_"):
            continue
        if callable(getattr(protocol, name, None)):
            members.add(name)
    return members


def test_fake_runtime_implements_every_protocol_member() -> None:
    fake = FakeRuntime()
    missing = sorted(
        name for name in _expected_members(Runtime) if not hasattr(fake, name)
    )
    assert missing == [], (
        f"FakeRuntime is missing Runtime members: {missing}. "
        "Add them to tests/scheduler/_fake_runtime.py."
    )
