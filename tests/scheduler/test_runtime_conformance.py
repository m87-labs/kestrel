"""Pin the scheduler↔runtime surface against drift.

If anyone adds a member to the ``Runtime`` protocol without updating
``FakeRuntime`` — or changes a method signature so the fake no longer
accepts what the protocol promises — this test catches it before the
next scheduler test fails with a confusing ``AttributeError`` /
``TypeError``.
"""

from __future__ import annotations

import inspect

from kestrel.runtime import AutoregressiveRuntime

from tests.scheduler._fake_runtime import FakeRuntime


def _protocol_members(protocol: type) -> set[str]:
    """Names declared on ``protocol`` — annotations + non-special methods."""

    members: set[str] = set()
    for klass in getattr(protocol, "__mro__", [protocol]):
        members.update(getattr(klass, "__annotations__", {}))
    for name in dir(protocol):
        if name.startswith("_"):
            continue
        if callable(getattr(protocol, name, None)):
            members.add(name)
    return members


def _params_excluding_self(func: object) -> list[inspect.Parameter]:
    sig = inspect.signature(func)  # type: ignore[arg-type]
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    return params


def test_fake_runtime_implements_every_protocol_member() -> None:
    fake = FakeRuntime()
    missing = sorted(
        name for name in _protocol_members(AutoregressiveRuntime) if not hasattr(fake, name)
    )
    assert missing == [], (
        f"FakeRuntime is missing Runtime members: {missing}. "
        "Add them to tests/scheduler/_fake_runtime.py."
    )


def _is_required(param: inspect.Parameter) -> bool:
    """A param is "required" if it has no default and isn't variadic."""

    if param.default is not inspect.Parameter.empty:
        return False
    return param.kind not in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )


def test_fake_runtime_method_signatures_accept_protocol_calls() -> None:
    """Every Protocol method's parameter list is satisfied by the fake.

    ``FakeRuntime`` may add extra *optional* parameters
    (Liskov-allowed), but:
    - every parameter ``Runtime`` declares must be present on the fake
      with the same kind, and the fake can't require a parameter the
      Protocol marks optional;
    - the fake can't add an extra required parameter the Protocol
      doesn't declare — Protocol-valid calls would raise ``TypeError``.
    """

    mismatches: list[str] = []
    for name in _protocol_members(AutoregressiveRuntime):
        proto_member = getattr(AutoregressiveRuntime, name, None)
        if not callable(proto_member):
            continue
        fake_member = getattr(FakeRuntime, name, None)
        if not callable(fake_member):
            mismatches.append(f"{name}: not callable on FakeRuntime")
            continue

        proto_params = {p.name: p for p in _params_excluding_self(proto_member)}
        fake_params = {p.name: p for p in _params_excluding_self(fake_member)}

        for pname, proto_p in proto_params.items():
            fake_p = fake_params.get(pname)
            if fake_p is None:
                mismatches.append(f"{name}: missing parameter {pname!r}")
                continue
            if proto_p.kind != fake_p.kind:
                mismatches.append(
                    f"{name}: parameter {pname!r} kind mismatch "
                    f"(proto={proto_p.kind.name} fake={fake_p.kind.name})"
                )
            if not _is_required(proto_p) and _is_required(fake_p):
                mismatches.append(
                    f"{name}: parameter {pname!r} is optional in Runtime but "
                    "required in FakeRuntime"
                )

        for pname, fake_p in fake_params.items():
            if pname in proto_params:
                continue
            if _is_required(fake_p):
                mismatches.append(
                    f"{name}: extra required parameter {pname!r} on "
                    "FakeRuntime that Runtime doesn't declare"
                )

    assert mismatches == [], (
        "FakeRuntime signatures drifted from Runtime:\n  - "
        + "\n  - ".join(mismatches)
    )
