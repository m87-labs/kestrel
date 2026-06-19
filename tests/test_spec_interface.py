"""Unit tests for the speculative-decoding contract (kestrel.runtime.spec).

Pure-CPU: exercises the inert scaffolding only (types + the capability hook).
No model construction, no GPU.
"""

from __future__ import annotations

from kestrel.runtime.spec import DraftResult, SpecDecodeCaps, SpecProposer


def test_draft_result_probs_default_none() -> None:
    dr = DraftResult(token_ids="<tensor>")  # token_ids is opaque here
    assert dr.draft_probs is None


class _StubProposer:
    """Minimal structural implementation of SpecProposer."""

    num_speculative_tokens = 16
    num_lookahead_tokens = 17

    def propose(self, ctx):  # noqa: ANN001 - ctx is intentionally open
        return DraftResult(token_ids="<tensor>")

    def commit_accept(self, ctx) -> None:  # noqa: ANN001
        return None


def test_stub_satisfies_spec_proposer_protocol() -> None:
    # runtime_checkable Protocol: structural conformance is enough.
    assert isinstance(_StubProposer(), SpecProposer)


def test_spec_decode_caps_defaults_to_no_hidden_layers() -> None:
    caps = SpecDecodeCaps(proposer=_StubProposer())
    assert caps.capture_hidden_layers == ()
    assert caps.proposer.num_lookahead_tokens == 17


def test_autoregressive_runtime_declares_spec_attribute() -> None:
    # The capability hook is part of the protocol surface, defaulting to None
    # on conforming runtimes. We assert the annotation exists rather than
    # constructing a runtime (which needs weights/GPU).
    from kestrel.runtime.protocol import AutoregressiveRuntime

    assert "spec" in AutoregressiveRuntime.__annotations__
