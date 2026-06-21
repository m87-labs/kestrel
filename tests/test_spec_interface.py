"""Unit tests for the speculative-decoding contract (kestrel.runtime.spec).

Pure-CPU: exercises the inert scaffolding only (types + the capability hook).
No model construction, no GPU.
"""

from __future__ import annotations

from kestrel.runtime.spec import (
    DraftResult,
    SpecDecodeCaps,
    SpecDecoder,
    SpecProposer,
    SpecStepResult,
)


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
    # The spec-step decoder is optional on the scaffolding (CPU tests build caps
    # that only advertise a proposer); it defaults to None.
    assert caps.decoder is None


class _StubState:
    def __init__(self) -> None:
        self.batch_idx = -1


class _StubDecoder:
    """Minimal structural implementation of the SpecDecoder contract."""

    num_speculative_tokens = 15

    @property
    def free_slots(self) -> int:
        return 4

    def admit(self, state, prompt_token_ids):  # noqa: ANN001
        state.batch_idx = 1
        return 42

    def step(self, states):  # noqa: ANN001
        return SpecStepResult(
            tokens=[[1, 2, 3] for _ in states],
            accept_counts=[2 for _ in states],
        )

    def retire(self, state) -> None:  # noqa: ANN001
        return None


def test_spec_step_result_lengths_match_accept_plus_one() -> None:
    res = SpecStepResult(tokens=[[7, 8]], accept_counts=[1])
    assert len(res.tokens[0]) == res.accept_counts[0] + 1


def test_stub_decoder_satisfies_protocol_and_sets_batch_idx() -> None:
    dec = _StubDecoder()
    assert isinstance(dec, SpecDecoder)
    state = _StubState()
    first = dec.admit(state, [1, 2, 3])
    assert first == 42
    assert state.batch_idx == 1  # admit assigns the pool row's batch index
    out = dec.step([state, state])
    assert len(out.tokens) == 2
    assert all(len(t) == a + 1 for t, a in zip(out.tokens, out.accept_counts))


def test_spec_decode_caps_carries_decoder() -> None:
    dec = _StubDecoder()
    caps = SpecDecodeCaps(
        proposer=_StubProposer(), capture_hidden_layers=(1, 5, 9), decoder=dec
    )
    assert caps.decoder is dec
    assert caps.capture_hidden_layers == (1, 5, 9)


def test_autoregressive_runtime_declares_spec_attribute() -> None:
    # The capability hook is part of the protocol surface, defaulting to None
    # on conforming runtimes. We assert the annotation exists rather than
    # constructing a runtime (which needs weights/GPU).
    from kestrel.runtime.protocol import AutoregressiveRuntime

    assert "spec" in AutoregressiveRuntime.__annotations__
