"""GPU state ownership and boundary tests for the generated TDT host loop."""

from types import SimpleNamespace

import pytest
import torch

from kestrel.models.parakeet_tdt.generated_decode import _TdtBatchGeneratedDecoder
from kestrel.models.parakeet_tdt.model import TdtState


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _decoder():
    device = torch.device("cuda")
    config = SimpleNamespace(
        blank_token_id=2,
        decoder_hidden_size=4,
        num_decoder_layers=2,
        durations=(0, 1, 2, 3),
        max_symbols_per_step=2,
        encoder=SimpleNamespace(max_position_embeddings=8),
    )

    def initial(tokens, state):
        batch = tokens.shape[0]
        return torch.zeros(batch, 1, 4, device=device), (
            torch.zeros(2, batch, 4, device=device),
            torch.zeros(2, batch, 4, device=device),
        )

    model = SimpleNamespace(
        config=config,
        decoder=initial,
        encoder_projector=SimpleNamespace(weight=torch.zeros(1, device=device)),
    )
    decoder = _TdtBatchGeneratedDecoder(
        model,
        max_batch=3,
        compute_stream=torch.cuda.current_stream(),
    )

    def static_launcher(slot, batch):
        def launch():
            rows = torch.arange(batch, device=device)
            frame = slot.frame_indices[:batch] - rows * 8
            data = slot.encoded[rows, frame]
            token, duration = data[:, 0].int(), data[:, 1].int()
            slot.decisions.gpu.view(2, -1)[0, :batch].copy_(token)
            slot.decisions.gpu.view(2, -1)[1, :batch].copy_(duration)
            emitted = slot.active[:batch] & (token != 2)
            for value in (slot.decoder_hidden, *slot.hidden, *slot.cell):
                value[:batch].add_(emitted[:, None])
            advance = torch.where((token == 2) & (duration == 0), 1, duration)
            following = frame + advance
            slot.frame_indices[:batch].copy_(
                torch.where(
                    slot.active[:batch],
                    rows * 8 + following.clamp_max(7),
                    slot.frame_indices[:batch],
                )
            )
            slot.active[:batch].logical_and_(following < slot.valid_lengths[:batch])

        return launch

    decoder._generated = SimpleNamespace(static_launcher=static_launcher)
    return decoder


@pytest.mark.parametrize("max_tokens", [None, 0, 1, 2, 3])
def test_stream_boundaries_budgets_and_owned_state(max_tokens):
    decoder = _decoder()
    encoded = torch.zeros(3, 8, 4, device="cuda")
    # Row 0 emits with zero duration until the token/step policy stops it.
    encoded[0, :, 0] = 1
    # Row 1 immediately overshoots its two-frame chunk by one frame.
    encoded[1, :, 0] = 1
    encoded[1, :, 1] = 3
    # Row 2 consists of blank zero-duration decisions (must advance one).
    encoded[2, :, 0] = 2
    valid = torch.ones(3, 8, dtype=torch.bool, device="cuda")
    output = decoder.generate_windows(
        encoded,
        valid,
        max_tokens=max_tokens,
        start_frames=[1, 2, 0],
        frame_counts=[2, 2, 2],
        states=[None] * 3,
    )
    limit = 4 if max_tokens is None else min(4, max_tokens)
    expected_emits = [limit, int(max_tokens is None or max_tokens > 0), 0]
    for row, result in enumerate(output):
        state = result.state
        assert state is not None
        for value in (state.decoder_hidden, state.hidden, state.cell):
            torch.testing.assert_close(
                value, torch.full_like(value, expected_emits[row])
            )
    assert output[1].state.carry == (1 if expected_emits[1] else 0)
    assert output[2].sequences.tolist() == ([[2, 2, 2]] if max_tokens != 0 else [[2]])

    snapshot = output[1].state.hidden.clone()
    # A different cohort reuses launch storage but must not alter returned state.
    decoder.generate_windows(
        encoded,
        valid,
        max_tokens=1,
        start_frames=[0] * 3,
        frame_counts=[1] * 3,
        states=[None] * 3,
    )
    torch.testing.assert_close(output[1].state.hidden, snapshot)

    # The next chunk can occupy a different batch row and resumes exactly the
    # saved recurrent state, including a duration that crossed its boundary.
    (resumed,) = decoder.generate_windows(
        encoded[1:2],
        valid[1:2],
        max_tokens=1,
        start_frames=[0],
        frame_counts=[2],
        states=[output[1].state],
    )
    torch.testing.assert_close(resumed.state.hidden, snapshot + 1)
    assert resumed.state.carry == output[1].state.carry + 1


def test_carry_skips_window_without_mutating_state():
    decoder = _decoder()
    prior = TdtState(
        torch.full((1, 1, 4), 7.0, device="cuda"),
        torch.full((2, 1, 4), 8.0, device="cuda"),
        torch.full((2, 1, 4), 9.0, device="cuda"),
        carry=3,
    )
    encoded = torch.ones(1, 8, 4, device="cuda")
    (result,) = decoder.generate_windows(
        encoded,
        torch.ones(1, 8, dtype=torch.bool, device="cuda"),
        max_tokens=None,
        start_frames=[7],
        frame_counts=[1],
        states=[prior],
    )
    assert result.sequences.tolist() == [[2]]
    assert result.durations.tolist() == [[1]]
    assert result.state.carry == 2
    for name in ("hidden", "cell", "decoder_hidden"):
        torch.testing.assert_close(getattr(result.state, name), getattr(prior, name))
