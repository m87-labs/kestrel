"""The patch operand is padded to the tensor-core alignment, exactly.

The SigLIP patch dimension is ``14*14*3 = 588``, which is only 4-element aligned. In
bf16 Hopper ``wgmma`` needs 8 elements (16 bytes), so at K=588 no sm90 tile in
cuBLAS's kernel set is eligible and the heuristic falls back to an **Ampere**
``s16816`` kernel on a Hopper part -- measured at 130.56 us against 36.96 us for the
sm90 kernel bound at K=592, i.e. +93.60 us per forward at nc=13.

Padding K to 592 fixes it and is EXACT, because the added columns are zero on both
operands. These tests pin that exactness, and pin the checkpoint path: a real
588-column checkpoint must keep loading with no re-export.
"""

import pytest
import torch

from kestrel.models.moondream.vision import (
    PATCH_DIM_ALIGN,
    aligned_patch_dim,
    create_patches,
)
from kestrel.models.moondream.weights import _copy_patch_emb_weight


def _naive_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """The obvious, unpadded implementation this optimisation must not change.

    Deliberately written the slow, readable way. `create_patches` writes through a
    strided view to avoid paying the copy twice, and clever strides are exactly the
    kind of code that needs an independent reference to check it.
    """
    bsz, channels, height, width = x.shape
    p1 = p2 = patch_size
    y = x.reshape(bsz, channels, height // p1, p1, width // p2, p2)
    y = y.permute(0, 2, 4, 1, 3, 5)
    return y.reshape(bsz, (height // p1) * (width // p2), channels * p1 * p2)


def test_aligned_patch_dim_rounds_up_to_the_wgmma_alignment():
    assert aligned_patch_dim(588) == 592
    assert PATCH_DIM_ALIGN == 8
    # Already-aligned dims must not move: padding an aligned operand would be pure
    # cost with no kernel change to pay for it.
    assert aligned_patch_dim(592) == 592
    assert aligned_patch_dim(1152) == 1152
    for raw in range(1, 200):
        out = aligned_patch_dim(raw)
        assert out % PATCH_DIM_ALIGN == 0
        assert 0 <= out - raw < PATCH_DIM_ALIGN


def test_padded_patches_are_bit_identical_to_the_naive_result():
    """The payload must survive the strided write BIT-EXACTLY, not approximately."""
    torch.manual_seed(0)
    x = torch.randn(3, 3, 378, 378)
    got = create_patches(x, 14)
    want = _naive_patches(x, 14)

    assert got.shape == (3, 729, 592)
    assert want.shape == (3, 729, 588)
    # Bit-identical, asserted by equality on the raw payload -- not a cosine. At
    # 1.3e6 elements a cosine cannot resolve identity from near-identity.
    assert torch.equal(got[:, :, :588], want)


def test_the_pad_columns_are_exactly_zero():
    """Zero is what makes the pad exact; anything else silently changes the model."""
    torch.manual_seed(1)
    x = torch.randn(2, 3, 378, 378)
    out = create_patches(x, 14)
    tail = out[:, :, 588:]
    assert tail.shape[-1] == 4
    assert torch.equal(tail, torch.zeros_like(tail))


def test_the_padded_linear_reproduces_the_unpadded_one_exactly_in_fp64():
    """End to end: pad the operand AND the weight, get the same answer.

    In fp64 the zero columns contribute exactly zero and the accumulation order for
    the shared prefix is unchanged, so this is an equality assertion, not a
    tolerance. (In bf16 the padded GEMM binds a different kernel with a different
    accumulation order, so the two agree to fp-noise rather than bitwise -- which is
    why the exactness argument is made here, in a dtype that can express it.)
    """
    torch.manual_seed(2)
    x = torch.randn(2, 3, 56, 56, dtype=torch.float64)
    raw, dim = 588, 592

    w_raw = torch.randn(1152, raw, dtype=torch.float64)
    b = torch.randn(1152, dtype=torch.float64)
    w_pad = torch.zeros(1152, dim, dtype=torch.float64)
    w_pad[:, :raw] = w_raw

    ref = torch.nn.functional.linear(_naive_patches(x, 14), w_raw, b)
    got = torch.nn.functional.linear(create_patches(x, 14), w_pad, b)
    assert torch.equal(ref, got)


def test_an_already_aligned_shape_is_not_padded_and_takes_the_fast_path():
    """No pad where none is needed: 2*14*14 = 392, and 392 % 8 == 0."""
    torch.manual_seed(3)
    x = torch.randn(1, 2, 28, 28)
    out = create_patches(x, 14)
    assert out.shape[-1] == 392
    assert torch.equal(out, _naive_patches(x, 14))


def test_other_unaligned_shapes_pad_and_round_trip_exactly():
    """The rule is derived from the tensor, not hardcoded for 588.

    1*14*14 = 196 = 4 * 49, so it is 4-element aligned like 588 is -- the same
    cliff, at a different width -- and rounds to 200.
    """
    torch.manual_seed(4)
    x = torch.randn(1, 1, 28, 28)
    out = create_patches(x, 14)
    assert out.shape[-1] == 200
    assert torch.equal(out[:, :, :196], _naive_patches(x, 14))
    assert torch.equal(out[:, :, 196:], torch.zeros_like(out[:, :, 196:]))


# --------------------------------------------------------------------------- #
# the checkpoint path -- a real 588-column checkpoint must load unchanged      #
# --------------------------------------------------------------------------- #

def test_a_raw_width_checkpoint_loads_into_the_padded_parameter():
    """This is the case that must not require a re-export."""
    param = torch.nn.Parameter(torch.full((1152, 592), 7.0))
    src = torch.randn(1152, 588)

    _copy_patch_emb_weight(param, src)

    assert torch.equal(param.data[:, :588], src)
    # The pad must be ZEROED, not left holding whatever the parameter was
    # initialised with -- nn.Linear's default init is random, and a random pad
    # column multiplies a zero activation today but would silently corrupt the
    # encoder the moment anything ever wrote a non-zero there.
    assert torch.equal(param.data[:, 588:], torch.zeros(1152, 4))


def test_an_already_padded_checkpoint_loads_unchanged():
    param = torch.nn.Parameter(torch.zeros(1152, 592))
    src = torch.randn(1152, 592)
    _copy_patch_emb_weight(param, src)
    assert torch.equal(param.data, src)


def test_a_genuinely_wrong_shape_refuses():
    """The one place a real checkpoint could silently mis-load.

    A wrong-shaped patch_emb yields a plausible-looking encoder that is quietly
    wrong, so the mismatch must raise rather than broadcast or truncate.
    """
    param = torch.nn.Parameter(torch.zeros(1152, 592))
    with pytest.raises(ValueError, match="not compatible"):
        _copy_patch_emb_weight(param, torch.randn(1152, 600))   # too wide
    with pytest.raises(ValueError, match="not compatible"):
        _copy_patch_emb_weight(param, torch.randn(768, 588))    # wrong out_features
    with pytest.raises(ValueError, match="not compatible"):
        _copy_patch_emb_weight(param, torch.randn(592))         # wrong rank


def test_a_freshly_built_model_has_a_ZEROED_pad_before_any_checkpoint_load():
    """The pad must be inert from BOTH sides, not just the activation side.

    `nn.Linear(592, ...)` random-inits all 592 columns. That is harmless only while
    `create_patches` keeps zeroing the matching activation columns -- a one-sided
    invariant. This pins the weight side so the two cannot drift apart, and so a
    built-only model matches a built-then-loaded one.
    """
    from kestrel.models.moondream.config import VisionConfig
    from kestrel.models.moondream.vision import build_vision_model

    cfg = VisionConfig()
    model = build_vision_model(cfg, torch.float32, device="cpu")
    pad = model["patch_emb"].weight[:, 588:]
    assert pad.shape[-1] == 4
    assert torch.equal(pad, torch.zeros_like(pad))
    # The real columns must NOT have been zeroed with it.
    assert model["patch_emb"].weight[:, :588].abs().sum() > 0


def test_build_vision_model_uses_the_aligned_width():
    from kestrel.models.moondream.config import VisionConfig
    from kestrel.models.moondream.vision import build_vision_model

    cfg = VisionConfig()
    model = build_vision_model(cfg, torch.float32, device="cpu")
    raw = cfg.enc_patch_size * cfg.enc_patch_size * cfg.in_channels
    assert raw == 588
    assert model["patch_emb"].weight.shape == (cfg.enc_dim, 592)
    # And the activation the tower feeds it must agree, or the GEMM will not run.
    x = torch.randn(1, cfg.in_channels, cfg.crop_size, cfg.crop_size)
    assert create_patches(x, cfg.enc_patch_size).shape[-1] == 592
