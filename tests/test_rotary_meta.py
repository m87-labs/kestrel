"""Meta model initialization must not import compilers for placeholder values."""

import subprocess
import sys

import pytest
import torch

from kestrel.ops.rotary import default_inv_freq, proportional_inv_freq


def test_meta_rotary_does_not_import_torch_compiler():
    code = '''
import sys
import torch
from kestrel.ops.rotary import default_inv_freq, proportional_inv_freq, MultidimensionalRotaryEmbedding
class RejectCompiler:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("torch._dynamo", "sympy"):
            raise AssertionError("unexpected compiler import: " + fullname)
sys.meta_path.insert(0, RejectCompiler())
assert "torch._dynamo" not in sys.modules
with torch.device("meta"):
    for fn in (default_inv_freq, proportional_inv_freq):
        result = fn(128, 10000, partial_rotary_factor=0.5)
        assert result.is_meta and result.dtype == torch.float32
    module = MultidimensionalRotaryEmbedding(128, 10000, dimensions=1)
    assert module.inv_freq.is_meta and module.inv_freq.shape == (64,)
assert "torch._dynamo" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path[:] = {sys.path!r}\n" + code],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fn", [default_inv_freq, proportional_inv_freq])
@pytest.mark.parametrize("partial", [0.25, 0.5, 1.0])
def test_meta_rotary_matches_real_metadata(fn, partial):
    real = fn(128, 10000, partial_rotary_factor=partial, factor=2.0)
    meta = fn(128, 10000, partial_rotary_factor=partial, factor=2.0,
              device=torch.device("meta"))
    assert meta.shape == real.shape and meta.dtype == real.dtype
    assert meta.is_meta
    pairs = int(partial * 128 // 2)
    denominator = int(128 * partial) if fn is default_inv_freq else 128
    expected = (1.0 / 10000 ** (torch.arange(0, pairs * 2, 2).float() / denominator)) / 2.0
    if fn is proportional_inv_freq:
        expected = torch.cat((expected, torch.zeros(64 - pairs)))
    torch.testing.assert_close(real, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fn", [default_inv_freq, proportional_inv_freq])
def test_meta_rotary_still_validates_dimensions(fn):
    with pytest.raises(ValueError):
        fn(127, 10000, device=torch.device("meta"))


@pytest.mark.parametrize("fn", [default_inv_freq, proportional_inv_freq])
@pytest.mark.parametrize("partial", [0.5, 1.0])
def test_meta_rotary_preserves_default_dtype_promotion(fn, partial):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        real = fn(128, 10000, partial_rotary_factor=partial)
        meta = fn(128, 10000, partial_rotary_factor=partial,
                  device=torch.device("meta"))
        assert meta.shape == real.shape and meta.dtype == real.dtype
    finally:
        torch.set_default_dtype(previous)
