"""Speculative-decoding correctness regression tests.

These guard the ReplaySSM ring-buffer **flush** + CUDA-graph **verify** path of
``SpecDecoder``. The committed token sequence is a pure function of the prompt
and the (target, drafter) weights -- it MUST NOT depend on:

  * the ring-buffer ``flush_cap`` (when the ring fills it is folded into the
    checkpoint via ``materialize_recurrent_from_replay`` and the cursor resets;
    this is a state-preserving optimization, so the result is invariant), nor
  * whether verify runs eagerly or via a replayed CUDA graph.

A regression in any of those (capture/replay vs the in-place flush, the verify
kernel's ring write/reconstruct at large ``init_len``, conv-window roll, etc.)
shows up as the committed tokens diverging between configurations. This is
exactly the shape of the ``flush_cap=64`` failure that produced wrong tokens
starting just past the first ring flush (``init_len`` ~48) on H100 SXM but was
silent at ``flush_cap=32`` -- it would have been caught here.

The invariance check is drafter-independent (verify is exact-greedy regardless
of draft quality), so a RANDOM-weight drafter is used -- no gated checkpoint
needed. It is also tie-free: every configuration drives the *same* verify/draft
kernels, so agreement must be bit-exact (unlike a spec-vs-torch comparison,
which can legitimately differ on bf16 argmax ties).

Cost note: the dominant wall-clock here is the one-time JIT of the ReplaySSM
``replay_verify``/``materialize`` CuTe kernels (~20-40s each, keyed by the ring
``flush_cap``), so the suite is structured to (a) build the target model +
drafter **once** (module fixture) and (b) exercise the **fewest distinct caps**
that still cover the regression: one small cap that folds the ring on most
blocks, one larger cap that folds rarely, each run both graphed and eagerly.
Adding a third cap roughly adds another ~30s of pure compile for no extra
coverage, so it is deliberately omitted.

Run on GPU (the verify/draft path is CUDA-only): ``pytest -k spec_decode`` on a
CUDA host, or via the Modal harness.
"""

from __future__ import annotations

import pytest
import torch

import kestrel.models.qwen35  # noqa: F401  (registers model specs)
from kestrel.config import RuntimeConfig
from kestrel.kv_cache import KVMemoryPool
from kestrel.models.qwen35.dflash import SpecDecoder
from kestrel.models.qwen35.dflash.model import DFlashConfig, DFlashDraftModel
from kestrel.models.qwen35.runtime import Qwen35Runtime

# The model the flush_cap=64 regression was found on. Any hybrid Qwen3.5 with
# GDN layers exercises the same flush/verify code; 4B keeps the test faithful to
# the reported failure.
_MODEL_ID = "Qwen/Qwen3.5-4B"
_BLOCK_SIZE = 16
# Enough new tokens that, with a low-acceptance random drafter (ring grows
# ~1/step), the ring is folded into the checkpoint several times at the small
# cap and a few times at the large cap -- i.e. the flush path is genuinely
# exercised -- while keeping the per-step loop short. (Was 96; halved since the
# fold path is already covered many times over within ~48 steps.)
_NEW_TOKENS = 48
# Two distinct ring caps span the regression: cap=32 folds the ring every other
# block (chunk = cap // block = 2), cap=64 folds it every fourth block. Both are
# >= block_size (required) and reuse the same verify-block size, so each adds
# exactly one ``replay_verify``/``materialize`` JIT. cap=64 is the cap the
# graph/verify divergence was reported on; it is the invariance reference.
_REF_CAP = 64
_FOLD_CAP = 32


def _random_drafter(rt: Qwen35Runtime, block_size: int):
    """A randomly-initialized DFlash drafter dimensionally matched to ``rt``.

    Quality is irrelevant -- verify is exact-greedy -- but the shapes (hidden
    size, vocab, head dims, target taps) must match the target model.
    """
    tc = getattr(rt.hf_config, "text_config", rt.hf_config)
    n_layers = int(tc.num_hidden_layers)
    step = max(1, n_layers // 8)
    target_layer_ids = tuple(range(1, n_layers, step))[:8]  # 8 valid residual taps
    head_dim = int(getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads))
    torch.manual_seed(0)
    dcfg = DFlashConfig(
        hidden_size=int(tc.hidden_size),
        intermediate_size=4096,
        num_hidden_layers=2,
        num_attention_heads=int(tc.num_attention_heads),
        num_key_value_heads=int(tc.num_attention_heads),
        head_dim=head_dim,
        vocab_size=int(tc.vocab_size),
        rope_theta=1e7,
        block_size=block_size,
        mask_token_id=0,
        target_layer_ids=target_layer_ids,
    )
    drafter = DFlashDraftModel(dcfg).to(torch.device("cuda"), torch.bfloat16).eval()
    return drafter, dcfg


@pytest.fixture(scope="module")
def spec_fixture():
    """Build the target runtime + random drafter once for the whole module.

    Constructing ``Qwen35Runtime`` (weights load + decode-graph capture) is the
    bulk of the fixed cost after JIT; sharing it across both regression tests
    halves the runtime without weakening either assertion (each test still
    builds its own ``SpecDecoder`` per config).
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    dev = torch.device("cuda")
    rt = Qwen35Runtime(
        RuntimeConfig(device="cuda", model=_MODEL_ID, max_batch_size=1),
        kv_pool=KVMemoryPool(device=dev),
    )
    drafter, dcfg = _random_drafter(rt, _BLOCK_SIZE)
    return rt, drafter, dcfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_decode_invariant_to_flush_cap_and_execution_mode(spec_fixture):
    """Committed tokens must be identical across flush_cap and eager-vs-graph.

    Regression guard for the flush_cap=64 graph/verify bug. The two caps fold
    the ring at different frequencies (cap=32 every other block, cap=64 every
    fourth) and each runs both graphed and eager; the eager runs exercise the
    verify without CUDA graphs. All four configurations must agree bit-exactly,
    so neither the fold frequency nor graph capture/replay may perturb the
    committed tokens.
    """
    rt, drafter, dcfg = spec_fixture
    prompt = rt.tokenizer.encode(
        "Explain how a transformer neural network works in detail."
    ).ids

    configs = [
        ("graph", _REF_CAP),
        ("graph", _FOLD_CAP),
        ("eager", _REF_CAP),
        ("eager", _FOLD_CAP),
    ]
    results: dict[tuple[str, int], list[int]] = {}
    with torch.inference_mode():
        for mode, cap in configs:
            spec = SpecDecoder(rt, drafter, dcfg, flush_cap=cap)
            results[(mode, cap)] = spec.generate(
                prompt, _NEW_TOKENS, eager=(mode == "eager")
            ).token_ids

    ref_key = ("graph", _REF_CAP)
    ref = results[ref_key]
    for key, toks in results.items():
        n = min(len(toks), len(ref))
        first_div = next((i for i in range(n) if toks[i] != ref[i]), -1)
        assert toks == ref, (
            f"spec-decode committed tokens depend on config: {key} diverges from "
            f"{ref_key} at index {first_div} "
            f"({toks[max(0, first_div - 1):first_div + 3]} vs "
            f"{ref[max(0, first_div - 1):first_div + 3]}). The result must be "
            f"invariant to flush_cap and eager-vs-graph execution."
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_decode_eager_equals_graph_past_flush(spec_fixture):
    """Focused guard: at flush_cap=64, eager verify == graphed verify.

    This is the minimal expression of the regression (graphed verify at cap=64
    diverged from the eager forward past the first flush). Kept separate from
    the broad invariance test so a failure points straight at graph-vs-eager.
    Reuses the module fixture and the already-compiled cap=64 kernels, so it
    adds only two short generations.
    """
    rt, drafter, dcfg = spec_fixture
    prompt = rt.tokenizer.encode("Describe the water cycle step by step.").ids
    with torch.inference_mode():
        graphed = SpecDecoder(rt, drafter, dcfg, flush_cap=_REF_CAP).generate(
            prompt, _NEW_TOKENS, eager=False
        ).token_ids
        eager = SpecDecoder(rt, drafter, dcfg, flush_cap=_REF_CAP).generate(
            prompt, _NEW_TOKENS, eager=True
        ).token_ids
    n = min(len(graphed), len(eager))
    first_div = next((i for i in range(n) if graphed[i] != eager[i]), -1)
    assert graphed == eager, (
        f"graphed verify diverges from eager verify at flush_cap={_REF_CAP}, index "
        f"{first_div} (the ring flush interacts with CUDA-graph capture/replay)."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("short_new", [1, 3])
def test_spec_decode_chunk_loop_stops_at_max_new_tokens(spec_fixture, short_new):
    """A ``flush_cap`` >> ``block_size`` must stop the inner loop at the request.

    Regression for the chunked single-sequence decode overrun: the inner loop ran
    a fixed ``flush_cap // block_size`` verify blocks per chunk and only the OUTER
    ``while`` checked ``n_committed >= max_new_tokens``, so with a large flush
    window (``chunk`` large) and a small ``max_new_tokens`` the inner loop kept
    drafting/verifying full blocks after the requested length was reached. The
    page-table row reserves only ``4*block_size+8`` headroom, so (with
    ``page_size=1``) those extra blocks could write KV past the reserved row.

    Reuses the module fixture's already-compiled ``_REF_CAP`` (=64) kernels, so
    ``chunk = 64 // 16 = 4`` blocks per chunk. Asking for only 1 or 3 new tokens,
    a pre-fix build ran all 4 blocks (committing >= 4 tokens) regardless; the fix
    caps the inner loop by the remaining output, so the result must be EXACTLY
    ``max_new_tokens`` long. Its prefix must also agree with the ``_FOLD_CAP``
    (=32, chunk=2) run -- the committed tokens are a pure function of the inputs,
    so capping *how far* the loop speculates may never change them. Both caps are
    pre-compiled by the invariance test above, so this adds only short
    generations.
    """
    rt, drafter, dcfg = spec_fixture
    prompt = rt.tokenizer.encode("List the planets of the solar system.").ids
    with torch.inference_mode():
        big_chunk = SpecDecoder(rt, drafter, dcfg, flush_cap=_REF_CAP).generate(
            prompt, short_new, eager=False
        ).token_ids
        small_chunk = SpecDecoder(rt, drafter, dcfg, flush_cap=_FOLD_CAP).generate(
            prompt, short_new, eager=False
        ).token_ids
    assert len(big_chunk) == short_new, (
        f"chunk=4 generate returned {len(big_chunk)} tokens for max_new_tokens="
        f"{short_new}; the inner chunk loop over-committed past the request."
    )
    assert len(small_chunk) == short_new, (
        f"chunk=2 generate returned {len(small_chunk)} tokens for max_new_tokens="
        f"{short_new}.")
    first_div = next(
        (i for i in range(short_new) if big_chunk[i] != small_chunk[i]), -1)
    assert big_chunk == small_chunk, (
        f"chunk=4 diverges from chunk=2 at index {first_div} for max_new_tokens="
        f"{short_new}; capping the inner loop by remaining output must not change "
        f"the committed tokens."
    )
