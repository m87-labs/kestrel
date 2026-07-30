"""Tests for the DFlash draft model.

- A pure-CPU shape/smoke test on a tiny config (always runs).
- A parity test of the kestrel port against the HF reference modeling code
  (``trust_remote_code``) on the published checkpoint — skipped unless CUDA,
  ``transformers``, and the (gated) checkpoint are available. Run in float32,
  where the port is bit-identical to the reference (bf16 differs only by
  rounding, ~2e-2 rel error on random inputs).
"""

from __future__ import annotations

import pytest
import torch

from kestrel.models.qwen35.dflash.model import DFlashConfig, DFlashDraftModel

REPO = "z-lab/Qwen3.5-4B-DFlash"


def _tiny_config() -> DFlashConfig:
    return DFlashConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=100,
        target_layer_ids=(0, 1, 2),
        block_size=4,
        mask_token_id=99,
    )


def _base_config_dict() -> dict:
    return {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 100,
    }


def test_from_dict_missing_dflash_config_keeps_defaults() -> None:
    # A checkpoint config that omits ``dflash_config`` entirely must preserve the
    # dataclass defaults for the nested keys rather than collapsing them to
    # ``None`` / ``()`` (which would build an unusable drafter).
    defaults = DFlashConfig()
    cfg = DFlashConfig.from_dict(_base_config_dict())
    assert cfg.mask_token_id == defaults.mask_token_id
    assert cfg.mask_token_id is not None
    assert cfg.target_layer_ids == defaults.target_layer_ids
    assert cfg.target_layer_ids != ()


def test_from_dict_empty_dflash_config_keeps_nested_defaults() -> None:
    # ``dflash_config`` present but with the nested keys absent: still fall back
    # to the dataclass defaults for those individual fields.
    data = {**_base_config_dict(), "dflash_config": {}}
    defaults = DFlashConfig()
    cfg = DFlashConfig.from_dict(data)
    assert cfg.mask_token_id == defaults.mask_token_id
    assert cfg.target_layer_ids == defaults.target_layer_ids


def test_from_dict_overrides_when_nested_keys_present() -> None:
    # When the nested keys ARE present they override the defaults.
    data = {
        **_base_config_dict(),
        "dflash_config": {"mask_token_id": 42, "target_layer_ids": [0, 1, 2]},
    }
    cfg = DFlashConfig.from_dict(data)
    assert cfg.mask_token_id == 42
    assert cfg.target_layer_ids == (0, 1, 2)


def test_dflash_forward_shape_cpu() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    model = DFlashDraftModel(cfg).eval()
    bsz, ctx_len, q_len = 2, 6, 5
    noise = torch.randn(bsz, q_len, cfg.hidden_size)
    target = torch.randn(bsz, ctx_len, len(cfg.target_layer_ids) * cfg.hidden_size)
    pos = torch.arange(ctx_len + q_len).unsqueeze(0).expand(bsz, -1)
    out = model(noise, target, pos)
    assert out.shape == (bsz, q_len, cfg.hidden_size)
    assert torch.isfinite(out).all()


def _load_hf_reference(device: torch.device):
    transformers = pytest.importorskip("transformers")
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    try:
        ref = transformers.AutoModel.from_pretrained(
            REPO,
            trust_remote_code=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
    except (GatedRepoError, HfHubHTTPError, OSError) as e:
        pytest.skip(f"DFlash checkpoint/reference unavailable: {type(e).__name__}")
    return ref.to(device).eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="parity test needs CUDA")
def test_dflash_matches_hf_reference() -> None:
    from kestrel.models.qwen35.dflash.loader import load_dflash_drafter

    device = torch.device("cuda")
    ref = _load_hf_reference(device)
    model, cfg = load_dflash_drafter(REPO, device=device, dtype=torch.float32)

    bsz, ctx_len, q_len = 2, 8, cfg.block_size
    torch.manual_seed(0)
    noise = torch.randn(bsz, q_len, cfg.hidden_size, device=device, dtype=torch.float32)
    target = torch.randn(
        bsz,
        ctx_len,
        len(cfg.target_layer_ids) * cfg.hidden_size,
        device=device,
        dtype=torch.float32,
    )
    pos = torch.arange(ctx_len + q_len, device=device).unsqueeze(0).expand(bsz, -1)

    with torch.no_grad():
        out = model(noise, target, pos)
        ref_out = ref(position_ids=pos, noise_embedding=noise, target_hidden=target)
        ref_out = ref_out if isinstance(ref_out, torch.Tensor) else ref_out.last_hidden_state

    assert out.shape == ref_out.shape
    rel = (out.float() - ref_out.float()).norm() / ref_out.float().norm()
    assert rel < 1e-4, f"relative error {rel.item():.4g} too large"


def _proposer(cfg: DFlashConfig):
    from torch import nn

    from kestrel.models.qwen35.dflash.proposer import DFlashProposer

    torch.manual_seed(0)
    drafter = DFlashDraftModel(cfg).eval()
    embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    return DFlashProposer(drafter, embed, lm_head, cfg)


def _ctx(cfg: DFlashConfig, bsz: int = 2, ctx_len: int = 6, **kw):
    from kestrel.models.qwen35.dflash.proposer import ProposeContext

    return ProposeContext(
        last_token_ids=torch.randint(0, cfg.vocab_size, (bsz,)),
        target_hidden=torch.randn(bsz, ctx_len, len(cfg.target_layer_ids) * cfg.hidden_size),
        position_ids=torch.arange(ctx_len + cfg.block_size).unsqueeze(0).expand(bsz, -1),
        **kw,
    )


def test_proposer_token_counts() -> None:
    p = _proposer(_tiny_config())
    assert p.num_speculative_tokens == 3  # block_size - 1
    assert p.num_lookahead_tokens == 4  # block_size


def test_proposer_greedy_shapes_and_determinism() -> None:
    cfg = _tiny_config()
    p = _proposer(cfg)
    ctx = _ctx(cfg)
    r1 = p.propose(ctx)
    r2 = p.propose(ctx)
    assert r1.token_ids.shape == (2, cfg.block_size - 1)
    assert r1.token_ids.dtype == torch.int32
    assert r1.draft_probs is None
    assert (r1.token_ids >= 0).all() and (r1.token_ids < cfg.vocab_size).all()
    assert torch.equal(r1.token_ids, r2.token_ids)  # greedy is deterministic


def test_proposer_greedy_matches_manual_composition() -> None:
    cfg = _tiny_config()
    p = _proposer(cfg)
    ctx = _ctx(cfg)
    bsz = ctx.last_token_ids.shape[0]
    block_ids = torch.full((bsz, cfg.block_size), cfg.mask_token_id, dtype=torch.long)
    block_ids[:, 0] = ctx.last_token_ids
    with torch.no_grad():
        hidden = p.drafter(p.embed_tokens(block_ids), ctx.target_hidden, ctx.position_ids)
        expected = p.lm_head(hidden[:, 1:, :]).argmax(dim=-1).to(torch.int32)
    assert torch.equal(p.propose(ctx).token_ids, expected)


def test_proposer_sampling_path() -> None:
    cfg = _tiny_config()
    p = _proposer(cfg)
    ctx = _ctx(cfg, temperature=torch.ones(2))
    r = p.propose(ctx)
    assert r.token_ids.shape == (2, cfg.block_size - 1)
    assert r.draft_probs.shape == (2, cfg.block_size - 1, cfg.vocab_size)
    assert (r.token_ids >= 0).all() and (r.token_ids < cfg.vocab_size).all()
