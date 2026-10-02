"""Explicit-device draft sessions preserve committed caches and greedy ordering."""

import copy
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35.dflash import DFlashConfig, DFlashContextCache, DFlashDraftModel
from kestrel.models.qwen35.distributed_draft import DistributedDFlashDraftSession
from kestrel.models.qwen35.draft_workspace import DFlashDraftGraphSession


def _config(**changes):
    values = dict(hidden_size=5120, intermediate_size=128, num_hidden_layers=1,
                  num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                  rms_norm_eps=1e-6, rope_theta=10000, block_size=16,
                  mask_token_id=0, target_layer_ids=(0,), layer_types=("full_attention",),
                  sliding_window=None)
    values.update(changes)
    return DFlashConfig(**values)


@pytest.mark.parametrize("devices,changes", [((0,), {}), ((0, 0), {}),
                                           ((0, 1), {"conv_kernel_size": 3}),
                                           ((0, 1), {"selector_rank": 16}),
                                           ((0, 1), {"num_key_value_heads": 3})])
def test_rejects_unsupported_placement_before_cuda(devices, changes):
    model = SimpleNamespace(config=_config(**changes))
    head = torch.nn.Linear(5120, 128, bias=False, device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible BF16 transformer"):
        DistributedDFlashDraftSession(model, [object()], lm_head=head, devices=devices)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="eight peer GPUs required")
@torch.inference_mode()
def test_explicit_devices_and_cache_rebind():
    devices = tuple(reversed(range(8)))
    primary, config = devices[0], _config()
    with torch.cuda.device(primary):
        torch.manual_seed(501)
        model = DFlashDraftModel(config).to(device=primary, dtype=torch.bfloat16).eval()
        head = torch.nn.Linear(config.hidden_size, 128, bias=False,
                               device=primary, dtype=torch.bfloat16).requires_grad_(False)
        noise = torch.randn((1, config.block_size, 5120), device=primary, dtype=torch.bfloat16)
        context = torch.randn((1, 4, 5120), device=primary, dtype=torch.bfloat16)
        cache = DFlashContextCache(64)
        model(noise, context, torch.arange(4 + config.block_size, device=primary)[None], context_cache=cache)
        session = DistributedDFlashDraftSession(model, [copy.deepcopy(cache)], lm_head=head, devices=devices)
        for rebind in (False, True):
            if rebind:
                session.rebind([copy.deepcopy(cache)])
            reference = DFlashDraftGraphSession(model, [copy.deepcopy(cache)], lm_head=head)
            for rows in (1, 16, 0, 3):
                context = torch.randn((1, rows, 5120), device=primary, dtype=torch.bfloat16)
                start = session.lengths[0]
                positions = torch.arange(start, start + rows + config.block_size, device=primary)[None]
                with reference.launch([noise], [context], [positions]) as expected:
                    reference.stream.synchronize()
                    expected = expected.clone()
                with session.launch([noise], [context], [positions]) as actual:
                    session.stream.synchronize()
                    assert (actual == expected).float().mean().item() >= .85
                for actual_layer, expected_layer in zip(session.caches[0].layers, reference.caches[0].layers):
                    for name in ("keys", "values"):
                        torch.testing.assert_close(getattr(actual_layer, name)[:, :session.lengths[0]],
                                                   getattr(expected_layer, name)[:, :session.lengths[0]],
                                                   rtol=.03, atol=.03)
            reference.shutdown()
        session.shutdown()
        session.shutdown()
