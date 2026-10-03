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
@pytest.mark.parametrize("deferred_consumer", [False, True])
@torch.inference_mode()
def test_explicit_devices_and_cache_rebind(deferred_consumer):
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
        consumer = None
        if deferred_consumer:
            from kestrel_kernels.graph_team import CudaGraphTeam
            from kestrel_kernels.peer_graph import PeerCopies

            proposals = torch.empty((1, config.block_size - 1), device=primary, dtype=torch.int32)
            streams = tuple(torch.cuda.Stream(device=device) for device in devices)
            buffers, transfers, graphs = [], [], []
            for rank, (device, stream) in enumerate(zip(devices, streams, strict=True)):
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    destination = torch.empty_like(proposals, device=device)
                    transfer = PeerCopies(device, (proposals,), (destination,))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        if rank == len(devices) - 1:
                            torch.cuda._sleep(500_000)
                        transfer.launch()
                        # Verification overwrites its input IDs; proposals must
                        # remain separate until the final acceptance readback.
                        destination.add_(rank + 1)
                    buffers.append(destination)
                    transfers.append(transfer)
                    graphs.append(graph)
            consumer = CudaGraphTeam(devices, graphs, streams, primary=session.stream,
                                     owners=(proposals, buffers, transfers))
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
                    if consumer is not None:
                        proposals.copy_(actual)
                        consumer.replay()
                    session.stream.synchronize()
                    assert (actual == expected).float().mean().item() >= .85
                    if consumer is not None:
                        torch.testing.assert_close(proposals, actual, rtol=0, atol=0)
                        for rank, destination in enumerate(buffers):
                            torch.testing.assert_close(destination.cpu(), proposals.cpu() + rank + 1,
                                                       rtol=0, atol=0)
                for actual_layer, expected_layer in zip(session.caches[0].layers, reference.caches[0].layers):
                    for name in ("keys", "values"):
                        torch.testing.assert_close(getattr(actual_layer, name)[:, :session.lengths[0]],
                                                   getattr(expected_layer, name)[:, :session.lengths[0]],
                                                   rtol=.03, atol=.03)
            reference.shutdown()
        if consumer is not None:
            consumer.close()
        session.shutdown()
        session.shutdown()
