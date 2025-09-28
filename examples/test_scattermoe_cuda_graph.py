from __future__ import annotations

import argparse

import torch

from kestrel.scattermoe import ScatterMoEWorkspace, kernels


def _dtype_from_name(name: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]


def _make_inputs(
    *,
    batch_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    inputs = torch.randn(batch_tokens, hidden_size, generator=g, device=device, dtype=dtype)
    weights = torch.randn(
        num_experts, hidden_size, hidden_size, generator=g, device=device, dtype=dtype
    )
    gates = torch.randn(batch_tokens, top_k, generator=g, device=device, dtype=dtype)

    expert_single = torch.zeros(batch_tokens, top_k, dtype=torch.long, device=device)
    ramp = torch.arange(top_k, device=device)
    offsets = torch.arange(batch_tokens, device=device)[:, None]
    expert_spread = (ramp + offsets) % num_experts

    return inputs, weights, gates, expert_single, expert_spread


def _scatter_gather(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    expert_idxs: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(expert_idxs)
    padded_block_idxs, _, _ = kernels.ops.padded_block_indices(
        sorted_expert_idxs, weights.size(0)
    )
    return kernels.ops.scatter2scatter(
        inputs,
        weights,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        top_k,
        padded_block_idxs,
    )


def run_dynamic(args: argparse.Namespace) -> int:
    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)

    (
        inputs,
        weights,
        _gates,
        expert_single,
        expert_spread,
    ) = _make_inputs(
        batch_tokens=args.batch_tokens,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    expected_single = _scatter_gather(inputs, weights, expert_single, top_k=args.top_k)

    static_inputs = inputs.clone()
    static_expert_idxs = expert_single.clone()
    static_out = torch.empty_like(expected_single)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        result = _scatter_gather(static_inputs, weights, static_expert_idxs, top_k=args.top_k)
        static_out.copy_(result)
    torch.cuda.synchronize()

    graph.replay()
    torch.cuda.synchronize()

    first_diff = (static_out - expected_single).abs().max()
    if first_diff > args.tol:
        print(
            f"[dynamic] capture mismatch on reference replay (max |diff|={first_diff.item():.3e})"
        )
        return 2

    # Change routing so the padded block layout differs from the captured shape.
    static_expert_idxs.copy_(expert_spread)
    expected_spread = _scatter_gather(inputs, weights, expert_spread, top_k=args.top_k)
    graph.replay()
    torch.cuda.synchronize()

    final_diff = (static_out - expected_spread).abs().max()
    if final_diff > args.tol:
        print(
            f"[dynamic] CUDA graph replay diverged after routing change (max |diff|={final_diff.item():.3e})"
        )
        return 1

    print("[dynamic] CUDA graph replay unexpectedly succeeded.")
    return 0


def run_static(args: argparse.Namespace) -> int:
    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)

    (
        inputs,
        weights,
        _gates,
        expert_single,
        expert_spread,
    ) = _make_inputs(
        batch_tokens=args.batch_tokens,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    expected_single = _scatter_gather(inputs, weights, expert_single, top_k=args.top_k)
    expected_spread = _scatter_gather(inputs, weights, expert_spread, top_k=args.top_k)

    workspace = ScatterMoEWorkspace.allocate(
        batch_tokens=args.batch_tokens,
        hidden_size=args.hidden_size,
        top_k=args.top_k,
        device=device,
        dtype=dtype,
    )

    static_inputs = inputs.clone()

    def _materialise_workspace(expert_idxs: torch.Tensor) -> None:
        sorted_expert_idxs, sorted_scattered_idxs = kernels.ops.flatten_and_sort(expert_idxs)
        workspace.sorted_expert_idxs.copy_(sorted_expert_idxs)
        workspace.sorted_scattered_idxs.copy_(sorted_scattered_idxs)
        workspace.compute_block_indices(args.num_experts)

    _materialise_workspace(expert_single)

    graph_out = workspace.output
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        kernels.ops.scatter2scatter(
            static_inputs,
            weights,
            workspace.sorted_expert_idxs,
            workspace.sorted_scattered_idxs,
            args.top_k,
            workspace.padded_block_idxs,
            out=graph_out,
        )
    torch.cuda.synchronize()

    graph.replay()
    torch.cuda.synchronize()

    first_diff = (graph_out - expected_single).abs().max()
    if first_diff > args.tol:
        print(
            f"[static] capture mismatch on reference routing (max |diff|={first_diff.item():.3e})"
        )
        return 2

    _materialise_workspace(expert_spread)
    graph.replay()
    torch.cuda.synchronize()

    final_diff = (graph_out - expected_spread).abs().max()
    if final_diff > args.tol:
        print(
            f"[static] CUDA graph replay mismatch (max |diff|={final_diff.item():.3e})"
        )
        return 3

    # Verify functional parity outside CUDA graphs using the static buffers.
    ref_from_workspace = kernels.ops.scatter2scatter(
        inputs,
        weights,
        workspace.sorted_expert_idxs,
        workspace.sorted_scattered_idxs,
        args.top_k,
        workspace.padded_block_idxs,
        out=torch.empty_like(graph_out),
    )
    parity_diff = (ref_from_workspace - expected_spread).abs().max()
    if parity_diff > args.tol:
        print(f"[static] Non-graph parity mismatch (max |diff|={parity_diff.item():.3e})")
        return 4

    print("[static] CUDA graph replay matched reference outputs.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ScatterMoE CUDA graph harness")
    parser.add_argument("--mode", choices=("dynamic", "static"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-tokens", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    if args.top_k > args.num_experts:
        raise ValueError("top_k must not exceed num_experts")

    torch.backends.cuda.matmul.allow_tf32 = False

    if args.mode == "dynamic":
        return run_dynamic(args)
    return run_static(args)


if __name__ == "__main__":
    raise SystemExit(main())
