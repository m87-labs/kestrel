# Fused MoE (vLLM-derived)

This directory vendors a minimal subset of the vLLM fused MoE implementation in order to accelerate Moondream decode on single H100 GPUs without taking a dependency on vLLM. Source material was taken from commit `4bf56c79cc252d285d0cb4f5edf323f02af735ca` of the vLLM repository under `ext/vllm`.

Key files:
- `kernels.py`: Triton kernels derived from `vllm/model_executor/layers/fused_moe/fused_moe.py` with quantization, EP/DP, and auxiliary code paths removed. Only the unquantized single-GPU path remains.
- `routing.py`: Pure PyTorch reimplementation of `moe_align_block_size` so that we do not depend on `vllm._custom_ops`.
- `module.py`: Thin wrapper that wires the kernels into the existing `ScatterMoEMLP` weights and exposes a drop-in backend for decode.

Any further updates to this directory should document the upstream source and justification for deviations so future contributors can re-sync changes if needed.
