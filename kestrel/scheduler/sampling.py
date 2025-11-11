import torch, flashinfer
from torch import Tensor


_EPS = 1e-6


@torch.library.custom_op("fi::topp_from_probs", mutates_args=())
def topp_from_probs(probs: Tensor, top_p: Tensor) -> Tensor:
    raise NotImplementedError("fi::topp_from_probs CPU path not implemented")

@topp_from_probs.register_fake
def _(probs, top_p):
    return probs.new_empty(probs.shape[:-1], dtype=torch.int32)

@topp_from_probs.register_kernel("cuda")
def _(probs: Tensor, top_p: Tensor) -> Tensor:
    return flashinfer.sampling.top_p_sampling_from_probs(probs, top_p)


def sample_tokens(logits: Tensor, temperatures: Tensor, top_ps: Tensor) -> Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (batch, vocab); received shape {logits.shape}")

    batch = logits.shape[0]
    if temperatures.shape[0] != batch or top_ps.shape[0] != batch:
        raise ValueError("Sampling parameters must match logits batch dimension")

    temps = torch.clamp(temperatures, min=0.0)
    topp = torch.clamp(top_ps, min=_EPS, max=1.0)

    greedy_ids = logits.argmax(dim=-1)

    # Avoid divide by zero, use 1.0 when temp = 0.
    eff_temp = torch.where(temps > _EPS, temps, torch.ones_like(temps)).unsqueeze(-1)

    scaled = logits / eff_temp
    probs = torch.softmax(scaled, dim=-1, dtype=torch.float32)

    row_invalid = ~torch.isfinite(probs).all(dim=-1)
    force_greedy = (temps <= _EPS) | row_invalid

    # Remove NaN/inf, zero out greedy, and set forced rows to 1.
    probs.nan_to_num_(0., posinf=0., neginf=0.)
    probs.masked_fill_(force_greedy.unsqueeze(1), 0.)
    probs.scatter_add_(1, greedy_ids.unsqueeze(1), force_greedy.to(probs.dtype).unsqueeze(1))

    # return flashinfer.sampling.top_p_sampling_from_probs(probs, topp)
    return torch.ops.fi.topp_from_probs(probs, topp)

