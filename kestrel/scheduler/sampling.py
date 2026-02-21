import torch

from kestrel_kernels.sampling import top_p_sampling_from_probs


_EPS = 1e-6

def sample_tokens(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
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

    return top_p_sampling_from_probs(
        probs, topp, generator=generator
    )
