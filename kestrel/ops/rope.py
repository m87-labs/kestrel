from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_D2": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_D2": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_D2": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_D2": 128}, num_warps=8, num_stages=3),
    ],
    key=["Hq", "Hk", "D", "R"],
)
@triton.jit
def _rope_interleaved_strided(
    q_in,
    q_bs,
    q_hs,
    q_ss,
    k_in,
    k_bs,
    k_hs,
    k_ss,
    q_out,
    q_out_bs,
    q_out_hs,
    q_out_ss,
    k_out,
    k_out_bs,
    k_out_hs,
    k_out_ss,
    cos,
    cos_bs,
    cos_ss,
    sin,
    sin_bs,
    sin_ss,
    seq_len,
    Hq: tl.constexpr,
    Hk: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    COS_BROADCAST: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D2: tl.constexpr,
):
    pid_x = tl.program_id(0).to(tl.int64)
    pid_y = tl.program_id(1)

    b = pid_x // seq_len
    s = pid_x % seq_len

    q_row_in = q_in + b * q_bs + s * q_ss
    k_row_in = k_in + b * k_bs + s * k_ss
    q_row_out = q_out + b * q_out_bs + s * q_out_ss
    k_row_out = k_out + b * k_out_bs + s * k_out_ss

    if COS_BROADCAST:
        cos_row = cos + s * cos_ss
        sin_row = sin + s * sin_ss
    else:
        cos_row = cos + b * cos_bs + s * cos_ss
        sin_row = sin + b * sin_bs + s * sin_ss

    h = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
    qh_mask = h < Hq
    kh_mask = h < Hk

    tl.static_assert(D % 2 == 0)
    tl.static_assert(R > 0)

    q_hbase_in = h[:, None] * q_hs
    k_hbase_in = h[:, None] * k_hs
    q_hbase_out = h[:, None] * q_out_hs
    k_hbase_out = h[:, None] * k_out_hs

    idx = tl.arange(0, BLOCK_D2)

    for d2 in tl.static_range(0, R, BLOCK_D2):
        j = d2 + idx
        m = j < R
        js = tl.where(m, j, R - 1).to(tl.int64)

        c = tl.load(cos_row + js, mask=m, other=0.0).to(tl.float32)
        s_ = tl.load(sin_row + js, mask=m, other=0.0).to(tl.float32)

        qmask = qh_mask[:, None] & m[None, :]
        qr = tl.load(q_row_in + q_hbase_in + js[None, :], mask=qmask, other=0.0)
        qi = tl.load(
            q_row_in + q_hbase_in + (R + js)[None, :],
            mask=qmask,
            other=0.0,
        )
        q_dtype = qr.dtype
        qr32 = qr.to(tl.float32)
        qi32 = qi.to(tl.float32)
        qr_rot = qr32 * c - qi32 * s_
        qi_rot = qr32 * s_ + qi32 * c
        tl.store(
            q_row_out + q_hbase_out + (2 * js)[None, :],
            qr_rot.to(q_dtype),
            mask=qmask,
        )
        tl.store(
            q_row_out + q_hbase_out + (2 * js + 1)[None, :],
            qi_rot.to(q_dtype),
            mask=qmask,
        )

        kmask = kh_mask[:, None] & m[None, :]
        kr = tl.load(k_row_in + k_hbase_in + js[None, :], mask=kmask, other=0.0)
        ki = tl.load(
            k_row_in + k_hbase_in + (R + js)[None, :],
            mask=kmask,
            other=0.0,
        )
        k_dtype = kr.dtype
        kr32 = kr.to(tl.float32)
        ki32 = ki.to(tl.float32)
        kr_rot = kr32 * c - ki32 * s_
        ki_rot = kr32 * s_ + ki32 * c
        tl.store(
            k_row_out + k_hbase_out + (2 * js)[None, :],
            kr_rot.to(k_dtype),
            mask=kmask,
        )
        tl.store(
            k_row_out + k_hbase_out + (2 * js + 1)[None, :],
            ki_rot.to(k_dtype),
            mask=kmask,
        )


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings using the Triton kernel.

    Tensors are expected in packed layout ``[B, H, S, D]`` with the first
    ``2 * R`` features interleaved on output to match the reference helper.
    """

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q,k must be rank-4 tensors [B,H,S,D]")

    B, Hq, S, D = q.shape
    if k.shape[0] != B or k.shape[2] != S or k.shape[3] != D:
        raise ValueError("q and k must share batch/seq/head_dim")

    R = cos.shape[-1]
    rot_dim = 2 * R
    if rot_dim > D:
        raise ValueError("rotary dim exceeds head dim")

    COS_BROADCAST = 1 if cos.shape[0] == 1 else 0

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    def grid(meta):
        return (B * S, triton.cdiv(max(Hq, k.shape[1]), meta["BLOCK_H"]))

    _rope_interleaved_strided[grid](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        q_out,
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        k_out,
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        cos.contiguous(),
        0 if COS_BROADCAST else cos.stride(0),
        cos.stride(-2),
        sin.contiguous(),
        0 if COS_BROADCAST else sin.stride(0),
        sin.stride(-2),
        S,
        Hq=Hq,
        Hk=k.shape[1],
        D=D,
        R=R,
        COS_BROADCAST=COS_BROADCAST,
    )

    if rot_dim < D:
        q_out[..., rot_dim:] = q[..., rot_dim:]
        k_out[..., rot_dim:] = k[..., rot_dim:]

    return q_out, k_out


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 1_500_000.0,
    dtype: torch.dtype = torch.float32,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if device is not None:
        device = torch.device(device)

    freq_indices = torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)]
    freqs = 1.0 / (theta ** (freq_indices / dim))
    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)
    freqs = t * freqs.unsqueeze(0)
    freqs = torch.exp(1j * freqs)
    return torch.stack([freqs.real, freqs.imag], dim=-1)


__all__ = ["apply_rotary_emb", "precompute_freqs_cis"]
