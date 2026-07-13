"""Minimal Moondream DFlash drafter inference path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor, nn

from .region import SpatialDecodeTables, spatial_decode_logits

DFLASH_CONDITION_SEGMENT_PROMPT = 0
DFLASH_CONDITION_SEGMENT_GENERATED = 1
DFLASH_CONDITION_SEGMENT_COUNT = 2


@dataclass(frozen=True)
class DFlashConfig:
    vocab_size: int
    block_size: int
    target_hidden_size: int
    target_layer_indices: tuple[int, ...]
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_multiplier: float
    max_context_tokens: int
    spatial_bin_count: int
    pad_token_id: int
    mask_token_id: int | None
    rope_theta: float
    dropout: float
    markov_rank: int
    markov_context: int
    confidence_head: bool
    refine_steps: int
    draft_attention: str
    oov_tail: bool

    @property
    def target_width(self) -> int:
        return self.block_size - 1

    @property
    def target_layer_count(self) -> int:
        return len(self.target_layer_indices)

    @classmethod
    def from_serving_config(cls, serving_config: dict[str, Any]) -> "DFlashConfig":
        raw = dict(serving_config["config"])
        if int(raw.get("prompt_summary_tokens", 0)) != 0:
            raise NotImplementedError("DFlash prompt summary rows are not supported")
        if bool(raw.get("mdn_box_head", False)):
            raise NotImplementedError("DFlash MDN box head is not supported")
        if int(raw.get("markov_context", 1)) != 1:
            raise NotImplementedError("DFlash markov_context > 1 is not supported")
        if str(raw.get("draft_attention", "bidirectional")) != "bidirectional":
            raise NotImplementedError(
                "Only bidirectional DFlash draft attention is supported"
            )
        if bool(raw.get("oov_tail", True)):
            raise NotImplementedError("DFlash OOV-tail serving checkpoints are not supported")
        return cls(
            vocab_size=int(raw["vocab_size"]),
            block_size=int(raw["block_size"]),
            target_hidden_size=int(raw["target_hidden_size"]),
            target_layer_indices=tuple(int(i) for i in raw["target_layer_indices"]),
            hidden_size=int(raw["hidden_size"]),
            num_layers=int(raw["num_layers"]),
            num_heads=int(raw["num_heads"]),
            ffn_multiplier=float(raw.get("ffn_multiplier", 4.0)),
            max_context_tokens=int(raw["max_context_tokens"]),
            spatial_bin_count=int(raw["spatial_bin_count"]),
            pad_token_id=int(raw.get("pad_token_id", 0)),
            mask_token_id=(
                None if raw.get("mask_token_id") is None else int(raw["mask_token_id"])
            ),
            rope_theta=float(raw.get("rope_theta", 1_000_000.0)),
            dropout=float(raw.get("dropout", 0.0)),
            markov_rank=int(raw.get("markov_rank", 0)),
            markov_context=int(raw.get("markov_context", 1)),
            confidence_head=bool(raw.get("confidence_head", False)),
            refine_steps=int(raw.get("refine_steps", 1)),
            draft_attention=str(raw.get("draft_attention", "bidirectional")),
            oov_tail=bool(raw.get("oov_tail", True)),
        )


@dataclass(frozen=True)
class DFlashMetadata:
    path: Path
    config: DFlashConfig
    step: int | None
    draft_vocab_size: int | None
    weight_dtype: str | None


@dataclass
class DFlashBatch:
    current_token_ids: Tensor
    target_hidden_states: Tensor
    target_hidden_mask: Tensor | None = None
    target_hidden_segment_ids: Tensor | None = None
    markov_prev_token_ids: Tensor | None = None


@dataclass
class DFlashOutput:
    token_logits: Tensor
    coord_logits: Tensor
    size_width_logits: Tensor
    size_height_logits: Tensor
    confidence_logits: Tensor | None
    first_pass_token_logits: Tensor | None
    target_hidden: Tensor


class DFlashHiddenWindow:
    """Right-aligned target-hidden context for one active sequence."""

    def __init__(
        self,
        config: DFlashConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.hidden = torch.zeros(
            (
                config.max_context_tokens,
                config.target_layer_count,
                config.target_hidden_size,
            ),
            device=device,
            dtype=dtype,
        )
        self.mask = torch.zeros(
            (config.max_context_tokens,),
            device=device,
            dtype=torch.bool,
        )
        self.segment_ids = torch.full(
            (config.max_context_tokens,),
            DFLASH_CONDITION_SEGMENT_GENERATED,
            device=device,
            dtype=torch.long,
        )
        self.count = 0

    def append(self, rows: Tensor) -> None:
        """Append one or more captured target rows shaped ``[N, L, C]``."""
        if rows.dim() == 2:
            rows = rows.unsqueeze(0)
        if rows.dim() != 3:
            raise ValueError("DFlash hidden rows must have shape [N, L, C]")
        expected = (
            self.config.target_layer_count,
            self.config.target_hidden_size,
        )
        if tuple(rows.shape[1:]) != expected:
            raise ValueError(
                f"DFlash hidden row shape mismatch: {tuple(rows.shape[1:])} != {expected}"
            )

        capacity = int(self.config.max_context_tokens)
        rows = rows.detach().to(device=self.hidden.device, dtype=self.hidden.dtype)
        if int(rows.shape[0]) > capacity:
            rows = rows[-capacity:]

        if self.count:
            current = self.hidden[capacity - self.count :]
            combined = torch.cat([current, rows], dim=0)
        else:
            combined = rows
        if int(combined.shape[0]) > capacity:
            combined = combined[-capacity:]

        count = int(combined.shape[0])
        start = capacity - count
        self.hidden.zero_()
        self.mask.zero_()
        self.hidden[start:].copy_(combined)
        self.mask[start:] = True
        self.segment_ids.fill_(DFLASH_CONDITION_SEGMENT_GENERATED)
        self.count = count


def stack_dflash_hidden_windows(
    windows: Sequence[DFlashHiddenWindow],
) -> tuple[Tensor, Tensor, Tensor]:
    if not windows:
        raise ValueError("windows must be non-empty")
    return (
        torch.stack([window.hidden for window in windows], dim=0),
        torch.stack([window.mask for window in windows], dim=0),
        torch.stack([window.segment_ids for window in windows], dim=0),
    )


def inspect_dflash_checkpoint(path: str | Path) -> DFlashMetadata:
    ckpt_path = Path(path).expanduser()
    with safe_open(str(ckpt_path), framework="numpy", device="cpu") as f:
        metadata = dict(f.metadata() or {})
    if metadata.get("format") != "dflash-serving-safetensors":
        raise ValueError(f"{ckpt_path} is not a DFlash serving safetensors file")
    serving_json = metadata.get("serving_config")
    if not serving_json:
        raise ValueError(f"{ckpt_path} has no serving_config metadata")
    serving_config = json.loads(serving_json)
    return DFlashMetadata(
        path=ckpt_path,
        config=DFlashConfig.from_serving_config(serving_config),
        step=int(metadata["step"]) if metadata.get("step") is not None else None,
        draft_vocab_size=(
            None
            if serving_config.get("draft_vocab_size") is None
            else int(serving_config["draft_vocab_size"])
        ),
        weight_dtype=metadata.get("weight_dtype"),
    )


class DFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x_float = x.float()
        scale = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * scale).to(dtype) * self.weight.to(dtype)


class DFlashRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, *, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            float(base)
            ** (
                torch.arange(0, int(head_dim), 2, dtype=torch.float32)
                / float(head_dim)
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: Tensor, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        inv_freq = self.inv_freq.to(device=position_ids.device)
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_len = int(q.shape[-2])
    q = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class DFlashKVInjectedBlock(nn.Module):
    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.hidden_size = hidden_size
        self.num_heads = int(config.num_heads)
        self.head_dim = hidden_size // self.num_heads
        self.dropout_p = float(config.dropout)
        self.attn_norm = DFlashRMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ffn_norm = DFlashRMSNorm(hidden_size)
        inner = int(hidden_size * float(config.ffn_multiplier))
        self.ffn_gate = nn.Linear(hidden_size, inner, bias=False)
        self.ffn_up = nn.Linear(hidden_size, inner, bias=False)
        self.ffn_down = nn.Linear(inner, hidden_size, bias=False)

    def forward(
        self,
        draft_hidden: Tensor,
        target_features: Tensor,
        target_mask: Tensor,
        rotary: tuple[Tensor, Tensor],
    ) -> Tensor:
        attn_input = self.attn_norm(draft_hidden)
        q = self._split_heads(self.q_proj(attn_input))
        draft_k = self.k_proj(attn_input)
        draft_v = self.v_proj(attn_input)
        target_k = self.k_proj(target_features)
        target_v = self.v_proj(target_features)
        k = self._split_heads(torch.cat([target_k, draft_k], dim=1))
        v = self._split_heads(torch.cat([target_v, draft_v], dim=1))
        q, k = _apply_rotary(q, k, rotary[0], rotary[1])

        q_len = int(draft_hidden.shape[1])
        draft_key_mask = torch.ones(
            (draft_hidden.shape[0], draft_k.shape[1]),
            dtype=torch.bool,
            device=draft_hidden.device,
        )
        key_mask = torch.cat([target_mask.to(torch.bool), draft_key_mask], dim=1)
        attn_mask = key_mask[:, None, None, :].expand(-1, 1, q_len, -1)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        draft_hidden = draft_hidden + self.out_proj(self._merge_heads(attn))
        ffn_input = self.ffn_norm(draft_hidden)
        ffn = self.ffn_down(
            F.gelu(self.ffn_gate(ffn_input), approximate="tanh")
            * self.ffn_up(ffn_input)
        )
        return draft_hidden + ffn

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _width = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _heads, seq_len, _head_dim = x.shape
        return x.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)


class MoondreamDFlashDrafter(nn.Module):
    def __init__(
        self,
        config: DFlashConfig,
        *,
        target_text: nn.Module,
        spatial_tables: SpatialDecodeTables,
        draft_vocab_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        object.__setattr__(self, "_target_text", target_text)
        object.__setattr__(self, "_spatial_tables", spatial_tables)

        self.register_buffer(
            "d2t",
            torch.empty(int(draft_vocab_size), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "t2d",
            torch.empty(int(config.vocab_size), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer("_draft_lm_weight", None, persistent=False)
        self.register_buffer("_draft_lm_bias", None, persistent=False)

        self.target_norm = DFlashRMSNorm(config.target_hidden_size)
        self.target_feature_proj = nn.Linear(
            config.target_hidden_size * config.target_layer_count,
            config.hidden_size,
            bias=False,
        )
        self.condition_segment_embedding = nn.Embedding(
            DFLASH_CONDITION_SEGMENT_COUNT,
            config.hidden_size,
        )
        self.condition_norm = DFlashRMSNorm(config.hidden_size)
        self.token_input_proj = nn.Linear(
            config.target_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.output_target_proj = nn.Linear(
            config.hidden_size,
            config.target_hidden_size,
            bias=False,
        )
        if config.mask_token_id is None:
            self.mask_target_embedding = nn.Parameter(
                torch.empty(config.target_hidden_size)
            )
        else:
            self.register_parameter("mask_target_embedding", None)
        self.rotary_emb = DFlashRotaryEmbedding(
            config.hidden_size // config.num_heads,
            base=config.rope_theta,
        )
        self.input_norm = DFlashRMSNorm(config.hidden_size)
        self.blocks = nn.ModuleList(
            [DFlashKVInjectedBlock(config) for _ in range(config.num_layers)]
        )
        self.output_norm = DFlashRMSNorm(config.hidden_size)
        self.markov_in = (
            nn.Embedding(int(draft_vocab_size), config.markov_rank)
            if config.markov_rank > 0
            else None
        )
        self.markov_out = (
            nn.Linear(config.markov_rank, int(draft_vocab_size), bias=False)
            if config.markov_rank > 0
            else None
        )
        self.confidence_proj = (
            nn.Linear(config.hidden_size, 1) if config.confidence_head else None
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        target_text: nn.Module,
        spatial_tables: SpatialDecodeTables,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "MoondreamDFlashDrafter":
        metadata = inspect_dflash_checkpoint(path)
        if metadata.draft_vocab_size is None:
            raise ValueError("DFlash checkpoint metadata is missing draft_vocab_size")
        model = cls(
            metadata.config,
            target_text=target_text,
            spatial_tables=spatial_tables,
            draft_vocab_size=int(metadata.draft_vocab_size),
        ).to(device=device, dtype=dtype)
        state = load_file(str(metadata.path), device=str(device))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if k not in {"_draft_lm_weight", "_draft_lm_bias"}]
        if missing or unexpected:
            raise ValueError(
                f"DFlash checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        model._slice_target_lm_head()
        return model.eval()

    def _slice_target_lm_head(self) -> None:
        text = self._target_text
        d2t = self.d2t.to(text.lm_head.weight.device)
        self._draft_lm_weight = text.lm_head.weight.index_select(0, d2t).detach()
        self._draft_lm_bias = text.lm_head.bias.index_select(0, d2t).detach()

    @torch.no_grad()
    def forward(self, batch: DFlashBatch) -> DFlashOutput:
        cfg = self.config
        target_hidden = batch.target_hidden_states
        if target_hidden.dim() != 4:
            raise ValueError("target_hidden_states must be [B, T, L, C]")
        if int(target_hidden.shape[2]) != cfg.target_layer_count:
            raise ValueError("target_hidden_states layer count mismatch")
        if int(target_hidden.shape[3]) != cfg.target_hidden_size:
            raise ValueError("target_hidden_states hidden width mismatch")
        target_mask = batch.target_hidden_mask
        if target_mask is None:
            target_mask = torch.ones(
                target_hidden.shape[:2],
                dtype=torch.bool,
                device=target_hidden.device,
            )
        segment_ids = batch.target_hidden_segment_ids
        if segment_ids is None:
            segment_ids = torch.full(
                target_hidden.shape[:2],
                DFLASH_CONDITION_SEGMENT_GENERATED,
                dtype=torch.long,
                device=target_hidden.device,
            )

        hidden = self.target_norm(target_hidden)
        condition = self.target_feature_proj(
            hidden.reshape(
                hidden.shape[0],
                hidden.shape[1],
                cfg.target_hidden_size * cfg.target_layer_count,
            )
        )
        condition = condition + self.condition_segment_embedding(segment_ids).to(
            dtype=condition.dtype
        )
        condition = self.condition_norm(condition)

        anchor_embedding = self._embed_token(batch.current_token_ids).unsqueeze(1)
        draft_length = 1 + cfg.target_width
        rotary = self.rotary_emb(
            self._position_ids(target_mask, segment_ids, draft_length=draft_length),
            dtype=condition.dtype,
        )

        slot_embeddings = self._embed_mask_tokens(
            batch_size=int(condition.shape[0]),
            device=target_hidden.device,
            dtype=target_hidden.dtype,
        )
        first_pass_token_logits: Tensor | None = None
        target_space_hidden: Tensor | None = None
        draft_hidden: Tensor | None = None
        prev_tokens = batch.markov_prev_token_ids

        for pass_idx in range(max(1, cfg.refine_steps)):
            x = self._run_segment(
                anchor_embedding,
                slot_embeddings,
                condition,
                target_mask,
                rotary,
            )
            draft_hidden = x[:, 1:]
            target_space_hidden = self.output_target_proj(draft_hidden)
            markov_bias = self._markov_bias(
                batch.current_token_ids,
                prev_tokens,
                width=cfg.target_width,
            )
            token_logits = self._token_logits(target_space_hidden, markov_bias)
            if pass_idx == 0:
                first_pass_token_logits = token_logits.detach()
            if pass_idx == cfg.refine_steps - 1:
                break
            next_ids = self.d2t[token_logits.argmax(dim=-1)]
            prev_tokens = torch.cat(
                [batch.current_token_ids.unsqueeze(1), next_ids[:, :-1]],
                dim=1,
            )
            slot_embeddings = self._embed_token(next_ids).to(dtype=target_hidden.dtype)

        assert target_space_hidden is not None
        assert draft_hidden is not None
        markov_bias = self._markov_bias(
            batch.current_token_ids,
            prev_tokens,
            width=cfg.target_width,
        )
        token_logits = self._token_logits(target_space_hidden, markov_bias)
        coord_logits, width_logits, height_logits = self._region_logits(
            target_space_hidden
        )
        confidence_logits = (
            self.confidence_proj(draft_hidden).squeeze(-1)
            if self.confidence_proj is not None
            else None
        )
        return DFlashOutput(
            token_logits=token_logits,
            coord_logits=coord_logits,
            size_width_logits=width_logits,
            size_height_logits=height_logits,
            confidence_logits=confidence_logits,
            first_pass_token_logits=first_pass_token_logits,
            target_hidden=target_space_hidden,
        )

    def _run_segment(
        self,
        anchor_embedding: Tensor,
        slot_embeddings: Tensor,
        condition: Tensor,
        target_mask: Tensor,
        rotary: tuple[Tensor, Tensor],
    ) -> Tensor:
        x = self.token_input_proj(torch.cat([anchor_embedding, slot_embeddings], dim=1))
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x, condition, target_mask, rotary)
        return self.output_norm(x)

    def _position_ids(
        self,
        target_mask: Tensor,
        segment_ids: Tensor,
        *,
        draft_length: int,
    ) -> Tensor:
        mask = target_mask.to(dtype=torch.bool)
        segments = segment_ids.to(device=target_mask.device, dtype=torch.long)
        prompt_mask = mask & (segments == DFLASH_CONDITION_SEGMENT_PROMPT)
        generated_mask = mask & (segments == DFLASH_CONDITION_SEGMENT_GENERATED)
        prompt_counts = prompt_mask.long().sum(dim=1)
        generated_counts = generated_mask.long().sum(dim=1)
        prompt_offsets = prompt_mask.long().cumsum(dim=1) - 1
        generated_offsets = generated_mask.long().cumsum(dim=1) - 1
        target_positions = torch.zeros_like(segment_ids, dtype=torch.long)
        target_positions = torch.where(
            prompt_mask,
            prompt_offsets.clamp_min(0),
            target_positions,
        )
        target_positions = torch.where(
            generated_mask,
            prompt_counts[:, None] + generated_offsets.clamp_min(0),
            target_positions,
        )
        anchor_positions = (prompt_counts + generated_counts - 1).clamp_min(0)
        draft_offsets = torch.arange(draft_length, device=target_mask.device)
        draft_positions = anchor_positions[:, None] + draft_offsets[None, :]
        return torch.cat([target_positions, draft_positions], dim=1)

    def _embed_token(self, token_ids: Tensor) -> Tensor:
        text = self._target_text
        ids = token_ids.clamp(0, self.config.vocab_size - 1)
        return F.embedding(ids, text.wte)

    def _embed_mask_tokens(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.config.mask_token_id is not None:
            ids = torch.full(
                (batch_size, self.config.target_width),
                int(self.config.mask_token_id),
                dtype=torch.long,
                device=device,
            )
            return self._embed_token(ids)
        if self.mask_target_embedding is None:
            raise RuntimeError("DFlash mask embedding is missing")
        return self.mask_target_embedding.to(device=device, dtype=dtype).view(
            1,
            1,
            self.config.target_hidden_size,
        ).expand(batch_size, self.config.target_width, self.config.target_hidden_size)

    def _markov_bias(
        self,
        current_token_ids: Tensor,
        prev_token_ids: Tensor | None,
        *,
        width: int,
    ) -> Tensor | None:
        if self.markov_in is None or self.markov_out is None:
            return None
        if prev_token_ids is None:
            prev_token_ids = current_token_ids[:, None].expand(-1, width)
        t2d = self.t2d.to(prev_token_ids.device)
        prev = prev_token_ids.clamp(0, self.config.vocab_size - 1)
        prev = t2d[prev].clamp(max=int(self.markov_in.num_embeddings) - 1)
        return self.markov_out(self.markov_in(prev))

    def _token_logits(self, hidden: Tensor, markov_bias: Tensor | None) -> Tensor:
        text = self._target_text
        hidden = F.layer_norm(
            hidden,
            (hidden.shape[-1],),
            weight=text.post_ln.weight,
            bias=text.post_ln.bias,
        )
        logits = F.linear(
            hidden,
            self._draft_lm_weight.to(hidden.dtype),
            self._draft_lm_bias.to(hidden.dtype),
        )
        if markov_bias is not None:
            logits = logits + markov_bias.to(logits.dtype)
        return logits

    def _region_logits(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, width, dim = hidden.shape
        coord, w, h = spatial_decode_logits(
            hidden.reshape(batch * width, dim),
            self._spatial_tables,
        )
        return (
            coord.reshape(batch, width, self.config.spatial_bin_count),
            w.reshape(batch, width, self.config.spatial_bin_count),
            h.reshape(batch, width, self.config.spatial_bin_count),
        )


__all__ = [
    "DFlashBatch",
    "DFlashConfig",
    "DFlashHiddenWindow",
    "DFlashMetadata",
    "DFlashOutput",
    "MoondreamDFlashDrafter",
    "inspect_dflash_checkpoint",
    "stack_dflash_hidden_windows",
]
