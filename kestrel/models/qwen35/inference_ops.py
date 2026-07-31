"""Inference-only attention and recurrent-state operations for Qwen 3.5."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from kestrel.ops.attention import repeat_kv


def torch_compilable_check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


class LinearAttentionState:
    def __init__(
        self,
        *,
        replay_capacity: int,
    ) -> None:
        self.conv_states: torch.Tensor | None = None
        self.recurrent_states: torch.Tensor | None = None
        self.replay_checkpoint_states: torch.Tensor | None = None
        self.replay_k: torch.Tensor | None = None
        self.replay_u: torch.Tensor | None = None
        self.replay_g: torch.Tensor | None = None
        self.replay_lengths: torch.Tensor | None = None
        self.replay_capacity = int(replay_capacity)
        self.is_conv_states_initialized = False
        self.is_recurrent_states_initialized = False
        self.has_previous_state = False

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        state_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_indices is not None and not self.is_conv_states_initialized:
            raise RuntimeError("Indexed conv state update requires initialized state")
        if not self.is_conv_states_initialized:
            self.dtype, self.device = conv_states.dtype, conv_states.device
            self.max_batch_size = conv_states.shape[0]
            self.conv_kernel_size = conv_states.shape[-1]
            self.conv_states = torch.empty_like(conv_states)
            self.is_conv_states_initialized = True
        if state_indices is not None:
            indices = state_indices.to(
                device=self.conv_states.device,
                dtype=torch.long,
            ).view(-1)
            if int(indices.shape[0]) != int(conv_states.shape[0]):
                raise ValueError("state_indices must match conv_states batch dimension")
            state_view = self.conv_states.index_select(0, indices).clone()
            if not self.has_previous_state:
                state_view.copy_(conv_states)
                self.has_previous_state = True
            else:
                num_new_tokens = conv_states.shape[-1]
                if num_new_tokens >= self.conv_kernel_size:
                    state_view.copy_(conv_states[..., -self.conv_kernel_size :])
                else:
                    state_view = state_view.roll(shifts=-num_new_tokens, dims=-1)
                    state_view[:, :, -num_new_tokens:] = conv_states
            self.conv_states.index_copy_(0, indices, state_view)
            return self.conv_states
        if not self.has_previous_state:
            self.conv_states.copy_(conv_states)
            self.has_previous_state = True
        else:
            num_new_tokens = conv_states.shape[-1]
            if num_new_tokens >= self.conv_kernel_size:
                self.conv_states.copy_(conv_states[..., -self.conv_kernel_size :])
            else:
                new_conv_states = self.conv_states.roll(shifts=-num_new_tokens, dims=-1)
                new_conv_states[:, :, -num_new_tokens:] = conv_states
                self.conv_states.copy_(new_conv_states)
        return self.conv_states

    def update_recurrent_state(
        self,
        recurrent_states: torch.Tensor,
        state_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_indices is not None and not self.is_recurrent_states_initialized:
            raise RuntimeError(
                "Indexed recurrent state update requires initialized state"
            )
        if not self.is_recurrent_states_initialized:
            self.recurrent_states = torch.empty_like(recurrent_states)
            self.is_recurrent_states_initialized = True
            # Allocate the ReplaySSM ring-buffer state alongside the recurrent
            # state pool (formerly done in lazy_initialization, which main
            # inlined away here).
            self._ensure_replay_state(recurrent_states)
        if state_indices is not None:
            indices = state_indices.to(
                device=self.recurrent_states.device,
                dtype=torch.long,
            ).view(-1)
            if int(indices.shape[0]) != int(recurrent_states.shape[0]):
                raise ValueError(
                    "state_indices must match recurrent_states batch dimension"
                )
            self.recurrent_states.index_copy_(0, indices, recurrent_states)
            self._reset_replay_rows(recurrent_states, indices)
            return self.recurrent_states
        self.recurrent_states.copy_(recurrent_states)
        self._reset_replay_rows(recurrent_states, None)
        return self.recurrent_states

    def _ensure_replay_state(self, recurrent_states: torch.Tensor) -> None:
        if recurrent_states.ndim != 4:
            return
        slots, value_heads, key_dim, value_dim = recurrent_states.shape
        # ReplaySSM indexes the key ring by value head; GQA applies the k-to-v
        # fan-out before the ring. Size by value_heads so k16/v32 Qwen3.5-4B is
        # consistent with both the native kernels and torch reference fold;
        # sizing it by key_heads aliased verify writes and tripped the
        # materialize gate -> torch fallback -> "bad replay_k shape" at flush.
        replay_k_dtype = (
            getattr(self, "dtype", torch.float16)
            if recurrent_states.device.type == "mps"
            else torch.bfloat16
        )
        specifications = {
            "replay_checkpoint_states": (
                (slots, value_heads, value_dim, key_dim),
                torch.float32,
            ),
            "replay_k": (
                (slots, self.replay_capacity, value_heads, key_dim),
                replay_k_dtype,
            ),
            "replay_u": (
                (slots, self.replay_capacity, value_heads, value_dim),
                torch.float32,
            ),
            "replay_g": (
                (slots, self.replay_capacity, value_heads),
                torch.float32,
            ),
            "replay_lengths": ((slots,), torch.int32),
        }
        existing = [getattr(self, name) for name in specifications]
        if any(tensor is not None for tensor in existing) and any(
            tensor is None for tensor in existing
        ):
            raise RuntimeError("Qwen replay state tensors are incomplete")
        for name, (shape, dtype) in specifications.items():
            tensor = getattr(self, name)
            if tensor is None:
                setattr(
                    self,
                    name,
                    torch.zeros(
                        shape,
                        dtype=dtype,
                        device=recurrent_states.device,
                    ),
                )
                continue
            if tuple(tensor.shape) != shape:
                raise RuntimeError(f"Qwen {name.replace('_', ' ')} shape changed")
            if tensor.dtype != dtype:
                raise RuntimeError(f"Qwen {name.replace('_', ' ')} dtype changed")

    def _reset_replay_rows(
        self,
        recurrent_states: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> None:
        if self.recurrent_states is None or recurrent_states.ndim != 4:
            return
        self._ensure_replay_state(self.recurrent_states)
        assert self.replay_checkpoint_states is not None
        assert self.replay_k is not None
        assert self.replay_u is not None
        assert self.replay_g is not None
        assert self.replay_lengths is not None
        checkpoint_rows = recurrent_states.transpose(-1, -2).contiguous()
        if indices is None:
            self.replay_checkpoint_states.copy_(checkpoint_rows)
            self.replay_lengths.zero_()
            return
        self.replay_checkpoint_states.index_copy_(0, indices, checkpoint_rows)
        self.replay_lengths.index_fill_(0, indices, 0)

    def allocate_zeroed(
        self,
        *,
        conv_shape: tuple[int, ...],
        recurrent_shape: tuple[int, ...],
        conv_dtype: torch.dtype,
        recurrent_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        for name, shape, dtype in (
            ("conv_states", conv_shape, conv_dtype),
            ("recurrent_states", recurrent_shape, recurrent_dtype),
        ):
            tensor = getattr(self, name)
            if tensor is None:
                setattr(
                    self,
                    name,
                    torch.zeros(shape, dtype=dtype, device=device),
                )
            elif tuple(tensor.shape) != shape:
                raise RuntimeError(f"Qwen GDN {name} shape changed")

        self.dtype = conv_dtype
        self.device = device
        self.max_batch_size = int(conv_shape[0])
        self.conv_kernel_size = int(conv_shape[-1])
        self.is_conv_states_initialized = True
        self.is_recurrent_states_initialized = True
        self._ensure_replay_state(self.recurrent_states)

    def clear(self, row: int | None = None) -> None:
        for tensor in (
            self.conv_states,
            self.recurrent_states,
            self.replay_checkpoint_states,
            self.replay_k,
            self.replay_u,
            self.replay_g,
            self.replay_lengths,
        ):
            if tensor is None:
                continue
            if row is None:
                tensor.zero_()
            else:
                tensor[int(row)].zero_()

    def reset_replay_tail(self, row_indices: torch.Tensor) -> None:
        if (
            self.replay_k is None
            or self.replay_u is None
            or self.replay_g is None
            or self.replay_lengths is None
        ):
            raise RuntimeError("Qwen replay state is not initialized")
        # replay_lengths is the validity boundary. Payload slots are overwritten
        # before the cursor advances, so clearing inaccessible K/U/G bytes would
        # only add three launches per recurrent layer.
        self.replay_lengths.index_fill_(0, row_indices, 0)

    def seed_replay_rows(self, row_indices: torch.Tensor) -> None:
        if self.recurrent_states is None or self.replay_checkpoint_states is None:
            raise RuntimeError("Qwen recurrent state is not initialized")
        checkpoint_rows = (
            self.recurrent_states.index_select(0, row_indices)
            .transpose(-1, -2)
            .contiguous()
        )
        self.replay_checkpoint_states.index_copy_(
            0,
            row_indices,
            checkpoint_rows,
        )
        self.reset_replay_tail(row_indices)

    def copy_rows_from(
        self,
        source: "LinearAttentionState",
        row_indices: torch.Tensor,
        *,
        copy_replay_payload: bool,
    ) -> None:
        batch_size = int(row_indices.shape[0])
        for name in ("conv_states", "recurrent_states"):
            destination = getattr(self, name)
            source_tensor = getattr(source, name)
            if destination is None or source_tensor is None:
                raise RuntimeError(f"Qwen GDN {name} is not initialized")
            destination.index_copy_(0, row_indices, source_tensor[:batch_size])
        if self.replay_checkpoint_states is None:
            return
        checkpoint_rows = (
            source.replay_checkpoint_states[:batch_size]
            if source.replay_checkpoint_states is not None
            else source.recurrent_states[:batch_size].transpose(-1, -2).contiguous()
        )
        self.replay_checkpoint_states.index_copy_(
            0,
            row_indices,
            checkpoint_rows,
        )
        if copy_replay_payload and all(
            getattr(source, name) is not None
            for name in ("replay_k", "replay_u", "replay_g", "replay_lengths")
        ):
            for name in ("replay_k", "replay_u", "replay_g", "replay_lengths"):
                destination = getattr(self, name)
                source_tensor = getattr(source, name)
                if destination is None or source_tensor is None:
                    raise RuntimeError(f"Qwen GDN {name} is not initialized")
                destination.index_copy_(
                    0,
                    row_indices,
                    source_tensor[:batch_size],
                )
        else:
            if self.replay_lengths is None:
                raise RuntimeError("Qwen replay length tensor is missing")
            self.replay_lengths.index_fill_(0, row_indices, 0)

    def materialize_recurrent_from_replay(
        self,
        indices: torch.Tensor | int | None = None,
        *,
        write_recurrent: bool = True,
    ) -> None:
        """Flush selected ReplaySSM ring rows into ``recurrent_states``.

        Single-token replay decode advances the replay representation
        (checkpoint + ring buffer) but leaves ``recurrent_states`` untouched.
        Before a multi-token (``seq_len > 1``) chunk-decode continuation reuses
        ``recurrent_states`` as its initial state, materialize the true current
        state so the chunk path does not read a stale value. The same fold also
        refreshes ``replay_checkpoint_states`` and clears only the selected
        replay cursors; the replay key/update buffers are left in place because
        ``replay_lengths == 0`` makes their contents inactive.
        """
        if (
            self.recurrent_states is None
            or self.replay_checkpoint_states is None
            or self.replay_k is None
            or self.replay_u is None
            or self.replay_g is None
            or self.replay_lengths is None
            or self.recurrent_states.ndim != 4
        ):
            return
        if indices is None:
            row_indices = torch.arange(
                self.recurrent_states.shape[0],
                device=self.recurrent_states.device,
                dtype=torch.long,
            )
        elif isinstance(indices, int):
            row_indices = torch.tensor(
                [indices],
                device=self.recurrent_states.device,
                dtype=torch.long,
            )
        else:
            row_indices = indices.to(
                device=self.recurrent_states.device,
                dtype=torch.long,
            ).view(-1)
        if int(row_indices.numel()) == 0:
            return

        # NOTE: the fast checkpoint-only Triton materializer
        # (materialize_replay_checkpoint_indexed_triton) uses a parallel
        # reduction whose accumulation order drifts from the sequential
        # recurrent decode at the bf16 floor, flipping argmax ties (observed as
        # cap64 != cap32 and spec != decode-kernel). The flush is ~0.3% of decode
        # time, so correctness wins: use the exact (sequential-order)
        # materializer below for the spec flush as well.
        from kestrel_kernels.gated_delta import materialize_replay_state_indexed

        materialize_replay_state_indexed(
            self.recurrent_states,
            self.replay_checkpoint_states,
            self.replay_k,
            self.replay_u,
            self.replay_g,
            self.replay_lengths,
            row_indices,
            write_recurrent=write_recurrent,
        )
        self.replay_lengths.index_fill_(0, row_indices, 0)


class LinearAttentionLayer(LinearAttentionState):
    def __init__(
        self,
        config: Any,
        *,
        replay_capacity: int,
    ) -> None:
        super().__init__(replay_capacity=replay_capacity)
        self.num_k_heads = int(config.linear_num_key_heads)
        self.num_v_heads = int(config.linear_num_value_heads)


def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, None]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    out = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
    )
    return out.transpose(1, 2).contiguous(), None


@lru_cache(maxsize=1)
def _kestrel_flash_attn_fwd() -> Callable[..., Any]:
    from kestrel_kernels import get_runtime

    return get_runtime().attention.flash_attn_fwd


def kestrel_vision_flash_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
) -> tuple[torch.Tensor, None]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "Kestrel Qwen vision attention expects query/key/value as [B, H, S, D]"
        )
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError(
            "Kestrel Qwen vision attention expects packed vision batch size 1"
        )

    device = query.device
    if cu_seq_lens_q.device != device:
        cu_seq_lens_q = cu_seq_lens_q.to(device=device, non_blocking=True)
    if cu_seq_lens_k.device != device:
        cu_seq_lens_k = cu_seq_lens_k.to(device=device, non_blocking=True)

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    out, _ = _kestrel_flash_attn_fwd()(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1),
        value[0].transpose(0, 1),
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k=cu_seq_lens_k,
        softmax_scale=scaling,
        causal=False,
        m_block_size=128,
        n_block_size=128,
    )
    return out.unsqueeze(0), None


def create_causal_mask(
    *,
    config: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None,
    position_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    batch_size, query_length = inputs_embeds.shape[:2]
    past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    key_length = past_length + query_length
    if attention_mask is not None:
        key_length = max(key_length, int(attention_mask.shape[-1]))
    min_value = torch.finfo(inputs_embeds.dtype).min
    mask = torch.full(
        (query_length, key_length),
        min_value,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    diagonal = 1 + past_length
    mask = torch.triu(mask, diagonal=diagonal)
    mask = mask[None, None, :, :].expand(batch_size, 1, query_length, key_length)
    if attention_mask is not None:
        pad = attention_mask[:, None, None, :].to(torch.bool)
        mask = mask.clone()
        mask = mask.masked_fill(~pad, min_value)
    return mask


def get_vision_position_ids(
    grid_thw: torch.Tensor,
    spatial_merge_size: int | torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if position_ids is not None:
        return position_ids
    device = grid_thw.device
    if isinstance(spatial_merge_size, int):
        merge_sizes = [int(spatial_merge_size)] * int(grid_thw.shape[0])
    else:
        merge_sizes = [int(value) for value in spatial_merge_size.tolist()]
    position_ids = []
    for (t, h, w), merge_size in zip(grid_thw.tolist(), merge_sizes):
        t, h, w, merge_size = int(t), int(h), int(w), int(merge_size)
        hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos_ids = (
            hpos_ids.reshape(h // merge_size, merge_size, w // merge_size, merge_size)
            .transpose(1, 2)
            .flatten()
        )
        wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos_ids = (
            wpos_ids.reshape(h // merge_size, merge_size, w // merge_size, merge_size)
            .transpose(1, 2)
            .flatten()
        )
        position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
    bilinear_indices: torch.Tensor | None = None,
    bilinear_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bilinear_indices is not None and bilinear_weights is not None:
        return bilinear_indices, bilinear_weights
    side = num_grid_per_side
    merge_size = spatial_merge_size
    device = grid_thw.device
    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side
        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]
        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (
            (h_idx[:, :, None, None] * w + w_idx[None, None, :, :])
            .transpose(1, 2)
            .flatten()
            .repeat(t)
        )
        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])
    return (
        torch.stack([torch.cat(p) for p in idx_parts]),
        torch.stack([torch.cat(p) for p in weight_parts]),
    )


def get_vision_cu_seqlens(
    grid_thw: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if cu_seqlens is not None:
        return cu_seqlens
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    return F.pad(cu_seqlens, (1, 0), value=0)


__all__ = [
    "create_causal_mask",
    "get_vision_bilinear_indices_and_weights",
    "get_vision_cu_seqlens",
    "get_vision_position_ids",
    "kestrel_vision_flash_attention_forward",
    "sdpa_attention_forward",
    "torch_compilable_check",
]
