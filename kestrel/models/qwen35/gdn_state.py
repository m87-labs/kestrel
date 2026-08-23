"""Recurrent-state storage for Qwen 3.5/3.6 GDN layers."""

from __future__ import annotations

import torch


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
        self.has_previous_state = False

    def _ensure_replay_state(self, recurrent_states: torch.Tensor) -> None:
        if recurrent_states.ndim != 4:
            return
        slots, value_heads, key_dim, value_dim = recurrent_states.shape
        # ReplaySSM indexes the key ring by value head; GQA applies the k-to-v
        # fan-out before the ring. Size by value_heads so k16/v32 Qwen3.5-4B is
        # consistent with both the native kernels and numeric oracle; sizing it
        # by key_heads aliases verify writes and breaks materialization.
        if recurrent_states.device.type == "mps" and self.conv_states is None:
            raise RuntimeError("Qwen replay state requires convolution state")
        replay_k_dtype = (
            self.conv_states.dtype
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
__all__ = ["LinearAttentionState"]
