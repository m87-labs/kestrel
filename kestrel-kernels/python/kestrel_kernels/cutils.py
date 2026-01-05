from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate
from cutlass.pipeline import (
    CooperativeGroup,
    PipelineAsync,
    PipelineOp,
    PipelineState,
    pipeline_init_wait,
)

from kestrel_kernels.flash_attn.cute.pipeline import PipelineTmaAsync


@dataclass(frozen=True)
class PipelineTmaCpAsync(PipelineTmaAsync):
    """
    PipelineTmaCpAsync is used for CpAsync + TMA producers and AsyncThread consumers.
    This mirrors quack's pipeline to support mixed cp.async (A) + TMA (B) loads.
    """

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        tidx: Optional[Int32] = None,
    ):
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.AsyncThread

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        if tidx is None:
            tidx, _, _ = cute.arch.thread_idx()
        if cta_layout_vmnk is None:
            cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        if const_expr(cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1):
            dst_rank = None
            is_signalling_thread = tidx % 128 == 0
        else:
            (
                dst_rank,
                is_signalling_thread,
            ) = PipelineTmaAsync.init_empty_barrier_arrive_signal(cta_layout_vmnk, tidx)

        producer_mask = None

        pipeline_init_wait(cta_layout_vmnk)

        return PipelineTmaCpAsync(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            dst_rank,
            is_signalling_thread,
        )

    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        if_generate(
            is_tma_warp,
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask),
        )

    def producer_cpasync_commit(self, state: PipelineState):
        cute.arch.cp_async_mbarrier_arrive_noinc(self.producer_get_barrier(state))
