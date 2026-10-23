"""Explicit native split attention with graph-cache-owned buffers."""
import torch


class VerificationAttention:
    _verification_scope = None

    def verification_attention(self, query, *, cache, paged_kv_layer, page_table,
                               used_k, sequence_lengths, scaling):
        scope = self._verification_scope
        if scope is None or not scope.active.get() or scope.attention_splits is None:
            return None
        if sequence_lengths is None:
            raise ValueError('split verification requires immutable packed query lengths')
        lengths = tuple(sequence_lengths)
        key = (lengths, page_table.data_ptr(), tuple(page_table.shape),
               used_k.data_ptr(), tuple(used_k.shape), query.device, scope.attention_splits)
        context = cache.verification_attention_context
        if context is None or context.key != key:
            context = NativeVerificationAttentionContext(key, lengths, page_table, used_k,
                paged_kv_layer.k_cache.shape[0], scope.attention_splits)
            cache.verification_attention_context = context
        return context.attend(self.layer_idx, query, paged_kv_layer, scaling)


class NativeVerificationAttentionContext:
    """One execution-cache owner, replaced when warmup graph inputs change.

    The existing allocator owns valid active page assignments and used-K staging.
    This context borrows those retained tensors; it never copies the page map.
    A cache has one context, and each context has at most one plan per layer.
    """
    def __init__(self, key, lengths, page_table, used_k, kv_pages, splits):
        from kestrel_kernels.flash_attn.cute.geometry import PackedAttentionGeometry
        from kestrel_kernels.flash_attn.cute.multiquery import PreparedMultiqueryLayout
        self.key, self.splits, self.plans = key, splits, {}
        self.workspaces = {}
        geometry = PackedAttentionGeometry(lengths, ((0, page_table.shape[1]),) * len(lengths))
        self.layout = PreparedMultiqueryLayout.bind_graph_inputs(
            geometry, page_table=page_table, used_k=used_k, kv_pages=kv_pages)

    def attend(self, layer_index, query, paged_kv_layer, scaling):
        from kestrel_kernels.flash_attn.cute.multiquery import (
            PreparedMultiqueryAttention, MultiqueryAttentionWorkspace)
        q = query.transpose(1, 2).flatten(0, 1)
        k = paged_kv_layer.k_cache.permute(0, 2, 1, 3)
        v = paged_kv_layer.v_cache.permute(0, 2, 1, 3)
        signature = (tuple(q.shape), q.dtype, k.data_ptr(), v.data_ptr(), float(scaling))
        previous = self.plans.get(layer_index)
        if previous is None or previous[0] != signature:
            with torch.cuda.device(q.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('prepare native verification attention during graph warmup')
            owned_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
            # Layers execute serially on this graph's stream. Keep outputs private,
            # but reuse the FP32 producer/combine scratch for compatible geometry.
            workspace_key = (tuple(q.shape), q.device, self.splits)
            workspace = self.workspaces.get(workspace_key)
            if workspace is None:
                workspace = MultiqueryAttentionWorkspace(owned_q, splits=self.splits)
                self.workspaces[workspace_key] = workspace
            plan = PreparedMultiqueryAttention(owned_q, k, v, self.layout,
                splits=self.splits, softmax_scale=scaling, workspace=workspace)
            self.plans[layer_index] = signature, plan
        else:
            plan = previous[1]
        plan.q.copy_(q)
        return plan().reshape(query.shape[0], query.shape[2], *q.shape[-2:])
