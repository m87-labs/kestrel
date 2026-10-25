"""Bounded shape-owned workspaces for native verification query metadata."""


def require_query_metadata(device):
    if device.type == 'cuda':
        from kestrel_kernels.paged_kv_metadata_ops.dispatch import require_packed_query_metadata
        require_packed_query_metadata(device)


def build_query_metadata(decoder, lengths, slots, starts):
    """Build metadata through the device-agnostic kernel workspace.

    Decoder execution is serialized. Outputs remain live until graph staging or
    eager verification consumes them; shape eviction cannot retire active work.
    """
    from kestrel_kernels.paged_kv_metadata_ops.dispatch import PackedQueryMetadata
    table = decoder.runtime.page_table.page_table
    key = (tuple(lengths), tuple(table.shape), table.stride(), decoder.runtime.page_size)
    workspaces = decoder._query_metadata_workspaces
    workspace = workspaces.pop(key, None)
    if workspace is None:
        workspace = PackedQueryMetadata(table, lengths, decoder.runtime.page_size)
        if len(workspaces) >= decoder.runtime.max_batch_size:
            workspaces.pop(next(iter(workspaces)))
    workspaces[key] = workspace
    return workspace.build(table, slots, starts)
