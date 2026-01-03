# Prefix Caching Design Document

This document describes the design for KV cache prefix sharing in Kestrel, enabling efficient reuse of computed key-value pairs across requests with common prefixes.

## Table of Contents

1. [Overview](#overview)
2. [Token Types and KV Mapping](#token-types-and-kv-mapping)
3. [Cache Key Design](#cache-key-design)
4. [Data Structures](#data-structures)
5. [Core Operations](#core-operations)
6. [Append Prefill](#append-prefill)
7. [Integration Points](#integration-points)
8. [Duplicate Prefill Prevention](#duplicate-prefill-prevention)
9. [Memory Management](#memory-management)
10. [LoRA Namespace Isolation](#lora-namespace-isolation)
11. [File Organization](#file-organization)
12. [Multi-Image Support (Future)](#multi-image-support-future)
13. [Test Plan](#test-plan)
14. [Future Considerations](#future-considerations)

---

## Overview

### Problem Statement

In multi-modal inference, many requests share common prefixes:
- Same system prompt with different user queries
- Same image with different questions
- Repeated prompt patterns in conversational flows

Without prefix caching, each request recomputes the full KV cache from scratch, including the vision encoder forward pass for images.

### Solution

Implement a radix tree-based prefix cache that:
- Stores computed KV cache pages keyed by token sequences
- Enables sharing of cached pages across requests with matching prefixes
- Uses dynamic memory management (no fixed cache budget)
- Supports all token types: text, image, coord, and size

### Design Principles

1. **Dynamic sizing**: Cache uses all available pages. Eviction happens on-demand when allocating new pages.
2. **Future-proof images**: Image KV length is stored per-key, not hardcoded.
3. **Unified token handling**: All token types (text, coord, size, image) are cacheable with discrete keys.
4. **Simple ownership model**: Tree nodes own pages. Active sequences hold locks preventing eviction.

### Concurrency Model

**All prefix cache operations are serialized through the scheduler thread.** This includes:

- `match_prefix()` (may split nodes)
- `insert()` (modifies tree structure)
- `lock()` / `unlock()` (modifies lock counts)
- `evict()` (removes nodes, returns pages to pool)
- `PageTable.allocate_pages()` (may trigger eviction)

This single-threaded model matches Kestrel's existing scheduler architecture. There is no mutex contention because only one thread accesses these structures.

**If multi-threaded access is needed in the future:**

1. Add a mutex around all prefix cache + page table allocation operations
2. Use consistent lock ordering: `prefix_cache_lock` → `page_table_lock` (if separated)
3. Consider read-write locks if read-heavy (match_prefix without splitting)

For now, the single-threaded model is correct and performant.

---

## Token Types and KV Mapping

Kestrel has four token types, each mapping to KV cache positions:

| Token Type | Fields | KV Positions | Cache Key (via `cache_key()`) |
|------------|--------|--------------|-----------------------------|
| `TextToken` | `token_id: int` | 1 | `(0, token_id)` |
| `CoordToken` | `pos: float`, `bin: int` | 1 | `(1, bin)` |
| `SizeToken` | `width: float`, `height: float`, `width_bin: int`, `height_bin: int` | 1 | `(2, width_bin, height_bin)` |
| `ImageToken` | `content_hash: int` (128-bit), `kv_length_: int` | Variable (729 today) | `(3, content_hash, kv_length_)` |

### Coord and Size Token Discretization

Coord and size values are derived from discrete sampling:
- **Coord**: 1024 bins, uniformly spaced from 0.0 to 1.0
- **Width/Height**: 1024 bins each, exponentially spaced from 2^-10 to 1.0

The continuous float values are used for embedding, while the discrete bin indices are used for cache keys. This enables exact matching for caching purposes.

### Sequence Structure

A typical multi-modal sequence has the structure:

```
Position:  [0]   [1 ... 729]  [730]  [731]  ... [730+N-1]
Content:   BOS   Image        Text₁  Text₂  ... TextN
KV slots:   1      729          1      1    ...    1
```

The cache token sequence mirrors this:
```
[TextToken(BOS_ID), ImageToken(hash, 729), TextToken(tok₁), TextToken(tok₂), ...]
```

---

## Cache Key Design

Instead of separate key types, we reuse the existing token types with a `cache_key()` method that extracts the hashable, discrete portion of each token.

### Token Type Modifications

```python
# kestrel/moondream/runtime.py

class TextToken(NamedTuple):
    token_id: int

    def cache_key(self) -> tuple:
        return (0, self.token_id)

    def kv_length(self) -> int:
        return 1


class CoordToken(NamedTuple):
    pos: float      # Continuous value for embedding
    bin: int        # Discrete bin index (0-1023) for cache key

    def cache_key(self) -> tuple:
        return (1, self.bin)  # Ignores pos - matching on bin only

    def kv_length(self) -> int:
        return 1


class SizeToken(NamedTuple):
    width: float        # Continuous value for embedding
    height: float       # Continuous value for embedding
    width_bin: int      # Discrete bin index (0-1023)
    height_bin: int     # Discrete bin index (0-1023)

    def cache_key(self) -> tuple:
        return (2, self.width_bin, self.height_bin)  # Ignores floats

    def kv_length(self) -> int:
        return 1


class ImageToken(NamedTuple):
    """Pseudo-token representing an image in cache key sequences."""
    content_hash: int   # First 16 bytes of SHA256 as int (128-bit)
    kv_length_: int     # 729 today, could change (trailing underscore avoids conflict with method)

    def cache_key(self) -> tuple:
        return (3, self.content_hash, self.kv_length_)

    def kv_length(self) -> int:
        return self.kv_length_


# Union type for cache key sequences
CacheToken = TextToken | CoordToken | SizeToken | ImageToken
```

The tuple prefix (0/1/2/3) in `cache_key()` ensures no hash collisions between different token types.

### Image Hash Computation

Image hashes are computed from raw image bytes at request submission time:

```python
# In engine.py, when processing incoming request
if image is not None:
    if isinstance(image, pyvips.Image):
        raw_bytes = image.write_to_buffer(".png")
    else:
        raw_bytes = image.tobytes()
    image_hash = hashlib.sha256(raw_bytes).digest()  # 32 bytes
```

The hash is stored with the request and passed through to `start_sequence()`.

**Why 128-bit (16 bytes) instead of 64-bit?**

We use the first 16 bytes of the SHA256 digest for cache keys:
- 64-bit: ~10^-8 collision probability at 1M entries (birthday paradox)
- 128-bit: ~10^-20 collision probability at 1M entries (negligible)

The cost difference is minimal (Python ints handle arbitrary precision), and 128-bit provides comfortable safety margin for non-adversarial workloads.

### Building Cache Token Sequences

Cache token sequences are built inline in `start_sequence()`. For the current single-image case:

```python
# In start_sequence():
cache_tokens: list[CacheToken] = [tokens[0]]  # BOS
if image_hash is not None:
    cache_tokens.append(ImageToken(
        content_hash=int.from_bytes(image_hash[:16], 'big'),  # 128-bit hash
        kv_length_=self.image_prefix_length,  # 729 today
    ))
cache_tokens.extend(tokens[1:])
```

This produces: `[TextToken(BOS), ImageToken(hash, 729), TextToken(tok₁), ...]`

The structure generalizes to multi-image (see [Multi-Image Support](#multi-image-support-future)).

---

## Data Structures

### TreeNode

```python
@dataclass(slots=True)
class TreeNode:
    """Radix tree node representing a cached prefix segment.

    Uses slots=True for memory efficiency - each node saves ~100-200 bytes
    by eliminating __dict__. With thousands of nodes, this adds up.
    """

    # Tree structure
    # Keys are token.cache_key() tuples for O(1) lookup
    children: dict[tuple, TreeNode] = field(default_factory=dict)
    parent: TreeNode | None = None

    # Segment this node represents (edge label from parent)
    # Stores actual tokens (not just cache keys) for kv_length calculation
    tokens: tuple[CacheToken, ...] = ()

    # Physical KV cache pages for this segment
    # Invariant: len(physical_pages) == sum(tok.kv_length() for tok in tokens)
    physical_pages: tuple[int, ...] = ()

    # Reference counting for eviction protection
    lock_ref: int = 0

    # Eviction policy metadata
    last_access_time: float = 0.0

    # Heap tracking (versioned for correct LRU behavior)
    # Version increments on each enqueue; stale heap entries are ignored on pop
    heap_version: int = 0

    # Namespace (only set for root nodes, used for pruning empty namespaces)
    namespace: CacheNamespace | None = None

    @property
    def total_kv_length(self) -> int:
        # O(1) via invariant: len(physical_pages) == sum(tok.kv_length())
        return len(self.physical_pages)

    def is_leaf(self) -> bool:
        return len(self.children) == 0
```

### MatchResult

```python
@dataclass(slots=True)
class MatchResult:
    """Result of prefix matching operation."""

    matched_pages: list[int]            # Physical pages to reuse
    matched_kv_length: int              # Total KV positions matched
    matched_token_count: int            # Number of tokens matched
    last_node: TreeNode | None          # Terminal node (for locking)
    unmatched_tokens: list[CacheToken]  # Remaining tokens needing prefill


@dataclass(slots=True)
class InsertResult:
    """Result of cache insertion operation.

    Critical for ownership correctness: the runtime must set cache_owned_page_count
    based on inserted_pages, NOT on prompt_len. If insert() traverses existing nodes
    without creating new ones, inserted_pages will be less than the pages passed in.
    """
    node: TreeNode                  # The node representing the full sequence
    inserted_pages: int             # Number of pages actually claimed by cache
                                    # (0 if sequence already existed or only traversed existing nodes)
```

### RadixPrefixCache

```python
class RadixPrefixCache:
    """
    Radix tree for KV cache prefix sharing.

    Integrates with PageTable for dynamic page management.
    Uses all available pages - evicts on demand.
    """

    def __init__(self, page_table: PageTable):
        assert page_table.page_size == 1, "RadixPrefixCache requires page_size=1"
        self._trees: dict[CacheNamespace, TreeNode] = {}
        self._default_namespace = CacheNamespace()  # lora_id=None, image_hash=None
        self.page_table = page_table
        self._total_cached_pages = 0

    @property
    def total_cached_pages(self) -> int:
        """Total pages currently owned by the cache (for invariant checking)."""
        return self._total_cached_pages

    def _get_root(self, namespace: CacheNamespace) -> TreeNode:
        if namespace not in self._trees:
            root = TreeNode(namespace=namespace)  # Track namespace for pruning
            self._trees[namespace] = root
        return self._trees[namespace]

    # Core operations (namespace required for LoRA isolation)
    def match_prefix(self, tokens: list[CacheToken],
                     namespace: CacheNamespace | None = None) -> MatchResult: ...
    def insert(self, tokens: list[CacheToken], pages: list[int],
               namespace: CacheNamespace | None = None,
               from_node: TreeNode | None = None,
               from_token_idx: int = 0, from_page_idx: int = 0) -> InsertResult: ...
    def lock(self, node: TreeNode | None) -> None: ...
    def unlock(self, node: TreeNode | None) -> None: ...
    def evict(self, needed_pages: int) -> int: ...
```

### SequenceState Extensions

`SequenceState` (defined in `kestrel/moondream/runtime.py`) tracks per-sequence metadata during inference - things like the batch index, current length, and hidden states. We extend it with fields to integrate prefix caching:

```python
@dataclass
class SequenceState:
    # Existing fields
    batch_idx: int                      # Index in page table
    length: int                         # Current sequence length
    max_length: int                     # Maximum allowed length
    prompt_length: int | None = None    # Length after prefill
    last_hidden: Tensor | None = None   # Last hidden state for spatial decode
    lora_slot: int = 0                  # LoRA adapter slot

    # New: Prefix cache integration
    cache_tokens: list[CacheToken] | None = None  # Full sequence as cache tokens

    # Lock/ownership tracking (separate concerns!)
    cache_lock_node: TreeNode | None = None       # Node we hold a lock on (exactly one per sequence)
    cache_owned_page_count: int = 0               # Pages owned by cache (NOT freed at teardown)
    reused_page_count: int = 0                        # Pages reused from cache at start

    # Image regions for attention masking (bidirectional within regions, causal elsewhere)
    # Single image: [(1, 730)] meaning positions 1-729 are bidirectional
    # Multi-image: [(1, 730), (750, 1479), ...] for future support
    image_regions: list[tuple[int, int]] = field(default_factory=list)
```

**New fields explained:**

- `cache_tokens`: The complete prompt converted to cache tokens (including `ImageToken` if applicable). Used to insert into the cache after prefill completes.

- `cache_lock_node`: Reference to the tree node we hold a lock on. **Exactly one lock per sequence** - acquired after cache insertion, released at teardown. This is the node we call `prefix_cache.unlock(cache_lock_node)` on.

- `cache_owned_page_count`: How many pages are owned by the cache and should NOT be freed at sequence teardown. This equals `skip_positions + insert_result.inserted_pages`, where `skip_positions` is the number of reused cached pages and `inserted_pages` is the count of new pages actually claimed by the cache. On full prompt match, this equals `skip_positions` (not `prompt_len`) because the forced private suffix page is not cache-owned. This is the *authoritative* value for determining what to free at teardown.

- `reused_page_count`: How many pages were reused from the cache at sequence start (0 if cache miss). Used to determine skip length for append prefill. **Not the same as `cache_owned_page_count`**.

- `image_regions`: List of (start, end) tuples indicating KV positions that should use bidirectional attention (image embeddings). For the current single-image model, this is `[(1, 730)]`. See [Multi-Image Support](#multi-image-support-future) for how this generalizes.

**Why separate `reused_page_count` from `cache_owned_page_count`?**

Consider a cache miss with full prefill:
- `reused_page_count = 0` (nothing reused)
- After insertion: `cache_owned_page_count = prompt_length` (entire prompt now cached)

Or a partial cache hit:
- `reused_page_count = 730` (reused BOS + image)
- After insertion: `cache_owned_page_count = prompt_length` (entire prompt now cached, including suffix)

**Migration note**: This replaces the existing `image_length` field. Code that currently reads `image_length` should be updated to use `image_regions`:

```python
# Before:
if state.image_length > 0:
    prefix_len = 1 + state.image_length  # BOS + image

# After:
if state.image_regions:
    # For now, validate we're in single-image mode
    assert len(state.image_regions) == 1, "Multi-image not yet supported"
    assert state.image_regions[0] == (1, 730), "Unexpected image region"
    prefix_len = state.image_regions[0][1]  # End of first region
```

Until multi-image is implemented, the runtime should validate that `image_regions` is either `[]` or `[(1, 730)]` and raise an error otherwise.

---

## Tree Structure: Multi-Token Nodes vs. Single-Token Nodes

The radix tree uses multi-token nodes with splitting (same as SGLang), rather than a simple trie with one token per node. This section explains the tradeoffs.

### Option A: Multi-token nodes with splitting (chosen approach)

```
Root → [BOS, Image, "Hello", "World"] → [" how"] → ...
                                      → [" are", "you"] → ...
```

Each node stores a sequence of tokens. When a new prefix branches mid-node, we split the node.

**Pros:**
- Fewer nodes = less memory overhead per sequence
- Fewer pointer traversals during matching
- Fewer `lock_ref` increments (locking walks up to root)
- Better cache locality (tokens in a tuple are contiguous)

**Cons:**
- Splitting logic adds complexity
- Need to track page boundaries within nodes

### Option B: Single-token nodes (simple trie)

```
Root → [BOS] → [Image] → ["Hello"] → ["World"] → [" how"] → ...
                                               → [" are"] → ["you"] → ...
```

Each node stores exactly one token. No splitting needed.

**Pros:**
- Much simpler implementation
- Perfect 1:1 alignment between nodes, tokens, and pages (with `page_size=1`)
- Insertion is trivial: just append child nodes

**Cons:**
- More nodes = significantly more memory overhead
- Each Python node: ~200+ bytes (dict, parent pointer, tuples, metadata)
- 1000-token sequence = 1000 nodes ≈ 200 KB overhead

### Memory comparison

For a 1000-token prefix with no branching:

| Approach     | Nodes | Overhead   |
|--------------|-------|------------|
| Multi-token  | 1     | ~200 bytes |
| Single-token | 1000  | ~200 KB    |

With branching, multi-token degrades toward single-token, but the common case for prefix caching (long shared prefixes with branching at the end) strongly favors multi-token.

### Decision

We use multi-token nodes because:
1. SGLang uses this approach successfully in production
2. The 1000x memory difference matters for long prefixes
3. The splitting complexity is manageable and well-understood

---

## Core Operations

### Prefix Matching

```python
def match_prefix(
    self,
    tokens: list[CacheToken],
    namespace: CacheNamespace | None = None,
) -> MatchResult:
    """Find longest cached prefix matching the given tokens in the specified namespace."""
    ns = namespace or self._default_namespace
    # Don't create root on miss - only insert() creates roots
    # This prevents unbounded _trees growth from match-only queries
    root = self._trees.get(ns)
    if root is None:
        return MatchResult([], 0, 0, None, tokens)
    node = root
    matched_pages: list[int] = []
    matched_kv_length = 0
    token_idx = 0
    last_matched_node: TreeNode | None = None

    while token_idx < len(tokens):
        cache_key = tokens[token_idx].cache_key()

        if cache_key not in node.children:
            break

        child = node.children[cache_key]
        child_tokens = child.tokens

        # Match as many tokens as possible in this child
        match_len = 0
        for i, child_token in enumerate(child_tokens):
            if token_idx + i >= len(tokens):
                break
            if tokens[token_idx + i].cache_key() != child_token.cache_key():
                break
            match_len += 1

        if match_len == 0:
            break

        if match_len < len(child_tokens):
            # Partial match - split the node
            child = self._split_node(child, match_len)

        # Accumulate matched pages
        matched_pages.extend(child.physical_pages)
        matched_kv_length += child.total_kv_length
        token_idx += len(child.tokens)
        last_matched_node = child
        node = child

    # Update access time for entire matched path (not just leaf)
    # This ensures ancestors get recent timestamps for better LRU behavior
    if last_matched_node:
        self._update_access_time_path(last_matched_node)

    # Post-condition: pages and kv_length must be in sync (page_size=1 invariant)
    assert len(matched_pages) == matched_kv_length

    return MatchResult(
        matched_pages=matched_pages,
        matched_kv_length=matched_kv_length,
        matched_token_count=token_idx,
        last_node=last_matched_node,
        unmatched_tokens=tokens[token_idx:],
    )
```

### Node Splitting

When a match ends partway through a node's token sequence, we split the node:

```python
def _split_node(self, node: TreeNode, split_at: int) -> TreeNode:
    """
    Split a node at the given token index.

    Before: parent -> node(tokens=[A,B,C,D], pages=[...])
    After:  parent -> new_node(tokens=[A,B], pages=[...])
                          -> node(tokens=[C,D], pages=[...])

    Returns the new parent node (containing the prefix).
    """
    assert 0 < split_at < len(node.tokens)

    # Calculate page split point
    page_split = sum(tok.kv_length() for tok in node.tokens[:split_at])

    # Create new parent with prefix
    # heap_version starts at 0 (fresh node, never enqueued)
    new_parent = TreeNode(
        parent=node.parent,
        tokens=node.tokens[:split_at],
        physical_pages=node.physical_pages[:page_split],
        lock_ref=node.lock_ref,
        last_access_time=node.last_access_time,
        heap_version=0,  # Fresh node, no stale heap entries
    )

    # Update original node to contain only suffix
    node.tokens = node.tokens[split_at:]
    node.physical_pages = node.physical_pages[page_split:]
    node.parent = new_parent
    # Invalidate any stale heap entries for the original node
    # (its tokens/pages changed, so old entries are invalid)
    node.heap_version += 1

    # Re-enqueue the suffix node if it's an unlocked leaf (maintains heap invariant)
    self._maybe_add_to_eviction_heap(node)

    # Verify radix invariant: len(pages) == sum(tok.kv_length() for tok in tokens)
    assert len(new_parent.physical_pages) == sum(t.kv_length() for t in new_parent.tokens)
    assert len(node.physical_pages) == sum(t.kv_length() for t in node.tokens)

    # Update parent's child pointer
    if new_parent.parent:
        first_key = new_parent.tokens[0].cache_key()
        new_parent.parent.children[first_key] = new_parent

    # New parent points to original node
    node_first_key = node.tokens[0].cache_key()
    new_parent.children[node_first_key] = node

    return new_parent
```

**Note on access semantics**: Splitting is a structural operation; it does not by itself count as "accessing" the suffix segment. The suffix node retains its original `last_access_time`. Only `_update_access_time_path()` (called during match) updates timestamps, and it only updates the matched path (which after a split is the new_parent and its ancestors, not the suffix).

### Insertion

```python
def insert(
    self,
    tokens: list[CacheToken],
    pages: list[int],
    namespace: CacheNamespace | None = None,
    from_node: TreeNode | None = None,
    from_token_idx: int = 0,
    from_page_idx: int = 0,  # Avoids O(N) re-summation; caller passes matched_kv_length
) -> InsertResult:
    """
    Insert a prefix into the cache.

    If from_node is provided, starts insertion from that point
    (used when extending an existing cached prefix). Caller should pass
    from_page_idx = matched_kv_length to avoid O(N) prefix summation.

    Returns InsertResult with the terminal node and count of pages actually
    claimed by the cache. The caller MUST use inserted_pages (not len(pages))
    to determine cache_owned_page_count for correct ownership tracking.
    """
    ns = namespace or self._default_namespace
    root = self._get_root(ns)

    if from_node is None:
        node = root
        token_idx = 0
        page_idx = 0
    else:
        node = from_node
        token_idx = from_token_idx
        page_idx = from_page_idx  # Caller provides this (typically matched_kv_length)

    # Traverse existing path
    while token_idx < len(tokens):
        cache_key = tokens[token_idx].cache_key()

        if cache_key in node.children:
            child = node.children[cache_key]

            # Check how much of child matches
            match_len = 0
            for i, child_token in enumerate(child.tokens):
                if token_idx + i >= len(tokens):
                    break
                if tokens[token_idx + i].cache_key() != child_token.cache_key():
                    break
                match_len += 1

            if match_len < len(child.tokens):
                child = self._split_node(child, match_len)

            token_idx += len(child.tokens)
            page_idx += child.total_kv_length
            node = child
        else:
            break

    # Create new node for remaining tokens
    if token_idx < len(tokens):
        remaining_tokens = tuple(tokens[token_idx:])
        remaining_pages = tuple(pages[page_idx:])

        # Verify radix invariant before creating node
        expected_pages = sum(t.kv_length() for t in remaining_tokens)
        assert len(remaining_pages) == expected_pages, (len(remaining_pages), expected_pages)

        new_node = TreeNode(
            parent=node,
            tokens=remaining_tokens,
            physical_pages=remaining_pages,
            last_access_time=time.monotonic(),
        )
        first_key = remaining_tokens[0].cache_key()
        node.children[first_key] = new_node
        self._total_cached_pages += len(remaining_pages)

        # New leaf is initially unlocked - add to eviction heap for correct LRU
        # (Callers typically lock immediately, making it ineligible, but this
        # keeps the heap accurate if they don't)
        self._maybe_add_to_eviction_heap(new_node)

        return InsertResult(node=new_node, inserted_pages=len(remaining_pages))

    # Sequence already existed - no new pages claimed
    return InsertResult(node=node, inserted_pages=0)
```

### Lock and Unlock

Locking prevents eviction of a prefix and all its ancestors:

```python
def lock(self, node: TreeNode | None) -> None:
    """Prevent eviction of this node and all ancestors."""
    while node is not None:
        node.lock_ref += 1
        node = node.parent

def unlock(self, node: TreeNode | None) -> None:
    """Release lock. Node eligible for eviction when lock_ref reaches 0."""
    leaf = node  # Track original node (which is the leaf we're unlocking)
    while node is not None:
        node.lock_ref -= 1
        assert node.lock_ref >= 0, "Lock reference underflow"
        node = node.parent

    # If the leaf is now evictable (unlocked, is a leaf, not root), add to heap
    # This ensures O(log n) eviction rather than requiring heap rebuilds
    if leaf is not None and leaf.lock_ref == 0:
        self._maybe_add_to_eviction_heap(leaf)
```

### Eviction

Eviction frees pages from unlocked leaf nodes using LRU policy with a min-heap for O(log n) victim selection:

```python
import heapq

class RadixPrefixCache:
    def __init__(self, page_table: PageTable):
        # ... existing init ...

        # LRU eviction heap: (last_access_time, counter, node, version)
        # counter is used for stable ordering when times are equal
        # version ensures stale entries (from re-access) are ignored
        self._eviction_heap: list[tuple[float, int, TreeNode, int]] = []
        self._heap_counter = 0  # Monotonic counter for stable ordering

    def _add_to_eviction_heap(self, node: TreeNode) -> None:
        """Add node to eviction heap if it's an evictable leaf."""
        # Use parent check instead of root comparison (works across namespaces)
        if node.is_leaf() and node.lock_ref == 0 and node.parent is not None:
            node.heap_version += 1  # Invalidate any previous heap entries
            heapq.heappush(
                self._eviction_heap,
                (node.last_access_time, self._heap_counter, node, node.heap_version)
            )
            self._heap_counter += 1

    def _pop_eviction_candidate(self) -> TreeNode | None:
        """Pop the next valid eviction candidate (lazy deletion via versioning)."""
        while self._eviction_heap:
            _, _, node, version = heapq.heappop(self._eviction_heap)
            # Version check: skip if node was re-enqueued (version incremented)
            if node.heap_version != version:
                continue
            # Basic validity checks
            if not node.is_leaf() or node.lock_ref != 0:
                continue
            # Forward-compatible: verify node is still in tree
            # (Guards against future structural optimizations like coalescing)
            parent = node.parent
            if parent is None:
                continue  # Root nodes are not evictable
            first_key = node.tokens[0].cache_key() if node.tokens else None
            if first_key is None or parent.children.get(first_key) is not node:
                continue  # Node was detached from tree
            return node
        return None

    def evict(self, needed_pages: int) -> int:
        """
        Evict unlocked leaves to free pages.

        Called by page allocator when free pool is insufficient.
        Returns number of pages actually freed.

        Uses lazy-deletion min-heap for O(log n) per eviction instead of O(tree_size).
        """
        freed = 0

        while freed < needed_pages:
            victim = self._pop_eviction_candidate()
            if victim is None:
                # No more evictable nodes - rebuild heap and try once more
                self._rebuild_eviction_heap()
                victim = self._pop_eviction_candidate()
                if victim is None:
                    break

            # Free pages back to pool
            self.page_table.free_pages_to_pool(victim.physical_pages)
            freed += len(victim.physical_pages)
            self._total_cached_pages -= len(victim.physical_pages)

            # Remove from tree
            self._remove_node(victim)

            # Parent may now be a leaf - add to heap
            if victim.parent and victim.parent.is_leaf():
                self._add_to_eviction_heap(victim.parent)

        return freed

    def _rebuild_eviction_heap(self) -> None:
        """Rebuild eviction heap from scratch (fallback for heap staleness)."""
        self._eviction_heap.clear()
        for leaf in self._collect_unlocked_leaves():
            self._add_to_eviction_heap(leaf)

    def _collect_unlocked_leaves(self) -> list[TreeNode]:
        """
        Collect all leaf nodes with lock_ref == 0 across all namespaces.

        Uses explicit stack to avoid recursion depth issues.
        """
        leaves = []
        # Iterate all namespace roots (global heap across namespaces)
        for root in self._trees.values():
            stack = [root]
            while stack:
                node = stack.pop()
                if node.is_leaf():
                    # Use parent check instead of root comparison
                    if node.lock_ref == 0 and node.parent is not None:
                        leaves.append(node)
                else:
                    # Add children to stack (explicit iteration, no recursion)
                    stack.extend(node.children.values())

        return leaves

    def _remove_node(self, node: TreeNode) -> None:
        """Remove a leaf node from the tree."""
        assert node.is_leaf()
        assert node.lock_ref == 0

        # No need to invalidate heap - version check handles stale entries

        parent = node.parent
        if parent:
            del parent.children[node.tokens[0].cache_key()]

            # Prune empty namespace roots to prevent unbounded _trees growth
            # A root is identified by: parent.parent is None and parent.namespace is not None
            if parent.parent is None and len(parent.children) == 0:
                ns = parent.namespace
                if ns is not None:
                    assert parent.lock_ref == 0, "Cannot prune locked root"
                    del self._trees[ns]

    def evictable_page_count(self) -> int:
        """
        Count pages that could be evicted (unlocked leaves).

        Used by PageTable.can_reserve_with_eviction() to check if
        allocation is possible without actually evicting.

        PERFORMANCE: O(tree_size) traversal. Only call on slow path
        (after available < pages_needed), not in hot scheduling loops.
        """
        total = 0
        for leaf in self._collect_unlocked_leaves():
            total += len(leaf.physical_pages)
        return total

    def _maybe_add_to_eviction_heap(self, node: TreeNode) -> None:
        """Add node to eviction heap if it's now evictable.

        Call sites (must be explicit for correctness):
        1. After unlock() causes lock_ref to reach 0
        2. After _remove_node() makes a parent become a leaf
        3. After insertion creates a new leaf that is initially unlocked
        """
        # Use parent check instead of root comparison (works across namespaces)
        if node.is_leaf() and node.lock_ref == 0 and node.parent is not None:
            self._add_to_eviction_heap(node)

    def _update_access_time(self, node: TreeNode) -> None:
        """Update access time and re-enqueue if evictable.

        Re-enqueueing with new access time invalidates old heap entries
        (via version increment), ensuring LRU accuracy.
        """
        node.last_access_time = time.monotonic()
        # Re-enqueue to update position in heap (old entry invalidated by version)
        self._maybe_add_to_eviction_heap(node)

    def _update_access_time_path(self, node: TreeNode) -> None:
        """Update access time for node and all ancestors.

        Called during match_prefix() to ensure shared prefixes maintain
        recent timestamps. Without this, ancestors could have stale
        last_access_time even when they're heavily used via descendant
        matches, leading to suboptimal eviction decisions.
        """
        current = node
        now = time.monotonic()
        while current is not None and current.parent is not None:
            current.last_access_time = now
            # Re-enqueue if this node is evictable
            self._maybe_add_to_eviction_heap(current)
            current = current.parent
```

**Heap-based eviction improvements:**

| Aspect | Before (O(n)) | After (O(log n)) |
|--------|---------------|------------------|
| Victim selection | Full tree traversal | Heap pop + lazy validation |
| Rebuild cost | N/A | Amortized across many evictions |
| Memory | None | ~24 bytes per leaf node |

**Why evict leaves first?**

The radix tree is designed for prefix sharing. If we have cached:
- `[A, B, C]`
- `[A, B, D]`

Evicting at the sequence level would free the shared `[A, B]` prefix unnecessarily. By evicting leaves first (`C` and `D`), we preserve shared prefixes for future matches.

---

## Append Prefill

### Problem

When a request matches a cached prefix, we need to:
1. Reuse cached KV for positions 0 to `skip-1`
2. Only compute KV for positions `skip` to end
3. Run attention where Q attends to both cached and new KV

### Industry Research

We surveyed how FlashInfer, SGLang, and vLLM handle this problem:

| System | Approach | Key Implementation |
|--------|----------|--------------------|
| **FlashInfer** | Write-then-Prefill | `append_paged_kv_cache()` writes to cache, then standard prefill |
| **SGLang** | Split Attention + LSE Merge | `ForwardMode.EXTEND` with two-phase attention |
| **vLLM** | Split Attention + LSE Merge | `BatchDCPPrefillWrapper` with `merge_attn_states()` |

**Key finding**: SGLang and vLLM use split attention with log-sum-exp merging. However, our FA3 kernel's causal mask already handles append attention natively, so we can use a simpler single-pass approach.

### Solution: Single-Pass Append Attention

The FA3 causal mask uses the formula `seqlen_k - seqlen_q` to compute the offset between Q and K positions. This **automatically right-aligns Q with K**, which is exactly what append attention needs:

```
Example: 150-token cached prefix + 50-token suffix
├── seqlen_q = 50 (suffix only)
├── seqlen_k = 200 (full KV cache after writing suffix)
├── offset = seqlen_k - seqlen_q = 150
└── Result: Q[0] attends to K[0:151], Q[49] attends to K[0:200] ✓
```

This means we can run a **single attention pass** instead of two-phase split attention:

```
┌────────────────────────────────────────────────────────────────┐
│  1. Write new K, V to paged cache at suffix positions          │
│  2. Run single causal attention:                               │
│     ├── Q: suffix tokens [B, suffix_len, H, D]                 │
│     ├── K, V: full paged cache (positions 0 to total_len-1)    │
│     ├── seqused_k: total_len (prefix + suffix)                 │
│     └── causal=True (mask auto-aligns via seqlen_k - seqlen_q) │
└────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
def _append_prefill_attention(
    self,
    q: Tensor,                    # [1, suffix_len, n_heads, head_dim]
    k: Tensor,                    # [1, suffix_len, n_kv_heads, head_dim]
    v: Tensor,                    # [1, suffix_len, n_kv_heads, head_dim]
    kv_cache,                     # Paged KV cache
    page_table: Tensor,           # [1, max_pages]
    skip_positions: int,          # Number of cached KV positions (prefix length)
    slot_mapping: Tensor,         # For writing new K, V
) -> Tensor:
    """
    Append prefill: suffix Q attends to cached prefix + new suffix KV.

    The causal mask formula (seqlen_k - seqlen_q) automatically right-aligns
    Q with K, so Q[0] at logical position skip_positions attends correctly.
    """
    suffix_len = q.shape[1]
    total_len = skip_positions + suffix_len

    # Write new K, V to paged cache at suffix positions
    suffix_positions = torch.arange(
        skip_positions, total_len,
        device=q.device, dtype=torch.int32
    )
    kv_cache.update(suffix_positions, k, v, slot_mapping)

    # Single attention pass: suffix Q attends to full KV cache
    # The causal mask offset (seqlen_k - seqlen_q = skip_positions)
    # automatically right-aligns Q positions
    k_cache = kv_cache.k_cache.permute(0, 2, 1, 3)  # [pages, page_size, heads, dim]
    v_cache = kv_cache.v_cache.permute(0, 2, 1, 3)

    out, _ = _flash_attn_fwd(
        q,
        k_cache, v_cache,
        page_table=page_table,
        seqused_k=torch.tensor([total_len], device=q.device, dtype=torch.int32),
        causal=True,
        k_scale=kv_cache.k_scale,
        v_scale=kv_cache.v_scale,
    )
    return out
```

### Why This Works

The key insight is that FA3's causal mask was designed for cross-attention scenarios where `seqlen_q != seqlen_k`. The offset `seqlen_k - seqlen_q` right-aligns Q with K:

| Q local position | Effective global position | Can attend to K positions |
|------------------|---------------------------|---------------------------|
| 0 | skip_positions | 0 to skip_positions |
| i | skip_positions + i | 0 to skip_positions + i |
| suffix_len - 1 | total_len - 1 | 0 to total_len - 1 |

This is exactly the causal masking pattern we need for append attention.

### Advantages Over Split Attention

| Aspect | Single-Pass | Two-Phase Split |
|--------|-------------|-----------------|
| Kernel launches | 1 | 2 + merge |
| Implementation | No new code | Merge kernel needed |
| Memory | Single output | Two outputs + merge |
| Complexity | Uses existing FA3 | Additional synchronization |

### Implementation Steps

1. **Modify `text_decoder` for append mode**:
   - Accept `skip_positions` parameter
   - Write suffix K/V to cache before attention
   - Call attention with `seqused_k = total_len`

### Compute Savings

For a 730-token cache hit with 20-token suffix:

| Component | Full Prefill | Append Prefill | Savings |
|-----------|--------------|----------------|---------|
| QKV linear | 750 tokens | 20 tokens | 97% |
| Attention FLOPs | 750 × 750 | 20 × 750 | 97% |
| MLP | 750 tokens | 20 tokens | 97% |

### Full Prompt Match Handling

When the entire prompt is already cached (`match.matched_kv_length == prompt_len`), we face a challenge: we need logits from the last prompt position to start generation, but KV cache alone doesn't give us the final hidden state. We also cannot compute into shared cached pages (would corrupt them for other sequences).

**Solution: Always compute at least one suffix KV position**

We cap reuse at `prompt_len - 1`, ensuring there's always at least one KV position to compute:

```python
# Cap reuse to ensure at least one suffix KV position
skip_positions = min(match.matched_kv_length, prompt_len - 1)
```

This guarantees:
1. We always have fresh pages to compute into (not shared cached pages)
2. We can compute valid logits for the first generated token
3. No writes occur into shared cached pages

**Single-image constraint**: In single-image mode, the suffix is always text tokens (1:1 with KV positions). The code asserts that image prompts include at least one text token after BOS, so backing off by one KV position always lands on a text token boundary.

**Future multi-image generalization**: For variable-length tokens (e.g., multiple images), the correct rule is to back off by one *cache token*, not one KV position. This would require computing the KV length of the last cache token and capping reuse at `prompt_len - last_token_kv_len`. Not needed for single-image Moondream where prompt templates guarantee text follows the image.

**Trade-off**: Full prompt matches still incur one-token prefill compute, and do not extend the cache; this is a deliberate choice to avoid storing per-leaf hidden state. We sacrifice one token of reuse (0.1% for 1000-token prompts), which is negligible compared to the complexity of caching hidden states on nodes.

**Ownership implications**: On full prompt match:
- `cache_owned_page_count = skip_positions` (not `prompt_len`) because the last token is computed into a private page
- No cache insertion occurs (the prompt is already fully cached)
- The private last-token page is freed at teardown via `erase()`

**Alternative considered (not chosen)**: Store `last_hidden` on leaf nodes and return it on full match. This adds complexity (hidden state storage, dtype/device matching, stale state handling) for minimal benefit.

---

## Integration Points

### PageTable Modifications

**File**: `kestrel/kv_cache.py`

#### Design Assumptions

**Page Size = 1**: We assume `page_size=1` for simplicity. This means:
- 1 physical page = 1 KV position
- No partial page handling needed
- `logical_page_idx == kv_position`

In this design, "KV positions" == "pages" because `page_size=1`; all ownership counters (e.g., `cache_owned_page_count`) are expressed in **pages**.

If we later want larger pages, we'd need to handle:
- Partial page reuse (cached prefix ends mid-page)
- Page boundary alignment for cache insertion
- Wasted space at sequence ends

**Relationship to Existing `reserve()`**: The current `reserve(batch_idx, seq_len)` allocates pages for a contiguous sequence starting from the current capacity. With prefix caching, we need more control:

| Operation | Use Case |
|-----------|----------|
| `reserve()` | Existing: extend capacity for decode tokens |
| `allocate_pages()` + `map_pages()` | New: allocate suffix pages after mapping cached prefix |

We keep `reserve()` for decode-time extension and add the new methods for prefill-time setup.

#### New Methods

```python
class PageTable:
    def __init__(self, ...):
        # ... existing init ...
        self.prefix_cache: RadixPrefixCache | None = None

    def set_prefix_cache(self, cache: RadixPrefixCache) -> None:
        """Attach prefix cache for on-demand eviction."""
        self.prefix_cache = cache

    def allocate_pages(self, count: int) -> list[int]:
        """
        Allocate physical pages from free pool, evicting from cache if needed.

        Unlike reserve(), this returns unbound physical pages that must be
        mapped via map_pages(). Used for suffix allocation after cache hit.
        """
        available = len(self.free_pages)

        if available < count and self.prefix_cache:
            needed = count - available
            freed = self.prefix_cache.evict(needed)
            available += freed

        if available < count:
            raise RuntimeError(
                f"Cannot allocate {count} pages: {available} available, "
                f"all cached prefixes locked by active sequences"
            )

        return [self.free_pages.pop() for _ in range(count)]

    def map_pages(
        self,
        batch_idx: int,
        logical_start: int,
        physical_pages: list[int],
    ) -> None:
        """
        Map physical pages into a batch's page table at specified positions.

        Used for:
        1. Mapping cached pages at logical_start=0
        2. Mapping newly allocated suffix pages at logical_start=skip_positions

        INVARIANT: map_pages() must not be called with a (batch_idx, logical_start, len)
        range that overlaps any previously mapped range for that batch. This is because
        page_table_cpu[batch_idx].extend() assumes non-overlapping, sequential mappings.
        Overlapping calls will corrupt the page list and cause incorrect frees at erase().

        If remapping is needed in the future (e.g., Design 2 alternative for full prompt
        match), implement a separate map_pages_overwrite() API that handles the list update.

        Args:
            batch_idx: Target batch slot
            logical_start: First logical page index (== first KV position with page_size=1)
            physical_pages: Physical page indices to map
        """
        # Enforce sequential, non-overlapping invariant
        assert logical_start == len(self.page_table_cpu[batch_idx]), (
            f"map_pages must be called sequentially: expected logical_start="
            f"{len(self.page_table_cpu[batch_idx])}, got {logical_start}"
        )
        for i, phys_page in enumerate(physical_pages):
            logical_page = logical_start + i
            self._page_table_cpu_tensor[batch_idx, logical_page] = phys_page
            self.physical_to_logical[batch_idx, phys_page] = logical_page

        end = logical_start + len(physical_pages)
        self._sync_page_table_row(batch_idx, start=logical_start, end=end)

        # Update capacity to cover all mapped pages
        new_capacity = end * self.page_size
        self.capacity[batch_idx] = max(self.capacity[batch_idx], new_capacity)

        # Track pages for this batch (for erase())
        # Note: For cached pages, we track them but won't free them
        self.page_table_cpu[batch_idx].extend(physical_pages)

    def get_pages(self, batch_idx: int, start: int, end: int) -> list[int]:
        """
        Get physical page indices for a range of logical pages.

        Used after prefill to get the physical pages for cache insertion.
        """
        return [
            self._page_table_cpu_tensor[batch_idx, i].item()
            for i in range(start, end)
        ]

    def free_pages_to_pool(self, pages: tuple[int, ...] | list[int]) -> None:
        """Return pages to free pool. Called by cache eviction."""
        self.free_pages.extend(reversed(pages))

    def can_reserve_with_eviction(self, size: int, batch_idx_int: int | None = None) -> bool:
        """
        Check if we can reserve pages, considering evictable cache pages.

        Extends existing can_reserve() to account for prefix cache.
        """
        if batch_idx_int is None:
            # New request: need batch slot + pages
            if len(self.free_batch_idx) == 0:
                return False
            pages_needed = (size + self.page_size - 1) // self.page_size
        else:
            # Existing request: just need pages
            current = self.capacity[batch_idx_int]
            if size <= current:
                return True
            pages_needed = (size - current + self.page_size - 1) // self.page_size

        available = self.pages_available
        if available >= pages_needed:
            return True

        # Check if eviction could free enough
        if self.prefix_cache:
            evictable = self.prefix_cache.evictable_page_count()
            return available + evictable >= pages_needed
        return False

    def erase(self, batch_idx: int, cached_page_count: int = 0) -> None:
        """
        Release a batch's resources.

        Args:
            batch_idx: Batch slot to release
            cached_page_count: Number of leading pages belonging to cache.
                               These are NOT returned to free pool.
        """
        self.free_batch_idx.append(batch_idx)

        allocated_pages = self.page_table_cpu[batch_idx]
        # Skip cached pages (they belong to the prefix cache tree)
        pages_to_free = allocated_pages[cached_page_count:]
        self.free_pages.extend(reversed(pages_to_free))

        self.page_table_cpu[batch_idx] = []
        self.capacity[batch_idx] = 0
```

**Optional defensive clearing**: The GPU-side tensors (`_page_table_cpu_tensor[batch_idx, :]` and `physical_to_logical[batch_idx, :]`) retain stale values after erase. This is safe because inactive batch indices should never be read, but clearing them can help catch bugs in tests:

```python
# Optional: clear GPU tensors to catch stale reads (useful in tests)
# self._page_table_cpu_tensor[batch_idx, :].fill_(-1)
# self._sync_page_table_row(batch_idx)
# self.physical_to_logical[batch_idx, :] = -1
```

#### Slot Mapping for Append Prefill

The existing `build_slot_mapping()` works correctly for append prefill because:

1. Cached pages are already mapped in the page table at positions 0..skip_positions-1
2. New pages are mapped at positions skip_positions..prompt_len-1
3. When we call `build_slot_mapping(batch_idx, suffix_positions)`, it looks up the physical pages from the page table

```python
# Example: 730 cached positions, 20 new tokens
# Page table after map_pages():
#   [0..729] -> cached physical pages
#   [730..749] -> newly allocated pages

suffix_positions = torch.arange(730, 750, device=device)
slot_mapping = page_table.build_slot_mapping(batch_idx, suffix_positions)
# slot_mapping correctly maps to the new physical pages
```

#### Invariants

1. **Page ownership (three states)**: A physical page is in exactly one of:
   - **Free pool**: In `free_pages`, available for allocation
   - **Cache-owned**: Owned by exactly one TreeNode in the prefix cache; may be mapped by multiple batch slots simultaneously (sharing cached KV)
   - **Batch-private**: Allocated to exactly one batch slot for suffix/decode pages; not yet owned by cache (will become cache-owned after insertion)

   Cache eviction only returns **cache-owned** pages to the free pool. Batch-private pages are returned via `erase()` when the sequence completes (unless they became cache-owned via insertion).

2. **Page table mapping**: Multiple batch slots can map the same physical page (sharing cached KV). The page is freed only when evicted from the cache tree.

3. **Capacity tracking**: `capacity[batch_idx]` reflects the total mapped KV positions, including both cached and new pages.

4. **Order preservation**: `page_table_cpu[batch_idx]` stores pages in logical order. With caching, cached pages come first, then newly allocated pages.

### Runtime Modifications

**File**: `kestrel/moondream/runtime.py`

```python
class MoondreamRuntime:
    def __init__(self, ...):
        # ... existing init ...
        self.prefix_cache = RadixPrefixCache(self.page_table)
        self.page_table.set_prefix_cache(self.prefix_cache)

    def start_sequence(
        self,
        tokens: list[Token],
        image: pyvips.Image | np.ndarray | None = None,
        image_crops: OverlapCropOutput | None = None,
        max_new_tokens: int = 256,
        *,
        image_hash: bytes | None = None,
    ) -> tuple[SequenceState, Tensor]:
        """Start a new sequence, potentially reusing cached KV prefix."""

        # Single-image mode invariants (machine-check the coupling we rely on)
        if image is None:
            assert image_hash is None, "image_hash must be None when image is None"
        else:
            assert image_hash is not None, "image_hash must be provided when image is not None"
            # Prompt templates always include text after the image (e.g., "<answer>").
            # This ensures full-prompt-match handling can back off by one text token.
            assert len(tokens) >= 2, "Image prompts must include at least one text token after BOS"

        # Build cache tokens (inline, generalizes to multi-image)
        image_kv_length = self.image_prefix_length if image else 0
        cache_tokens: list[CacheToken] = [tokens[0]]  # BOS
        if image_hash is not None:
            cache_tokens.append(ImageToken(
                content_hash=int.from_bytes(image_hash[:16], 'big'),  # 128-bit hash
                kv_length_=image_kv_length,
            ))
        cache_tokens.extend(tokens[1:])

        # Image regions for attention masking
        # Single image: [(1, 1 + image_kv_length)]
        # Multi-image: would have multiple regions
        image_regions = [(1, 1 + image_kv_length)] if image else []

        # Try to match prefix (with LoRA + image namespace)
        # BOS KV depends on image content (training bug), so different images need separate trees
        image_hash_int = int.from_bytes(image_hash[:16], 'big') if image_hash else None
        namespace = CacheNamespace(
            lora_id=lora_adapter_id,  # None for base model
            image_hash=image_hash_int,
        )
        match = self.prefix_cache.match_prefix(cache_tokens, namespace=namespace)

        # Compute prompt_len early (needed for full prompt match handling)
        prompt_len = len(tokens) + (image_kv_length if image else 0)

        # Allocate batch slot
        batch_idx = self.page_table.allocate()
        batch_tensor = torch.tensor([batch_idx], device=self.device, dtype=torch.int32)

        # Determine reuse eligibility
        can_reuse_any = match.matched_kv_length > 0

        # INVARIANT: In the current single-image design with image-hash namespacing,
        # any hit in an image namespace MUST include BOS+image. This is because:
        # 1. BOS KV depends on image content (training bug), so each image has its own tree
        # 2. The image token immediately follows BOS in every sequence
        # 3. Therefore, any match in an image namespace necessarily includes BOS+image
        #
        # If this invariant ever fails, it means the namespace design changed and the
        # reuse logic must be redesigned. We cannot safely "fall back" to full prefill
        # after mapping pages (violates map_pages() non-overlap invariant).
        if image and can_reuse_any:
            assert match.matched_kv_length >= (1 + image_kv_length), (
                f"Invariant violated: image namespace hit ({match.matched_kv_length} KV) "
                f"must include BOS+image ({1 + image_kv_length} KV)"
            )

        # Temporary lock on matched prefix (will be released after insertion)
        temp_lock_node: TreeNode | None = None

        # Full prompt cached ONLY matters if we're taking the hit path
        # Critical: must be gated behind can_reuse_any to avoid skipping insertion on miss path
        full_prompt_cached = can_reuse_any and (match.matched_kv_length >= prompt_len)

        if can_reuse_any:
            # Calculate skip_positions FIRST (before mapping)
            # Cap at prompt_len - 1 to ensure at least one suffix KV position (see Full Prompt Match section)
            skip_positions = min(match.matched_kv_length, prompt_len - 1)

            # Map ONLY the pages we actually reuse (not all matched pages)
            # Critical: with full prompt match, skip_positions < matched_kv_length
            cached_pages = match.matched_pages[:skip_positions]
            self.page_table.map_pages(batch_idx, 0, cached_pages)

            # Temporary lock to protect matched prefix during prefill
            temp_lock_node = match.last_node
            self.prefix_cache.lock(temp_lock_node)

            # Derive suffix tokens from skip_positions
            if image:
                # KV layout: [BOS(1)] [Image(image_kv_length)] [Text tokens...]
                # Invariant guarantees skip_positions >= 1 + image_kv_length
                prefix_kv = 1 + image_kv_length
                text_kv_cached = skip_positions - prefix_kv
                assert 0 <= text_kv_cached <= len(tokens) - 1, f"text_kv_cached {text_kv_cached} out of bounds"
                # tokens[0] = BOS, tokens[1:] = text after image
                suffix_tokens = tokens[1 + text_kv_cached:]
                image_embed = None  # Image is cached, skip vision encode
            else:
                # No image: KV position == token index
                suffix_tokens = tokens[skip_positions:]
                image_embed = None

            # Invariant: full prompt match must still yield at least one suffix token
            if full_prompt_cached:
                assert len(suffix_tokens) >= 1, "Full prompt match must compute at least one suffix token"

            use_append_prefill = True
        else:
            # No meaningful cache hit - full prefill
            skip_positions = 0
            suffix_tokens = tokens
            image_embed = self.encode_image(image, overlap=image_crops) if image else None
            use_append_prefill = False
            match = MatchResult([], 0, 0, None, cache_tokens)  # Reset match

        # Allocate pages for suffix
        target_length = self._compute_target_length(tokens, image, max_new_tokens)
        suffix_pages_needed = target_length - skip_positions

        if suffix_pages_needed > 0:
            new_pages = self.page_table.allocate_pages(suffix_pages_needed)
            self.page_table.map_pages(batch_idx, skip_positions, new_pages)

        # Run prefill (prompt_len computed earlier)

        if use_append_prefill and skip_positions > 0:
            # Append prefill: only compute suffix, attend to cached + new KV
            suffix_embed = self._embed_tokens(suffix_tokens)
            hidden, logits = self._append_prefill(
                suffix_embed,
                batch_idx,
                start_position=skip_positions,
                total_length=prompt_len,
                use_prefix_attn=bool(image),
            )
        else:
            # Full prefill path
            hidden, logits = self._full_prefill(
                tokens, image_embed, batch_idx, bool(image)
            )

        # Insert into cache (or skip if full prompt already cached)
        # With full_prompt_cached, the prompt already exists in cache - we computed
        # the last token into a private page, NOT a cache-owned page.
        if full_prompt_cached:
            # Full prompt already cached - do NOT insert (would be a no-op anyway)
            # Keep temp_lock_node as our cache lock
            # cache_owned_page_count = skip_positions (not prompt_len!) because:
            #   - positions 0..skip_positions-1 are cache-owned (shared pages)
            #   - position skip_positions..prompt_len-1 is our private recomputed page
            cache_lock_node = temp_lock_node
            cache_owned_page_count = skip_positions
        else:
            # Normal case: insert prompt into cache
            # cache_tokens was built from prompt only, so use it directly
            # (no need for _count_prompt_tokens - that would be a footgun if misimplemented)
            prompt_pages = self.page_table.get_pages(batch_idx, 0, prompt_len)
            insert_result = self.prefix_cache.insert(
                cache_tokens, prompt_pages,
                namespace=namespace,
                from_node=match.last_node,
                from_token_idx=match.matched_token_count,
                from_page_idx=skip_positions,  # Avoids O(N) re-summation
            )

            # Lock transfer: ensure exactly one lock held at end
            # IMPORTANT: Handle identity case where insert_result.node IS temp_lock_node
            # This can happen if insert() found the sequence already exists (inserted_pages=0)
            if temp_lock_node is None:
                # Miss path: acquire the one long-lived lock
                self.prefix_cache.lock(insert_result.node)
                cache_lock_node = insert_result.node
            elif insert_result.node is temp_lock_node:
                # Identity case: insert returned the same node we already locked
                # Keep existing lock, don't lock+unlock (which would leave us with no lock)
                cache_lock_node = temp_lock_node
            else:
                # Partial hit path: transfer lock from matched node to inserted node
                self.prefix_cache.lock(insert_result.node)
                self.prefix_cache.unlock(temp_lock_node)
                cache_lock_node = insert_result.node

            # CRITICAL: cache_owned_page_count is reused pages + newly inserted pages
            # This handles the case where insert() traversed existing nodes without
            # creating new ones (e.g., if multiple sequences race to insert the same prefix)
            cache_owned_page_count = skip_positions + insert_result.inserted_pages

        # Create sequence state
        state = SequenceState(
            batch_idx=batch_idx,
            length=prompt_len,
            max_length=target_length,
            prompt_length=prompt_len,
            last_hidden=hidden[:, -1, :].squeeze(0).detach(),
            cache_tokens=cache_tokens,
            cache_lock_node=cache_lock_node,                # Exactly one lock
            cache_owned_page_count=cache_owned_page_count,  # May be < prompt_len for full match
            reused_page_count=skip_positions,                   # For debugging/metrics
            image_regions=image_regions,
        )

        self.active_sequences[batch_idx] = state
        return state, logits
```

### Scheduler Modifications

**File**: `kestrel/scheduler/scheduler.py`

```python
def _release_sequence(self, seq: ScheduledSequence) -> None:
    """Release sequence resources."""
    state = seq.state

    # Unlock the cached prefix (exactly one unlock per sequence)
    if state.cache_lock_node:
        self.runtime.prefix_cache.unlock(state.cache_lock_node)

    # Release batch slot
    # Don't free cache-owned pages - they belong to the prefix cache tree
    self.runtime.page_table.erase(state.batch_idx, state.cache_owned_page_count)

    # Release adapter slot
    self.runtime.release_adapter_slot(state.lora_slot)
```

### Spatial Sampling Modifications

**File**: `kestrel/scheduler/spatial.py`

The `compute_spatial_values` function needs to return bin indices alongside continuous values:

```python
def compute_spatial_values(...) -> SpatialOutput:
    # ... existing sampling code ...

    coord_bins = torch.argmax(coord_logits, dim=-1)
    width_bins = torch.argmax(width_logits, dim=-1)
    height_bins = torch.argmax(height_logits, dim=-1)

    coord_values = tables.coord_value_lut[coord_bins]
    width_values = tables.size_value_lut[width_bins]
    height_values = tables.size_value_lut[height_bins]

    return SpatialOutput(
        coord_values=coord_values,
        coord_bins=coord_bins,
        width_values=width_values,
        width_bins=width_bins,
        height_values=height_values,
        height_bins=height_bins,
    )
```

### Engine Modifications

**File**: `kestrel/engine.py`

Add image hash to pending requests:

```python
@dataclass
class _PendingRequest:
    # ... existing fields ...
    image_hash: bytes | None = None

def _submit_request(self, ..., image: pyvips.Image | np.ndarray | None, ...):
    image_hash = None
    if image is not None:
        if isinstance(image, pyvips.Image):
            raw_bytes = image.write_to_buffer(".png")
        else:
            raw_bytes = image.tobytes()
        image_hash = hashlib.sha256(raw_bytes).digest()

    request = _PendingRequest(
        ...,
        image=image,
        image_hash=image_hash,
    )
```

---

## Duplicate Prefill Prevention

### Problem

Without coordination, concurrent requests with the same prefix could run full prefill:

```
Timeline (BAD - if prefills were parallel):
  A arrives → full prefill → decode → complete → insert cache
  B arrives → full prefill (WASTED!) → decode → complete
```

### Solution

Insert into cache **immediately after prefill**, before decode begins:

```
Timeline (GOOD):
  A arrives → prefill → insert+lock → decode...
  B arrives ──────────→ cache hit! → append prefill → decode...
```

This is implemented in the `start_sequence` flow shown above:

1. After prefill completes, immediately call `prefix_cache.insert()`
2. Lock the inserted node to prevent eviction during decode
3. Subsequent requests will match this cached prefix

### Why Duplicate Prefills Can't Happen

Kestrel runs prefills sequentially - one sequence at a time. The cache check happens *before* prefill starts:

```
Scheduler loop:
  1. Pick next request from queue
  2. Check prefix cache → hit or miss
  3. Run prefill (full or append)
  4. Insert into cache
  5. Start decode
  6. Go to step 1
```

Since step 4 completes before the next request's step 2, there's no window for duplicate prefills. Request B will always see A's cached result.

**Note**: If Kestrel later adds parallel prefills (multiple prefills running concurrently), this assumption would break. In that case, the design would need explicit handling for the case where `insert()` finds an existing node - the duplicate's pages should be freed rather than claimed as cache-owned.

---

## Memory Management

### Dynamic Sizing

The cache has no fixed size budget. It uses all available pages:

1. Active sequences allocate pages from the free pool
2. Completed sequences' pages transfer to cache (owned by tree nodes)
3. When free pool is empty and allocation is needed:
   - Evict unlocked leaves from cache
   - Return their pages to free pool
   - Retry allocation

### Page Ownership Model

**Invariant**: A page is in exactly one of three states:

| State | Owner | Freed by |
|-------|-------|----------|
| **Free pool** | None | N/A (already free) |
| **Cache-owned** | Exactly one TreeNode | Cache eviction |
| **Batch-private** | Exactly one batch slot | `erase()` at sequence completion |

- Active sequences hold **locks** on nodes, not ownership
- Locks prevent eviction but don't affect ownership
- Decode extension pages start as batch-private, become cache-owned if the sequence is later cached (not currently implemented - decode tokens are not cached)

### Ownership Policy: Always Reuse Cached Prefix

**Critical design decision**: When a cache hit occurs (`matched_kv_length > 0`), we **always** reuse the cached pages and only compute suffix pages. We never do "full prefill into fresh pages" when a cached prefix exists.

**Why this matters**: If we did full prefill and then called `insert()`, the tree would only claim pages for the *new* suffix portion (since the prefix path already exists). The runtime would incorrectly set `cache_owned_page_count = prompt_len`, causing the "duplicate" prefix pages to leak - they're neither freed at teardown (marked as cache-owned) nor actually owned by the cache tree.

**Example of the bug we avoid**:
```
1. Cache contains [BOS] (page P1, owned by existing BOS node)
2. New request [BOS, A, B] does full prefill into FRESH pages [P2, P3, P4]
   - P2 = computed BOS (duplicate of cached P1)
   - P3 = A
   - P4 = B
3. insert() traverses existing BOS node (P1), creates new [A, B] node with [P3, P4]
   (Only the suffix pages get claimed - the tree already has a BOS node)
4. insert_result.inserted_pages = 2 (only [A, B] suffix pages)
5. cache_owned_page_count = skip_positions + inserted_pages = 0 + 2 = 2
6. page_table_cpu[batch_idx] = [P2, P3, P4]  (our prefill pages)
7. At teardown: erase(batch_idx, cached_page_count=2)
   - Treats first 2 pages [P2, P3] as cache-owned → NOT freed
   - Frees remaining pages [P4] → returned to free pool

   BUG: P2 is our duplicate BOS page, NOT owned by cache tree (P1 is).
        P2 is never freed → LEAKED!
        P4 IS owned by cache tree, but gets freed → CORRUPTION risk!
```

**The fix**: By always reusing cached prefix:
```
1. Cache contains [BOS] (page P1)
2. New request [BOS, A, B] maps P1, allocates [P2, P3] for suffix
3. insert() creates new node for [A, B] with pages [P2, P3]
4. insert_result.inserted_pages = 2
5. cache_owned_page_count = skip_positions + inserted_pages = 1 + 2 = 3
6. At teardown, erase(batch_idx, cached_page_count=3) frees nothing → correct!
```

No duplicate pages are ever computed, so no pages can leak.

### Eviction Policy

**LRU (Least Recently Used)** at the node level:

1. Collect all unlocked leaf nodes (`lock_ref == 0`)
2. Sort by `last_access_time` (ascending)
3. Evict oldest leaves until enough pages are freed
4. Parent nodes may become leaves after children are evicted (eligible for future eviction)

### Memory Pressure Handling

If eviction cannot free enough pages (all leaves are locked):

```python
def allocate_pages(self, count: int) -> list[int]:
    available = len(self.free_pages)

    if available < count and self.prefix_cache:
        freed = self.prefix_cache.evict(count - available)
        available += freed

    if available < count:
        raise RuntimeError(
            f"Cannot allocate {count} pages: {available} available, "
            f"all cached prefixes are locked by active sequences"
        )

    return [self.free_pages.pop() for _ in range(count)]
```

This error indicates the system is over-subscribed: too many concurrent sequences relative to total KV cache capacity.

---

## LoRA Namespace Isolation

Different LoRA adapters produce different KV values for the same tokens. Without namespace isolation, requests with different adapters could incorrectly share cached KV pairs, producing wrong outputs.

### Problem

Consider two requests:
1. Request A: prompt "Hello world" with LoRA adapter `finetune-v1`
2. Request B: prompt "Hello world" with LoRA adapter `finetune-v2`

Without namespacing, B would match A's cached prefix and get incorrect KV values.

### Solution: Namespace Keying

Include adapter identity in the cache key lookup:

```python
@dataclass(frozen=True, slots=True)
class CacheNamespace:
    """Identifies the context that produced KV values.

    IMPORTANT: The BOS token's KV values depend on the image content.

    This is due to a bug in the attention mask used during training: the BOS
    position (index 0) was allowed to attend to the image embedding region
    (indices 1-729) bidirectionally. As a result, the BOS hidden state - and
    thus its KV values - are influenced by the specific image present in the
    sequence.

    Consequence: sequences with different images (or no image vs. image)
    cannot share BOS KV values. We enforce this by including the image hash
    in the namespace, giving each unique image its own radix tree.

    Note: This bug will be fixed in future model training, at which point
    image_hash can be removed from the namespace.
    """
    lora_id: int | None = None       # None = base model; non-None = adapter ID
                                     # IMPORTANT: 0 is NOT a special value (use None for base)
    image_hash: int | None = None    # 128-bit hash of image, or None for text-only

    # frozen=True makes it hashable automatically (no need for __hash__/__eq__)


class RadixPrefixCache:
    def __init__(self, page_table: PageTable):
        assert page_table.page_size == 1, "RadixPrefixCache requires page_size=1"
        # Separate tree root per namespace
        self._trees: dict[CacheNamespace, TreeNode] = {}
        self._default_namespace = CacheNamespace()  # lora_id=None, image_hash=None
        self.page_table = page_table

    def _get_root(self, namespace: CacheNamespace) -> TreeNode:
        if namespace not in self._trees:
            root = TreeNode(namespace=namespace)  # Track namespace for pruning
            self._trees[namespace] = root
        return self._trees[namespace]

    def match_prefix(
        self,
        tokens: list[CacheToken],
        namespace: CacheNamespace | None = None,
    ) -> MatchResult:
        """Find longest cached prefix in the given namespace."""
        ns = namespace or self._default_namespace
        # Don't create root on miss - only insert() creates roots
        root = self._trees.get(ns)
        if root is None:
            return MatchResult([], 0, 0, None, tokens)
        # ... rest of matching logic uses this root ...
```

### Integration

```python
# In start_sequence():
image_hash_int = int.from_bytes(image_hash[:16], 'big') if image_hash else None
namespace = CacheNamespace(
    lora_id=lora_adapter_id,  # None for base model
    image_hash=image_hash_int,
)
match = self.prefix_cache.match_prefix(cache_tokens, namespace=namespace)
# ...
insert_result = self.prefix_cache.insert(
    cache_tokens, pages,
    namespace=namespace,
    from_node=match.last_node,
    from_token_idx=match.matched_token_count,
    from_page_idx=match.matched_kv_length,
)
```

### Memory Implications

Each namespace gets its own tree. With `(lora_id, image_hash)` keying:

- **LoRA isolation**: Different adapters cannot share KV (correct by design)
- **Image isolation**: Different images cannot share BOS KV (necessary due to training bug)
- **Text-only isolation**: Text-only requests have their own tree (`image_hash=None`)

Memory impact:
- One root TreeNode per unique `(lora_id, image_hash)` pair (~200 bytes each)
- Workloads with many unique images will have many small trees
- Workloads with repeated images benefit fully from prefix sharing

**Optimization for high-cardinality image workloads**: If image diversity is very high (each request has unique image), consider disabling prefix caching entirely or implementing a per-namespace cache size limit with aggressive eviction.

### What's in the Namespace

The namespace contains factors that affect KV values and can change per-request:

| Factor | In Namespace? | Rationale |
|--------|---------------|-----------|
| LoRA adapter ID | ✅ Yes (`None` = base model) | Changes per-request, affects all KV values |
| Image hash | ✅ Yes (`None` = text-only) | BOS KV depends on image (training bug, see above) |
| Base model weights | ❌ No | Fixed at startup |
| RoPE scaling config | ❌ No | Fixed at startup |
| Tokenizer version | ❌ No | Fixed at startup |
| Image encoder weights | ❌ No | Fixed at startup |
| KV cache dtype (FP8/BF16) | ❌ No | Fixed at startup |

**Assumption**: A single Kestrel process runs one base model configuration. If this changes (e.g., hot-swapping model weights), the namespace would need to include a model version identifier.

---

## File Organization

Inspired by SGLang's modular structure, we organize prefix caching code as follows:

### Directory Structure

```
kestrel/
├── prefix_cache/                    # New package for prefix caching
│   ├── __init__.py                  # Public API exports
│   ├── base.py                      # Abstract base class + protocols
│   ├── radix_cache.py               # RadixPrefixCache, TreeNode, MatchResult
│   ├── namespace.py                 # CacheNamespace + namespace management
│   └── eviction.py                  # LRU eviction policy (extensible later)
│
├── kv_cache.py                      # Existing - add page allocation methods
├── moondream/
│   ├── runtime.py                   # Existing - add cache integration
│   └── text.py                      # Existing - add append prefill attention
│
└── scheduler/
    └── scheduler.py                 # Existing - add cache unlock on release

kestrel-kernels/
```

### Package Design

**`kestrel/prefix_cache/__init__.py`**:
```python
from .radix_cache import RadixPrefixCache, TreeNode, MatchResult
from .namespace import CacheNamespace
from .base import BasePrefixCache

__all__ = [
    "RadixPrefixCache",
    "TreeNode",
    "MatchResult",
    "CacheNamespace",
    "BasePrefixCache",
]
```

**`kestrel/prefix_cache/base.py`**:
```python
from abc import ABC, abstractmethod
from typing import Protocol

class BasePrefixCache(ABC):
    """Abstract base for prefix cache implementations."""

    @abstractmethod
    def match_prefix(
        self,
        tokens: list[CacheToken],
        namespace: CacheNamespace | None = None,
    ) -> MatchResult:
        """Find longest cached prefix in the given namespace."""
        ...

    @abstractmethod
    def insert(
        self,
        tokens: list[CacheToken],
        pages: list[int],
        namespace: CacheNamespace | None = None,
        from_node: TreeNode | None = None,
        from_token_idx: int = 0,
        from_page_idx: int = 0,
    ) -> InsertResult:
        """Insert a prefix into the cache, optionally starting from a known node.

        If from_node is provided, caller should also pass from_page_idx
        (typically matched_kv_length or skip_positions) to avoid O(N) summation.

        Returns InsertResult with node and inserted_pages count. The caller MUST
        use inserted_pages to correctly set cache_owned_page_count.
        """
        ...

    @abstractmethod
    def lock(self, node: TreeNode | None) -> None:
        """Prevent eviction of node and ancestors."""
        ...

    @abstractmethod
    def unlock(self, node: TreeNode | None) -> None:
        """Release eviction lock."""
        ...

    @abstractmethod
    def evict(self, needed_pages: int) -> int:
        """Evict to free pages. Returns pages freed."""
        ...
```

### Why This Structure

| Aspect | Rationale |
|--------|-----------|
| **Separate package** | Encapsulates complexity; clear public API |
| **Base class** | Enables future cache variants (e.g., chunk-based, hierarchical) |
| **Namespace module** | Isolates LoRA-specific logic; easy to extend |
| **Eviction module** | Strategy pattern for LRU/LFU/priority policies |
| **Kernel in kestrel-kernels** | Consistent with existing Triton kernels location |

### Files Changed (Updated)

| File | Changes |
|------|---------|
| **New: Prefix Cache Package** | |
| `kestrel/prefix_cache/__init__.py` | Package exports |
| `kestrel/prefix_cache/base.py` | `BasePrefixCache` ABC |
| `kestrel/prefix_cache/radix_cache.py` | `RadixPrefixCache`, `TreeNode`, `MatchResult`, `match_prefix()`, `insert()`, `lock()`, `unlock()`, `evict()`, `evictable_page_count()` |
| `kestrel/prefix_cache/namespace.py` | `CacheNamespace` dataclass |
| `kestrel/prefix_cache/eviction.py` | LRU eviction policy (extensible) |
| **Modified: KV Cache** | |
| `kestrel/kv_cache.py` (PageTable) | Add `set_prefix_cache()`, `allocate_pages()`, `map_pages()`, `get_pages()`, `free_pages_to_pool()`, `can_reserve_with_eviction()`. Modify `erase(cached_page_count)` |
| **Modified: Runtime** | |
| `kestrel/moondream/runtime.py` | Add `cache_tokens`, `cache_lock_node`, `cache_owned_page_count`, `reused_page_count` to `SequenceState`. Modify `start_sequence()` for cache lookup/insert. Add `_append_prefill()` |
| `kestrel/moondream/text.py` | Add `_append_prefill_attention()` using single-pass append attention |
| **Modified: Scheduler** | |
| `kestrel/scheduler/scheduler.py` | Modify `_release_sequence()` to call `prefix_cache.unlock()` and `erase(cached_page_count)` |
| `kestrel/scheduler/spatial.py` | Return bin indices from `compute_spatial_values()` |
| **Modified: Engine** | |
| `kestrel/engine.py` | Add `image_hash` to `_PendingRequest`. Compute SHA256 hash in `_submit_request()` |

---

## Multi-Image Support (Future)

The current implementation assumes zero or one image per request, always at position 1 (after BOS). This section documents how the design extends to multiple images at arbitrary positions.

### What's Already Multi-Image Compatible

| Component | Status | Notes |
|-----------|--------|-------|
| **Cache key sequence** | ✅ Ready | Multiple `ImageToken`s in sequence just work |
| **Radix tree matching** | ✅ Ready | Matches any token sequence |
| **Page table / allocation** | ✅ Ready | Position-based, not image-aware |
| **Append attention** | ✅ Ready | Single-pass with right-aligned causal mask |
| **Attention masks** | ⚠️ Needs work | Currently hardcoded for single 730-token prefix |

The cache doesn't care how many images there are - it just sees a sequence of cache keys. The complexity is in attention masking.

### Cache Token Construction (Multi-Image)

```python
@dataclass
class ImageInfo:
    position: int      # Token position where image appears
    hash: bytes        # SHA256 hash of image bytes
    kv_length: int     # KV positions this image produces (729 today)

def build_cache_tokens(
    tokens: list[Token],
    images: list[ImageInfo],
) -> tuple[list[CacheToken], list[tuple[int, int]]]:
    """
    Build cache token sequence and image regions for multi-image.

    Returns:
        cache_tokens: Sequence with ImageTokens at correct positions
        image_regions: [(start, end), ...] for bidirectional attention
    """
    cache_tokens: list[CacheToken] = []
    image_regions: list[tuple[int, int]] = []

    images_by_pos = {img.position: img for img in images}
    kv_offset = 0  # Track KV position offset from inserted images

    for i, tok in enumerate(tokens):
        if i in images_by_pos:
            img = images_by_pos[i]
            cache_tokens.append(ImageToken(
                content_hash=int.from_bytes(img.hash[:16], 'big'),  # 128-bit hash
                kv_length_=img.kv_length,
            ))
            # Record region for bidirectional attention
            region_start = len(cache_tokens) - 1 + kv_offset
            region_end = region_start + img.kv_length
            image_regions.append((region_start, region_end))
            kv_offset += img.kv_length - 1  # -1 because ImageToken is 1 cache token

        cache_tokens.append(tok)

    return cache_tokens, image_regions
```

### Attention Mask (Multi-Image)

The current `cute_prefix_lm_mask_730` is hardcoded for a single 730-token bidirectional region. Multi-image requires a dynamic mask:

```python
def build_multi_region_mask(regions: list[tuple[int, int]]):
    """
    Build attention mask with multiple bidirectional regions.

    Positions within a region attend bidirectionally.
    Positions outside regions use causal attention.
    Cross-region attention is causal (later regions can attend to earlier).
    """
    @cute.jit
    def multi_region_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        aux_tensors,
    ) -> cute.TensorSSA:
        # Check if both positions are in the same region (bidirectional)
        same_region = False
        for start, end in regions:
            in_region_m = (m_idx >= start) & (m_idx < end)
            in_region_n = (n_idx >= start) & (n_idx < end)
            same_region = same_region | (in_region_m & in_region_n)

        # Causal: m can attend to n if n <= m
        causal = m_idx >= n_idx

        return same_region | causal

    return multi_region_mask
```

**Note**: This is a conceptual sketch. The actual implementation would need to handle the regions as kernel parameters rather than Python closures for efficiency.

### Implementation Path

When adding multi-image support:

1. **Engine**: Accept `images: list[ImageInfo]` instead of single `image`
2. **Runtime**: Use `build_cache_tokens()` to construct cache tokens and regions
3. **SequenceState**: Populate `image_regions` from the build step
4. **Attention**: Replace `cute_prefix_lm_mask_730` with dynamic mask based on `image_regions`
5. **Vision encoder**: Process multiple images, interleave embeddings at correct positions

The prefix caching components (radix tree, page table, append attention) require **no changes**.

---

## Test Plan

### A) Radix Tree Correctness

**1. Split correctness**
```python
def test_split_correctness():
    cache = RadixPrefixCache(page_table)
    # Insert [A,B,C,D]
    cache.insert([A, B, C, D], pages=[0, 1, 2, 3])

    # Match [A,B,X] triggers split at 2
    match = cache.match_prefix([A, B, X])

    # Verify structure:
    # - parent has [A,B] with pages [0,1]
    # - child has [C,D] with pages [2,3]
    assert match.matched_token_count == 2
    assert match.matched_pages == [0, 1]
    parent = match.last_node
    assert len(parent.tokens) == 2
    child = list(parent.children.values())[0]
    assert len(child.tokens) == 2
    assert child.physical_pages == (2, 3)
```

**2. Image token kv_length accounting**
```python
def test_image_kv_length():
    cache = RadixPrefixCache(page_table)
    # Insert [BOS, Image(k=729), t1, t2] with 1+729+1+1=732 pages
    tokens = [TextToken(BOS), ImageToken(hash, 729), TextToken(1), TextToken(2)]
    pages = list(range(732))
    cache.insert(tokens, pages)

    # Match should return correct KV lengths
    match = cache.match_prefix(tokens)
    assert match.matched_kv_length == 732
    assert len(match.matched_pages) == 732

    # Partial match respects kv_length boundaries
    partial = cache.match_prefix([TextToken(BOS), ImageToken(hash, 729)])
    assert partial.matched_kv_length == 730  # BOS + image
```

**3. Lock ref invariants**
```python
def test_lock_invariants():
    cache = RadixPrefixCache(page_table)
    result = cache.insert([A, B, C], pages=[0, 1, 2])
    node = result.node

    cache.lock(node)
    assert node.lock_ref == 1
    assert node.parent.lock_ref == 1  # Ancestors also locked

    cache.unlock(node)
    assert node.lock_ref == 0
    assert node.parent.lock_ref == 0

    # Verify no underflow
    with pytest.raises(AssertionError):
        cache.unlock(node)  # Should fail
```

### B) Page Ownership / Freeing

**4. Cache miss full prefill → pages stay cached**
```python
def test_cache_miss_page_ownership():
    page_table = PageTable(n_pages=100, page_size=1, max_batch_size=10)
    cache = RadixPrefixCache(page_table)
    page_table.set_prefix_cache(cache)

    initial_free = page_table.pages_available

    # Simulate cache miss full prefill
    batch_idx = page_table.allocate()
    pages = page_table.allocate_pages(10)  # Prompt pages
    page_table.map_pages(batch_idx, 0, pages)

    # Insert into cache
    insert_result = cache.insert(tokens, pages)
    cache.lock(insert_result.node)

    # cache_owned_page_count = skip_positions + inserted_pages = 0 + 10 = 10
    cache_owned_page_count = insert_result.inserted_pages
    assert cache_owned_page_count == 10

    # Simulate sequence completion - erase with cache_owned_page_count=10
    cache.unlock(insert_result.node)
    page_table.erase(batch_idx, cached_page_count=cache_owned_page_count)

    # Verify: prompt pages still NOT in free pool (owned by cache)
    assert page_table.pages_available == initial_free - 10

    # After eviction, pages return to pool
    cache.evict(10)
    assert page_table.pages_available == initial_free
```

**5. Partial hit → all prompt pages cached**
```python
def test_partial_hit_page_ownership():
    # ... setup with cached prefix ...

    # Partial hit: 5 pages reused, 5 new
    match = cache.match_prefix(tokens[:5])
    skip_positions = match.matched_kv_length  # 5
    new_pages = page_table.allocate_pages(5)
    page_table.map_pages(batch_idx, skip_positions, new_pages)

    # Insert full sequence (only new suffix gets claimed)
    all_pages = match.matched_pages + new_pages
    insert_result = cache.insert(
        tokens, all_pages,
        from_node=match.last_node,
        from_token_idx=5,
        from_page_idx=skip_positions,  # Must pass page index, not just token index
    )

    # insert_result.inserted_pages = 5 (only suffix pages)
    # cache_owned_page_count = skip_positions + inserted_pages = 5 + 5 = 10
    cache_owned_page_count = skip_positions + insert_result.inserted_pages
    assert cache_owned_page_count == 10

    # At teardown: cache_owned_page_count = 10 (entire prompt)
    page_table.erase(batch_idx, cached_page_count=cache_owned_page_count)

    # All 10 prompt pages should be cache-owned, not freed
```

### C) Insert Idempotency and Lock Transfer

**6. Inserting same sequence twice returns existing node with inserted_pages=0**
```python
def test_insert_idempotency():
    cache = RadixPrefixCache(page_table)

    # Insert sequence
    result_a = cache.insert(tokens, pages)
    assert result_a.inserted_pages == len(pages), "First insert claims all pages"

    # Insert same sequence again (e.g., if there was a race)
    result_b = cache.insert(tokens, pages)

    # Should return same node with inserted_pages=0
    assert result_b.node is result_a.node
    assert result_b.inserted_pages == 0, "Second insert claims no new pages"
    assert cache.total_cached_pages == len(pages)  # Not doubled
```

**7. Lock transfer identity case (critical edge case)**
```python
def test_lock_transfer_identity():
    """When insert_result.node is temp_lock_node, we must keep exactly one lock."""
    cache = RadixPrefixCache(page_table)

    # Insert and lock a sequence (simulating first request)
    tokens = [TextToken(1), TextToken(2), TextToken(3)]
    pages = [0, 1, 2]
    result = cache.insert(tokens, pages)
    cache.lock(result.node)
    assert result.node.lock_ref == 1
    assert result.inserted_pages == 3

    # Unlock (simulating first request completion)
    cache.unlock(result.node)
    assert result.node.lock_ref == 0

    # Second request: full prompt match
    match = cache.match_prefix(tokens)
    assert match.last_node is result.node
    assert match.matched_kv_length == 3  # Full match

    # Simulate start_sequence with full match - full_prompt_cached path
    # (The real code skips insert() entirely for full match)
    temp_lock_node = match.last_node
    cache.lock(temp_lock_node)

    # For full prompt match, we don't call insert() - we keep temp_lock_node
    # This test verifies the identity case works if we did call insert()
    insert_result = cache.insert(tokens, pages)
    assert insert_result.node is temp_lock_node  # Same node!
    assert insert_result.inserted_pages == 0     # No new pages claimed

    # Lock transfer: identity case - keep existing lock, don't lock+unlock
    if insert_result.node is temp_lock_node:
        cache_lock_node = temp_lock_node  # Keep existing lock
    else:
        cache.lock(insert_result.node)
        cache.unlock(temp_lock_node)
        cache_lock_node = insert_result.node

    # Verify exactly one lock
    assert cache_lock_node.lock_ref == 1

    # Simulate teardown - one unlock
    cache.unlock(cache_lock_node)
    assert cache_lock_node.lock_ref == 0  # No underflow!
```

**8. Full prompt match forces suffix computation**
```python
def test_full_prompt_match_suffix():
    """Full prompt match still computes at least one suffix token."""
    runtime = create_test_runtime()
    tokens = [TextToken(BOS), TextToken(1), TextToken(2)]
    prompt_len = 3

    # First request: populates cache
    state1, _ = runtime.start_sequence(tokens)
    runtime.release_sequence(state1)

    # Second request: same prompt (full match)
    state2, logits = runtime.start_sequence(tokens)

    # Verify: skip_positions capped at prompt_len - 1
    assert state2.reused_page_count == prompt_len - 1  # 2, not 3

    # Verify: we got valid logits (generation can start)
    assert logits is not None
    assert logits.shape[-1] == vocab_size

    # Verify: no writes to shared cached pages
    # (The last token was computed into private pages at position prompt_len-1)
```

**9. Multiple batches share cached pages without corruption**
```python
def test_multi_batch_shared_pages():
    """Multiple sequences can map same cached pages and decode independently."""
    runtime = create_test_runtime()
    prompt = [TextToken(BOS), ImageToken(hash, 729), TextToken(1)]

    # First request populates cache
    state1, _ = runtime.start_sequence(prompt, image=img)
    # state1 is still active (decoding)

    # Second request matches cached prefix while first is active
    state2, _ = runtime.start_sequence(prompt, image=img)

    # Both map the same physical pages for cached prefix
    pages1 = page_table.get_pages(state1.batch_idx, 0, 730)
    pages2 = page_table.get_pages(state2.batch_idx, 0, 730)
    assert pages1 == pages2  # Same physical pages

    # Generate tokens for both - writes go to DIFFERENT decode pages
    for _ in range(10):
        runtime.decode_step(state1)
        runtime.decode_step(state2)

    # Verify no corruption: outputs should be independent
    # (Each writes to its own decode pages, not shared cached pages)
    assert state1.length == state2.length
    # Decode pages are different
    decode_pages1 = page_table.get_pages(state1.batch_idx, 730, state1.length)
    decode_pages2 = page_table.get_pages(state2.batch_idx, 730, state2.length)
    assert set(decode_pages1).isdisjoint(set(decode_pages2))
```

**10. Full prompt match: page ownership and leak prevention**
```python
def test_full_prompt_match_page_ownership():
    """Full prompt match maps only reused pages and correctly sets ownership."""
    page_table = PageTable(n_pages=100, page_size=1, max_batch_size=10)
    cache = RadixPrefixCache(page_table)
    page_table.set_prefix_cache(cache)

    tokens = [TextToken(BOS), TextToken(1), TextToken(2), TextToken(3)]
    prompt_len = 4

    # First request: full prefill, populates cache with 4 pages
    batch_idx_1 = page_table.allocate()
    pages = page_table.allocate_pages(prompt_len)
    page_table.map_pages(batch_idx_1, 0, pages)
    insert_result = cache.insert(tokens, pages)
    cache.lock(insert_result.node)

    # Complete first request
    cache.unlock(insert_result.node)
    page_table.erase(batch_idx_1, cached_page_count=insert_result.inserted_pages)

    initial_free = page_table.pages_available  # All non-cached pages

    # Second request: full prompt match
    batch_idx_2 = page_table.allocate()
    match = cache.match_prefix(tokens)
    assert match.matched_kv_length == prompt_len  # Full match!

    # Apply full_prompt_cached logic: cap skip_positions, map only reused pages
    skip_positions = min(match.matched_kv_length, prompt_len - 1)  # = 3
    assert skip_positions == prompt_len - 1

    # Map only the cached pages we reuse (NOT all matched pages)
    cached_pages = match.matched_pages[:skip_positions]
    page_table.map_pages(batch_idx_2, 0, cached_pages)

    # Verify: mapped only skip_positions pages, not prompt_len
    mapped_count = len(page_table.page_table_cpu[batch_idx_2])
    assert mapped_count == skip_positions  # 3, not 4

    # Allocate private page for recomputed last token
    private_pages = page_table.allocate_pages(1)
    page_table.map_pages(batch_idx_2, skip_positions, private_pages)

    # Lock the matched node (full prompt already cached, no insert needed)
    cache.lock(match.last_node)
    cache_lock_node = match.last_node
    cache_owned_page_count = skip_positions  # NOT prompt_len! (no insert, so inserted_pages=0)

    # Verify: logical page prompt_len-1 is NOT a cached physical page
    private_page = page_table.get_pages(batch_idx_2, prompt_len - 1, prompt_len)[0]
    assert private_page in private_pages  # It's our private page
    assert private_page not in match.matched_pages  # NOT from cache

    # Teardown: unlock and erase
    cache.unlock(cache_lock_node)
    page_table.erase(batch_idx_2, cached_page_count=cache_owned_page_count)

    # Verify: private page returned to free pool
    assert private_pages[0] in page_table.free_pages

    # Cache pages still owned by cache (not freed)
    assert page_table.pages_available == initial_free  # Back to same as before request 2
```

**11. Tiny hit (text-only): minimal prefix sharing**
```python
def test_tiny_hit_text_only():
    """
    Test the most common "always reuse" case: second request shares only BOS
    (or BOS + one token), so skip_positions is small but non-zero.

    This validates that even minimal cache hits don't leak pages.
    """
    page_table = PageTable(n_pages=100, page_size=1, max_batch_size=10)
    cache = RadixPrefixCache(page_table)
    page_table.set_prefix_cache(cache)

    total_pages = 99  # Excluding reserved page 0
    initial_free = page_table.pages_available

    def check_invariant(context: str):
        free = page_table.pages_available
        cached = cache.total_cached_pages
        assert free + cached == total_pages, f"Leak at {context}: {free} + {cached} != {total_pages}"

    # First request: long prompt [BOS, A, B, C, D, E, F, G, H, I] (10 tokens)
    prompt1 = [TextToken(BOS)] + [TextToken(i) for i in range(1, 10)]
    batch_idx_1 = page_table.allocate()
    pages_1 = page_table.allocate_pages(10)
    page_table.map_pages(batch_idx_1, 0, pages_1)
    result_1 = cache.insert(prompt1, pages_1)
    cache.lock(result_1.node)
    cache_owned_1 = result_1.inserted_pages
    assert cache_owned_1 == 10

    # Complete first request
    cache.unlock(result_1.node)
    page_table.erase(batch_idx_1, cached_page_count=cache_owned_1)
    check_invariant("after first request")

    # Second request: shares only BOS [BOS, X, Y] (3 tokens, only BOS matches)
    prompt2 = [TextToken(BOS), TextToken(100), TextToken(101)]  # Different tokens after BOS
    batch_idx_2 = page_table.allocate()
    match = cache.match_prefix(prompt2)

    # Tiny hit: only BOS matched
    assert match.matched_kv_length == 1, f"Expected 1, got {match.matched_kv_length}"
    skip_positions = match.matched_kv_length  # = 1

    # Map cached BOS page, allocate suffix pages
    page_table.map_pages(batch_idx_2, 0, match.matched_pages)  # [P_bos]
    suffix_pages = page_table.allocate_pages(2)  # For X, Y
    page_table.map_pages(batch_idx_2, skip_positions, suffix_pages)

    # Lock temporarily
    cache.lock(match.last_node)

    # Insert: creates new node for [X, Y] with suffix_pages
    all_pages = match.matched_pages + suffix_pages
    result_2 = cache.insert(
        prompt2, all_pages,
        from_node=match.last_node,
        from_token_idx=1,
        from_page_idx=skip_positions,  # = 1 (same as token_idx for text-only, but explicit)
    )
    assert result_2.inserted_pages == 2, "Only suffix pages should be claimed"

    # Lock transfer (using identity-safe pattern)
    if result_2.node is match.last_node:
        cache_lock_node = match.last_node
    else:
        cache.lock(result_2.node)
        cache.unlock(match.last_node)
        cache_lock_node = result_2.node

    # cache_owned_page_count = reused + inserted
    cache_owned_2 = skip_positions + result_2.inserted_pages
    assert cache_owned_2 == 3, f"Expected 3, got {cache_owned_2}"

    # Complete second request
    cache.unlock(cache_lock_node)
    page_table.erase(batch_idx_2, cached_page_count=cache_owned_2)
    check_invariant("after tiny hit request")

    # Final verification: evict all and return to initial state
    cache.evict(total_pages)
    assert page_table.pages_available == total_pages, "Should return to empty cache state"
```

### D) Numerical Equivalence

**12. Logit consistency across paths**
```python
def test_logit_consistency():
    # Run same prompt through:
    # 1. Full prefill (no cache)
    # 2. Full prefill with cache insertion
    # 3. Append prefill (cache hit)

    logits_baseline = run_full_prefill(prompt, use_cache=False)
    logits_cached = run_full_prefill(prompt, use_cache=True)
    logits_append = run_append_prefill(prompt)  # After cache is populated

    # Should match within FP8 tolerance
    assert torch.allclose(logits_baseline, logits_cached, atol=1e-2)
    assert torch.allclose(logits_cached, logits_append, atol=1e-2)
```

### E) Stress Tests

**13. Memory pressure with locked pages**
```python
def test_memory_pressure():
    # Fill cache completely
    for i in range(100):
        insert_and_lock_sequence(i)

    # All pages locked - allocation should fail gracefully
    with pytest.raises(RuntimeError, match="all cached prefixes are locked"):
        page_table.allocate_pages(10)

    # Unlock some sequences
    for i in range(50):
        unlock_sequence(i)

    # Now allocation should succeed (evicts unlocked)
    pages = page_table.allocate_pages(10)
    assert len(pages) == 10
```

**14. No double-free or page ownership violation**
```python
def test_no_double_free():
    # Run many sequences with varying cache hit/miss patterns
    for _ in range(1000):
        tokens = random_tokens()
        state = runtime.start_sequence(tokens, image)
        # ... generate ...
        scheduler.release_sequence(state)

    # Verify invariants
    assert page_table.pages_available + cache.total_cached_pages == total_pages
    # No page in both free pool and cache
```

**15. Page conservation: no leaks after sequence completion**
```python
def test_page_conservation():
    """
    Verifies that pages are properly accounted for after sequence completion.

    The key invariant:
        free_pages + cache_owned_pages + active_batch_pages == total_pages

    This test catches ownership bugs where:
    - Pages are marked as cache-owned but not actually in the tree
    - Pages are leaked (neither free, nor in cache, nor in active batch)
    - Double-free attempts
    """
    page_table = PageTable(n_pages=100, page_size=1, max_batch_size=10)
    cache = RadixPrefixCache(page_table)
    page_table.set_prefix_cache(cache)

    total_pages = 100 - 1  # Page 0 reserved
    initial_free = page_table.pages_available
    assert initial_free == total_pages, "All pages should be free initially"

    def check_invariant(context: str):
        free = page_table.pages_available
        cached = cache.total_cached_pages
        # No active batches after all sequences released
        assert free + cached == total_pages, f"Invariant violated at {context}: {free} + {cached} != {total_pages}"

    # Test 1: Simple sequence - cache miss, then release
    prompt1 = [TextToken(BOS), TextToken(1), TextToken(2)]
    state1, _ = runtime.start_sequence(prompt1)
    runtime.release_sequence(state1)
    check_invariant("after simple sequence")

    # Test 2: Cache hit with full match - verify private page freed
    state2, _ = runtime.start_sequence(prompt1)  # Full match
    assert state2.reused_page_count == len(prompt1) - 1, "Should reuse all but last position"
    runtime.release_sequence(state2)
    check_invariant("after full match sequence")

    # Test 3: Partial cache hit
    prompt3 = [TextToken(BOS), TextToken(1), TextToken(3), TextToken(4)]  # Shares first 2 tokens
    state3, _ = runtime.start_sequence(prompt3)
    assert state3.reused_page_count == 2, "Should reuse BOS and token 1"
    runtime.release_sequence(state3)
    check_invariant("after partial hit sequence")

    # Test 4: Stress test - many sequences with varying patterns
    for i in range(50):
        prompt = [TextToken(BOS)] + [TextToken(j) for j in range(i % 10 + 1)]
        state, _ = runtime.start_sequence(prompt)
        runtime.release_sequence(state)

    check_invariant("after stress test")

    # Verify: can evict all cached pages and return to initial state
    evicted = cache.evict(total_pages)
    assert page_table.pages_available == total_pages, "Should be able to return to empty state"
```

**16. Namespace pruning prevents memory leaks**
```python
def test_namespace_pruning():
    """
    Verifies that empty namespace roots are pruned from _trees.

    Without pruning, each unique image creates a namespace entry that
    persists forever, causing unbounded memory growth in high-cardinality
    image workloads.
    """
    page_table = PageTable(n_pages=100, page_size=1, max_batch_size=10)
    cache = RadixPrefixCache(page_table)
    page_table.set_prefix_cache(cache)

    # Create 10 unique image namespaces
    namespaces = []
    for i in range(10):
        ns = CacheNamespace(image_hash=i)
        namespaces.append(ns)
        tokens = [TextToken(BOS), TextToken(1), TextToken(2)]
        pages = page_table.allocate_pages(3)
        cache.insert(tokens, pages, namespace=ns)

    assert len(cache._trees) == 10, "Should have 10 namespace roots"

    # Evict all cached pages
    cache.evict(100)

    # All namespace roots should be pruned
    assert len(cache._trees) == 0, "Empty namespaces should be pruned"

    # Can still insert into previously-used namespaces (recreates root)
    tokens = [TextToken(BOS), TextToken(1)]
    pages = page_table.allocate_pages(2)
    cache.insert(tokens, pages, namespace=namespaces[0])
    assert len(cache._trees) == 1, "Should recreate namespace on new insert"
```

**17. Match-only queries don't create namespace roots**
```python
def test_match_only_no_root_creation():
    """
    Verifies that match_prefix() does not create roots for unknown namespaces.

    This prevents unbounded _trees growth from match-only queries in workloads
    that query many unique namespaces without inserting.
    """
    page_table = PageTable(n_pages=100, page_size=1, max_batch_size=10)
    cache = RadixPrefixCache(page_table)
    page_table.set_prefix_cache(cache)

    assert len(cache._trees) == 0, "Should start empty"

    # Query 100 unique namespaces (without inserting)
    tokens = [TextToken(BOS), TextToken(1), TextToken(2)]
    for i in range(100):
        ns = CacheNamespace(image_hash=i)
        result = cache.match_prefix(tokens, namespace=ns)
        # Should return empty match
        assert result.matched_kv_length == 0
        assert result.last_node is None

    # No roots should have been created
    assert len(cache._trees) == 0, "match_prefix should not create roots"

    # Insert into one namespace
    ns = CacheNamespace(image_hash=42)
    pages = page_table.allocate_pages(3)
    cache.insert(tokens, pages, namespace=ns)
    assert len(cache._trees) == 1, "insert should create root"

    # Now match should find it
    result = cache.match_prefix(tokens, namespace=ns)
    assert result.matched_kv_length == 3
```

**18. from_page_idx correctly aligns pages with variable-KV tokens**
```python
def test_from_page_idx_alignment():
    """
    Verifies that from_page_idx correctly aligns page slicing when
    tokens have variable kv_length (e.g., ImageToken with kv_length=729).

    This test catches the bug where token_idx and page_idx drift apart.
    """
    page_table = PageTable(n_pages=1000, page_size=1, max_batch_size=10)
    cache = RadixPrefixCache(page_table)
    page_table.set_prefix_cache(cache)

    # Create a prefix with variable-length token: BOS(1) + Image(729) = 730 pages
    image_kv_length = 729
    prefix_tokens = [
        TextToken(BOS),
        ImageToken(content_hash=12345, kv_length_=image_kv_length),
    ]
    prefix_pages = list(range(730))  # Pages 0-729
    cache.insert(prefix_tokens, prefix_pages)

    # Create full sequence: prefix + 10 text tokens
    suffix_text_tokens = [TextToken(i) for i in range(10)]
    full_tokens = prefix_tokens + suffix_text_tokens
    full_pages = list(range(740))  # Pages 0-739

    # Match should find the prefix (730 KV positions, 2 tokens)
    result = cache.match_prefix(full_tokens)
    assert result.matched_kv_length == 730
    assert result.matched_token_count == 2

    # Now insert the suffix using from_page_idx
    # This is the critical test: from_page_idx must equal matched_kv_length (730),
    # not matched_token_count (2)
    insert_result = cache.insert(
        full_tokens, full_pages,
        from_node=result.last_node,
        from_token_idx=result.matched_token_count,  # 2
        from_page_idx=result.matched_kv_length,      # 730 (NOT 2!)
    )

    # Should have inserted 10 new pages (for the 10 suffix text tokens)
    assert insert_result.inserted_pages == 10

    # Verify the new node has the correct pages (730-739, not 2-11)
    new_node = insert_result.node
    assert len(new_node.physical_pages) == 10
    assert list(new_node.physical_pages) == list(range(730, 740))
```

---

## Future Considerations

### Node Coalescing

After evicting a leaf node, its parent may become a single-child node. Over time, the tree can accumulate unnecessary intermediate nodes:

```
Before eviction: root → [A,B] → [C,D] (leaf, evicted)
                              → [C,E] (leaf, kept)

After eviction:  root → [A,B] → [C,E]  ← parent now has single child
```

Coalescing would merge `[A,B]` with `[C,E]` into `[A,B,C,E]`, reducing node count and traversal depth. This is an optimization, not a correctness issue - the tree works correctly without it.

### Generated Token Caching

Currently, we cache the prompt after prefill. We could extend this to include generated tokens:

1. After generation completes, extend the cache with output tokens
2. Future requests matching prompt + partial response could skip more compute

Trade-off: Increased cache size vs. benefit from matching generated prefixes.

### Hierarchical Caching (GPU → CPU → Disk)

For very large working sets:

1. **Hot** (GPU): Active sequences + recently used prefixes
2. **Warm** (CPU pinned): Evicted from GPU but still in memory
3. **Cold** (Disk): Long-term storage for system prompts

Prefetch from warm/cold to GPU when request arrives.

### Prefix Prediction

If request patterns are predictable (e.g., same images repeat), proactively prefetch prefixes before requests arrive.
