# Pipelined Decoding — Design + Implementation Plan

## The Problem

Today, Kestrel's decode loop has a GPU bubble between steps:

```
CURRENT (blocking):

CPU:  [plan] [launch] [====== wait =======] [commit] [plan] [launch] ...
GPU:         [=== forward/sample ===] [D2H]                 [=== forward/sample ===]
                                           ↑
                                      GPU IDLE
                               (waiting for CPU to plan next step)
```

The GPU finishes a step, then sits idle while the CPU commits results, plans the next step, and launches it. This "between-steps bubble" limits throughput — the GPU is our expensive resource, and we're not keeping it busy.

## The Solution

Pipeline the work so CPU and GPU run concurrently:

```
PROPOSED (pipelined):

CPU:  [plan t] [launch t] [plan t+1] [launch t+1] [commit t] [plan t+2] ...
GPU:           [==== forward t ====] [==== forward t+1 ====] [==== forward t+2 ====]
                          ↑                       ↑
               CPU plans t+1 while          CPU commits t while
               GPU runs t                   GPU runs t+1
```

Key techniques:
- **Ping-pong slots:** Two sets of GPU buffers, alternated each step. While one is in use by the GPU, the other can be planned/committed.
- **Deferred D2H:** Don't wait for GPU→CPU copy immediately. Let it run in the background; block only when we actually need the results.
- **Block-to-commit-previous:** When we need to wait, wait on the *oldest* in-flight step while the GPU works on newer steps.

## TL;DR (for the impatient)

1. Two `DecodeSlot` objects with independent FlashInfer contexts, CUDA graphs, and staging buffers
2. A 2-deep queue of in-flight steps; FIFO completion order
3. Per-sequence `finalized` + `inflight_refs` tracking replaces explicit zombie propagation
4. Scheduling is a pure selector; refcounts mutated only at execution time
5. Engine loop uses simple blocking (`wait()`) rather than polling (`ready()`)
6. Unified forward/sample split model handles both constrained and unconstrained batches

---

## Overview

This document proposes an in-place refactor to make Kestrel's decoding fully asynchronous:

- **Non-blocking GPU dispatch:** Launch forward/sampling without blocking the scheduler thread.
- **Deferred D2H copies:** Initiate device→host copies on a dedicated stream; synchronize only when results are needed.
- **Pipelined planning:** Overlap FlashInfer `plan()` for step t+1 with GPU execution of step t.

The design preserves Kestrel's single-owner-thread constraint: one thread owns all CUDA state and drives `MoondreamRuntime`.

---

## Goals / Non-Goals

### Goals (testable)

- Remove the per-token-step CPU stall caused by `torch.cuda.Event.synchronize()` in the decode loop.
- Allow the scheduler thread to keep doing CPU work (draining queues, promoting crops, delivering results) while GPU execution and D2H copies are in-flight.
- Hide FlashInfer `plan(...)` latency by **planning and launching forward for step t+1 while step t commits**; sampling for t+1 is gated by mask availability for constrained sequences.
- Preserve behavior:
  - Same streaming semantics (ordering, request isolation),
  - Same KV-cache capacity and LoRA slot invariants.
- Keep the public API unchanged (`InferenceEngine`, `EngineStream`, HTTP server behavior).

### Non-goals (explicitly out of scope for this refactor)

- Replacing the scheduler thread with a pure-`asyncio` engine (the server loop should remain unblocked).
- Multi-process "engine core" architecture like vLLM v1 (ZMQ / separate process).
- Speculative decoding / multi-token-per-step.
- Moving structured-skill interpretation fully onto GPU (eliminating CPU `Token` materialization + `SkillState.consume_step`).
- Bit-exact RNG determinism across refactor. Zombie rows may change RNG consumption order vs the current implementation; we accept this.

---

## 1. Current Decoding (what matters for async)

### 1.1 Control flow (threading + queues)

**Async-facing layer**: `kestrel/engine.py`

- Clients call `await InferenceEngine.query/caption/point/detect/segment(...)`.
- `_submit_request(...)` packages a `_PendingRequest` and puts it into `self._queue` (an `asyncio.Queue`).
- `_worker_loop(...)` runs as an asyncio task and forwards requests into a **thread-safe** `queue.Queue` (`self._scheduler_queue`) used by the scheduler thread.

**Scheduler thread**: `InferenceEngine._scheduler_loop()` (`kestrel/engine.py`)

Responsibilities inside the `while True` loop:

1. Drain `_PendingRequest`s from `self._scheduler_queue`.
2. If a request has an image, compute overlap crops (CPU) via `ThreadPoolExecutor` and admit the request once crops are ready.
3. Admit work into `GenerationScheduler` and drive it via `scheduler.advance()`.
4. Convert `SchedulerResult` → `EngineResult`, and deliver:
   - final results via `future.set_result` on the asyncio loop,
   - streaming updates via `call_soon_threadsafe(queue.put_nowait, StreamUpdate)`.

### 1.2 Decode state model (scheduler)

Core types (`kestrel/scheduler/types.py`):

- `GenerationRequest`: immutable request metadata + optional `SkillState`.
- `ScheduledSequence`: groups:
  - runtime `SequenceState` (KV page allocation, length, last_hidden),
  - the `GenerationRequest`,
  - the `SkillState`.

The scheduler owns two FIFO queues:

- `waiting`: requests not yet admitted (no KV reserved).
- `running`: admitted sequences; each has a staged `pending_token` plus a per-sequence packed id/value slot used as the next `decode_batch` input.

### 1.3 The blocking point

Even though GPU work itself is asynchronous, the decode loop currently forces a CPU wait **on every step**:

- `transfer.wait()` calls `torch.cuda.Event.synchronize()` inside `TransferHandle.wait()`.
- This blocks the scheduler thread, preventing:
  - admission of new requests (even if crops ready),
  - draining scheduler input queue promptly,
  - delivering results/streaming updates promptly,
  - overlapping CPU bookkeeping with GPU execution and copies.

There are other sync points (not addressed in this first refactor):

- FlashInfer batch metadata construction + `FlashInferDecodeContext.plan(...)` happens every step and can dominate the “between steps” bubble.
- Streaming text delta for query/caption decodes the entire prefix each step (O(n²) CPU).

---

## 2. Structured decoding constraint (why we must be careful)

Kestrel’s “structured decoding” today is implemented as **dynamic token masking** in `GenerationScheduler._sample_batch(...)`:

- Each `SkillState` can return `allowed_token_ids(runtime)` (e.g. point/detect).
- The scheduler prunes logits to `-inf` for disallowed ids before sampling.

Important dependency:

- The allowed-token set for step **t+1** depends on skill state after consuming token **t**.
- Therefore, we must ensure:
  - token **t** is staged and `consume_step()` has run before sampling **t+1** for that sequence.

This refactor preserves that constraint, but with a key insight: **forward doesn't need the mask, only sampling does**.

- For **unconstrained** sequences: both forward and sampling can run ahead of commit.
- For **constrained** sequences: **forward** may run ahead, but **sampling** is gated by commit (must wait for `consume_step()` to compute the mask).

The split model exploits this by decoupling forward from sampling:
1. Launch forward t+1 (doesn't need mask)
2. Commit step t (runs `consume_step()`)
3. Compute mask for t+1
4. Launch sampling t+1 (now has mask)

This achieves full 2-deep pipelining even for constrained batches. See §4.7 for timing details.

Note: With the packed id/value decode path (`token_ids`, `coord_values`, `size_values`) available to the runtime, the *next* forward no longer needs CPU materialization of typed `Token` objects. That makes forward-only pipelining feasible for all batches.

---

## 3. vLLM v1 patterns we reuse

We follow two well-tested patterns from vLLM v1:

- Deferred output copies (launch D2H on a copy stream; synchronize only when needed).
- A bounded in-flight batch queue (block only when the queue is full).

See §8 for the specific reference locations in `ext/vllm`.

---

## 4. Proposed Design (in-place, minimal)

### Hard Invariants

The following invariants are **non-negotiable** throughout the implementation. They are referenced throughout this section but collected here for clarity:

| ID | Invariant | Rationale |
|----|-----------|-----------|
| **I1** | Single shared `decode_compute_stream` across both slots | Preserves sequential token dependencies (`_pending_*` write/read, KV ordering) |
| **I2** | FIFO completion (oldest step committed first) | Zombie skipping, mask computation, D2H ordering all depend on commit order |
| **I3** | Per-step `step_done_event` anchors D2H (never `wait_stream`) | `wait_stream` captures dependency on all enqueued work; event anchors to exactly one step |
| **I4** | `0 <= inflight_refs <= 2`; only scheduler mutates it | Prevents double-increment/decrement; `PipelineState` does queue/slot bookkeeping only |
| **I5** | Prefill only when pipeline is drained | Freezes decode batch membership; ensures clean state for synchronous prefill |
| **I6** | Slot reusable only after `commit_step()` (not just D2H) | Pinned host buffers and per-slot state must not be overwritten before commit consumes them |

When implementing or reviewing changes, verify these invariants are preserved. Debug assertions for I4 are recommended (see §4.5).

### 4.1 Reuse existing `RenderBuffer` (vLLM-style deferred output)

Kestrel already has `RenderBuffer` (`kestrel/scheduler/transfer.py`) which provides:

- pinned CPU destination tensors for sampled ids + decoded coord/size values,
- a copy stream, and a `torch.cuda.Event` recorded after the copy,
- a `TransferHandle` that currently blocks via `wait()`.

To support async completion, `TransferHandle` needs:

- `wait()` — blocks until D2H transfer completes (already exists). **Currently returns the CPU tensors directly**, which is fine. No need to change this.
- `view() -> tuple[torch.Tensor, ...]` — optional accessor if we want to separate "wait for completion" from "get data". Since the current `wait()` already returns tensors, this is only needed if we want `commit_step()` to call `wait()` for synchronization and `view()` later for data access. **For simplicity, keep current behavior: `wait()` returns tensors.**

Optional (add only if profiling shows benefit):
- `ready() -> bool` — non-blocking check via `event.query()`. The current design uses blocking `wait()` exclusively; D2H is fast enough that polling provides no measurable benefit.

**Transfer dependency contract (critical for pipelining):**

`render.start_transfer(...)` must wait on decode completion for the *current step only*, not on any later work that may have been enqueued on the compute stream. The naive approach (`copy_stream.wait_stream(compute_stream)`) is fragile because `wait_stream` captures a dependency on *everything enqueued so far*—if next-step `plan()` H2D copies are already enqueued, the D2H will wait on those too, destroying overlap.

**Required implementation:** Use a per-step event:

```py
def start_transfer(self, ready_event: torch.cuda.Event, ...) -> TransferHandle:
    """Start D2H transfer, waiting only on ready_event (not all compute work)."""
    with torch.cuda.stream(self._copy_stream):
        self._copy_stream.wait_event(ready_event)  # wait only on this step's sampling
        # ... D2H copies ...
        self._done_event.record()
    return TransferHandle(...)
```

Call site records `ready_event` immediately after sampling completes:

```py
# In finalize_sampling:
with torch.cuda.stream(slot.compute_stream):
    # ... forward, sampling ...
    step_done_event = torch.cuda.Event()
    step_done_event.record()  # captures exactly "this step's GPU writes are done"

transfer = slot.render.start_transfer(ready_event=step_done_event, ...)
```

This makes D2H independent of any later compute-stream enqueues (next-step planning, next-step forward, etc.).

### 4.2 DecodeSlot encapsulation

We bundle all per-slot resources into a single `DecodeSlot` object to reduce bookkeeping errors and make slot ownership trivially visible:

```py
@dataclass
class CudaGraphBundle:
    workspace: _GraphWorkspace               # input/output buffers for graph replay
    cuda_graphs: dict[int, CUDAGraph]        # batch_size -> captured graph

@dataclass
class DecodeMetaBuffers:
    """Per-slot pinned host metadata buffers for H2D copies."""
    batch_idx: CpuGpuBuffer      # int64 [max_batch] — sequence batch indices
    input_pos: CpuGpuBuffer      # int32 [max_batch] — token positions
    lora_slot_ids: CpuGpuBuffer  # int32 [max_batch] — LoRA slot assignments

@dataclass
class DecodeSlot:
    meta: DecodeMetaBuffers                  # per-slot pinned metadata (H2D sources)
    render: RenderBuffer                     # pinned host destination for D2H
    sampled_ids: torch.Tensor                # GPU staging for sampled token IDs
    coord_staging: torch.Tensor              # GPU staging for coord values
    size_staging: torch.Tensor               # GPU staging for size values
    logits: torch.Tensor                     # GPU buffer for forward output logits
    hidden_last: torch.Tensor                # GPU buffer for last hidden states (spatial decode)
    flashinfer_ctx: FlashInferDecodeContext  # FlashInfer decode context + workspace
    graph: Optional[CudaGraphBundle]         # CUDA graph workspace + captures (if enabled)
    compute_stream: torch.cuda.Stream        # reference to shared decode compute stream
```

**Forward output lifetime (critical for delayed sampling):**

In constrained mode, forward runs ahead and sampling is delayed until mask computation completes. Forward outputs (`logits`, `hidden_last`) must remain valid until `finalize_sampling()` consumes them.

- **Invariant:** Forward outputs live in per-slot buffers (`slot.logits`, `slot.hidden_last`). `LaunchHandle` carries `(kind, sequences, payload)`; for decode the payload contains `slot_id`, and sampling looks up outputs via `decode_slots[handle.slot_id]`.
- **Invariant:** No new forward may run on a slot until any pending sampling for that slot has completed. This is enforced by the slot allocation logic: a slot is "in use" while `launch_handle` or any `PendingCommit` in `batch_queue` references it.

**Decode compute stream topology (hard invariant):**

All decode forwards and sampling operations across **both ping-pong slots** are enqueued on a **single runtime-owned `decode_compute_stream`**. Both slots reference the same stream object via `slot.compute_stream`.

This preserves sequential token dependencies:
- `_pending_*` write/read ordering (sample t writes → forward t+1 reads)
- KV-cache write/read ordering across steps
- The implicit "decode steps are sequential" assumption

**⚠️ Per-slot distinct compute streams are unsupported (invariant I1).** Separate streams would require explicit cross-stream event chaining for every inter-step dependency.

```py
# CORRECT: single shared stream
self._decode_compute_stream = torch.cuda.Stream(device=self.device)
slot_a.compute_stream = self._decode_compute_stream
slot_b.compute_stream = self._decode_compute_stream  # same object

# WRONG: per-slot streams (breaks _pending_* and KV ordering)
slot_a.compute_stream = torch.cuda.Stream(...)  # stream A
slot_b.compute_stream = torch.cuda.Stream(...)  # stream B — BROKEN
```

**CUDA graph keying:** Each slot owns its own `dict[int, CUDAGraph]` keyed by batch size. This replaces the current runtime-level `_cuda_graphs: dict[int, CUDAGraph]`. Since graphs capture tensor pointers, and each slot has different staging buffer addresses, graphs cannot be shared across slots. Graph selection during execution: `slot.graph.cuda_graphs[batch_size]`.

Each in-flight step "owns" the slot it used; alternating slots prevents buffer clobbering. The 2-deep queue guarantees a slot is free before reuse.

**Ownership model:**

- **Runtime owns both DecodeSlots.** `MoondreamRuntime` creates and holds the two slots, including their FlashInfer contexts, staging buffers, and CUDA graph state. This is consistent with the runtime owning all GPU resources.
- **Engine loop tracks which slot is free.** The scheduler loop maintains `batch_queue: deque[PendingCommit]` and uses `free_slot_id(batch_queue, launch_handle)` to determine which slot is available via bitmask.
- **Scheduler is backend-agnostic.** `launch_forward_async(plan, slot_id)` receives the slot ID as a parameter; the scheduler looks up `decode_slots[slot_id]` internally.

```py
# In MoondreamRuntime.__init__:
self._decode_slots = [DecodeSlot(...), DecodeSlot(...)]  # slot 0, slot 1

@property
def decode_slots(self) -> list[DecodeSlot]:
    return self._decode_slots

# In engine._scheduler_loop:
decode_slots = runtime.decode_slots
batch_queue: deque[PendingCommit] = deque()
launch_handle: LaunchHandle | None = None
# ...
slot_id = free_slot_id(batch_queue, launch_handle)
launch_handle = scheduler.launch_forward_async(plan, slot_id)
```

**CUDA graph duplication:**

When CUDA graphs are enabled, each `DecodeSlot` owns its own graph workspace (`_GraphWorkspace`) and captured graphs (`dict[int, CUDAGraph]`). Graph capture for both slots happens **eagerly during `MoondreamRuntime` initialization** (not lazily on first use), ensuring no capture latency on the first decode step for either slot.

This doubles graph memory compared to the current single-context design (see §7.2). The tradeoff is correctness: CUDA graph replay requires stable tensor pointers, and each slot's staging buffers / FlashInfer workspace have different addresses. Sharing graphs between slots would require re-capturing on every slot switch, defeating the purpose of graph caching.

**Consolidated capture path:** Both initialization and `rebuild_cuda_graphs()` use the same per-slot capture logic. The current runtime has separate `_ensure_cuda_graphs_ready()` and `rebuild_cuda_graphs()` methods; with ping-pong slots, we consolidate into a single `_capture_graphs_for_slot(slot)` that both paths call:

```py
def _capture_graphs_for_slot(self, slot: DecodeSlot) -> None:
    """Capture CUDA graphs for all batch sizes using this slot's resources."""
    # Clear any existing graph state for this slot
    slot.flashinfer_ctx._graph_states.clear()
    slot.graph.cuda_graphs.clear()
    # Capture graphs using slot's workspace and FlashInfer context
    for batch_size in self._graph_batch_sizes:
        self._capture_single_graph(slot, batch_size)

def _ensure_cuda_graphs_ready(self) -> None:
    if not self._use_cuda_graphs:
        return
    for slot in self._decode_slots:
        if not slot.graph.cuda_graphs:  # not yet captured
            self._capture_graphs_for_slot(slot)

def rebuild_cuda_graphs(self) -> None:
    """Re-capture graphs for both slots (e.g., after weight updates)."""
    if not self._use_cuda_graphs:
        return
    with self.graph_capture_lock:
        for slot in self._decode_slots:
            self._capture_graphs_for_slot(slot)
```

**CUDA graph capture boundaries (forward/sample split):**

Under pipelined decoding with constrained batches, sampling may be delayed relative to forward. This affects what can be graph-captured:

- **CUDA graphs capture forward only.** The captured graph includes: FlashInfer attention, MLP, and any other decode kernels up to (but not including) sampling.
- **Sampling runs outside the graph** (or via a separate sampling graph that takes a mask tensor as an input buffer).

This separation is required because:
1. The mask for constrained sequences is not known at forward dispatch time.
2. Graph replay requires all inputs to be in stable buffers at capture time; a per-step mask violates this.

If sampling is also graph-captured, use a separate graph with `slot.logits` and a pre-allocated `mask_buffer` as inputs. The mask is copied into `mask_buffer` before replay.

**Slot selection via bitmask (not identity comparison):**

Each decode `PendingCommit` and decode `LaunchHandle` stores a `slot_id: int` in its payload rather than a reference to the `DecodeSlot` object. This enables simple bitmask-based allocation:

```
Ping-pong with 2-slot queue:

batch_queue (newest → oldest):
┌─────────────────────────────────────────────────────────────────────────┐
│  Step t+1 [slot 1]  ←──  Step t [slot 0]                                │
│     (newest)              (oldest, completed next)                      │
└─────────────────────────────────────────────────────────────────────────┘
        ↑                         ↓
    appendleft()                pop()

After completing step t (slot 0 freed):
┌─────────────────────────────────────────────────────────────────────────┐
│  Step t+1 [slot 1]                                                      │
└─────────────────────────────────────────────────────────────────────────┘

free_slot_id() returns 0 → schedule step t+2 on slot 0
```

Slot allocation is handled by `PipelineState.free_slot_id()` (see §5 Phase 2). This is simpler than identity comparison (`step.slot is slot`) and handles all transients correctly, including the split forward/sample case where both `batch_queue` and `launch_handle` may reference slots.

### 4.3 Staging buffer invariants

**D2H sources:** The D2H copy reads from per-slot GPU staging tensors (`slot.sampled_ids`, `slot.coord_staging`, `slot.size_staging`). If step t+1's sampling overwrites these while step t's D2H is in flight, the copy reads corrupted data. Per-slot ownership prevents this.

**Invariant:** D2H copies **must only** read from the per-slot staging buffers (never from `_pending_*`).

**`_pending_*` buffers remain runtime-level (not per-slot):** These buffers (`_pending_token_ids`, `_pending_coord_values`, `_pending_size_values`) store next-step *inputs* keyed by `batch_idx`. They are:
- Written after sampling (step t writes inputs for step t+1)
- Read at the start of the next step's forward pass

These are GPU-resident inputs, not outputs being D2H copied. With 2-deep pipelining:
1. Step t samples → writes to `_pending_*[seq.batch_idx]`
2. Step t+1 forward → reads from `_pending_*[seq.batch_idx]`
3. Step t+1 samples → overwrites `_pending_*[seq.batch_idx]`

Since all decode steps (across both ping-pong slots) are serialized on the shared `decode_compute_stream`, the write-before-read ordering is preserved. No per-slot duplication needed—the `batch_idx` key ensures each sequence's pending input is distinct.

**H2D metadata buffers (per-slot, critical for correctness):**

Metadata H2D copies (`batch_idx`, `input_pos`, `lora_slot_ids`) use `CpuGpuBuffer.copy_to_gpu(..., non_blocking=True)` from **pinned host memory**. With pipelining, the CPU may overwrite the pinned source buffer for step t+1 before step t's H2D copy executes on the GPU.

- **Stream ordering is NOT sufficient** for pinned host buffer reuse. Stream ordering only orders GPU-side operations; it does not prevent the CPU from overwriting the pinned source memory while the DMA is pending.
- **Per-slot pinned metadata buffers are required.** Each `DecodeSlot` owns its own `DecodeMetaBuffers` with independent pinned CPU tensors.
- **Slot lifetime guarantees safety.** A slot is not reused until `commit_step()` (invariant I6), so the prior H2D copy has completed before the CPU writes new metadata.

Inside `launch_forward_async(plan, slot_id)`:
```python
slot = decode_slots[slot_id]
slot.meta.batch_idx.np[:B] = [seq.state.batch_idx for seq in plan.sequences]
slot.meta.input_pos.np[:B] = [seq.state.length for seq in plan.sequences]
slot.meta.lora_slot_ids.np[:B] = [seq.state.lora_slot for seq in plan.sequences]

with torch.cuda.stream(slot.compute_stream):
    batch_idx = slot.meta.batch_idx.copy_to_gpu(B)
    input_pos = slot.meta.input_pos.copy_to_gpu(B)
    lora_slot_ids = slot.meta.lora_slot_ids.copy_to_gpu(B)
    # ... forward kernels use these tensors ...
```

### 4.4 Stream ordering

The async copy flow relies on correct CUDA stream semantics:

```
Stream ordering for one decode step:

Time ──────────────────────────────────────────────────────────────────────────►

Compute      ┌──────────────┐ ┌────────┐ ┌────────┐ ┌───────────────┐
Stream:      │   forward    │ │ sample │ │ spatial│ │ record        │
             └──────────────┘ └────────┘ │ +stage │ │ step_done_evt │
                                         └────────┘ └───────────────┘
                                                            │
                                                            │ copy_stream.wait_event(step_done_event)
                                                            ▼
Copy         ·····························────────┌─────────┐ ┌───────────────┐
Stream:                                           │ D2H copy│ │ record event  │
                                                  └─────────┘ └───────────────┘
                                                                       │
                                                                       │ event.synchronize()
                                                                       ▼
CPU:         ·····················································────│wait() blocks│ [commit]
```

1. **Compute stream** runs forward, sampling, spatial decode, and staging writes to per-slot buffers.
2. **Step-done event** is recorded on compute stream after **all GPU writes complete** (sampling + spatial decode + staging). This captures the dependency for exactly this step, before any later work is enqueued.
3. **Copy stream** waits on `step_done_event` (NOT `wait_stream`): `copy_stream.wait_event(step_done_event)`.
4. **D2H copies** are enqueued on the copy stream, reading from staging buffers into pinned host buffers.
5. **Completion event** is recorded on the copy stream after D2H finishes.
6. **`wait()`** synchronizes on the completion event; it blocks CPU until D2H is done.

**⚠️ Why event-based, not `wait_stream`:** `wait_stream(other_stream)` captures a dependency on *everything enqueued on `other_stream` at call time*. If next-step `plan()` H2D copies are already enqueued before `wait_stream` is called, the D2H will wait on those too—destroying pipelining. The per-step `step_done_event` anchors the dependency to exactly "this step's GPU writes," independent of later enqueues.

**Copy-stream topology:** We use a **single shared copy stream** across both slots (simpler ordering guarantees; D2H is typically short). Step t's copies are enqueued before step t+1's copies, and we complete steps strictly FIFO (oldest first), so commit ordering is correct even though both slots share the stream. Per-slot copy streams would also work but add complexity without significant benefit for short D2H transfers.

**⚠️ FIFO completion is mandatory (invariant I2).** See Hard Invariants table.

**Implementation: shared streams:**

Create runtime-level streams and pass them to both slot constructors:

```py
# In MoondreamRuntime.__init__:
self._decode_compute_stream = torch.cuda.Stream(device=self.device)
self._copy_stream = torch.cuda.Stream(device=self.device)

self._decode_slots = [
    DecodeSlot(
        render=RenderBuffer(..., copy_stream=self._copy_stream),
        compute_stream=self._decode_compute_stream,  # SAME stream for both slots
        ...
    )
    for _ in range(2)
]
```

**⚠️ Both slots MUST reference the same `_decode_compute_stream` object (invariant I1).**

Currently `RenderBuffer` creates its own stream in `__init__`. Change it to accept an optional `copy_stream` parameter, defaulting to creating one if not provided (backwards compatibility).

### 4.5a Engine pause/resume handling

The engine's `pause()` and `resume()` methods must interact correctly with the pipelined decode state. `pause()` is used for hot-loading weights, rebuilding CUDA graphs, and configuration changes—operations that may invalidate in-flight state.

**Hazard:** If we pause mid-pipeline, the following state may become invalid after configuration changes:
- `launch_handle` (forward dispatched, not yet sampled)
- Steps in `batch_queue` (reference old slot/graph state)

**Rule: Drain pipeline before acknowledging pause.**

When `paused_flag` is set, the scheduler loop must drain in the correct order (same as prefill drain):

```python
if paused_flag.is_set():
    # 1. Commit any sampled steps first (updates grammar state via consume_step)
    #    This must happen BEFORE finalizing launch_handle, because the mask
    #    for t+1 depends on grammar state after committing t.
    while step := pipeline.pop_oldest():
        scheduler.commit_step(step)
        pipeline.on_step_completed()

    # 2. Then finalize any pending forward using the updated grammar state
    if pipeline.launch_handle is not None:
        mask = scheduler.compute_mask(pipeline.launch_handle.sequences)
        step = scheduler.finalize_sampling(pipeline.launch_handle, mask)
        pipeline.on_pending_commit(step)
        scheduler.commit_step(pipeline.pop_oldest())
        pipeline.on_step_completed()

    # 3. Now safe to sync and acknowledge
    with runtime.graph_capture_lock:
        torch.cuda.synchronize(runtime.device)
    paused_event.set()
    run_gate.wait(timeout=0.1)
    continue
```

**Why drain is necessary:**
- CUDA graph rebuild changes workspace tensor addresses; old plans reference stale pointers
- Weight updates may invalidate FlashInfer attention metadata
- Draining ensures resume starts with clean state

**Cost:** Pause takes slightly longer (must complete in-flight steps). This is acceptable because pause/resume is rare (hot-reload, config changes) and correctness is critical.

**Resume behavior:** After `resume()`, the loop continues normally. The first decode iteration will plan from scratch since the pipeline is fully drained.

**CUDA graph interaction:**

If decode is graph-captured, ensure the "copy stream wait + copy + record event" sequence is either:
- Outside the captured region (preferred), or
- Captured in a slot-consistent way where buffer pointers and streams are stable across replays.

Each ping-pong slot owns its own staging buffers and (if graph-captured) its own graph workspace, so slot A's graph replay uses slot A's buffers exclusively.

### 4.5 Scheduler refactor: split "schedule/execute/complete"

We refactor `GenerationScheduler` from a single blocking `advance()` into three operations. The key simplification is that **zombie behavior emerges from per-sequence state** rather than explicit propagation between steps.

**Two readiness concepts:**

- **GPU-ready**: A sequence has valid next-step *GPU inputs* in `_pending_*` (or will have them by stream order). This is what's needed to enqueue `decode_batch`.
- **CPU-ready-to-commit**: The D2H transfer for that sequence's most recent step is complete, allowing the scheduler to materialize tokens, call `consume_step`, and emit streaming.

With GPU run-ahead, a sequence can be GPU-ready for step t+1 before step t is CPU-ready-to-commit.

**Bounded commit-lag invariant:**

- For **all sequences**: eligible for forward scheduling iff `inflight_refs < 2`. A sequence may appear in at most 2 in-flight steps simultaneously, so GPU progress may lead CPU commit by up to 2 steps.
- The difference between constrained and unconstrained is **when sampling happens**, not eligibility:
  - **Unconstrained:** Sampling is finalized immediately after forward dispatch.
  - **Constrained:** Sampling is deferred until after committing the previous step (to compute the mask).

This constraint is enforced in `schedule_decode_step()` (see §4.5).

**Length / position bookkeeping:**

With GPU run-ahead, length and position counters can diverge:

- `input_pos` and KV length (used for attention indexing) must track **GPU progress**—they may be up to 2 steps ahead of CPU commit (bounded by the in-flight queue depth).
- Streaming, stop conditions, and skill consumption must use **CPU-committed** state only.

In Kestrel today:
- `SequenceState.length` reflects GPU progress (used for attention/KV indexing).
- `SkillState.token_count` reflects CPU-committed tokens (used for streaming, stop conditions, `consume_step`).

This split prevents off-by-one bugs where attention uses stale positions or streaming emits uncommitted tokens.

**Per-sequence finalization tracking (replaces explicit zombie propagation):**

Instead of propagating zombie state between steps, we track two fields per sequence:

```py
seq.finalized: bool = False   # True once EOS or length cap reached
seq.inflight_refs: int = 0    # how many in-flight steps include this sequence
```

**Where these fields live:** On `ScheduledSequence` (or a new `SequenceTracker` wrapper). The key lifecycle:

1. **Prefill creates** `ScheduledSequence`, adds to `running`, `inflight_refs = 0`.
2. **Execution** increments `inflight_refs` for each sequence in the step.
3. **Completion** decrements `inflight_refs`. If `finalized and inflight_refs == 0` → release.
4. **Finalization** (EOS/length cap): set `finalized = True`, remove from `running`.

**Zombie lifecycle example:**

```
Sequence A hits EOS at step t. Step t+1 was already planned with A included.

Step t          Step t+1        After t+1
┌─────────┐     ┌─────────┐
│ A (eos) │     │ A       │     A released
│ B       │     │ B       │     (inflight_refs=0)
│ C       │     │ C       │
└─────────┘     └─────────┘

Timeline:
─────────────────────────────────────────────────────────────────────────────
execute(t)     execute(t+1)    complete(t)           complete(t+1)
A.refs=1       A.refs=2        A.refs=1              A.refs=0 → RELEASE
                               A.finalized=True       (skip commit, was zombie)
                               A removed from
                               running                B,C continue
─────────────────────────────────────────────────────────────────────────────
```

After finalization, the sequence is **not in `running`** but still exists because `PendingCommit.sequences` holds a reference. No separate "zombies" container needed—zombies are just finalized sequences with `inflight_refs > 0`, reachable only via the in-flight queue.

**Capacity tracking:** Derive from page table state, not `len(running)`:

```py
@property
def num_allocated(self) -> int:
    """Number of KV rows currently allocated (includes zombies)."""
    return self.runtime.page_table.num_allocated
```

**Implementation note:** `PageTable.num_allocated` does not exist in the current codebase and must be added. It can be computed as `max_batch_size - len(free_batch_idx) - 1` (subtracting 1 for reserved `batch_idx=0`). Alternatively, maintain a counter incremented by `allocate()` and decremented by `erase()`.

The page table is the source of truth for KV capacity. It already tracks which `batch_idx` slots are in use via `allocate()` / `erase()`. Deriving capacity from it:
- Cannot drift (single source of truth)
- Automatically accounts for zombies (they hold `batch_idx` until released)
- Matches the actual resource constraint (KV memory, not scheduling state)

**⚠️ `len(running)` is wrong for capacity checks.** Zombies are not in `running` but still hold KV capacity. This is a common bug:

```py
# WRONG: misses zombies, allows over-allocation
if len(self.running) < max_batch_size:
    admit_new_request()

# CORRECT: includes zombies
if self.num_allocated < max_batch_size:
    admit_new_request()
```

Consider adding an assertion at scheduler construction: `assert not hasattr(self, '__len__')` or similar to prevent accidental `len(scheduler)` usage. The fairness check must use `num_allocated < max_batch_size`.

This eliminates `CompletedStepInfo`, `apply_zombies()`, `zombie_seq_ids`, and `row_of_seq`. Zombie behavior becomes emergent:

- On execution (`launch_forward_async`): `seq.inflight_refs += 1` for each sequence in the step.
- On completion (`commit_step`): `seq.inflight_refs -= 1` for each sequence.
- If `seq.finalized` is already true at commit time → skip committing (this is the "zombie" behavior).
- If `seq.finalized` becomes true at this step (EOS/length cap) → set it, finalize API semantics.
- If `seq.finalized and seq.inflight_refs == 0` → release resources immediately.

**Why this works:** FIFO completion order guarantees step t is committed before step t+1. When committing step t+1, we've already processed step t and set `finalized=True` for any EOS'd sequences. We simply check the flag and skip.

**⚠️ FIFO completion is invariant I2.** Out-of-order completion breaks zombie skipping.

**In-flight step record:**

Track each scheduled step in a `PendingCommit` record:
  - `sequences: list[ScheduledSequence]`
  - `payload.slot_id: int` (0 or 1, which ping-pong slot this step used)
  - `transfer: TransferHandle`

**Row order invariant:** `PendingCommit.sequences[i]` corresponds to row `i` in the render/staging buffers for that step. This mapping is implicit (no separate `row_of_seq` needed) because:
- Forward and sampling write outputs in batch order
- `commit_step` iterates `sequences` in order to match outputs to sequences
- Maintaining this invariant is the responsibility of `launch_forward_async` and `finalize_sampling`

Note: `zombie_seq_ids` and `row_of_seq` are no longer needed.

**Capacity invariant:**

At any time, the number of **allocated KV rows / active `batch_idx`** must be ≤ `max_batch_size`. Sequences with `finalized=True` but `inflight_refs > 0` still count against capacity until released. Capacity is counted by allocated sequence slots, not by in-flight-step multiplicity.

**StepPlan type:**

```py
@dataclass
class StepPlan:
    sequences: list[ScheduledSequence]  # sequences to include in this step
```

Note: `StepPlan` contains only the sequence selection. Slot selection is derived from queue occupancy by the engine and passed separately to `launch_forward_async`. This avoids redundancy and prevents any mismatch where a plan is executed on the wrong slot.

**Scheduler methods:**

Decode and prefill have separate APIs. This makes Phase 2's "prefill only when decode queue is empty" rule explicit in the type system and prevents accidentally scheduling prefill into the decode queue.

- `schedule_decode_step() -> StepPlan | None`
  - **Pure selector—does not mutate sequence state.**
  - Chooses a decode batch over currently-ready sequences.
  - Returns `None` if no decode work is schedulable.
  - Filters candidates using `can_dispatch(seq)` (defined in §4.7), which excludes finalized sequences, sequences at their `inflight_refs` limit (2), and sequences that would exceed their token budget if dispatched again.
  - **Eligibility:** A sequence is eligible iff `seq.inflight_refs < 2`.
  - Must enforce `max_batch_size` as unique allocated KV rows.

- `try_run_prefill_sync(request) -> ScheduledSequence`
  - Runs prefill synchronously. Called only when decode queue is empty.
  - Handles KV allocation, model prefill, sampling, D2H, commit.
  - Returns the newly-admitted sequence (now in `running`).

- `launch_forward_async(plan: StepPlan, slot_id: int) -> LaunchHandle`
  - **Increments `seq.inflight_refs` for each sequence in the plan.** This is the commit point—refcounts are only mutated when the step is actually launched.
  - Runs FlashInfer `plan()` and model forward on GPU using `decode_slots[slot_id].compute_stream`.
  - **Calls `seq.state.advance()` immediately after dispatching forward**, before returning. This updates KV length for next-step attention indexing.
  - Returns immediately with a `LaunchHandle(kind="decode")` whose payload carries `slot_id`.
  - **Error handling:** See §4.8 for the exception model.

- `finalize_sampling(handle: LaunchHandle, mask: torch.Tensor | None) -> PendingCommit`
  - Launches sampling kernel on `decode_slots[handle.slot_id].compute_stream` (stream ordering ensures forward completes first).
  - If `mask` is provided, applies token constraints; if `None`, samples without constraints.
  - Records `step_done_event` (after all GPU writes: sampling + spatial decode + staging) and kicks off non-blocking D2H copy via `slot.render.start_transfer(ready_event=step_done_event, ...)`.
  - Returns `PendingCommit(kind="decode")` whose payload carries `slot_id`, plus `sequences` and `transfer`.

- `commit_step(inflight) -> None`
  - **Owns the `wait()` call.** Blocks until D2H transfer completes by calling `inflight.transfer.wait()` internally. Callers never call `wait()` directly—this ensures wait ownership is centralized and prevents double-wait bugs.
  - For each `seq` in `inflight.sequences`:
    - Decrement `seq.inflight_refs`.
    - If `seq.finalized`: skip commit (zombie), check if `inflight_refs == 0` → release.
    - Else: commit token, check for EOS/length cap → if finalized, check if `inflight_refs == 0` → release or keep for next step.
  - **Length-cap releases:** Sequences hitting predictable caps were excluded from the next plan (via filter predicate), so `inflight_refs` drops to 0 immediately → release now.
  - **EOS releases:** If `inflight_refs > 0`, the sequence is in the next planned step (zombie). It will be released when that step completes. If `inflight_refs == 0`, release now.
  - If a sequence becomes finalized at this step (EOS/length cap): set `finalized=True`, remove from `running`, finalize the request (resolve future, emit final stream event).

**Helper methods (used by engine loop):**

- `has_pending_prefill() -> bool`
  - Returns `True` if there are requests in the waiting queue ready for prefill.

- `peek_pending_prefill() -> GenerationRequest | None`
  - Returns the next prefill request without removing it from the queue.
  - Used to check `can_reserve()` before committing to prefill.

- `plan_needs_mask(sequences: list[ScheduledSequence]) -> bool`
  - Returns `True` if any non-finalized sequence uses token constraints.
  - Implementation: `any(not seq.finalized and seq.skill_state.allowed_token_ids(self.runtime) is not None for seq in sequences)`

- `compute_mask(sequences: list[ScheduledSequence]) -> torch.Tensor`
  - Builds the token constraint mask for sampling.
  - Treats finalized sequences as unconstrained (all tokens allowed).

**`inflight_refs` invariant:**

- **Invariant:** `0 <= seq.inflight_refs <= 2` for all sequences.
- **Ownership:** Only scheduler methods (`launch_forward_async`, `commit_step`) mutate `inflight_refs`. `PipelineState` does queue/slot bookkeeping only—it never touches refcounts.
- **Debug assertion (recommended):** After each state transition, verify `seq.inflight_refs` matches the actual membership count across `batch_queue` and `launch_handle`. In debug builds:
  - `assert 0 <= seq.inflight_refs <= 2`
  - `assert seq.inflight_refs == membership_count(seq, pipeline.batch_queue, pipeline.launch_handle)`

  This catches refcount drift bugs early without duplicating state mutation responsibilities.

### 4.6 Engine scheduler loop: unified pipeline

We update `InferenceEngine._scheduler_loop` to maintain:

- `batch_queue: deque[PendingCommit]` — fully-sampled steps awaiting completion (at most 2)
- `launch_handle: LaunchHandle | None` — current forward pass in-flight but not yet sampled
- `decode_slots: list[DecodeSlot]` — the two ping-pong slots (slot 0 and slot 1)

**Queue ordering invariant:** `batch_queue[-1]` is the **oldest** in-flight step; `batch_queue[0]` is the newest. We `appendleft(new)` and `pop()` to complete oldest first (FIFO).

**Total in-flight invariant:** `len(batch_queue) + (1 if launch_handle else 0) <= 2`

**Unified loop (handles both constrained and unconstrained):**

The loop delegates all state transitions to `PipelineState`, which is the canonical state machine. Queue depth emerges naturally from slot availability and the gating rule.

- For **unconstrained** batches: `finalize_sampling` is called immediately after `launch_forward_async`, so both slots can be in use.
- For **constrained** batches: `finalize_sampling` is delayed until after committing the previous step, which naturally limits queue depth.

```py
pipeline = PipelineState(num_slots=2)

while True:
    drain incoming requests / promote crops

    # ──────────────────────────────────────────────────────────────────────
    # PREFILL HANDLING
    # Drain pipeline before prefilling to ensure clean state.
    # Gate includes can_reserve() to prevent spinning when KV pages are
    # exhausted but row slots are available.
    # ──────────────────────────────────────────────────────────────────────
    next_prefill = scheduler.peek_pending_prefill()
    prefill_wanted = (
        next_prefill is not None
        and scheduler.num_allocated < max_batch_size
        and runtime.can_reserve(next_prefill.target_length)
    )

    if prefill_wanted:
        # 1) Commit any sampled steps first
        while step := pipeline.pop_oldest():
            scheduler.commit_step(step)
            pipeline.on_step_completed()

        # 2) If there's a pending forward, finalize and commit it
        if pipeline.launch_handle is not None:
            mask = scheduler.compute_mask(pipeline.launch_handle.sequences)
            step = scheduler.finalize_sampling(pipeline.launch_handle, mask)
            pipeline.on_pending_commit(step)
            scheduler.commit_step(pipeline.pop_oldest())
            pipeline.on_step_completed()

        # 3) Now prefill is safe
        scheduler.try_run_prefill_sync(next_prefill)
        continue

    # ──────────────────────────────────────────────────────────────────────
    # DECODE: Launch forward if slot available and no forward in-flight
    # ──────────────────────────────────────────────────────────────────────
    if pipeline.launch_handle is None:
        slot_id = pipeline.free_slot_id()

        if slot_id is None:
            # Both slots busy → must complete oldest sampled step to free a slot
            step = pipeline.pop_oldest()
            if step is not None:
                scheduler.commit_step(step)
                pipeline.on_step_completed()
            continue

        plan = scheduler.schedule_decode_step()
        if plan is not None:
            handle = scheduler.launch_forward_async(plan, slot_id)
            pipeline.on_launch(handle)

    # ──────────────────────────────────────────────────────────────────────
    # DECODE: Finalize sampling for the in-flight forward
    # ──────────────────────────────────────────────────────────────────────
    if pipeline.launch_handle is not None:
        needs_mask = scheduler.plan_needs_mask(pipeline.launch_handle.sequences)

        if needs_mask:
            # Constrained: commit previous step first to update grammar state
            step = pipeline.pop_oldest()
            if step is not None:
                scheduler.commit_step(step)
                pipeline.on_step_completed()

            mask = scheduler.compute_mask(pipeline.launch_handle.sequences)
            step = scheduler.finalize_sampling(pipeline.launch_handle, mask)
        else:
            # Unconstrained: finalize immediately
            step = scheduler.finalize_sampling(pipeline.launch_handle, mask=None)

        pipeline.on_pending_commit(step)
        continue

    # ──────────────────────────────────────────────────────────────────────
    # DRAIN: No forward to process, complete oldest step
    # ──────────────────────────────────────────────────────────────────────
    step = pipeline.pop_oldest()
    if step is not None:
        scheduler.commit_step(step)
        pipeline.on_step_completed()
        continue

    # Truly idle: sleep until new work arrives.
    wake_event.wait()
    wake_event.clear()
```

**`plan_needs_mask(sequences)` helper:**

Returns `True` if any non-finalized sequence in the list uses token constraints:

```py
def plan_needs_mask(self, sequences: list[ScheduledSequence]) -> bool:
    return any(
        not seq.finalized and seq.skill_state.allowed_token_ids(self.runtime) is not None
        for seq in sequences
    )
```

**`compute_mask(sequences)` zombie handling:**

`compute_mask` may be called when some sequences in the list are already `finalized=True` (zombies). This happens when:
1. Forward t+1 is launched
2. Step t commits and marks some sequences finalized
3. Mask is computed for t+1's sampling

`compute_mask` treats finalized sequences as unconstrained ("all tokens allowed") to avoid querying skill state after termination. This keeps mask shapes stable and prevents use-after-finalization bugs.

**Why queue depth emerges naturally:**

- **Unconstrained batch:** Forward t+1 launches, immediately finalized → `batch_queue` grows to 2. Both slots can be used.
- **Constrained batch:** Forward t+1 launches, but finalization waits for commit t. This means `batch_queue` drains to 0 before step t+1's `PendingCommit` is added. Effective depth is 1.
- **Mixed batch:** Treated as constrained (delayed finalization).

No explicit `has_constrained_active()` gate or `max_decode_inflight` knob needed.

**Why this loop is correct:**

1. **Latency hiding:** Forward t+1 runs on GPU while CPU waits on and commits step t. The overlap is preserved.
2. **FIFO completion:** We always complete `batch_queue[-1]` (oldest). This guarantees step t commits before step t+1.
3. **Liveness:** After completing a step, we `continue` to retry scheduling—completions may free sequences.
4. **Prefill fairness:** `prefill_wanted` triggers a full drain before prefilling.
5. **Correct masking:** For constrained batches, `compute_mask` is called after `commit_step`, ensuring grammar state is updated.

**Wake sources:** Every transition that can make work schedulable must signal `wake_event`:
- New request arrival (already signals via `_scheduler_event.set()` in `engine.py`)
- Crop-future completion (already signals via a done-callback)

**Note on blocking duration:** This design always blocks via `wait()` rather than polling `ready()`. In the steady state, by the time we wait on step t's transfer, step t+1's forward is running, and step t should already be complete. During transients (queue ramp-up, prefill drain), blocks may be longer. If profiling shows this is a problem, a non-blocking `ready()` check can be added.

### 4.7 Batch stability for pipelined planning (FlashInfer)

If step-to-step latency is dominated by FlashInfer metadata build + `FlashInferDecodeContext.plan(...)`, the biggest win is to overlap **planning for step *t+1*** with **GPU forward/sampling for step *t***.

Mechanism:

- Keep two FlashInfer decode contexts (`active` and `next`) and ping-pong them:
  - run step *t* on `active`,
  - while step *t* runs on GPU, build metadata and call `next.plan(...)` for step *t+1* on CPU,
  - swap contexts at the step boundary.

**No dedicated planning stream needed:** Analysis of FlashInfer's `plan()` implementation shows it performs only:
1. CUDA runtime API calls for device property queries (synchronous, no kernels)
2. Pure CPU computation to build index arrays
3. Small async H2D copies (`cudaMemcpyAsync`) of metadata (~KB)

Since `plan()` launches no compute kernels, a dedicated planning stream provides no benefit.

**H2D ordering requirement:** Although `plan()` launches no compute, its async H2D copies must complete before the decode kernels that consume the metadata.

We execute `slot.flashinfer_ctx.plan(...)` while `slot.compute_stream` is the current stream:

```py
with torch.cuda.stream(slot.compute_stream):
    slot.flashinfer_ctx.plan(...)  # H2D metadata copies on compute_stream
    # ... launch forward kernels on same stream ...
```

Any H2D metadata copies performed by `plan()` are therefore ordered before the decode kernels enqueued on the same stream. FlashInfer's `plan()` uses `cudaMemcpyAsync` on the current stream, so this ordering is implicit.

**⚠️ Implementation constraint:** Always call `plan()` and launch kernels within the same `with torch.cuda.stream(slot.compute_stream):` block. If `plan()` is accidentally called on default stream while kernels launch on `slot.compute_stream`, metadata may not be ready when kernels read it.

**Clarification: planning overlap is CPU-side only.** Because `plan()`'s H2D copies are enqueued on the same compute stream as decode kernels, those copies will not execute until prior decode kernels complete. This is expected and okay — the overlap we achieve is:

- **CPU:** FlashInfer metadata build + `plan()` CPU work
- **GPU:** forward/sampling for the previous step

The H2D copies are tiny (~KB) and stream-ordered; they do not reintroduce a meaningful bubble. Do not attempt to "optimize" by putting H2D on a separate stream — that would break the ordering guarantee.

**When planning overlaps with GPU work:**

The overlap happens naturally through the engine loop structure, not through explicit threading:

```
Timeline (simplified):

CPU:  [plan t] [launch t] [plan t+1] [launch t+1] [plan t+2] ...
GPU:           [────forward t────]   [────forward t+1────]  ...
                     ↑                      ↑
              CPU does plan t+1       CPU does plan t+2
              while GPU runs t        while GPU runs t+1
```

Concretely, within `launch_forward_async(plan, slot_id)` + `finalize_sampling(handle, mask)`:
1. Build FlashInfer metadata for this step
2. Call `slot.flashinfer_ctx.plan(...)` — CPU work, no GPU compute
3. Launch forward pass — async, returns `LaunchHandle`
4. (Later, possibly after commit) Call `finalize_sampling`:
5. Launch sampling — async
6. Launch D2H transfer — async
7. Return `PendingCommit`

After step 6 returns, the engine loop immediately calls `schedule_decode_step()` for the next step. While the GPU is still executing step t's forward/sampling, the CPU is free to build metadata and call `plan()` for step t+1 on the **other slot's** FlashInfer context. No blocking occurs because:
- Step t's GPU work is in-flight (async)
- Step t's D2H hasn't been waited on yet
- The two slots have independent FlashInfer contexts

The ping-pong benefit: we don't need to wait for step t to finish before calling `plan()` for t+1, because each slot has its own context. With a single shared context, we'd have to wait for t's `run()` to complete before calling `plan()` again.

**CPU bookkeeping ordering:** Planning step t+1 requires `input_pos` values that are +1 from step t's start. This works because `seq.state.advance()` is called **inside `launch_forward_async`**, before returning—not at completion time. The sequence:

1. `launch_forward_async` launches forward (async)
2. `launch_forward_async` calls `seq.state.advance()` for each sequence (CPU, inline)
3. `finalize_sampling` launches sampling + D2H (async)
4. Control returns to engine loop
5. Engine loop calls `schedule_decode_step()` → reads updated `seq.state.length`
6. Engine loop calls `launch_forward_async` for step t+1 → uses correct positions

No additional synchronization needed: CPU bookkeeping is synchronous and happens before the function returns. GPU work is in-flight but positions are already updated on CPU.

Because Kestrel’s next-step inputs are already represented as packed GPU tensors (`token_ids`, `coord_values`, `size_values`), step *t+1* forward does not require CPU token materialization; only stateful masking forces a sync point before launching the next step.

CUDA graphs constraint:

- If decode is CUDA-graph-captured, graph replay pins tensor/workspace pointers. Each `DecodeSlot` owns all per-slot resources (see §4.2), ensuring pointer stability.
- We alternate slots each step. Graph capture happens per-slot; both slots need graphs for each batch size.

Constraint:

- FlashInfer plans are for a specific packed batch (metadata like `kv_indptr`/`kv_indices` is tightly packed). We cannot “use part of” a plan if membership changes.

To keep pipelined planning valid without replanning, we enforce exactly three batch-stability rules:

1) **Predictable drops are excluded before planning step *t+1*.**
   - If a sequence will hit `max_length` / `max_new_tokens` at the end of step *t*, we know it before launching *t*.
   - Since the sequence is not in step *t+1*, its `inflight_refs` drops to 0 when step *t* completes → release immediately.

   **Budgeted tokens (critical for correctness under run-ahead):**

   Under 2-deep pipelining, CPU-committed tokens and GPU-dispatched tokens diverge by up to 2. Scheduling must use **budgeted** counts to prevent overscheduling beyond caps:

   - **Committed tokens:** `seq.skill_state.token_count` — tokens fully committed (D2H complete, `consume_step` called). Used for streaming, stop decisions.
   - **Budgeted tokens:** `committed + seq.inflight_refs` — accounts for dispatched-but-not-yet-committed steps.

   The scheduling eligibility predicate:

   ```py
   def can_dispatch(seq: ScheduledSequence) -> bool:
       if seq.finalized:
           return False
       if seq.inflight_refs >= 2:
           return False

       # Absolute max length (includes prompt); GPU length is authoritative for KV indexing.
       if seq.state.length >= seq.state.max_length:
           return False

       # Max new tokens: prevent overscheduling by accounting for in-flight steps.
       if seq.request.max_new_tokens is not None:
           committed = seq.skill_state.token_count
           if committed + seq.inflight_refs >= seq.request.max_new_tokens:
               return False

       return True
   ```

   This ensures sequences are excluded from planning *before* they would exceed their caps, even with 2 steps in flight.

2) **EOS sequences remain in the already-planned step (zombie behavior).**
   - EOS is discovered after sampling step *t*. The sequence stays in the already-planned *t+1* batch, then is dropped from *t+2*.
   - **Tracking via refcounts:** When step *t* completes, we set `seq.finalized = True` and decrement `inflight_refs`. Since the sequence is still in step *t+1*, `inflight_refs > 0` → don't release yet.
   - **Zombie behavior is emergent:** When step *t+1* completes, `commit_step` sees `seq.finalized = True` and skips committing (discards outputs). Then `inflight_refs` drops to 0 → release.
   - Result finalization (resolve future, emit final stream) happens at step *t*. Only resource release is deferred.
   - Batch capacity: sequences with `finalized=True, inflight_refs > 0` count against `max_batch_size`; new admissions resume at *t+2*.
   - **EOS is the only termination condition that is not predictable at plan time.** Length-based caps are predictable and handled via rule (1).

3) **Prefill admission rule: prefill only when pipeline is drained.**
   - New sequences are admitted via prefill, which runs synchronously.
   - Prefill only runs when `batch_queue` is empty and `launch_handle is None`.
   - This naturally freezes decode batch membership: once a forward is launched, no new sequences can join until the pipeline drains for the next prefill.

**Constrained-active policy (resolves masking vs pipelined planning):**

Constrained sequences require grammar state to be updated (`consume_step()`) before computing the mask for the next token. This creates a dependency:

```
Step t sample → commit → consume_step() → allowed_token_ids() → Step t+1 sample
```

The key insight: **forward doesn't need the mask, only sampling does**. We can decouple them:

```
Constrained async pipeline:

Time ──────────────────────────────────────────────────────────────────────────────────────►

GPU:    ┌───────────┐ ┌──────┐ ┌────┐   ┌───────────┐           ┌──────┐ ┌────┐
        │ forward t │ │samp t│ │D2H │   │forward t+1│           │samp  │ │D2H │
        └───────────┘ └──────┘ └────┘   └───────────┘           │ t+1  │ └────┘
                                 │             ▲                └──────┘
                                 │             │                    ▲
                                 │             │ launched           │ mask arrived,
                                 ▼             │ before mask!       │ sampling can run
CPU:    ························│wait│ [commit t] [mask t+1] ·······
                                       consume()   send H2D

Step-by-step:
1. GPU runs forward t, sample t
2. D2H t starts (async on copy stream)
3. CPU launches forward t+1 ← no mask yet, but forward doesn't need it!
4. D2H t completes, CPU commits step t, calls consume_step()
5. CPU computes mask for t+1, sends to GPU (fast H2D)
6. Forward t+1 finishes (took longer than steps 4-5)
7. Sample t+1 runs using the mask
```

This achieves full 2-deep pipelining for constrained batches, but requires splitting the GPU pipeline:
- **Current (fused):** `forward → sample → D2H` launched as one unit
- **New (split):** `forward` → [CPU: commit, mask, H2D] → `sample → D2H`

The unified loop handles both constrained and unconstrained batches — constrained sequences have delayed sampling finalization, unconstrained finalize immediately. No separate "phases" or policy knobs needed.

**Mask H2D stream wiring:** The mask is copied to GPU on `decode_compute_stream` immediately before launching the sampling kernel:

```python
with torch.cuda.stream(decode_compute_stream):
    mask_gpu.copy_(mask_cpu, non_blocking=True)  # H2D on compute stream
    sample_tokens(logits, mask_gpu, ...)          # same stream → ordered after copy
```

Stream ordering guarantees the copy completes before sampling reads the mask. No separate event or cross-stream synchronization is needed.

**Fallback for short sequences:** If forward t+1 completes before mask t+1 arrives (very short sequences or slow commit), the GPU simply idles briefly until the CPU enqueues the mask copy and sampling kernel. In the worst case, this degrades to synchronous behavior, but correctness is preserved.

**Mixed batches:** If a batch contains both constrained and unconstrained sequences, use the delayed finalization path (wait for mask). The mask for unconstrained sequences is "all tokens allowed" (no-op mask or skip masking entirely for those rows).

### 4.8 Exception and fault model

CUDA errors can surface asynchronously—often at the **next synchronization point** rather than at dispatch. For example, a kernel launch failure or illegal memory access during step t may only raise an exception when we call `transfer.wait()` for step t, which happens after step t+1 is already in-flight.

**Consequences of async failures:**

If an exception surfaces at `transfer.wait()` (inside `commit_step`), we may have:
- `inflight_refs` already incremented for step t+1
- Step t+1 already in `batch_queue`
- Slot t+1 is using already "in-flight" with GPU kernels reading its buffers

Trying to "partially recover" from this state risks:
- Wrong refcounts (sequences never released, or released too early)
- Leaked KV rows (allocated but never freed)
- Hung requests (futures never resolved)
- Data corruption if slots are reused while GPU is still accessing them

**Rule: Treat CUDA errors as engine-fatal.**

On any CUDA exception at a sync boundary (`transfer.wait()`, or during `launch_forward_async` / `finalize_sampling`):

```python
def _handle_cuda_failure(self, exc: Exception) -> NoReturn:
    """Handle CUDA failure by cleaning up and raising EngineUnhealthy."""
    # 1. Synchronize streams to flush/surface all pending errors
    torch.cuda.synchronize()  # or per-stream sync

    # 2. Resolve ALL outstanding requests to prevent hangs
    #    This includes:
    #    - Sequences in batch_queue (in-flight sampled steps)
    #    - Sequences in launch_handle (forward dispatched, not yet sampled)
    #    - Admitted sequences (scheduler.running)
    #    - Waiting requests (accepted but not yet prefilled)
    #
    #    The exact iteration pattern depends on the container types; the key requirement
    #    is: every request that the engine has accepted must either complete normally
    #    or receive an error. No request may be left hanging.

    self._resolve_all_inflight_requests(exc)   # batch_queue + launch_handle
    self._resolve_all_admitted_sequences(exc)  # scheduler.running
    self._resolve_all_waiting_requests(exc)    # scheduler.waiting

    # 3. Clear all state
    self.batch_queue.clear()
    self.launch_handle = None
    self.scheduler.running.clear()
    self.scheduler.waiting.clear()

    # 4. Release all KV allocations
    # NOTE: release_all() must be implemented. It should:
    #   - Iterate all allocated batch_idx and call page_table.erase()
    #   - Call runtime.release_adapter_slot() for each to free LoRA slots
    #   - Reset free_batch_idx and free_pages to initial state
    self.runtime.release_all_sequences()

    # 5. Require explicit restart
    raise EngineUnhealthy("CUDA failure requires restart") from exc
```

**Implementation note:** `release_all_sequences()` does not exist and must be added to `MoondreamRuntime`. It should iterate `active_sequences` and call `release_sequence()` for each, which handles both page table cleanup and adapter slot release.

This is simpler and more robust than attempting surgical recovery from undefined CUDA behavior.

**Slot safety under exceptions:**

Once any slot-owned GPU-visible buffer has been mutated for a step, that slot is **unsafe to reuse** until we can prove the step is no longer in-flight. The hazard:

```
launch_forward_async:
    slot.flashinfer_ctx.plan(...)     # H2D copies enqueued
    slot.graph.replay(...)            # kernels launched
    <exception here>
    # We return without creating LaunchHandle
    # Slot appears "free" (not in batch_queue or launch_handle)
    # But GPU is still executing kernels that read from slot buffers!
    # Next iteration reuses slot → data race / memory clobber
```

**Resolution:** With the engine-fatal policy above, this is moot—any exception after GPU work starts leads to full cleanup and restart. If a more granular recovery is ever desired, the rule would be: *on any exception after dispatch/plan writes, synchronize the slot's streams before allowing slot reuse.*

**Pre-dispatch vs post-dispatch exception handling:**

Distinguish between two exception classes:

1. **Pre-dispatch exceptions** (before any GPU-visible work): Safe to rollback and continue.
   - Examples: Python logic errors, validation failures, capacity checks
   - Rollback: decrement refcounts, mark sequences failed, release if `inflight_refs == 0`
   - Engine can continue with next iteration

2. **Post-dispatch exceptions** (after GPU work has started): Must treat as CUDA-fatal.
   - Examples: CUDA OOM during kernel launch, FlashInfer errors after `plan()` H2D
   - Any exception after `slot.flashinfer_ctx.plan()` or kernel dispatch
   - Must call `_handle_cuda_failure()` and abort

```python
def launch_forward_async(self, plan: StepPlan, slot_id: int) -> LaunchHandle:
    slot = self._decode_slots[slot_id]
    gpu_work_started = False
    try:
        for seq in plan.sequences:
            seq.inflight_refs += 1

        # Pre-dispatch: pure Python, safe to rollback
        self._validate_plan(plan)

        # Post-dispatch boundary: after this point, treat exceptions as CUDA-fatal
        gpu_work_started = True
        slot.flashinfer_ctx.plan(...)  # H2D copies enqueued
        # ... forward launch ...
        return LaunchHandle(kind="decode", sequences=..., payload=DecodeLaunch(slot_id=slot_id))

    except Exception as e:
        if gpu_work_started:
            # GPU work may be in-flight — cannot safely continue
            self._handle_cuda_failure(e)  # does not return
        else:
            # Pre-dispatch: safe to rollback and continue
            for seq in plan.sequences:
                seq.inflight_refs -= 1
                seq.finalized = True
                if seq.inflight_refs == 0:
                    self._release(seq)
            raise
```

This separation ensures slot safety: we only attempt recovery when we can prove no GPU work is in-flight.

---

## 5. Implementation Plan (phased)

### Phase 1 — Conservative pipelining (queue depth ≤ 1)

Goal: Introduce the full Phase 2 infrastructure (PipelineState, LaunchHandle, unified loop) with a conservative constraint: **at most 1 sampled step in queue**. This validates the pipelining mechanism while limiting risk.

**Why this approach:**

Instead of "total in-flight = 1" (which prevents overlap), we constrain **queue depth** only:
- `launch_handle` may coexist with a single queued sampled step
- Always use constrained path ordering (commit previous before finalizing next)
- This delivers real forward overlap while keeping the "at most 1 sampled step awaiting commit" safety profile

Phase 2 relaxes Phase 1 by (a) allowing immediate finalize for unconstrained batches and (b) allowing up to 2 sampled steps in the queue.

**Implementation:**

- Add two `DecodeSlot` objects (§4.2) in `MoondreamRuntime`
- Implement `PipelineState` with all methods (§5 Phase 2)
- Use the unified loop from §4.6, but **always commit previous step before finalizing sampling**

```py
# Phase 1: queue depth <= 1, always commit-before-finalize
pipeline = PipelineState(num_slots=2)

while True:
    drain incoming requests / promote crops

    if prefill_wanted:
        # Drain before prefill
        while step := pipeline.pop_oldest():
            scheduler.commit_step(step)
            pipeline.on_step_completed()
        if pipeline.launch_handle is not None:
            mask = scheduler.compute_mask(pipeline.launch_handle.sequences)
            step = scheduler.finalize_sampling(pipeline.launch_handle, mask)
            pipeline.on_pending_commit(step)
            scheduler.commit_step(pipeline.pop_oldest())
            pipeline.on_step_completed()
        scheduler.try_run_prefill_sync(next_prefill)
        continue

    # Launch forward if none in-flight
    if pipeline.launch_handle is None:
        plan = scheduler.schedule_decode_step()
        if plan is not None:
            slot_id = pipeline.free_slot_id()
            handle = scheduler.launch_forward_async(plan, slot_id)
            pipeline.on_launch(handle)

    # Finalize sampling (always commit previous first — Phase 1 conservative path)
    if pipeline.launch_handle is not None:
        # Commit previous step first (if any)
        step = pipeline.pop_oldest()
        if step is not None:
            scheduler.commit_step(step)
            pipeline.on_step_completed()

        mask = scheduler.compute_mask(pipeline.launch_handle.sequences)
        step = scheduler.finalize_sampling(pipeline.launch_handle, mask)
        pipeline.on_pending_commit(step)
        continue

    # Drain
    step = pipeline.pop_oldest()
    if step is not None:
        scheduler.commit_step(step)
        pipeline.on_step_completed()
        continue

    wake_event.wait()
    wake_event.clear()
```

**Phase 1 delivers real overlap:**

```
Time ──────────────────────────────────────────────────────────────────────────►

GPU:    ┌───────────┐ ┌──────┐ ┌────┐   ┌───────────┐           ┌──────┐
        │ forward t │ │samp t│ │D2H │   │forward t+1│           │samp  │
        └───────────┘ └──────┘ └────┘   └───────────┘           │ t+1  │
                                 │             ▲                └──────┘
                                 │             │
                                 ▼             │ forward t+1 runs
CPU:    ························│wait│ [commit t] [mask t+1] ·····
                                       consume()   while CPU commits!
```

Key: forward t+1 is launched **before** we commit step t. While CPU waits and commits, the GPU runs forward t+1.

**Phase 1 provides:**
- Two slots with independent FlashInfer contexts and CUDA graphs
- `PipelineState` tested and working
- Full forward/commit overlap (the main latency win)
- `seq.finalized` and `seq.inflight_refs` tracking
- Conservative safety: at most 1 sampled step awaiting commit

**Phase 1 limitations (removed in Phase 2):**
- Always commit previous before finalizing (even for unconstrained batches)
- Queue depth capped at 1 — cannot have 2 sampled steps in flight

Acceptance:
- Measurable reduction in per-step latency from forward/commit overlap.
- No behavior regressions.
- If Phase 1 shows no improvement, the bottleneck is elsewhere (investigate before Phase 2).

### Phase 2 — Full async decode pipeline

Goal: 2-deep queue for all decode batches (constrained and unconstrained) using the unified forward/sample split model.

**Implementation:**

- Bundle per-slot resources into `DecodeSlot` objects (§4.2); slot selection via `free_slot_id()` bitmask.
- Implement the scheduler API split (§4.5):
  - `schedule_decode_step()` — pure selector, returns `StepPlan | None`
  - `launch_forward_async(plan, slot_id)` — increments `seq.inflight_refs`, launches forward, returns `LaunchHandle`
  - `finalize_sampling(handle, mask)` — launches sampling + D2H, returns `PendingCommit`
  - `commit_step(inflight)` — blocks on transfer, decrements refcounts, handles zombies
  - `try_run_prefill_sync(request)` — runs prefill synchronously when decode queue is empty
- Add `seq.finalized` and `seq.inflight_refs` tracking to `ScheduledSequence`.
- Update `InferenceEngine._scheduler_loop` to use the unified loop (§4.6).

**Implemented: `kestrel/scheduler/pipeline.py`**

The `PipelineState` class is now implemented in `kestrel/scheduler/pipeline.py`. The engine loop delegates to this state machine for all pipeline state transitions, making it the single source of truth.

Key features:
- **Two-phase completion:** `pop_oldest()` moves step to `committing_step` (slot stays in use), `on_step_completed()` frees the slot after scheduler finishes. This enforces invariant I6.
- **Defensive assertions:** `on_launch()` verifies slot is free, `on_pending_commit()` verifies slot_id matches.
- **FIFO ordering:** Queue uses `appendleft`/`pop` for correct completion order.

See `kestrel/scheduler/pipeline.py` for the full implementation and comprehensive docstrings.

**Tests: `tests/scheduler/test_pipeline.py`**

43 unit tests verify queue/slot bookkeeping, two-phase completion, slot assertions, and ping-pong scenarios. No CUDA required — tests use mock sequences and transfers.

**Prefill handling:**

The 2-slot ping-pong resources are reserved for **decode steps only**.

**Rule: Prefill only when decode queue is empty.** Before running prefill, drain all in-flight decode steps. Prefill then runs synchronously.

**Fairness policy:** If a prefill is ready **and** `num_allocated < max_batch_size`, drain the decode queue (≤2 steps) before prefilling. Use `num_allocated`, not `len(running)`, to account for zombies.

**Timing and overlap (constrained batches):**

```
Time ──────────────────────────────────────────────────────────────────────────────────────►

GPU:    ┌───────────┐ ┌──────┐ ┌────┐   ┌───────────┐           ┌──────┐ ┌────┐
        │ forward t │ │samp t│ │D2H │   │forward t+1│           │samp  │ │D2H │
        └───────────┘ └──────┘ └────┘   └───────────┘           │ t+1  │ └────┘
                                 │             ▲                └──────┘
                                 │             │                    ▲
                                 │             │ launched           │ mask arrived,
                                 ▼             │ before mask!       │ sampling runs
CPU:    ························│wait│ [commit t] [mask t+1] ·······
                                       consume()   send H2D
```

Forward t+1 launches before we wait on step t's D2H. While CPU waits and commits step t, forward t+1 runs on GPU. After commit, we compute mask and send to GPU, then finalize sampling.

**Timing requirement:** Forward must take longer than (D2H wait + commit + mask compute + mask H2D). For typical batch sizes:
- Forward: ~10-50ms
- D2H: ~10-100μs
- Commit + mask: ~100μs - 1ms
- Mask H2D: ~10-50μs

This should be satisfied for most practical workloads. If not, the GPU simply idles briefly between forward and sampling (no CPU involvement), gracefully degrading without blocking the scheduler thread.

**Invariant tests:**
- **Device staging buffer correctness**: Force step t+1 to enqueue before step t transfer completes. Assert step t host outputs are correct.
- **Refcount-based zombie behavior**: Verify EOS'd sequences with `inflight_refs > 0` are released only after their last in-flight step completes.
- **Pure scheduling API**: `schedule_decode_step()` doesn't mutate refcounts.
- **Prefill fairness**: Prefill begins within ≤2 decode completions.

Acceptance:
- 2-deep queue works for both constrained and unconstrained workloads.
- Queue depth naturally emerges from the gating rule (see §4.6).
- No deadlocks, correct streaming order.
- Graceful degradation if timing requirement not satisfied.

### Phase 3 — Follow-ups (streaming + optimization)

- Reduce streaming detokenization cost (incremental decode / chunked streaming).
- Profile Phase 2 to measure actual timing margins (forward duration vs commit + mask latency).
- If `consume_step()` / `allowed_token_ids()` are bottlenecks, consider:
  - FSM-based constraint compilation (outlines/lm-format-enforcer style)
  - Caching computed masks for repeated grammar states
  - Parallelizing mask computation with other CPU work

---

## 6. Expected code touch points

- `kestrel/scheduler/pipeline.py`: NEW — `PipelineState` (canonical production state machine, no CUDA dependencies).
- `kestrel/scheduler/types.py`: Add `DecodeSlot`, `StepPlan`, `PendingCommit`, `LaunchHandle`; extend `ScheduledSequence` with `finalized`, `inflight_refs`.
- `kestrel/scheduler/transfer.py`: Extend with event-based `start_transfer()` API.
- `kestrel/scheduler/scheduler.py`: Scheduler API split (`schedule_decode_step`, `launch_forward_async`, `finalize_sampling`, `commit_step`).
- `kestrel/moondream/runtime.py`: Two `DecodeSlot` objects, ping-pong FlashInfer contexts.
- `kestrel/engine.py`: Unified loop in `_scheduler_loop`, delegates to `PipelineState` for state transitions.

**Tests:**

- `tests/scheduler/test_pipeline.py`: NEW — Unit tests for state machine (no CUDA).
- `tests/scheduler/test_transfer.py`: Integration tests for D2H transfers.
- Existing integration tests: Verify sync/async equivalence.

---

## 7. Open Questions / Risks

1. **CPU commit overhead (structured skills)**
   - Remaining overhead is primarily the host-side token commit path (D2H copy + Python `Token` objects + `consume_step`), not coord/size value decoding itself.
   - If this becomes dominant, consider: smaller "decision copies" (e.g., EOS flags) + batching commits, or pushing more of skill interpretation onto GPU.

2. **CUDA graph memory cost**
   - Two ping-pong slots means 2× the graph workspace memory and 2× captured graphs.
   - Measure the added memory cost vs graph replay speedup before finalizing batch-size coverage.

---

## 8. References

- vLLM async copy helpers: `ext/vllm/vllm/v1/worker/gpu/async_utils.py`
- vLLM execute/sample split: `ext/vllm/vllm/v1/worker/gpu/model_runner.py`
- vLLM batch queue loop: `ext/vllm/vllm/v1/engine/core.py` (`step_with_batch_queue`)
- Kestrel engine scheduler loop: `kestrel/engine.py` (`_scheduler_loop`)
- Kestrel scheduler: `kestrel/scheduler/scheduler.py`

---

## Appendix A — Current decode details (reference)

### A.1 Prefill (today)

Location: `GenerationScheduler._try_prefill()` (`kestrel/scheduler/scheduler.py`)

Steps per admitted request:

1. Check KV capacity (`runtime.can_reserve(request.target_length)`).
2. Run `runtime.start_sequence(...)`:
   - allocates a KV “row” (`page_table.allocate()`),
   - reserves pages for the target sequence length,
   - runs model prefill and returns `(SequenceState, logits)`.
3. Sample the first token from `logits` on GPU via `_sample_batch(...)`.
4. Decode coord/size **values** for the sampled ids on GPU (so next-step embeddings do not require CPU token materialization).
5. Write the sampled ids + decoded values into per-sequence “pending” GPU slots keyed by `SequenceState.batch_idx`.
6. Start a D2H copy of the sampled ids + decoded values into pinned host buffers (copy stream + event).
7. **Synchronize immediately** via `transfer.wait()` and stage the typed tokens:
   - materialize `Token` objects (TextToken/CoordToken/SizeToken) on CPU from `(token_id, coord_value, size_value)`,
   - call `SkillState.consume_step(...)`,
   - emit optional streaming callback.
8. Push the sequence into `running`.

### A.2 Decode step (today)

Location: `GenerationScheduler._decode_step()` (`kestrel/scheduler/scheduler.py`)

Per step:

1. Take all sequences from `running`.
2. Gather packed decode inputs from per-sequence GPU “pending” slots:
   - `token_ids` (LongTensor),
   - `coord_values` (float tensor shaped `[B, 1]`),
   - `size_values` (float tensor shaped `[B, 2]`).
3. Run `runtime.decode_batch(states, token_ids, coord_values, size_values)` and get `(logits, hidden_last)`.
4. Sample token ids on GPU (`_sample_batch(...)`).
5. Decode coord/size values for sampled ids on GPU (from `hidden_last`) and write the next-step packed inputs back into the per-sequence “pending” slots.
6. Start D2H copy of `(sampled_ids, coord_values, size_values)` into pinned host buffers (copy stream + event).
7. Do tiny CPU work (advance lengths): `seq.state.advance()`.
8. **Synchronize immediately**: `transfer.wait()`.
9. Materialize `Token` objects on CPU, call `stage_token` (`consume_step`, streaming), finalize finished sequences, re-queue the rest.
