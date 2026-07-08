# Scheduler Overview

The scheduler owns batched prefill/decode for Moondream inference. It sits between the high-level engine (`kestrel.engine.InferenceEngine`) and the low-level runtime (`kestrel.models.moondream.runtime.MoondreamRuntime`). The engine prepares validated `GenerationRequest` objects and corresponding `SkillState` instances, then hands them to the scheduler, which multiplexes work on a shared runtime instance.

## Responsibilities

- Maintain FIFO queues for pending and active `GenerationRequest`s while respecting the runtime's KV-cache capacity.
- Perform staged execution: prefill each request, transition it into decode waves, and drain sequences until completion.
- Surface streaming updates and final `SchedulerResult`s back to the engine without interpreting the skill payloads.
- Keep the scheduler skill-agnostic; it only forwards sampled tokens to the provided `SkillState` and invokes `finalize` when a sequence ends.

## Relationship to Other Components

- **InferenceEngine**: constructs `GenerationRequest`s, instantiates `SkillState`s (using skill-specific request contexts), and enqueues them via `GenerationScheduler.enqueue_request`. The engine also supplies optional stream callbacks for token updates.
- **MoondreamRuntime**: executes `prepare_sequence` + `launch_prepared_batch` for prefill and `decode_with_slot` for decode. The scheduler ensures batch sizes and decode steps respect runtime limits.
- **Skills**: provide `SkillState` implementations that buffer tokens, perform per-skill logic, and produce final results. The scheduler never inspects skill-specific data.

## Execution Flow

1. **Submission**: `GenerationScheduler.enqueue_request(...)` pushes a prepared request/skill-state pair into the waiting queue.
2. **Prefill** (`_prepare_prefill` + `_launch_prefill_step` + `_finalize_prefill`): whenever capacity allows, the scheduler prepares requests on CPU (`runtime.prepare_sequence`), then launches one batched GPU prefill (`runtime.launch_prepared_batch`) and samples token0 into shared pending buffers. Tokens are committed later via `commit_step` (pipelined like decode).
3. **Decode Loop** (pipelined in `advance`): batches active sequences, feeds pending tokens into `runtime.decode_with_slot`, stages new tokens on each `SkillState`, and re-queues sequences until they finish or hit limits.
4. **Finalization** (`_finalize_sequence`): once a sequence ends, the scheduler releases runtime resources, asks the `SkillState` to `finalize`, and records a `SchedulerResult`.

## KV Reservation and Recompute Preemption

Requests reserve KV cache incrementally. Prefill reserves the prompt plus a
runtime-chosen decode window, not the full generation cap. Before each decode
launch, `schedule_decode_step` asks the runtime to grow every selected row for
the upcoming write; a row enters the GPU launch batch only after that reservation
succeeds.

If decode growth fails, the scheduler recovers by recompute preemption. A row is
preemptible only when speculative decoding is disabled, it has no in-flight
decode work, it is not finished or finalized, it owns a live `SequenceState`, and
it still has generation budget left. The scheduler first preempts newer idle rows
so older work keeps priority. If no newer idle row can help and decode work is
still in flight, the scheduler waits for that work to commit. If no other row
can free capacity, the blocked row preempts itself.

Preemption releases the row's KV pages, records the committed generated tokens
and logprobs in `request.generated_prefix`, resets runtime-owned lifecycle
state, and pushes the request to the front of the waiting queue. When that
request launches again, prefill recomputes the prompt plus generated prefix and
decode resumes from the committed tokens.

This makes incremental reservation deadlock-free for the scheduler as long as
launched GPU work eventually commits or fails and the runtime releases KV for
completed or preempted rows. Each failed decode-growth attempt either reserves
capacity, frees capacity by preempting a finite idle row, waits for finite
in-flight work, or removes the blocked row from the running set so it no longer
holds KV while waiting. A request whose recompute prefill can never fit is a hard
capacity error, not a scheduler deadlock.

## Internal API Summary

- `GenerationScheduler.enqueue_request(request, skill_state)`: enqueue a fully prepared request/skill state duo. Used by the engine when it wants full control over state creation.
- `GenerationScheduler.waiting` / `GenerationScheduler.running`: FIFO queues (`RequestQueue` / `RunningQueue`) tracking pending vs. active sequences.
- `RequestLifecycle`: groups the runtime `SequenceState`, the user-facing `GenerationRequest`, and the `SkillState`. The scheduler only mutates decode bookkeeping on this container.
- `GenerationRequest`: immutable request metadata plus the associated `skill_state`/`request_context` set during submission.
- `StreamCallback`: optional callable that receives `StreamUpdate` events when the scheduler stages new tokens; used by the engine to power streaming APIs.

## Notes and Best Practices

- Always ensure `GenerationRequest.skill_state` is set before enqueuing; the scheduler assumes it is present when prefill begins.
- Keep skill-specific behavior inside `SkillState.consume_step` and `SkillState.finalize` so the scheduler remains generic.
- Batch sizing is governed by `MoondreamRuntime.max_batch_size` (effective sequences; batch_idx 0 is reserved) and KV-cache availability.
- Stream callbacks run on the engine's loop via `call_soon_threadsafe`; they should avoid long blocking operations.
- Tokens staged on `SkillState`s are typed (`TextToken`, `CoordToken`, `SizeToken`). The scheduler asks the runtime to render sampled ids into these structures before handing them to skills, and the runtime re-embeds them (including region values) on the next decode step.
