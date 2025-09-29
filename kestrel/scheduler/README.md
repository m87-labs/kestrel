# Scheduler Overview

The scheduler owns batched prefill/decode for Moondream inference. It sits between the high-level engine (`kestrel.engine.InferenceEngine`) and the low-level runtime (`kestrel.moondream.runtime.MoondreamRuntime`). The engine prepares validated `GenerationRequest` objects and corresponding `SkillState` instances, then hands them to the scheduler, which multiplexes work on a shared runtime instance.

## Responsibilities

- Maintain FIFO queues for pending and active `GenerationRequest`s while respecting the runtime's KV-cache capacity.
- Perform staged execution: prefill each request, transition it into decode waves, and drain sequences until completion.
- Surface streaming updates and final `SchedulerResult`s back to the engine without interpreting the skill payloads.
- Keep the scheduler skill-agnostic; it only forwards sampled tokens to the provided `SkillState` and invokes `finalize` when a sequence ends.

## Relationship to Other Components

- **InferenceEngine**: constructs `GenerationRequest`s, instantiates `SkillState`s (using skill-specific context), and enqueues them via `GenerationScheduler.enqueue_request` or `submit`. The engine also supplies optional stream callbacks for token updates.
- **MoondreamRuntime**: executes `start_sequence` and `decode_batch` calls. The scheduler ensures batch sizes and decode steps respect runtime limits.
- **Skills**: provide `SkillState` implementations that buffer tokens, perform per-skill logic, and produce final results. The scheduler never inspects skill-specific data.

## Execution Flow

1. **Submission**: `GenerationScheduler.submit(...)` (or `enqueue_request`) pushes a request and associated `SkillState` into the waiting queue.
2. **Prefill** (`_try_prefill`): whenever capacity allows, the scheduler pops waiting requests, runs `runtime.start_sequence`, and primes the `SkillState` with the first sampled token.
3. **Decode Loop** (`_decode_step`): batches active sequences, feeds pending tokens into `runtime.decode_batch`, stages new tokens on each `SkillState`, and re-queues sequences until they finish or hit limits.
4. **Finalization** (`_finalize_sequence`): once a sequence ends, the scheduler releases runtime resources, asks the `SkillState` to `finalize`, and records a `SchedulerResult`.
5. **Completion**: `run()` drains all work and returns the list of finished results to the engine.

## Internal API Summary

- `GenerationScheduler.submit(...)`: convenience helper that builds prompt tokens (if needed), constructs a `GenerationRequest`, creates the `SkillState`, and enqueues the pair.
- `GenerationScheduler.enqueue_request(request, skill_state)`: enqueue a fully prepared request/skill state duo. Used by the engine when it wants full control over state creation.
- `GenerationScheduler.run() -> list[SchedulerResult]`: main driver; processes queues until no runnable work remains.
- `GenerationScheduler.submit_many(...)`: bulk submission wrapper that calls `submit` for multiple prompts.
- `GenerationScheduler.waiting` / `GenerationScheduler.running`: FIFO queues (`RequestQueue` / `RunningQueue`) tracking pending vs. active sequences.
- `ScheduledSequence`: groups the runtime `SequenceState`, the user-facing `GenerationRequest`, and the `SkillState`. The scheduler only mutates decode bookkeeping on this container.
- `GenerationRequest`: immutable request metadata plus an attached `skill_state` set during submission.
- `StreamCallback`: optional callable that receives `StreamUpdate` events when the scheduler stages new tokens; used by the engine to power streaming APIs.

## Notes and Best Practices

- Always ensure `GenerationRequest.skill_state` is set before enqueuing; the scheduler assumes it is present when prefill begins.
- Keep skill-specific behavior inside `SkillState.consume_step` and `SkillState.finalize` so the scheduler remains generic.
- Batch sizing is governed by `MoondreamRuntime.max_batch_size` and KV-cache availability. If `run()` exits early with work still queued, the engine should investigate capacity issues.
- Stream callbacks run on the engine's loop via `call_soon_threadsafe`; they should avoid long blocking operations.
