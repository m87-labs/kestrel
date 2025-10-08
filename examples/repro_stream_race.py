"""Demonstrate the streaming completion race in the inference engine."""

from __future__ import annotations

import asyncio


async def _collect(queue: asyncio.Queue[str]) -> list[str]:
    results: list[str] = []
    while True:
        item = await queue.get()
        results.append(item)
        if item == "<done>":
            break
    return results


async def demonstrate_direct_completion() -> list[str]:
    """Reproduce the bug: completion enqueued before pending updates."""

    queue: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Schedule token updates on the loop; they will run once we yield control.
    for idx in range(3):
        loop.call_soon(queue.put_nowait, f"token-{idx}")

    # Original implementation called ``queue.put_nowait`` directly, which makes
    # the completion arrive before any of the scheduled updates have a chance
    # to run when the consumer awaits the next item.
    queue.put_nowait("<done>")

    return await _collect(queue)


async def demonstrate_scheduled_completion() -> list[str]:
    """Show the fix: schedule completion on the loop as well."""

    queue: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    for idx in range(3):
        loop.call_soon(queue.put_nowait, f"token-{idx}")

    # By scheduling the completion on the loop we ensure it runs after the
    # previously queued token callbacks, matching the behaviour of the fix.
    loop.call_soon(queue.put_nowait, "<done>")

    return await _collect(queue)


async def main() -> None:
    direct = await demonstrate_direct_completion()
    scheduled = await demonstrate_scheduled_completion()

    print("Direct completion (buggy order):", direct)
    print("Scheduled completion (correct order):", scheduled)


if __name__ == "__main__":
    asyncio.run(main())
