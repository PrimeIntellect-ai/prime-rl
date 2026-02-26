import asyncio


async def safe_cancel(task: asyncio.Task) -> None:
    """Safely cancels and awaits an asyncio.Task."""
    task.cancel()
    try:
        await task
    except Exception:
        pass
