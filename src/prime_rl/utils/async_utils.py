import asyncio


async def safe_cancel(task: asyncio.Task) -> None:
    """Safely cancels and awaits an asyncio.Task."""
    task.cancel()
    try:
        await task
    except BaseException:
        pass


async def safe_cancel_all(tasks: list[asyncio.Task]) -> None:
    """Safely cancels and awaits all asyncio.Tasks."""
    await asyncio.gather(*[safe_cancel(task) for task in tasks])
