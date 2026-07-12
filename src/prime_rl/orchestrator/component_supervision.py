"""Failure propagation for orchestrator background components."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")

# Cleanup must not extend a user-facing operation timeout indefinitely. A task
# that suppresses cancellation is retained below and observed when it settles.
SUPERVISED_OPERATION_CANCEL_GRACE_SECONDS = 1.0
_ORPHANED_OPERATIONS: set[asyncio.Task] = set()


def _observe_operation(task: asyncio.Task) -> None:
    _ORPHANED_OPERATIONS.discard(task)
    try:
        task.exception()
    except BaseException:
        pass


async def _cancel_operation_with_grace(task: asyncio.Task) -> asyncio.CancelledError | None:
    task.cancel()
    loop = asyncio.get_running_loop()
    grace_elapsed = loop.create_future()
    caller_cancellation: asyncio.CancelledError | None = None

    def finish_grace() -> None:
        if not grace_elapsed.done():
            grace_elapsed.set_result(None)

    timer = loop.call_later(SUPERVISED_OPERATION_CANCEL_GRACE_SECONDS, finish_grace)
    try:
        while not task.done() and not grace_elapsed.done():
            try:
                await asyncio.wait((task, grace_elapsed), return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError as error:
                # Finish bounded cleanup before propagating caller cancellation.
                # Repeated cancellation must not make cleanup unbounded.
                if caller_cancellation is None:
                    caller_cancellation = error
                continue
    finally:
        timer.cancel()

    if task.done():
        _observe_operation(task)
    else:
        _ORPHANED_OPERATIONS.add(task)
        task.add_done_callback(_observe_operation)
    return caller_cancellation


def raise_if_component_failed(tasks: Sequence[asyncio.Task]) -> None:
    """Raise when a background loop exits while the main loop is still live."""
    for task in tasks:
        if not task.done():
            continue
        name = task.get_name()
        if task.cancelled():
            raise RuntimeError(f"Orchestrator component {name!r} was cancelled unexpectedly")
        error = task.exception()
        if error is not None:
            raise error
        raise RuntimeError(f"Orchestrator component {name!r} stopped unexpectedly")


async def run_with_component_supervision(
    operation: Callable[[], Awaitable[T]],
    component_tasks: Sequence[asyncio.Task],
    *,
    timeout: float | None = None,
    timeout_description: str | None = None,
) -> T | None:
    """Run one operation while racing every supervised component.

    A component failure wins when both sides complete in the same event-loop
    turn. ``None`` denotes timeout unless ``timeout_description`` is supplied,
    in which case timeout raises after cancelling the operation.
    """
    raise_if_component_failed(component_tasks)
    task = asyncio.create_task(operation())
    component_failure_selected = False
    try:
        await asyncio.wait(
            [task, *component_tasks],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout,
        )
        try:
            raise_if_component_failed(component_tasks)
        except BaseException:
            component_failure_selected = True
            raise
        if task.done():
            return await task
        if timeout_description is not None:
            raise TimeoutError(f"{timeout_description} timed out after {timeout} seconds")
        return None
    finally:
        if not task.done():
            caller_cancellation = await _cancel_operation_with_grace(task)
            if caller_cancellation is not None and not component_failure_selected:
                raise caller_cancellation
