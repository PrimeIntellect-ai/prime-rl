"""Failure propagation for orchestrator background components."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence


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
