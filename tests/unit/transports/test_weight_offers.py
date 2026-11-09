import asyncio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prime_rl.configs.shared import FileSystemWeightBroadcastConfig
from prime_rl.transports.weights.base import FINISHED_MARKER, SENDER_READY_MARKER, OfferSuperseded
from prime_rl.transports.weights.filesystem import FileSystemWeightReceiver


def _offer(broadcast_dir: Path, step: int, offer: str) -> Path:
    """Offer v{step} like the trainer does: a fresh step dir with a new offer id."""
    step_dir = broadcast_dir / f"step_{step}"
    shutil.rmtree(step_dir, ignore_errors=True)
    step_dir.mkdir(parents=True)
    (step_dir / SENDER_READY_MARKER).write_text(offer)
    return step_dir


def test_failed_offer_waits_for_a_new_one_and_a_reoffer_supersedes_a_receive(tmp_path: Path):
    admin_plane = MagicMock(update_weights=AsyncMock(side_effect=RuntimeError("install failed")))
    receiver = FileSystemWeightReceiver(tmp_path, FileSystemWeightBroadcastConfig(), admin_plane, "model")
    (_offer(tmp_path, 3, "a") / FINISHED_MARKER).touch()
    with pytest.raises(RuntimeError):
        asyncio.run(receiver.apply(3))
    assert receiver.next_version(2) == 2

    # The restarted trainer offers v3 again: it is published, and a third offer supersedes its receive
    _offer(tmp_path, 3, "b")
    assert receiver.next_version(2) == 3

    async def supersede():
        apply = asyncio.create_task(receiver.apply(3))
        await asyncio.sleep(0.2)
        _offer(tmp_path, 3, "c")
        with pytest.raises(OfferSuperseded):
            await asyncio.wait_for(apply, 5)

    asyncio.run(supersede())
    assert receiver.is_published(3)
