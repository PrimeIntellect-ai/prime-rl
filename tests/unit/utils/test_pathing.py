import asyncio
import threading
import time

import pytest

from prime_rl.utils.pathing import sync_wait_for_path, validate_output_dir, wait_for_path


def test_nonexistent_dir_passes(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_empty_dir_passes(tmp_path):
    output_dir = tmp_path / "empty"
    output_dir.mkdir()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_only_logs_passes(tmp_path):
    output_dir = tmp_path / "has_logs"
    output_dir.mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "logs" / "trainer").mkdir(parents=True)
    (output_dir / "logs" / "trainer" / "rank_0.log").touch()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_raises(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains checkpoints"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_passes_when_resuming(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    validate_output_dir(output_dir, resuming=True, clean=False)


def test_dir_with_checkpoints_cleaned_when_flag_set(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    (output_dir / "logs").mkdir()

    validate_output_dir(output_dir, resuming=False, clean=True)

    assert not output_dir.exists()


def test_clean_on_nonexistent_dir_is_noop(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=True)
    assert not output_dir.exists()


class TestSyncWaitForPathTimeout:
    """Tests that sync_wait_for_path blocks forever when a path never appears.

    Reproduces a production issue where the shared trainer gets permanently
    stuck in pathing.py:106 polling for rollout files from a run that has
    been stopped. With no timeout parameter, the trainer can never break out
    of the wait loop to discover new runs.

    See: 4b trainer stuck waiting for step_127 for 4.8 days,
         35b trainer stuck waiting for step_329 for 3.5 days.
    """

    def test_sync_wait_blocks_indefinitely_on_missing_path(self, tmp_path):
        """sync_wait_for_path has no timeout and blocks forever when a path
        never appears. This reproduces a production issue where the shared
        trainer gets permanently stuck polling for rollout files from a
        stopped run (4b stuck 4.8 days on step_127, 35b stuck 3.5 days on
        step_329)."""
        missing = tmp_path / "will_never_exist" / "rank_0.bin"
        returned = threading.Event()

        def target():
            sync_wait_for_path(missing, interval=1)
            returned.set()

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=3)

        assert not t.is_alive(), (
            "sync_wait_for_path should support a timeout parameter so the "
            "trainer can break out of the poll loop when a run is stopped. "
            "Currently it blocks indefinitely with no escape."
        )

    def test_sync_wait_returns_when_path_appears(self, tmp_path):
        """sync_wait_for_path should return promptly when the path exists."""
        target_file = tmp_path / "rollouts" / "step_0" / "rank_0.bin"

        def create_after_delay():
            time.sleep(0.5)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.touch()

        creator = threading.Thread(target=create_after_delay)
        creator.start()

        sync_wait_for_path(target_file, interval=1)
        creator.join()
        assert target_file.exists()


class TestAsyncWaitForPathTimeout:
    """Same issue for the async variant."""

    def test_async_wait_blocks_indefinitely_on_missing_path(self, tmp_path):
        """wait_for_path has no timeout and blocks forever when a path
        never appears. Same bug as the sync variant."""
        missing = tmp_path / "will_never_exist" / "rank_0.bin"

        async def run():
            await asyncio.wait_for(
                wait_for_path(missing, interval=1),
                timeout=3,
            )

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(run())

    def test_async_wait_returns_when_path_appears(self, tmp_path):
        """wait_for_path should return promptly when the path exists."""
        target_file = tmp_path / "rollouts" / "step_0" / "rank_0.bin"

        async def run():
            async def create_after_delay():
                await asyncio.sleep(0.5)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.touch()

            asyncio.create_task(create_after_delay())
            await wait_for_path(target_file, interval=1)

        asyncio.run(run())
        assert target_file.exists()
