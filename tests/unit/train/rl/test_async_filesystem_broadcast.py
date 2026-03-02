import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from prime_rl.configs.trainer import (
    AsyncFileSystemWeightBroadcastConfig,
    FileSystemWeightBroadcastConfig,
    NCCLWeightBroadcastConfig,
)
from prime_rl.utils.validation import _broadcast_type_family


def test_async_filesystem_config_defaults():
    config = AsyncFileSystemWeightBroadcastConfig()
    assert config.type == "async_filesystem"
    assert config.save_format == "safetensors"
    assert config.save_sharded is True


def test_async_filesystem_config_custom():
    config = AsyncFileSystemWeightBroadcastConfig(save_format="torch", save_sharded=False)
    assert config.save_format == "torch"
    assert config.save_sharded is False


def test_config_discriminator_dispatch():
    """The WeightBroadcastConfig union should correctly dispatch on type."""
    from pydantic import TypeAdapter

    from prime_rl.configs.trainer import WeightBroadcastConfig

    adapter = TypeAdapter(WeightBroadcastConfig)

    fs = adapter.validate_python({"type": "filesystem"})
    assert isinstance(fs, FileSystemWeightBroadcastConfig)

    async_fs = adapter.validate_python({"type": "async_filesystem"})
    assert isinstance(async_fs, AsyncFileSystemWeightBroadcastConfig)

    nccl = adapter.validate_python({"type": "nccl"})
    assert isinstance(nccl, NCCLWeightBroadcastConfig)


def test_broadcast_type_family():
    assert _broadcast_type_family("filesystem") == "filesystem"
    assert _broadcast_type_family("async_filesystem") == "filesystem"
    assert _broadcast_type_family("nccl") == "nccl"


def test_setup_weight_broadcast_dispatches_async():
    """setup_weight_broadcast should return AsyncFileSystemWeightBroadcast for async_filesystem config."""
    config = AsyncFileSystemWeightBroadcastConfig()

    with (
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_world") as mock_world,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_multi_run_manager") as mock_mrm,
    ):
        mock_world.return_value = MagicMock(is_master=True)
        mock_mrm.return_value = MagicMock()

        from prime_rl.trainer.rl.broadcast import setup_weight_broadcast
        from prime_rl.trainer.rl.broadcast.async_filesystem import AsyncFileSystemWeightBroadcast

        broadcast = setup_weight_broadcast(Path("/tmp/test"), config)
        assert isinstance(broadcast, AsyncFileSystemWeightBroadcast)
        broadcast.shutdown()


def test_setup_weight_broadcast_dispatches_sync():
    """setup_weight_broadcast should return FileSystemWeightBroadcast for filesystem config."""
    config = FileSystemWeightBroadcastConfig()

    with (
        patch("prime_rl.trainer.rl.broadcast.filesystem.get_world") as mock_world,
        patch("prime_rl.trainer.rl.broadcast.filesystem.get_multi_run_manager") as mock_mrm,
    ):
        mock_world.return_value = MagicMock(is_master=True)
        mock_mrm.return_value = MagicMock()

        from prime_rl.trainer.rl.broadcast import setup_weight_broadcast
        from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast

        broadcast = setup_weight_broadcast(Path("/tmp/test"), config)
        assert isinstance(broadcast, FileSystemWeightBroadcast)


def test_wait_for_pending_returns_zero_when_no_pending():
    config = AsyncFileSystemWeightBroadcastConfig()
    with (
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_world") as mock_world,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_multi_run_manager") as mock_mrm,
    ):
        mock_world.return_value = MagicMock(is_master=True)
        mock_mrm.return_value = MagicMock()

        from prime_rl.trainer.rl.broadcast.async_filesystem import AsyncFileSystemWeightBroadcast

        broadcast = AsyncFileSystemWeightBroadcast(Path("/tmp/test"), config)
        assert broadcast._pending is None
        wait = broadcast._wait_for_pending()
        assert wait == 0.0
        broadcast.shutdown()


def test_wait_for_pending_blocks_on_inflight():
    config = AsyncFileSystemWeightBroadcastConfig()
    with (
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_world") as mock_world,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_multi_run_manager") as mock_mrm,
    ):
        mock_world.return_value = MagicMock(is_master=True)
        mock_mrm.return_value = MagicMock()

        from prime_rl.trainer.rl.broadcast.async_filesystem import AsyncFileSystemWeightBroadcast

        broadcast = AsyncFileSystemWeightBroadcast(Path("/tmp/test"), config)

        executor = ThreadPoolExecutor(max_workers=1)
        broadcast._pending = executor.submit(time.sleep, 0.2)

        start = time.perf_counter()
        wait = broadcast._wait_for_pending()
        elapsed = time.perf_counter() - start

        assert wait > 0.1
        assert elapsed >= 0.15
        assert broadcast._pending is None

        executor.shutdown(wait=True)
        broadcast.shutdown()


def test_wait_for_pending_reraises_exception():
    config = AsyncFileSystemWeightBroadcastConfig()
    with (
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_world") as mock_world,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_multi_run_manager") as mock_mrm,
    ):
        mock_world.return_value = MagicMock(is_master=True)
        mock_mrm.return_value = MagicMock()

        from prime_rl.trainer.rl.broadcast.async_filesystem import AsyncFileSystemWeightBroadcast

        broadcast = AsyncFileSystemWeightBroadcast(Path("/tmp/test"), config)

        def failing_task():
            raise RuntimeError("disk full")

        executor = ThreadPoolExecutor(max_workers=1)
        broadcast._pending = executor.submit(failing_task)
        time.sleep(0.05)

        with pytest.raises(RuntimeError, match="disk full"):
            broadcast._wait_for_pending()

        assert broadcast._pending is None

        executor.shutdown(wait=True)
        broadcast.shutdown()


def test_write_and_notify_saves_and_touches_stable(tmp_path):
    config = AsyncFileSystemWeightBroadcastConfig()
    with (
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_world") as mock_world,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_multi_run_manager") as mock_mrm,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.save_state_dict") as mock_save,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_broadcast_dir") as mock_bdir,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_step_path") as mock_step_path,
    ):
        mock_world.return_value = MagicMock(is_master=True)
        mock_mrm_inst = MagicMock()
        mock_mrm_inst.get_orchestrator_config.return_value = MagicMock()
        mock_mrm.return_value = mock_mrm_inst

        from prime_rl.trainer.rl.broadcast.async_filesystem import AsyncFileSystemWeightBroadcast

        broadcast = AsyncFileSystemWeightBroadcast(tmp_path, config)

        save_dir = tmp_path / "broadcast" / "step-0"
        save_dir.mkdir(parents=True)
        mock_step_path.return_value = save_dir
        mock_bdir.return_value = tmp_path / "broadcast"

        run_dir = tmp_path / "run_default"
        state_dict = {"weight": torch.randn(4, 4)}
        run_metadata = {0: (run_dir, 0)}

        broadcast._write_and_notify(
            state_dicts={0: state_dict},
            run_metadata=run_metadata,
            adapter_only=False,
            lora_configs=None,
            model=None,
            cuda_event=None,
        )

        mock_save.assert_called_once_with(state_dict, save_dir, "safetensors", True, adapter=False)
        assert (save_dir / "STABLE").exists()
        assert broadcast._last_broadcast_time > 0.0

        broadcast.shutdown()


def test_write_and_notify_syncs_cuda_event(tmp_path):
    config = AsyncFileSystemWeightBroadcastConfig()
    with (
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_world") as mock_world,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_multi_run_manager") as mock_mrm,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.save_state_dict"),
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_broadcast_dir") as mock_bdir,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_step_path") as mock_step_path,
    ):
        mock_world.return_value = MagicMock(is_master=True)
        mock_mrm_inst = MagicMock()
        mock_mrm_inst.get_orchestrator_config.return_value = MagicMock()
        mock_mrm.return_value = mock_mrm_inst

        from prime_rl.trainer.rl.broadcast.async_filesystem import AsyncFileSystemWeightBroadcast

        broadcast = AsyncFileSystemWeightBroadcast(tmp_path, config)

        save_dir = tmp_path / "broadcast" / "step-0"
        save_dir.mkdir(parents=True)
        mock_step_path.return_value = save_dir
        mock_bdir.return_value = tmp_path / "broadcast"

        mock_event = MagicMock()
        broadcast._write_and_notify(
            state_dicts={0: {"w": torch.randn(2, 2)}},
            run_metadata={0: (tmp_path / "run", 0)},
            adapter_only=False,
            lora_configs=None,
            model=None,
            cuda_event=mock_event,
        )

        mock_event.synchronize.assert_called_once()
        broadcast.shutdown()


def test_shutdown_waits_for_pending():
    config = AsyncFileSystemWeightBroadcastConfig()
    with (
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_world") as mock_world,
        patch("prime_rl.trainer.rl.broadcast.async_filesystem.get_multi_run_manager") as mock_mrm,
    ):
        mock_world.return_value = MagicMock(is_master=True)
        mock_mrm.return_value = MagicMock()

        from prime_rl.trainer.rl.broadcast.async_filesystem import AsyncFileSystemWeightBroadcast

        broadcast = AsyncFileSystemWeightBroadcast(Path("/tmp/test"), config)

        executor = ThreadPoolExecutor(max_workers=1)
        broadcast._pending = executor.submit(time.sleep, 0.15)

        start = time.perf_counter()
        broadcast.shutdown()
        elapsed = time.perf_counter() - start

        assert elapsed >= 0.1
        executor.shutdown(wait=True)
