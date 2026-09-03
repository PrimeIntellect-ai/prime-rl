import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig
from prime_rl.inference.vllm.worker.filesystem import FileSystemWeightUpdateWorker
from prime_rl.transports.weights.base import FINISHED_MARKER
from prime_rl.transports.weights.filesystem import FileSystemWeightReceiver


def test_filesystem_receiver_uses_collective_rpc_after_topology_check(tmp_path):
    step_dir = tmp_path / "step_3"
    step_dir.mkdir()
    (step_dir / FINISHED_MARKER).touch()
    topology_guard = AsyncMock()
    receiver = FileSystemWeightReceiver(
        tmp_path,
        FileSystemWeightBroadcastConfig(),
        [],
        "Qwen/Qwen3-0.6B",
        use_collective_rpc=True,
        topology_guard=topology_guard,
    )

    with patch(
        "prime_rl.transports.weights.filesystem.update_weights",
        new_callable=AsyncMock,
    ) as update:
        asyncio.run(receiver.receive(3))

    topology_guard.assert_awaited_once_with()
    update.assert_awaited_once_with(
        [],
        step_dir,
        step=3,
        use_collective_rpc=True,
    )


def test_filesystem_worker_uses_native_vllm_reload(tmp_path):
    worker = MagicMock()

    FileSystemWeightUpdateWorker.update_weights_from_path(worker, str(tmp_path))

    worker.reload_weights.assert_called_once_with(weights_path=str(tmp_path))
