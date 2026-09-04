from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")

from prime_rl.inference.vllm.worker.filesystem import FileSystemWeightUpdateWorker


def test_filesystem_worker_uses_native_vllm_reload(tmp_path):
    worker = MagicMock()

    FileSystemWeightUpdateWorker.update_weights_from_path(worker, str(tmp_path))

    worker.reload_weights.assert_called_once_with(weights_path=str(tmp_path))
