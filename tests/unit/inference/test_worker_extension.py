from unittest.mock import patch

import pytest

from prime_rl.inference.vllm.worker_extension import worker_extension_class_path


def test_worker_extension_class_path_resolves_nccl_automatically():
    assert worker_extension_class_path("nccl") == ("prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker")


def test_worker_extension_import_validation_reports_missing_module():
    with (
        patch("prime_rl.inference.vllm.worker_extension.import_module", side_effect=ModuleNotFoundError("missing")),
        pytest.raises(ImportError, match="Could not import the nccl worker extension"),
    ):
        worker_extension_class_path("nccl", validate_import=True)
