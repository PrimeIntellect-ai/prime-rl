from types import SimpleNamespace
from unittest.mock import patch

from torch.nn import Module

from prime_rl.inference.vllm.worker import nccl
from prime_rl.inference.vllm.worker.nccl import NCCLWeightUpdateWorker


def test_nccl_update_logs_global_rank_settlement():
    worker = object.__new__(NCCLWeightUpdateWorker)
    worker.quantize_in_weight_transfer = False
    worker.model_runner = SimpleNamespace(model=Module())
    worker.nccl_broadcast_receiver = SimpleNamespace(
        receive_state_dicts=lambda: iter(())
    )
    worker._prime_rl_global_inference_rank = 7
    worker._prime_rl_inference_world_size = 16

    with patch.object(nccl.logger, "info") as log_info:
        worker.update_weights_from_path("/run/broadcasts/step_1")

    log_info.assert_called_once_with(
        "Completed NCCL weight update "
        "[global_rank=7 inference_world_size=16 "
        "weight_dir=/run/broadcasts/step_1]"
    )
