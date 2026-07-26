from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, call, patch

from torch.nn import Module

from prime_rl.inference.vllm.worker import nccl
from prime_rl.inference.vllm.worker.nccl import NCCLWeightUpdateWorker


def test_nccl_update_logs_global_rank_settlement():
    state_iter = iter((("weight", object()),))
    worker = object.__new__(NCCLWeightUpdateWorker)
    worker.quantize_in_weight_transfer = False
    worker.model_runner = SimpleNamespace(model=Module(), model_config=object())
    worker.vllm_config = object()
    worker.nccl_broadcast_receiver = SimpleNamespace(
        receive_state_dicts=lambda: iter((state_iter,))
    )
    worker._prime_rl_global_inference_rank = 7
    worker._prime_rl_inference_world_size = 16

    events = MagicMock()
    with (
        patch.object(
            nccl,
            "load_weights_checkpoint_layerwise",
            side_effect=lambda model, state_iter, model_config, vllm_config: list(state_iter),
        ) as load,
        patch.object(nccl.logger, "info") as log_info,
    ):
        events.attach_mock(load, "load")
        events.attach_mock(log_info, "log")
        worker.update_weights_from_path("/run/broadcasts/step_1")

    assert events.mock_calls == [
        call.load(
            worker.model_runner.model,
            ANY,
            worker.model_runner.model_config,
            worker.vllm_config,
        ),
        call.log(
            "Completed NCCL weight update "
            "[global_rank=7 inference_world_size=16 "
            "weight_dir=/run/broadcasts/step_1]"
        ),
    ]
