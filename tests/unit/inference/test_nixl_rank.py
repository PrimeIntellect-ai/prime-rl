from types import SimpleNamespace
from unittest.mock import patch

from prime_rl.inference.vllm.worker import nixl


def _initialize_worker(*, data_parallel_index: int, device_index: int) -> int:
    worker = object.__new__(nixl.NIXLWeightUpdateWorker)
    worker.device = SimpleNamespace(index=device_index)
    worker.rank = 0
    worker.parallel_config = SimpleNamespace(
        data_parallel_index=data_parallel_index,
        data_parallel_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
    )

    with (
        patch.object(nixl, "set_ucx_env_defaults"),
        patch.object(nixl, "NixlAgent"),
        patch.object(nixl, "MxClient"),
        patch.object(nixl, "ModelExpressSession") as session,
    ):
        worker.init_broadcaster(
            host="model-express",
            port=8001,
            rank_offset=4,
            inference_world_size=8,
            timeout=120,
            session_id="test-session",
            engine_world_size=4,
        )

    return session.call_args.kwargs["rank"]


def test_init_broadcaster_uses_logical_rank_when_device_ordinals_repeat():
    first_rank = _initialize_worker(data_parallel_index=0, device_index=3)
    second_rank = _initialize_worker(data_parallel_index=1, device_index=3)

    assert (first_rank, second_rank) == (4, 6)
