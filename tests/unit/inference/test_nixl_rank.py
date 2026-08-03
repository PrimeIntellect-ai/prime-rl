from types import SimpleNamespace
from unittest.mock import patch

from prime_rl.inference.vllm.worker import nixl


def test_init_broadcaster_uses_topology_aware_global_rank():
    worker = object.__new__(nixl.NIXLWeightUpdateWorker)
    worker.device = SimpleNamespace(index=3)
    worker.rank = 5
    worker.parallel_config = SimpleNamespace(
        data_parallel_index=2,
        data_parallel_size=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
    )

    with (
        patch.object(nixl, "global_inference_rank", return_value=7) as compute_rank,
        patch.object(nixl, "set_ucx_env_defaults"),
        patch.object(nixl, "NixlAgent") as agent,
        patch.object(nixl, "MxClient") as client,
        patch.object(nixl, "ModelExpressSession") as session,
    ):
        worker.init_broadcaster(
            host="model-express",
            port=8001,
            rank_offset=4,
            inference_world_size=12,
            timeout=120,
            session_id="test-session",
            engine_world_size=8,
        )

    compute_rank.assert_called_once_with(
        rank_offset=4,
        data_parallel_index=2,
        data_parallel_size=1,
        worker_rank=5,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        inference_world_size=12,
        engine_world_size=8,
    )
    agent.assert_called_once_with(nixl.make_agent_name("inference", 7))
    client.assert_called_once_with(server_url="model-express:8001")
    session.assert_called_once_with(
        client=client.return_value,
        role="inference",
        rank=7,
        session_id="test-session",
        worker_id="inference-7",
    )
    assert worker.weight_transfer_timeout == 120
    assert worker.weight_transfer_plan is None
