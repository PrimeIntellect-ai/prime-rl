from types import SimpleNamespace
from unittest.mock import Mock

import torch.nn as nn

from prime_rl.transports.weights.dynamo_nccl import DynamoNCCLWeightBroadcast
from prime_rl.utils.dynamo import DynamoWorker


class NoVersionArgumentEngine:
    def __init__(self) -> None:
        self.called = False

    def send_weights(self) -> None:
        self.called = True


def test_dynamo_nccl_broadcast_uses_main_vllm_api_and_commits_version_explicitly(tmp_path, monkeypatch):
    broadcaster = object.__new__(DynamoNCCLWeightBroadcast)
    broadcaster.output_dir = tmp_path
    broadcaster.world = SimpleNamespace(is_master=True, world_size=1)
    broadcaster.source = Mock()
    broadcaster.engine = NoVersionArgumentEngine()
    broadcaster.client = Mock()
    model = nn.Linear(2, 2)
    monkeypatch.setattr("prime_rl.transports.weights.dynamo_nccl.sync_wait_for_path", Mock())

    broadcaster.broadcast_weights(model, step=7)

    broadcaster.source.set_model.assert_called_once_with(model)
    assert broadcaster.engine.called
    broadcaster.client.update_weight_version.assert_called_once_with("7")


def test_dynamo_nccl_broadcaster_constructs_against_pinned_vllm_main(tmp_path, monkeypatch):
    from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

    worker = DynamoWorker(
        namespace="dynamo",
        component="backend",
        instance_id=1,
        model="Qwen/Qwen3-0.6B",
        system_url="http://backend-1:8080",
        admin_base_url="http://backend-1:8120",
        world_size=1,
        weight_transfer_backend="nccl",
        routes=(),
    )
    dynamo = SimpleNamespace(
        headers={},
        headers_from_env={},
        api_key_var="PRIME_API_KEY",
        discovery_url="http://frontend:8000",
        model_name=worker.model,
    )
    config = SimpleNamespace(
        dynamo=dynamo,
        timeout=30,
        host="trainer",
        port=12345,
        inference_world_size=1,
    )
    trainer_engine = Mock()
    trainer_init = Mock(return_value=trainer_engine)
    monkeypatch.setattr("prime_rl.transports.weights.dynamo_nccl._discover", lambda _: (worker,))
    monkeypatch.setattr(
        "prime_rl.transports.weights.dynamo_nccl.get_world",
        lambda: SimpleNamespace(rank=0, is_master=True, world_size=1),
    )
    monkeypatch.setattr("prime_rl.utils.nccl.disable_nccl_p2p_if_unavailable", Mock())
    monkeypatch.setattr(WeightTransferTrainerFactory, "trainer_init", trainer_init)

    broadcaster = DynamoNCCLWeightBroadcast(tmp_path, config)

    assert broadcaster.engine is trainer_engine
    init_info = trainer_init.call_args.args[0]
    assert isinstance(init_info, NCCLTrainerInitInfo)
    assert init_info.master_address == "trainer"
    assert init_info.master_port == 12345
    assert init_info.world_size == 2
