import multiprocessing as mp

import pytest
import torch
from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

pytestmark = [pytest.mark.gpu]


@pytest.mark.skip(reason="Skipping NCCL broadcast as it fail only in ci")
def test_nccl_weight_transfer(free_port):
    host = "localhost"
    port = free_port()

    def send():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        group = NCCLWeightTransferEngine.trainer_init({
            "master_address": host,
            "master_port": port,
            "world_size": 2,
        })

        # Create model with known weights
        model = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device=device) for _ in range(10)]
        )
        for param in model.parameters():
            param.data = torch.ones_like(param.data)

        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=((n, p) for n, p in model.named_parameters()),
            group=group,
        )

    def receive():
        device = torch.device("cuda:1")
        torch.cuda.set_device(device)

        config = WeightTransferConfig(backend="nccl")
        parallel_config = ParallelConfig(world_size=1, rank=0, data_parallel_rank=0)
        engine = NCCLWeightTransferEngine(config, parallel_config)
        engine.init_transfer_engine(engine.parse_init_info({
            "master_address": host,
            "master_port": port,
            "rank_offset": 1,
            "world_size": 2,
        }))

        # Create a model with same architecture to get expected parameter info
        model = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10, device=device) for _ in range(10)]
        )
        names = [n for n, _ in model.named_parameters()]
        dtype_names = [str(p.dtype).replace("torch.", "") for _, p in model.named_parameters()]
        shapes = [list(p.shape) for _, p in model.named_parameters()]

        received_weights = []

        def load_weights(weights):
            received_weights.extend(weights)

        engine.receive_weights(
            engine.parse_update_info({
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "is_checkpoint_format": False,
            }),
            load_weights=load_weights,
        )

        for name, weight in received_weights:
            assert weight.allclose(torch.ones_like(weight)), f"Weight {name} mismatch"

    processes = [mp.Process(target=send), mp.Process(target=receive)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
        assert process.exitcode == 0, f"Process {process.name} exited with code {process.exitcode}"
