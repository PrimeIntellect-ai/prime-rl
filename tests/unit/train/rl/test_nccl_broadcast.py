import multiprocessing as mp

import pytest
import torch

from prime_rl.trainer.rl.broadcast.nccl import NCCLBroadcastReceiver, NCCLBroadcastSender
from prime_rl.utils.logger import get_logger

pytestmark = [pytest.mark.gpu]


@pytest.mark.skip(reason="Skipping NCCL broadcast as it fail only in ci")
def test_nccl_broadcast(free_port):
    logger = get_logger()

    host = "localhost"

    def send():
        device = torch.device(f"cuda:{0}")

        logger.info("Sending weights")
        nccl_broadcast = NCCLBroadcastSender(
            host=host, port=free_port, rank=0, world_size=2, device=device, logger=logger, timeout=10
        )

        class SubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.layers = torch.nn.ModuleList([torch.nn.Linear(10, 10) for _ in range(10)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SubModel()

            def forward(self, x):
                return self.model(x)

        model = Model().to(device)
        for param in model.parameters():
            param.data = torch.ones_like(param.data)

        nccl_broadcast.broadcast_state_dict(model)

    def receive():
        device = torch.device(f"cuda:{1}")
        logger.info("Receiving weights")
        nccl_broadcast = NCCLBroadcastReceiver(
            host=host, port=free_port, rank=1, world_size=2, device=device, logger=logger, timeout=10
        )

        for key, value in nccl_broadcast.receive_state_dict():
            assert value.allclose(torch.ones_like(value))

    processes = [mp.Process(target=send), mp.Process(target=receive)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
        assert process.exitcode == 0, f"Process {process.name} exited with code {process.exitcode}"
