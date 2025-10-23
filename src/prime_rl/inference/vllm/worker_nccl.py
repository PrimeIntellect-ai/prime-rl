from typing import TYPE_CHECKING

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class NCCLBroadcastWorker(Worker):
    """
    This is an extension of a vLLM worker that allows for broadcasting weights
    using NCCL across multiple GPUs.
    """

    def init_broadcaster(self, host: str, port: int, rank: int, world_size: int) -> None:
        """Initialize the process group for NCCL broadcast."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.communicator = PyNcclCommunicator(pg, device=self.device)

        print(f"NCCL broadcast initialized for rank {rank} and world size {world_size}")

    def update_weights(self, weight_dir: str) -> None:
        """Update weights from a specified path pointing to a .pt file."""
        import torch

        tensor = torch.zeros(1000, dtype=torch.float32, device=self.device)
        self.communicator.broadcast(tensor, src=0)
        assert tensor.mean().item() == 1, "Tensor should be all ones"
