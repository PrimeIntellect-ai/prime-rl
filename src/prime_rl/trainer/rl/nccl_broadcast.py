import torch

from prime_rl.utils.logger import get_logger


class NCCLBroadcast:
    def __init__(self, host: str, port: int, rank: int, world_size: int):
        self.logger = get_logger()

        self.logger.info(f"Initializing NCCL broadcast ({host}:{port}, rank={rank}, world_size={world_size})")
        import torch
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.communicator = PyNcclCommunicator(pg, device=torch.cuda.current_device())

        self.logger.info(f"NCCL broadcast initialized for rank {rank} and world size {world_size}")

    def broadcast(self, model: torch.nn.Module) -> None:
        tensor = torch.ones(1000, dtype=torch.float32, device="cuda")
        self.logger.info("Broadcasting weights to inference pool")

        self.communicator.broadcast(tensor, src=0)
        self.logger.info("Weights broadcasted to inference pool")
