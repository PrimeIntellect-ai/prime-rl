import pickle

import torch
from torch.distributed.tensor import DTensor

from prime_rl.trainer.rl.broadcast.utils import init_tensor_from_string_description, tensor_string_description
from prime_rl.trainer.utils import get_world
from prime_rl.trainer.weights import convert_tt_to_hf_moe, has_tt_moe_layers


class NCCLBroadcastTrainer:
    def __init__(
        self, host: str, port: int, rank: int, world_size: int, device, logger, dtype: torch.dtype = torch.bfloat16
    ):
        self.logger = logger

        self.training_rank = get_world().rank

        if self.training_rank == 0:
            self.logger.info(
                f"Initializing NCCL broadcast ({host}:{port}, rank={get_world().rank}, world_size={world_size})"
            )
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup

            pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
            self.communicator = PyNcclCommunicator(pg, device=device)

            self.logger.info(f"NCCL broadcast initialized for rank {rank} and world size {world_size}")

        self.device = device
        self.dtype = dtype

    def broadcast_state_dict(self, model: torch.nn.Module) -> None:
        self.logger.debug("Broadcasting weights to inference pool")

        state_dict = model.state_dict()

        if has_tt_moe_layers(state_dict):
            convert_tt_to_hf_moe(state_dict)

        if self.training_rank == 0:
            state = pickle.dumps({key: tensor_string_description(value) for key, value in state_dict.items()})
            size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
            self.communicator.broadcast(size_tensor, src=0)
            state_tensor = torch.ByteTensor(list(state)).cuda()
            self.communicator.broadcast(state_tensor, src=0)

        # TODO(SAMI): there are two performance optimization we should do here:
        # 1. we should bucket more tensor into one broadcast call
        # 2. we should make sure both gather and broadcast are done in parallel

        for key, value in state_dict.items():
            if isinstance(value, DTensor):
                value = value.to(self.dtype)
                # only gather after the downcast to dtype as it will be faster
                value = value.full_tensor()
            else:
                value = value.to(self.dtype)

            if self.training_rank == 0:
                value = value.to(self.dtype)
                self.communicator.broadcast(value, src=0)

        self.logger.info("Weights broadcasted to inference pool")


class NCCLBroadcastInference:
    def __init__(
        self, host: str, port: int, rank: int, world_size: int, device, logger, dtype: torch.dtype = torch.bfloat16
    ):
        self.logger = logger

        self.logger.info(f"Initializing NCCL broadcast ({host}:{port}, rank={rank}, world_size={world_size})")
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.communicator = PyNcclCommunicator(pg, device=device)

        self.device = device
        self.dtype = dtype

    def receive_state_dict(self):
        size_tensor = torch.tensor([10], dtype=torch.long).to(self.device)
        self.communicator.broadcast(size_tensor, src=0)
        state_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8).to(self.device)
        self.communicator.broadcast(state_tensor, src=0)

        state = pickle.loads(bytes(state_tensor.cpu().numpy()))

        for key, value in state.items():
            tensor = init_tensor_from_string_description(value, self.device, self.dtype)
            self.communicator.broadcast(tensor, src=0)
            yield key, tensor
