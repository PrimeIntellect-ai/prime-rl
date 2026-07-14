"""Shared NCCL weight-broadcast receiver.

One implementation for every process that receives the trainer's full-state-dict
broadcast: the vLLM workers (``inference/vllm/worker/nccl.py`` re-exports from here) and
the TTT service ranks (``ttt/server_v2.py``). The vLLM communicator wrappers are imported
lazily so importing this module stays cheap for non-receiving callers.
"""

import pickle
from typing import TYPE_CHECKING, Generator, cast

import torch

from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def receive_integer(communicator: "PyNcclCommunicator") -> int:
    """Receive an integer from the trainer master rank using NCCL communicator."""
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(integer_tensor, src=0)
    return cast(int, integer_tensor.item())


def receive_state_dict(communicator: "PyNcclCommunicator") -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream tensors in a state dict broadcasted over NCCL."""
    size_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8).to(communicator.device)
    communicator.broadcast(state_tensor, src=0)

    metadata = pickle.loads(bytes(state_tensor.cpu().numpy()))

    # Receive concatenated tensors per dtype and split them back
    for dtype, tensor_info_list in metadata.items():
        # Receive concatenated tensor for this dtype
        total_elements = sum(numel for _, _, numel in tensor_info_list)
        concatenated = torch.empty(total_elements, dtype=dtype, device=communicator.device)
        communicator.broadcast(concatenated, src=0)

        # Split concatenated tensor back into individual tensors
        offset = 0
        for key, shape, numel in tensor_info_list:
            tensor = concatenated[offset : offset + numel].view(shape).clone()
            offset += numel
            try:
                yield key, tensor
            finally:
                del tensor

        del concatenated


class NCCLWeightBroadcastReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
    ):
        # Lazy: vLLM is heavy and only receiver processes need its communicator wrappers.
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        get_logger().info(f"Initializing NCCL broadcast receiver ({host}:{port}, rank={rank}, world_size={world_size})")
        disable_nccl_p2p_if_unavailable()

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
        self.communicator = PyNcclCommunicator(pg, device=device)

    @torch.no_grad()
    def receive_state_dict(self):
        """Receives the state dict of a model from the trainer master rank using NCCL communicator."""
        logger = get_logger()
        logger.info("Receiving weights from trainer")
        num_state_dict_to_receive = receive_integer(self.communicator)
        logger.info(f"Receiving {num_state_dict_to_receive} layer state dicts")
        for layer_id in range(num_state_dict_to_receive):
            logger.info(f"Receiving state dict {layer_id + 1}/{num_state_dict_to_receive}")
            for key, value in receive_state_dict(self.communicator):
                yield key, value
