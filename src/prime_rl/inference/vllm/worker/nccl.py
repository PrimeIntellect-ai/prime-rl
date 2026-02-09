import pickle
from typing import TYPE_CHECKING, Generator, cast

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")


def receive_integer(communicator: PyNcclCommunicator) -> int:
    """Receive an integer from the trainer master rank using NCCL communicator."""
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(integer_tensor, src=0)
    return cast(int, integer_tensor.item())


def receive_state_dict(communicator: PyNcclCommunicator) -> Generator[tuple[str, torch.Tensor], None, None]:
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
        logger.info(f"Initializing NCCL broadcast receiver ({host}:{port}, rank={rank}, world_size={world_size})")

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
        self.communicator = PyNcclCommunicator(pg, device=device)

    @torch.no_grad()
    def receive_state_dict(self):
        """Receives the state dict of a model from the trainer master rank using NCCL communicator."""
        logger.info("Receiving weights from trainer")
        num_state_dict_to_receive = receive_integer(self.communicator)
        logger.info(f"Receiving {num_state_dict_to_receive} layer state dicts")
        for layer_id in range(num_state_dict_to_receive):
            logger.info(f"Receiving state dict {layer_id + 1}/{num_state_dict_to_receive}")
            for key, value in receive_state_dict(self.communicator):
                yield key, value


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using NCCL."""

    def init_broadcaster(self, host: str, port: int, server_rank: int, num_inference_server: int, timeout: int) -> None:
        """Initialize the NCCL broadcast receiver."""
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        global_rank_inference = (server_rank * tp_size * dp_size) + (dp_rank * tp_size) + tp_rank
        global_inference_world_size = num_inference_server * tp_size * dp_size

        logger.info(
            f"Worker [tp={tp_rank} dp={dp_rank} server_rank={server_rank}] -> [global_rank={global_rank_inference} global_world_size={global_inference_world_size}]"
        )

        self.nccl_broadcast_receiver = NCCLWeightBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,  # +1 as the trainer broadcaster is on rank 0
            world_size=global_inference_world_size + 1,  # +1 as the trainer broadcaster is on rank 0
            device=self.device,
            timeout=timeout,
        )

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Update weights with the nccl communicator."""
        model_runner = self.model_runner
        model = model_runner.model.runnable
        assert isinstance(model, Module)

        state_iter = self.nccl_broadcast_receiver.receive_state_dict()
        model.load_weights(state_iter)  # type: ignore

        # # Process weights after loading (important for some models)
        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)
