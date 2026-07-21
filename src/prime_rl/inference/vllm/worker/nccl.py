import pickle
from typing import TYPE_CHECKING, Generator, cast

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.ranks import global_inference_rank
from prime_rl.inference.vllm.worker.weight_transfer import (
    load_weights_checkpoint_layerwise,
    load_weights_kernel,
    update_mla_absorbed_weights,
)
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

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
        disable_nccl_p2p_if_unavailable()

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
        self.communicator = PyNcclCommunicator(pg, device=device)

    @torch.no_grad()
    def receive_state_dicts(
        self,
    ) -> Generator[Generator[tuple[str, torch.Tensor], None, None], None, None]:
        """Receive each trainer-broadcast state dict as a separate stream."""
        logger.info("Receiving weights from trainer")
        num_state_dict_to_receive = receive_integer(self.communicator)
        logger.info(f"Receiving {num_state_dict_to_receive} layer state dicts")
        for layer_id in range(num_state_dict_to_receive):
            logger.info(f"Receiving state dict {layer_id + 1}/{num_state_dict_to_receive}")
            yield receive_state_dict(self.communicator)

    @torch.no_grad()
    def receive_state_dict(self):
        """Receive trainer weights as one flat stream for kernel-format loading."""
        for state_dict in self.receive_state_dicts():
            yield from state_dict


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using NCCL."""

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
        engine_world_size: int | None = None,
    ) -> None:
        """Initialize the NCCL broadcast receiver.

        Args:
            rank_offset: Starting GPU offset for this server in the global inference group.
            inference_world_size: Total number of inference GPUs across all servers.
            engine_world_size: Number of inference GPUs assigned to this server.
        """
        self.quantize_in_weight_transfer = quantize_in_weight_transfer
        parallel_config = self.parallel_config
        data_parallel_index = parallel_config.data_parallel_index
        global_rank_inference = global_inference_rank(
            rank_offset=rank_offset,
            data_parallel_index=data_parallel_index,
            worker_rank=self.rank,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            prefill_context_parallel_size=getattr(parallel_config, "prefill_context_parallel_size", 1),
            inference_world_size=inference_world_size,
            engine_world_size=engine_world_size,
        )

        logger.info(
            f"Worker [worker_rank={self.rank} data_parallel_index={data_parallel_index} "
            f"rank_offset={rank_offset}] "
            f"-> [global_rank={global_rank_inference} inference_world_size={inference_world_size}]"
        )

        self.nccl_broadcast_receiver = NCCLWeightBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,  # +1 as the trainer broadcaster is on rank 0
            world_size=inference_world_size + 1,  # +1 as the trainer broadcaster is on rank 0
            device=self.device,
            timeout=timeout,
        )

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Update weights with the nccl communicator."""
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        if self.quantize_in_weight_transfer:
            state_iter = self.nccl_broadcast_receiver.receive_state_dict()
            load_weights_kernel(model, state_iter)
            update_mla_absorbed_weights(model)
            return

        for state_iter in self.nccl_broadcast_receiver.receive_state_dicts():
            load_weights_checkpoint_layerwise(
                model,
                state_iter,
                self.model_runner.model_config,
                self.vllm_config,
            )
