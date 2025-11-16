import pickle
from typing import Generator

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.trainer.utils import get_world
from prime_rl.trainer.weights import convert_tt_layer_to_hf, get_max_layer_num, has_tt_moe_layers, quantize_layer_params


def create_nccl_communicator(
    host: str, port: int, rank: int, world_size: int, device: torch.device, timeout: int
) -> PyNcclCommunicator:
    pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
    return PyNcclCommunicator(pg, device=device)


def send_state_dict(state_dict: dict[str, torch.Tensor], communicator: PyNcclCommunicator | dist.ProcessGroup) -> None:
    """
    Get a state dict of tensor and broadcast it to the other ranks using NCCL.
    """
    # Group tensors by dtype
    dtype_groups: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for key, value in state_dict.items():
        assert not isinstance(value, DTensor), (
            "DTensor is not supported for broadcast, should have been converted to tensor already"
        )
        dtype = value.dtype
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append((key, value))

    # Build metadata: for each dtype group, store keys and shapes
    metadata = {}
    for dtype, items in dtype_groups.items():
        metadata[dtype] = [(key, value.shape, value.numel()) for key, value in items]

    # Send metadata
    state = pickle.dumps(metadata)
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.ByteTensor(list(state)).cuda()
    communicator.broadcast(state_tensor, src=0)

    # Concatenate and broadcast tensors grouped by dtype
    for dtype, items in dtype_groups.items():
        # Flatten all tensors and concatenate
        flat_tensors = [value.flatten() for _, value in items]
        concatenated = torch.cat(flat_tensors)
        communicator.broadcast(concatenated, src=0)
        del concatenated
        # Clean up individual tensors
        for _, value in items:
            del value


def receive_state_dict(
    communicator: PyNcclCommunicator | dist.ProcessGroup,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream tensors in a state dict broadcasted over NCCL."""
    size_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8).to(communicator.device)
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


def send_integer(integer: int, communicator: PyNcclCommunicator | dist.ProcessGroup) -> None:
    """
    Send an integer to the other ranks using NCCL.
    """
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    communicator.broadcast(integer_tensor, src=0)


def receive_integer(communicator: PyNcclCommunicator | dist.ProcessGroup) -> int:
    """
    Receive an integer from the other ranks using NCCL.
    """
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(integer_tensor, src=0)
    return integer_tensor.item()


def filter_state_dict_by_layers(
    state_dict: dict[str, torch.Tensor], num_layers: int
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """
    Yield a generator of state dicts for each layer as well as the remaining weights.
    """
    buckets: list[dict[str, torch.Tensor]] = [dict() for _ in range(num_layers + 1)]

    for key, value in state_dict.items():
        if key.startswith("model.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_idx = int(parts[2])
                if 1 <= layer_idx <= num_layers:
                    buckets[layer_idx][key] = value
                    continue
        buckets[0][key] = value

    for idx, bucket in enumerate(buckets):
        yield idx, bucket


class NCCLBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device,
        logger,
        timeout: int,
        dtype: torch.dtype = torch.bfloat16,
        quantization_config: dict | None = None,
    ):
        self.logger = logger

        self.training_rank = get_world().rank

        if self.training_rank == 0:
            self.communicator = create_nccl_communicator(host, port, rank, world_size, device, timeout)
            self.logger.info(f"NCCL broadcast initialized for rank {rank} and world size {world_size}")

        self.device = device
        self.dtype = dtype
        self.quantization_config = quantization_config

    @torch.no_grad()
    def broadcast_state_dict(self, model: torch.nn.Module) -> None:
        
        if self.training_rank != 0:
            return

        self.logger.debug("Broadcasting weights to inference pool")

        state_dict = model.state_dict()
        #
        num_layers = get_max_layer_num(state_dict)

        num_state_dict_to_send = num_layers + 1  # we send all layer plus the remaining weights
        
        send_integer(num_state_dict_to_send, self.communicator)

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} layer state dicts")

        for i, state_dict in filter_state_dict_by_layers(state_dict, num_layers):
            self.logger.debug(f"Sending layer {i}/{num_state_dict_to_send} state dict")
            for key, value in list(state_dict.items()):
                if isinstance(value, DTensor):
                    value = value.to(self.dtype).full_tensor()

                state_dict[key] = value

            if has_tt_moe_layers(state_dict):
                convert_tt_layer_to_hf(state_dict, i)

            # Quantize layer parameters if quantization config is provided
            if self.quantization_config is not None:
                quantize_layer_params(state_dict, i, self.quantization_config)

            send_state_dict(state_dict, self.communicator)

        self.logger.info("Weights broadcasted to inference pool")


class NCCLBroadcastReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device,
        logger,
        timeout: int,
    ):
        self.logger = logger

        self.logger.info(f"Initializing NCCL broadcast receiver ({host}:{port}, rank={rank}, world_size={world_size})")
        self.communicator = create_nccl_communicator(host, port, rank, world_size, device, timeout)

        self.device = self.communicator.device

    @torch.no_grad()
    def receive_state_dict(self):
        num_state_dict_to_receive = receive_integer(self.communicator)

        self.logger.info(f"Receiving {num_state_dict_to_receive} state dicts")
        for i in range(num_state_dict_to_receive):
            self.logger.info(
                f"Receiving state dict {i}/{num_state_dict_to_receive}, peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
            )
            for key, value in receive_state_dict(self.communicator):
                yield key, value
