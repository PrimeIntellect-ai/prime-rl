import pickle
from typing import TYPE_CHECKING, Generator, cast

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

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

# NemotronH mixer.D is dropped by vLLM 0.22's layerwise online-reload path (left uninitialized).
_RELOAD_CORRUPTED_SUFFIXES = (".mixer.D",)


def _restore_reload_corrupted_params(model: Module, received: dict[str, torch.Tensor]) -> None:
    """Work around a vLLM 0.22 layerwise-reload bug for NemotronH.

    The online reload drops the weight load for every Mamba layer's ``mixer.D`` (the SSD skip
    connection), leaving it as uninitialized ``empty_strided`` memory -- it reads back as garbage
    (NaN/inf) and the logits go NaN after a weight update. The received broadcast value is correct,
    so restore D from it via the param's own ``weight_loader``. Remove once the upstream bug is fixed.
    """

    def _layer_key(name: str) -> str:
        index = name.find("layers.")
        return name[index:] if index >= 0 else name

    received_by_key = {_layer_key(name): tensor for name, tensor in received.items()}
    for name, param in model.named_parameters():
        if not name.endswith(_RELOAD_CORRUPTED_SUFFIXES):
            continue
        tensor = received_by_key.get(_layer_key(name))
        if tensor is None:
            continue
        tensor = tensor.to(device=param.device)
        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is not None:
            weight_loader(param, tensor)
        elif tensor.shape == param.shape:
            param.data.copy_(tensor.to(param.dtype))


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

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
    ) -> None:
        """Initialize the NCCL broadcast receiver.

        Args:
            rank_offset: Starting GPU offset for this server in the global inference group.
            inference_world_size: Total number of inference GPUs across all servers.
        """
        # Use the worker's device index directly as the local rank.
        # The previous dp_group-based computation broke in vLLM v1 multiprocess
        # DP mode where each worker is a separate process with a singleton
        # DP group (rank_in_group is always 0).
        local_rank = self.device.index
        global_rank_inference = rank_offset + local_rank
        requested_config = {
            "host": host,
            "port": port,
            "rank_offset": rank_offset,
            "inference_world_size": inference_world_size,
            "timeout": timeout,
            "quantize_in_weight_transfer": quantize_in_weight_transfer,
            "local_rank": local_rank,
        }
        current_config = getattr(self, "_nccl_broadcaster_config", None)
        if current_config is not None:
            if current_config == requested_config:
                logger.info("NCCL broadcast receiver already initialized with matching config; skipping")
                self.quantize_in_weight_transfer = quantize_in_weight_transfer
                return
            raise RuntimeError(
                f"NCCL broadcast receiver already initialized with {current_config}; requested {requested_config}"
            )
        if hasattr(self, "nccl_broadcast_receiver"):
            raise RuntimeError("NCCL broadcast receiver already initialized without a recorded config")

        self.quantize_in_weight_transfer = quantize_in_weight_transfer

        logger.info(
            f"Worker [local_rank={local_rank} rank_offset={rank_offset}] "
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
        self._nccl_broadcaster_config = requested_config

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

        state_iter = self.nccl_broadcast_receiver.receive_state_dict()
        if self.quantize_in_weight_transfer:
            load_weights_kernel(model, state_iter)
            update_mla_absorbed_weights(model)
            return

        # vLLM 0.22's layerwise reload drops NemotronH mixer.D's weight load (see
        # _restore_reload_corrupted_params). Capture the correct received value to restore after.
        received_reload_fix: dict[str, torch.Tensor] = {}

        def _capture_reload_fix(weights):
            for name, tensor in weights:
                if name.endswith(_RELOAD_CORRUPTED_SUFFIXES):
                    received_reload_fix[name] = tensor.detach().to("cpu", copy=True)
                yield name, tensor

        load_weights_checkpoint_layerwise(
            model,
            _capture_reload_fix(state_iter),
            self.model_runner.model_config,
            self.vllm_config,
        )
        _restore_reload_corrupted_params(model, received_reload_fix)
