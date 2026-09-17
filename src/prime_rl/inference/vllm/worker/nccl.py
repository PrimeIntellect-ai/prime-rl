from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import load_weights_checkpoint_layerwise
from prime_rl.transports.wire import receive_integer, receive_state_dict
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")


def _reject_eplb(vllm_config) -> None:
    """The checkpoint-format reload maps experts with the *initial* EPLB
    assignment, while the router keeps routing on the current rebalanced map —
    after a rebalance this would silently write a logical expert into a
    physical slot owned by another one. Ordinary expert parallelism is fine:
    its mapping never changes."""
    if vllm_config.parallel_config.enable_eplb:
        raise ValueError(
            "NCCL weight transfer does not support enable_eplb=true: the "
            "checkpoint-format reload path maps expert weights with the initial "
            "EPLB assignment, not the live rebalanced one. Use plain "
            "enable_expert_parallel until live-map-aware reloading is implemented."
        )


def _validate_serialized_fp8_quantization(vllm_config) -> None:
    """The fp8 wire format (fp8 e4m3 weight + fp32 ``weight_scale_inv``) can
    only be applied by engines whose quantization config is a serialized
    blockwise-FP8 checkpoint layout — i.e. engines serving the model's
    blockwise-FP8 checkpoint. An engine quantizing a bf16 checkpoint on the
    fly (``quantization="fp8"`` without a serialized fp8 checkpoint) holds
    bf16-source online/per-tensor quant methods: it cannot consume fp8 codes
    plus block scales, and feeding it ``weight_scale_inv`` tensors fails the
    load mid-apply."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    quant_config = vllm_config.quant_config
    if not isinstance(quant_config, Fp8Config) or not quant_config.is_checkpoint_fp8_serialized:
        raise ValueError(
            "quantize_in_weight_transfer requires an engine initialized from a "
            "serialized blockwise-FP8 checkpoint (the model's FP8 release). This "
            "engine's quantization config does not match: fp8 wire tensors "
            "(weight + weight_scale_inv) cannot be applied by an engine that "
            "quantizes a bf16 checkpoint on the fly. Serve the FP8 checkpoint of "
            "the model or disable quantize_in_weight_transfer."
        )
    if list(quant_config.weight_block_size or []) != [128, 128]:
        raise ValueError(
            "quantize_in_weight_transfer requires [128, 128] fp8 weight blocks "
            "(the grid of the wire's weight_scale_inv tensors), got "
            f"{quant_config.weight_block_size}"
        )


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
        session_id: str = "default",
    ) -> None:
        """Initialize the NCCL broadcast receiver.

        Args:
            rank_offset: Starting GPU offset for this server in the global inference group.
            inference_world_size: Total number of inference GPUs across all servers.
        """
        del session_id
        self.quantize_in_weight_transfer = quantize_in_weight_transfer
        # Fail fast, before any rank can strand in a blocking NCCL read, when
        # this engine cannot safely apply the streamed checkpoint-format weights.
        _reject_eplb(self.vllm_config)
        if quantize_in_weight_transfer:
            _validate_serialized_fp8_quantization(self.vllm_config)
        # Use the worker's device index directly as the local rank.
        # The previous dp_group-based computation broke in vLLM v1 multiprocess
        # DP mode where each worker is a separate process with a singleton
        # DP group (rank_in_group is always 0).
        local_rank = self.device.index
        global_rank_inference = rank_offset + local_rank

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

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Update weights with the nccl communicator.

        The incoming stream is checkpoint-format weights — bf16 when
        ``quantize_in_weight_transfer`` is off, fp8 e4m3 plus fp32
        ``weight_scale_inv`` scales when it is on. Both flow through vLLM's own
        checkpoint weight-loading path (``model.load_weights`` under the
        layerwise reload lifecycle), which owns the TP/EP slicing, the fused
        parameter layouts, and the quantization post-processing — identical to
        how the engine loads weights from disk.
        """
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        if self.quantize_in_weight_transfer:
            _validate_serialized_fp8_quantization(self.vllm_config)

        state_iter = self.nccl_broadcast_receiver.receive_state_dict()
        load_weights_checkpoint_layerwise(
            model,
            state_iter,
            self.model_runner.model_config,
            self.vllm_config,
        )
