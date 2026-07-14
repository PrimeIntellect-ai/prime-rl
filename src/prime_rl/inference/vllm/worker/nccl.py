from typing import TYPE_CHECKING

from torch.nn import Module
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import (
    load_weights_checkpoint_layerwise,
    load_weights_kernel,
    update_mla_absorbed_weights,
)

# Shared receiver implementation (also used by the TTT service); re-exported here to keep
# the vLLM worker's public surface unchanged.
from prime_rl.utils.nccl_receiver import (  # noqa: F401
    NCCLWeightBroadcastReceiver,
    receive_integer,
    receive_state_dict,
)

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")


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
        ttt_world_size: int = 0,
    ) -> None:
        """Initialize the NCCL broadcast receiver.

        Args:
            rank_offset: Starting GPU offset for this server in the global inference group.
            inference_world_size: Total number of inference GPUs across all servers.
            ttt_world_size: Extra TTT-service receiver GPUs in the same group (they occupy
                ranks after all inference GPUs); every member must agree on the world size.
        """
        self.quantize_in_weight_transfer = quantize_in_weight_transfer
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
            world_size=inference_world_size + ttt_world_size + 1,  # +1 as the trainer broadcaster is on rank 0
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

        state_iter = self.nccl_broadcast_receiver.receive_state_dict()
        if self.quantize_in_weight_transfer:
            load_weights_kernel(model, state_iter)
            update_mla_absorbed_weights(model)
            return

        load_weights_checkpoint_layerwise(
            model,
            state_iter,
            self.model_runner.model_config,
            self.vllm_config,
        )
