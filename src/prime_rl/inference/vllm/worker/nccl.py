from typing import TYPE_CHECKING

from torch.nn import Module
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from prime_rl.trainer.rl.broadcast.nccl_broadcast import NCCLBroadcastReceiver

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class NCCLWeightUpdateWorker(Worker):
    """
    This is an vLLM worker extension for updating weights to an updated RL policy model using NCCL.
    """

    def init_broadcaster(self, host: str, port: int, server_rank: int, num_inference_server: int, timeout: int) -> None:
        """Initialize the process group for NCCL broadcast."""
        logger = init_logger("vllm.inference.vllm.worker_nccl")
        self.tp_rank = get_tp_group().rank

        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank

        global_rank_inference = (server_rank * tp_size) + tp_rank
        global_inference_world_size = num_inference_server * tp_size

        logger.info(
            f"Worker [tp={tp_rank} server_rank={server_rank}] -> [global_rank={global_rank_inference} global_world_size={global_inference_world_size}]"
        )

        self.nccl_broadcast = NCCLBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,  # +1 as the trainer broadcaster is rank 0
            world_size=global_inference_world_size + 1,  # +1 for the trainer broadcaster
            device=self.device,
            logger=logger,
            timeout=timeout,
        )

    def update_weights(self, weight_dir: str) -> None:
        """Update weights with the nccl communicator."""
        model_runner = self.model_runner
        model = model_runner.model
        assert isinstance(model, Module)

        state_iter = self.nccl_broadcast.receive_state_dict()
        model.load_weights(state_iter)  # type: ignore

        # # Process weights after loading (important for some models)
        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)
