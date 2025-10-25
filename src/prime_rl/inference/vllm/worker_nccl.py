from typing import TYPE_CHECKING

from vllm.distributed.parallel_state import get_dp_group, get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from prime_rl.trainer.rl.broadcast.nccl_broadcast import NCCLBroadcastInference

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class NCCLBroadcastWorker(Worker):
    """
    This is an extension of a vLLM worker that allows for broadcasting weights
    using NCCL across multiple GPUs.
    """

    def init_broadcaster(self, host: str, port: int, rank: int, world_size: int) -> None:
        """Initialize the process group for NCCL broadcast."""
        logger = init_logger("vllm.inference.vllm.worker_nccl")
        self.tp_rank = get_tensor_model_parallel_rank()
        self.dp_rank = get_dp_group().rank
        self.dp_world_size = get_dp_group().world_size

        logger.info(f"Worker TP rank: {self.tp_rank}")

        if self.tp_rank == 0:
            self.nccl_broadcast = NCCLBroadcastInference(
                host=host,
                port=port,
                rank=rank + self.dp_rank,
                world_size=(world_size - 1) * self.dp_world_size + 1,
                device=self.device,
                logger=logger,
            )

    def update_weights(self, weight_dir: str) -> None:
        """Update weights from a specified path pointing to a .pt file."""
        ...
        if self.tp_rank == 0:
            model_runner = self.model_runner
            model = model_runner.model

            self.nccl_broadcast.receive_state_dict()

            model.load_weights(self.nccl_broadcast.receive_state_dict())

            # # Process weights after loading (important for some models)
            device = next(model.parameters()).device
            process_weights_after_loading(model, self.model_runner.model_config, device)
