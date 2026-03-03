from typing import TYPE_CHECKING

from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using NCCL."""

    def init_broadcaster(self, host: str, port: int, server_rank: int, num_inference_server: int, timeout: int) -> None:
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        global_rank = (server_rank * tp_size * dp_size) + (dp_rank * tp_size) + tp_rank
        global_world_size = num_inference_server * tp_size * dp_size

        logger.info(
            f"Worker [tp={tp_rank} dp={dp_rank} server_rank={server_rank}] -> "
            f"[global_rank={global_rank} global_world_size={global_world_size}]"
        )

        self.init_weight_transfer_engine({
            "master_address": host,
            "master_port": port,
            "rank_offset": global_rank + 1,  # +1 as the trainer is on rank 0
            "world_size": global_world_size + 1,
        })

    def update_weights_from_path(self, weight_dir: str) -> None:
        model = self.model_runner.model
        self.weight_transfer_engine.receive_weights(None, load_weights=model.load_weights)

        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)
