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
    """vLLM worker extension that computes NCCL rank offsets and delegates to vLLM's built-in weight transfer engine."""

    packed: bool = True

    def init_broadcaster(self, host: str, port: int, server_rank: int, num_inference_server: int, timeout: int, packed: bool = True) -> None:
        """Initialize vLLM's native NCCL weight transfer engine with correct rank offsets."""
        tp_size = get_tp_group().world_size
        dp_size = get_dp_group().world_size
        workers_per_server = tp_size * dp_size
        rank_offset = 1 + server_rank * workers_per_server
        world_size = 1 + num_inference_server * workers_per_server

        logger.info(
            f"Initializing weight transfer engine "
            f"[server_rank={server_rank} rank_offset={rank_offset} world_size={world_size}]"
        )

        self.init_weight_transfer_engine({
            "master_address": host,
            "master_port": port,
            "rank_offset": rank_offset,
            "world_size": world_size,
        })
        self.packed = packed

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Receive weights via NCCL and load them directly into the model.

        Uses is_checkpoint_format=False to bypass vLLM's layerwise reload machinery,
        which causes excessive memory usage. Instead, we receive weights and feed them
        incrementally to model.load_weights(), then run process_weights_after_loading()
        ourselves — matching the old custom NCCL implementation.
        """
        engine = self.weight_transfer_engine
        update_info = engine.parse_update_info({
            "receive_metadata": True,
            "packed": self.packed,
            "is_checkpoint_format": False,
        })

        model = self.model_runner.model
        engine.receive_weights(update_info, load_weights=model.load_weights)

        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)
