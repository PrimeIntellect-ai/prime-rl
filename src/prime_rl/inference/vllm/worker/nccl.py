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
    """vLLM worker extension for NCCL weight transfer via PrimeNCCLWeightTransferEngine."""

    def init_broadcaster(self, host: str, port: int, server_rank: int, num_inference_server: int, timeout: int, packed: bool = True) -> None:
        tp_size = get_tp_group().world_size
        dp_size = get_dp_group().world_size
        workers_per_server = tp_size * dp_size
        rank_offset = 1 + server_rank * workers_per_server
        world_size = 1 + num_inference_server * workers_per_server

        logger.info(f"Initializing weight transfer [rank_offset={rank_offset} world_size={world_size}]")

        self.init_weight_transfer_engine({
            "master_address": host,
            "master_port": port,
            "rank_offset": rank_offset,
            "world_size": world_size,
        })
        self._packed = packed

    def update_weights_from_path(self, weight_dir: str) -> None:
        # Metadata placeholder — PrimeNCCLWeightTransferEngine fills it from NCCL
        update_info = self.weight_transfer_engine.parse_update_info({
            "names": [], "dtype_names": [], "shapes": [],
            "packed": self._packed,
            "is_checkpoint_format": False,
        })

        model = self.model_runner.model
        self.weight_transfer_engine.receive_weights(update_info, load_weights=model.load_weights)

        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)
