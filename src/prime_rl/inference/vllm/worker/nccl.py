from typing import TYPE_CHECKING

from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.logger import init_logger

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
        """Receive weights via NCCL using vLLM's built-in weight transfer engine.

        Metadata (names, dtypes, shapes) is received via NCCL from the trainer,
        so no out-of-band metadata passing is needed.
        """
        self.update_weights({"receive_metadata": True, "packed": self.packed})
