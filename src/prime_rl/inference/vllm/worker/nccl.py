import json
from typing import TYPE_CHECKING

import torch
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for NCCL weight transfer with in-band metadata.

    Manages its own NCCL process group directly, bypassing vLLM's config-driven
    WeightTransferEngine creation (no WeightTransferConfig or Literal type changes needed).

    The trainer broadcasts weight metadata (names, dtypes, shapes) as a JSON header
    via NCCL before the weight tensors, so no out-of-band metadata passing is needed.
    """

    def init_broadcaster(self, host: str, port: int, server_rank: int, num_inference_server: int, timeout: int, packed: bool = True) -> None:
        """Initialize NCCL process group for weight transfer."""
        tp_size = get_tp_group().world_size
        dp_size = get_dp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dp_rank = get_dp_group().rank_in_group

        workers_per_server = tp_size * dp_size
        local_rank = dp_rank * tp_size + tp_rank
        rank = 1 + server_rank * workers_per_server + local_rank
        world_size = 1 + num_inference_server * workers_per_server

        logger.info(
            f"Initializing NCCL weight transfer "
            f"[server_rank={server_rank} rank={rank} world_size={world_size}]"
        )

        self._nccl_group = NCCLWeightTransferEngine._stateless_init_process_group(
            host, port, rank, world_size, torch.cuda.current_device()
        )
        self._packed = packed

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Receive metadata + weights via NCCL and load into the model.

        Protocol:
        1. Receive JSON metadata header (names, dtypes, shapes) via NCCL
        2. Receive weight tensors via NCCL (packed or unpacked)
        3. Feed weights to model.load_weights() incrementally
        4. Run process_weights_after_loading() for quantization etc.
        """
        metadata = self._receive_metadata()

        update_info = NCCLWeightTransferUpdateInfo(
            names=metadata["names"],
            dtype_names=metadata["dtype_names"],
            shapes=metadata["shapes"],
            packed=self._packed,
        )

        # Use a temporary engine instance just for receive_weights
        # (it only needs model_update_group to be set)
        engine = NCCLWeightTransferEngine.__new__(NCCLWeightTransferEngine)
        engine.model_update_group = self._nccl_group

        model = self.model_runner.model
        engine.receive_weights(update_info, load_weights=model.load_weights)

        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)

    def _receive_metadata(self) -> dict:
        """Receive weight metadata as a JSON header via NCCL broadcast."""
        group = self._nccl_group
        size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        group.broadcast(size_tensor, src=0, stream=torch.cuda.current_stream())
        metadata_tensor = torch.zeros(int(size_tensor.item()), dtype=torch.uint8, device="cuda")
        group.broadcast(metadata_tensor, src=0, stream=torch.cuda.current_stream())
        return json.loads(bytes(metadata_tensor.cpu().numpy()))
