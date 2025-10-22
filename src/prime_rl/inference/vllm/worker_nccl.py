from typing import TYPE_CHECKING

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

    def init_process_group(self) -> None:
        """Initialize the process group for NCCL broadcast."""
        # torch.distributed.init_process_group(backend="nccl", device_id=torch.cuda.current_device())
        raise NotImplementedError("NCCL broadcast is not implemented yet.")

    def update_weights(self, weight_path: str) -> None:
        """Update weights from a specified path pointing to a .pt file."""
        # torch.distributed.broadcast(torch.tensor(np.array(pd.read_parquet(weight_path))), src=0)
        raise NotImplementedError("NCCL broadcast is not implemented yet.")
