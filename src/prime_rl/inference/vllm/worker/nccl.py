from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for prime-rl administrative probes."""

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None
