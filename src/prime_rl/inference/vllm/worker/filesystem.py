from typing import TYPE_CHECKING

from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("prime_rl.inference.vllm.worker.filesystem")


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem."""

    def init_broadcaster(self) -> None:
        """Initialize the broadcaster."""
        ...

    def update_weights(self, weight_path: str) -> None:
        """Update weights from a specified path in shared filesystem containing a HF-compatible checkpoint."""
        # Get vLLM model runner and model
        # When enforce_eager=True, model isn't wrapped by torch.compile so no .runnable attr
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        # Get vLLM model loader
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        local_source = DefaultModelLoader.Source(
            weight_path,
            revision=None,  # TODO: Check that this is correct or if we should use the default (model_config.revision)
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        weights_iterator = model_loader._get_weights_iterator(local_source)
        model.load_weights(weights_iterator)  # type: ignore

        # Process weights after loading (important for some models)
        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)

    def broadcast_to_peer(self, nccl_port: int = 29600) -> None:
        """
        Broadcast model weights to a peer pod via NCCL.

        Called when a newly scaled pod requests weights. This pod acts as the
        TCP store master (rank 0), the requesting peer is rank 1.

        Uses NCCL which auto-detects the best transport:
        - NVLink for same-node (~900 GB/s)
        - InfiniBand for cross-node with RDMA (~400 GB/s)
        - TCP sockets as fallback (~1-25 GB/s depending on network)

        Args:
            nccl_port: Port for NCCL communication
        """
        from prime_rl.inference.vllm.peer_broadcast import broadcast_weights_to_peer

        # Get model from worker
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        logger.info(f"Broadcasting weights to peer on port {nccl_port}")
        broadcast_weights_to_peer(model, nccl_port=nccl_port)

    def fetch_weights_from_peer(
        self,
        headless_service: str | None = None,
        namespace: str | None = None,
        peer_ip: str | None = None,
        http_port: int = 8000,
        nccl_port: int = 29600,
    ) -> bool:
        """
        Fetch model weights from a healthy peer pod via NCCL.

        Called when starting with --load-format dummy to quickly receive weights
        from an existing pod instead of loading from disk.

        Can use either K8s DNS discovery OR direct peer IP:
        - K8s mode: provide headless_service and namespace
        - Direct mode: provide peer_ip

        Args:
            headless_service: K8s headless service name for peer discovery (optional)
            namespace: K8s namespace (optional)
            peer_ip: Direct peer IP address to fetch from (optional)
            http_port: HTTP port for health checks and broadcast trigger
            nccl_port: Port for NCCL communication

        Returns:
            True if weights were fetched, False if no peer available
        """
        from prime_rl.inference.vllm.peer_broadcast import fetch_weights_from_peer

        # Get model from worker
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        if peer_ip:
            logger.info(f"Attempting to fetch weights from peer {peer_ip}")
        else:
            logger.info(f"Attempting to fetch weights from peer in {namespace}/{headless_service}")

        success = fetch_weights_from_peer(
            model,
            headless_service=headless_service,
            namespace=namespace,
            peer_ip=peer_ip,
            http_port=http_port,
            nccl_port=nccl_port,
        )

        if success:
            # Process weights after loading
            device = next(model.parameters()).device
            process_weights_after_loading(model, self.model_runner.model_config, device)
            logger.info("Weights from peer loaded and processed successfully")

        return success
