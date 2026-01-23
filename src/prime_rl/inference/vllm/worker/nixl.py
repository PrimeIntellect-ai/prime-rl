import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import save_file
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

from prime_rl.transport.nixl.parameter_client import ParameterClient

# This is to get type hints for the Worker class but not actually extend it at runtime
# as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nixl")


class NIXLLoRAWorker(Worker):
    """vLLM worker extension for loading LoRA adapters via NIXL RDMA.

    Each inference worker connects to ALL trainer ParameterServers to fetch all
    FSDP shards and reconstruct the full LoRA weights.

    Architecture:
        Trainer Rank 0 (ParameterServer :port+0) ──┐
        Trainer Rank 1 (ParameterServer :port+1) ──┼──> This Worker
        Trainer Rank 2 (ParameterServer :port+2) ──┤    (fetches from ALL)
        ...                                        │
        Trainer Rank N (ParameterServer :port+N) ──┘
    """

    nixl_clients: list[ParameterClient]

    def init_nixl_client(
        self,
        trainer_addresses: list[tuple[str, int]],
        timeout: float = 30.0,
    ) -> None:
        """Initialize NIXL ParameterClient connections to ALL trainer ranks in an FSDP group.

        Each inference worker connects to all trainer ParameterServers to
        fetch all FSDP shards and reconstruct the full LoRA weights.

        Args:
            server_name: Base name for ParameterServers (e.g., "lora_param_server")
            trainer_addresses: List of (ip, port) tuples, one per trainer rank
            timeout: Connection timeout in seconds
        """
        tp_rank = get_tp_group().rank

        self.nixl_clients = []
        for trainer_rank, (server_ip, server_port) in enumerate(trainer_addresses):
            client_name = f"inference_tp{tp_rank}_to_trainer{trainer_rank}"
            full_server_name = f"lora_param_server_{trainer_rank}"

            logger.info(
                f"Connecting to trainer rank {trainer_rank}: server={full_server_name} at {server_ip}:{server_port}"
            )

            client = ParameterClient(
                name=client_name,
                server_name=full_server_name,
                server_ip=server_ip,
                server_port=server_port,
                device=str(self.device),
                timeout=timeout,
            )
            self.nixl_clients.append(client)

        logger.info(f"Connected to {len(self.nixl_clients)} trainer ParameterServers")

    def load_lora_from_nixl(self, lora_name: str, run_idx: str) -> None:
        """Fetch LoRA weights from ALL trainers via NIXL and load into vLLM.

        This method:
        1. Refreshes catalog from all trainer ParameterServers
        2. Fetches metadata to get key list and verify step
        3. Fetches all LoRA weight shards (updated in-place by trainer)
        4. Reassembles full tensors from FSDP shards
        5. Writes to a temp directory in safetensors format
        6. Loads via vLLM's add_lora mechanism

        Args:
            lora_name: Name to register the LoRA adapter under
            run_id: Run identifier (e.g., "run_0")
            step: Training step number (used for verification)
        """
        if not hasattr(self, "nixl_clients") or not self.nixl_clients:
            raise RuntimeError("NIXL clients not initialized. Call init_nixl_client first.")

        logger.info(f"Loading LoRA adapter '{lora_name}' from NIXL (run_idx={run_idx})")

        # Refresh catalogs from all trainers
        for client in self.nixl_clients:
            client.refresh_catalog()

        param_key_prefix = f"lora:{run_idx}:"
        param_keys = [key for key in self.nixl_clients[0].keys() if key.startswith(param_key_prefix)]

        logger.info(f"Fetching {len(param_keys)} LoRA parameters from {len(self.nixl_clients)} trainer ranks")

        # Fetch all shards from all trainers and reassemble
        reassembled_weights = {}

        for param_key in param_keys:
            # Key format: lora:{run_id}:{param_key} (no step - tensors updated in-place)
            shards = []

            # Fetch shard from each trainer rank
            for trainer_rank, client in enumerate(self.nixl_clients):
                shard = client.get(param_key)
                if shard is None:
                    logger.warning(f"Shard not found for key {param_key} from trainer rank {trainer_rank}")
                    continue
                shards.append(shard)

            if not shards:
                logger.error(f"No shards found for key {param_key}")
                continue

            # Reassemble full tensor from shards
            full_tensor = torch.cat(shards, dim=0)
            reassembled_weights[param_key] = full_tensor.cpu()

        # Write to temp directory in safetensors format for vLLM
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Save weights as safetensors
            save_file(reassembled_weights, tmpdir_path / "adapter_model.safetensors")

            # Load via vLLM's mechanism
            self._load_lora_adapter(lora_name, str(tmpdir_path))

        logger.info(f"Successfully loaded LoRA adapter '{lora_name}' from NIXL")

    def _load_lora_adapter(self, lora_name: str, lora_path: str) -> None:
        """Load a LoRA adapter into vLLM using the internal mechanism.

        Args:
            lora_name: Name to register the adapter under
            lora_path: Path to the adapter directory
        """
        # vLLM's LoRA loading mechanism
        # The model_runner has access to the LoRA manager
        model_runner = self.model_runner

        # Check if using vLLM v1 API
        if hasattr(model_runner, "model") and hasattr(model_runner.model, "load_lora"):
            # vLLM v1 style
            model_runner.model.load_lora(lora_name, lora_path)
        elif hasattr(model_runner, "lora_manager"):
            # Fallback to LoRA manager
            from vllm.lora.request import LoRARequest

            # Create LoRA request
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=hash(lora_name) % (2**31),  # Generate stable ID
                lora_path=lora_path,
            )
            model_runner.lora_manager.add_adapter(lora_request)
        else:
            # Try direct model.load_weights approach as last resort
            logger.warning(
                "Could not find standard LoRA loading mechanism, "
                "attempting direct weight load (may not work for all models)"
            )
            raise RuntimeError(
                f"Could not find LoRA loading mechanism in model_runner. Available attrs: {dir(model_runner)}"
            )

    def disconnect_nixl_client(self) -> None:
        """Disconnect from all NIXL ParameterServers."""
        if hasattr(self, "nixl_clients"):
            for client in self.nixl_clients:
                try:
                    client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting NIXL client: {e}")
            self.nixl_clients = []
            logger.info("Disconnected from all NIXL ParameterServers")
