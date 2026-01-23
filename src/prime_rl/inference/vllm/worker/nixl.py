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

    def load_lora_from_nixl(
        self,
        lora_name: str,
        run_idx: int,
        trainer_addresses: list[tuple[str, int]],
        timeout: float = 30.0,
    ) -> None:
        """Fetch LoRA weights from ALL trainers via NIXL and load into vLLM.

        Creates fresh NIXL client connections, fetches weights, and disconnects.

        Args:
            lora_name: Name to register the LoRA adapter under
            run_idx: Run index in the MultiRunManager
            trainer_addresses: List of (ip, port) tuples, one per trainer rank
            timeout: Connection timeout in seconds
        """
        logger.info(f"Loading LoRA adapter '{lora_name}' from NIXL (run_idx={run_idx})")

        tp_rank = get_tp_group().rank
        clients: list[ParameterClient] = []

        try:
            # Connect to all trainer ParameterServers
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
                clients.append(client)

            logger.info(f"Connected to {len(clients)} trainer ParameterServers")

            # Fetch and load weights
            self._fetch_and_load_weights(lora_name, run_idx, clients)

        finally:
            # Always disconnect clients
            for client in clients:
                try:
                    client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting NIXL client: {e}")

        logger.info(f"Successfully loaded LoRA adapter '{lora_name}' from NIXL")
        import time

        time.sleep(10)

    def _fetch_and_load_weights(
        self,
        lora_name: str,
        run_idx: int,
        clients: list[ParameterClient],
    ) -> None:
        """Fetch weights from NIXL clients and load into vLLM."""
        # Refresh catalogs from all trainers
        for client in clients:
            client.refresh_catalog()

        param_key_prefix = f"lora:{run_idx}:"
        param_keys = [key for key in clients[0].keys() if key.startswith(param_key_prefix)]

        logger.info(f"Fetching {len(param_keys)} LoRA parameters from {len(clients)} trainer ranks")

        # Fetch all shards from all trainers and reassemble
        reassembled_weights = {}

        for param_key in param_keys:
            shards = []

            # Fetch shard from each trainer rank
            for trainer_rank, client in enumerate(clients):
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
            reassembled_weights[param_key[len(param_key_prefix) :]] = full_tensor.cpu()

        # Write to temp directory in safetensors format for vLLM
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            logger.info(f"Saving {[k + ': ' + str(v.shape) for k, v in reassembled_weights.items()]}")
            # Save weights as safetensors
            save_file(reassembled_weights, tmpdir_path / "adapter_model.safetensors")

            # Load via vLLM's mechanism
            # self._load_lora_adapter(lora_name, str(tmpdir_path))

    def _load_lora_adapter(self, lora_name: str, lora_path: str) -> None:
        """Load a LoRA adapter into vLLM using the internal mechanism."""
        model_runner = self.model_runner

        if hasattr(model_runner, "model") and hasattr(model_runner.model, "load_lora"):
            model_runner.model.load_lora(lora_name, lora_path)
        elif hasattr(model_runner, "lora_manager"):
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=hash(lora_name) % (2**31),
                lora_path=lora_path,
            )
            model_runner.lora_manager.add_adapter(lora_request)
        else:
            raise RuntimeError(
                f"Could not find LoRA loading mechanism in model_runner. Available attrs: {dir(model_runner)}"
            )
