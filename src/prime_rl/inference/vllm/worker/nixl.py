import json
import os
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

# Directory for storing fetched LoRA adapters
NIXL_LORA_CACHE_DIR = Path(os.environ.get("NIXL_LORA_CACHE_DIR", "/tmp/nixl_lora_cache"))


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
        lora_alpha: float,
        timeout: float = 30.0,
    ) -> str:
        """Fetch LoRA weights from ALL trainers via NIXL and save to disk.

        Creates fresh NIXL client connections, fetches weights, saves to a
        deterministic path, and disconnects. Returns the path for loading
        via vLLM's standard LoRA mechanism.

        Args:
            lora_name: Name to register the LoRA adapter under
            run_idx: Run index in the MultiRunManager
            trainer_addresses: List of (ip, port) tuples, one per trainer rank
            lora_alpha: LoRA alpha scaling parameter
            timeout: Connection timeout in seconds

        Returns:
            Path to the saved LoRA adapter directory
        """
        logger.info(f"Fetching LoRA adapter '{lora_name}' from NIXL (run_idx={run_idx})")

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

            # Fetch weights and save to disk
            lora_path = self._fetch_and_save_weights(lora_name, run_idx, clients, lora_alpha)

        finally:
            # Always disconnect clients
            for client in clients:
                try:
                    client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting NIXL client: {e}")

        logger.info(f"Successfully fetched LoRA adapter '{lora_name}' from NIXL to {lora_path}")
        return lora_path

    def _fetch_and_save_weights(
        self,
        lora_name: str,
        run_idx: int,
        clients: list[ParameterClient],
        lora_alpha: float,
    ) -> str:
        """Fetch weights from NIXL clients and save to disk.

        Returns:
            Path to the saved LoRA adapter directory
        """
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

        # Save to deterministic path based on lora_name
        lora_dir = NIXL_LORA_CACHE_DIR / lora_name
        lora_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter config
        self._save_adapter_config(lora_dir, lora_alpha, reassembled_weights)

        logger.info(
            f"Saving {[k + ': ' + str(v.shape) for k, v in reassembled_weights.items()]} to {lora_dir / 'adapter_model.safetensors'}"
        )
        # Save weights as safetensors
        save_file(reassembled_weights, lora_dir / "adapter_model.safetensors")

        return str(lora_dir)

    def _save_adapter_config(self, lora_dir: Path, lora_alpha: float, weights: dict[str, torch.Tensor]) -> None:
        """Save adapter_config.json in PEFT format.

        Infers rank and target_modules from weights, uses provided lora_alpha.
        """
        target_modules = set()
        rank = None

        for key, tensor in weights.items():
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "lora_A":
                    if i > 0:
                        target_modules.add(parts[i - 1])
                    if rank is None:
                        rank = tensor.shape[0]
                elif part == "lora_B" and i > 0:
                    target_modules.add(parts[i - 1])

        adapter_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": sorted(list(target_modules)),
        }

        config_path = lora_dir / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)
        logger.info(f"Saved adapter config to {config_path}")
