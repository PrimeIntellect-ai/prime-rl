import threading
import time
from pathlib import Path

import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.config import LoRAConfig
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.config import NIXLWeightBroadcastConfig
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.world import get_world
from prime_rl.transport.nixl.parameter_server import ParameterServer
from prime_rl.utils.utils import get_broadcast_dir, get_step_path


class NIXLWeightBroadcast(WeightBroadcast):
    """Broadcast LoRA weights to inference servers via NIXL RDMA.

    Each trainer rank runs its own ParameterServer exposing its FSDP-sharded LoRA weights.
    Inference workers connect to ALL trainer ParameterServers to fetch all shards and
    reassemble the full weights.
    This allows the trainer to be passive and not have to coordinate with the inference servers.

    Tensors are registered once when a run is created and updated in-place by the optimizer.
    broadcast_weights() only updates metadata and signals readiness.

    Architecture:
        Trainer Rank 0 (ParameterServer :port+0) ──┐
        Trainer Rank 1 (ParameterServer :port+1) ──┼──> Each Inference Worker
        Trainer Rank 2 (ParameterServer :port+2) ──┤    fetches from ALL
        ...                                        │
        Trainer Rank N (ParameterServer :port+N) ──┘
    """

    def __init__(
        self,
        output_dir: Path,
        config: NIXLWeightBroadcastConfig,
        lora_config: LoRAConfig,
    ):
        super().__init__(output_dir, lora_config)
        self.config = config
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()

        # Each rank creates its own ParameterServer on a unique port
        self.port = config.port + self.world.rank
        self.server_name = f"{config.server_name}_{self.world.rank}"
        self.param_server = ParameterServer(self.server_name, port=self.port)

        # Start server in background thread
        self.server_thread = threading.Thread(target=self.param_server.run, daemon=True)
        self.server_thread.start()

        # Register MultiLoRA parameters for all runs
        for idx in range(self.multi_run_manager.max_runs):
            self._register_multilora_parameters(idx)

        self.logger.info(
            f"NIXL broadcast initialized (rank={self.world.rank}, server={self.server_name}, port={self.port})"
        )

    def _register_multilora_parameters(self, idx: int):
        """Register MultiLoRA tensors with ParameterServer"""
        named_params = self.multi_run_manager.get_named_parameters_for_run(idx)
        self.logger.debug(f"Registering {len(named_params)} NIXL tensors for run {idx}")

        for param_key, param in named_params:
            if isinstance(param, DTensor):
                # Get local shard from DTensor - this shares storage with the DTensor
                # so updates to the DTensor will be reflected here
                local_tensor = param.to_local()
                if not local_tensor.is_contiguous():
                    raise ValueError(f"Local tensor for {param_key} is not contiguous")
            else:
                if not param.is_contiguous():
                    raise ValueError(f"Parameter {param_key} is not contiguous")
                local_tensor = param.contiguous()

            # Key format without step - tensors are updated in-place
            nixl_key = f"lora:{idx}:{param_key}"
            self.param_server.put(nixl_key, local_tensor)

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Signal that LoRA weights are ready for fetching.

        Tensors are already registered and updated in-place by the optimizer.
        This method only updates metadata and writes the STABLE marker.
        """
        if not self.world.is_master:
            return
        self.logger.debug("Notifying NIXL weight readiness")
        start_time = time.perf_counter()

        for idx in self.multi_run_manager.ready_to_update_idxs:
            run_id = self.multi_run_manager.idx_2_id[idx]
            current_step = self.multi_run_manager.progress[idx].step
            self.logger.debug(
                f"Notifying NIXL weights ready for run {idx} (run_id={run_id}, step={current_step}, ready_to_update={self.multi_run_manager.ready_to_update[idx]})"
            )

            # Update metadata with current step
            save_dir = get_step_path(
                get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                current_step,
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            self._notify_orchestrator(save_dir)

            self.multi_run_manager.ready_to_update[idx] = False

        self.logger.debug(f"NIXL weights notified in {time.perf_counter() - start_time:.2f}s")

    def _notify_orchestrator(self, save_dir: Path):
        """Notify the orchestrator that the weights have been broadcast by writing a 'STABLE' file to a shared filesystem."""
        stable_file = save_dir / "STABLE"
        stable_file.touch()

    def maybe_clean(self, max_async_level: int, interval_to_keep: int | None) -> None:
        """Clean old broadcast directories (only filesystem metadata)."""
        from prime_rl.trainer.utils import maybe_clean

        for idx in self.multi_run_manager.used_idxs:
            maybe_clean(
                get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                self.multi_run_manager.progress[idx].step,
                max_async_level,
                interval_to_keep,
            )

    def shutdown(self) -> None:
        """Cleanup resources and stop the ParameterServer."""
        self.logger.info(f"Shutting down NIXL broadcast (rank={self.world.rank})")
        self.param_server.shutdown()
