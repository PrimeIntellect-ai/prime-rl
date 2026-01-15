import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import tomli
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.config import OrchestratorConfig
    from prime_rl.trainer.models.layers.lora import MultiLoRALinear


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0


class Runs:
    """This class stores information about the runs in the system."""

    def __init__(self, output_dir: Path, max_runs: int, device: torch.device = torch.device("cpu")):
        self.output_dir = output_dir
        self.max_runs = max_runs
        self.logger = get_logger()

        self.idx_2_id: dict[int, str] = {}
        self.id_2_idx: dict[str, int] = {}
        self.unused_idxs = {i for i in range(self.max_runs)}

        self.progress: dict[int, Progress] = {}
        self.config: dict[int, "OrchestratorConfig"] = {}
        self.ready_to_update = [False] * max_runs

        self._creation_hooks: list[Callable[[int, str], None]] = []
        self._create_run_data_hooks: list[Callable[[int, str, "OrchestratorConfig"], None]] = []
        self._delete_run_data_hooks: list[Callable[[int, str], None]] = []
        self._config_validation_hooks: list[Callable[["OrchestratorConfig"], tuple[bool, str]]] = []
        self._scaling_hook: Callable[["OrchestratorConfig"], float] | None = None

        # We use the store to keep other ranks in sync with master
        self.store = c10d._get_default_store()
        self.world = get_world()
        # Track id_2_idx state at last sync_runs to calculate diffs
        self._last_synced_id_2_idx: dict[str, int] = {}

        # Store modules with their FQN prefixes for parameter management
        self._modules: list[tuple[str, "MultiLoRALinear"]] = []

        # Initialize lora globals on device so runs.* ARE the global tensors
        from prime_rl.trainer.models.layers.lora import (
            get_lora_num_tokens,
            get_multilora_scaling,
            set_lora_num_tokens,
            set_multilora_scaling,
        )

        # We first set to None so we dont check the shape
        # Might be better to not have the check at all but this will do for now
        set_lora_num_tokens(None, reset_reference=True)
        set_multilora_scaling(None, reset_reference=True)

        set_lora_num_tokens(torch.zeros(max_runs, dtype=torch.int32, device=device), reset_reference=True)
        self.lora_num_tokens = get_lora_num_tokens()

        set_multilora_scaling(
            torch.ones(max_runs, dtype=torch.bfloat16, device=device) * 1000_000.0, reset_reference=True
        )
        self.scaling_factors = get_multilora_scaling()

    def register_config_validation_hook(self, hook: Callable[["OrchestratorConfig"], tuple[bool, str]]) -> None:
        """Register a hook to validate orchestrator config when a run is created.

        Args:
            hook: A callable that takes (config: OrchestratorConfig) and returns
                  (is_valid: bool, error_message: str). Error message is used when invalid.
        """
        self._config_validation_hooks.append(hook)

    def register_scaling_hook(self, hook: Callable[["OrchestratorConfig"], float]) -> None:
        """Register a hook to compute LoRA scaling factor for a run.

        Args:
            hook: A callable that takes (config: OrchestratorConfig) and returns the scaling factor.
        """
        self._scaling_hook = hook

    def register_create_run_data_hook(self, hook: Callable[[int, str, "OrchestratorConfig"], None]) -> None:
        """Register a hook to be called when run data is created (master only).

        Args:
            hook: A callable that takes (idx: int, run_id: str, config: OrchestratorConfig).
                  Called only on master rank when a new run's data structures are initialized.
        """
        self._create_run_data_hooks.append(hook)

    def register_delete_run_data_hook(self, hook: Callable[[int, str], None]) -> None:
        """Register a hook to be called when run data is deleted (master only).

        Args:
            hook: A callable that takes (idx: int, run_id: str).
                  Called only on master rank when a run's data structures are being deleted.
        """
        self._delete_run_data_hooks.append(hook)

    def get_orchestrator_config(self, run_id: str) -> Optional["OrchestratorConfig"]:
        """Load and validate orchestrator config for a run.

        Returns None if config doesn't exist, fails to parse, or fails validation.
        Writes error to config dir for orchestrator to consume.
        """
        config_path = self.output_dir / run_id / "configs" / "orch.toml"
        config_dir = config_path.parent
        error_path = config_dir / "error.txt"

        if not config_path.exists():
            if not error_path.exists():
                config_dir.mkdir(parents=True, exist_ok=True)
                with open(error_path, "w") as f:
                    f.write(f"No orchestrator config found at {config_path}\n")
            self.logger.error(f"Run {run_id}: No orchestrator config found at {config_path}")
            return None

        try:
            with open(config_path, "rb") as f:
                config_dict = tomli.load(f)

            from prime_rl.orchestrator.config import OrchestratorConfig

            config = OrchestratorConfig(**config_dict)
        except Exception as e:
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(error_path, "w") as f:
                f.write(f"Error parsing orchestrator config:\n{str(e)}\n")
            self.logger.error(f"Run {run_id}: Error parsing orchestrator config: {e}")
            return None

        # Run registered validation hooks
        for hook in self._config_validation_hooks:
            is_valid, error_message = hook(config)
            if not is_valid:
                self.logger.error(f"Run {run_id}: {error_message}")
                config_dir.mkdir(parents=True, exist_ok=True)
                with open(error_path, "w") as f:
                    f.write(f"{error_message}\n")
                return None

        # Config is valid, remove any stale error file
        if error_path.exists():
            error_path.unlink()

        return config

    def _delete_run_data(self, deleted_run: str, deleted_idx: int) -> None:
        """Update data structures for a deleted run"""
        # Call delete hooks before removing data
        for hook in self._delete_run_data_hooks:
            hook(deleted_idx, deleted_run)

        del self.progress[deleted_idx]
        if deleted_idx in self.config:
            del self.config[deleted_idx]

        # A big value might help make it error loudly
        self.scaling_factors[deleted_idx] = 1000_000.0

        # Process mappings
        self.unused_idxs.add(deleted_idx)
        del self.idx_2_id[deleted_idx]
        del self.id_2_idx[deleted_run]

    def _create_run_data(self, new_run: str, new_id: int, config: "OrchestratorConfig") -> None:
        """Update data structures for a new run (no hooks or param reset)."""
        self.id_2_idx[new_run] = new_id
        self.unused_idxs.remove(new_id)
        self.idx_2_id[new_id] = new_run

        # Get progress
        self.progress[new_id] = Progress()

        prev_ckpt_steps = [
            int(i.stem.split("_")[-1]) for i in (self.get_run_dir(new_id) / "checkpoints").glob("step_*")
        ]
        self.progress[new_id].step = max(prev_ckpt_steps) if prev_ckpt_steps else 0

        # Store the parsed config
        self.config[new_id] = config

    def _create_run_hooks(self, new_id: int, new_run: str) -> None:
        """Reset parameters and call creation hooks for a run."""
        self.reset_run_parameters(new_id)
        for hook in self._creation_hooks:
            hook(new_id, new_run)

    def check_for_changes(self) -> None:
        """Detect run changes and update data structures. Must be followed by sync_runs().

        Only updates mappings and data structures. Hooks and parameter resets
        are deferred to sync_runs() so all ranks execute them together.
        """
        run_ids = {run_path.stem for run_path in self.output_dir.glob("run_*")}
        deleted_runs = self.id_2_idx.keys() - run_ids
        new_runs = run_ids - self.id_2_idx.keys()

        for deleted_run in deleted_runs:
            deleted_idx = self.id_2_idx[deleted_run]
            self._delete_run_data(deleted_run, deleted_idx)

        for new_run in new_runs:
            try:
                new_id = next(iter(self.unused_idxs))

                config = self.get_orchestrator_config(new_run)
                if config is None:
                    continue

                self._create_run_data(new_run, new_id, config)
                for hook in self._create_run_data_hooks:
                    hook(new_id, new_run, config)
            except StopIteration:
                continue

    def sync_runs(self) -> None:
        """Sync run state across ranks and execute hooks.

        Master calculates what changed since last sync using _last_synced_id_2_idx.
        This matches what non-master ranks will calculate as new/deleted runs.
        All ranks then execute hooks and parameter resets together.
        """

        if self.world.is_master:
            # Compute scaling factors for new runs before sync
            new_runs = self.id_2_idx.keys() - self._last_synced_id_2_idx.keys()
            for new_run in new_runs:
                new_id = self.id_2_idx[new_run]
                if self._scaling_hook is not None:
                    self.scaling_factors[new_id] = self._scaling_hook(self.config[new_id])

            # Include configs for new runs so non-master ranks have them
            new_configs = {new_run: self.config[self.id_2_idx[new_run]] for new_run in new_runs}

            sync_data = {
                "id_2_idx": self.id_2_idx,
                "ready_to_update": self.ready_to_update,
                "scaling_factors": self.scaling_factors.cpu(),
                "new_configs": new_configs,
            }
            self.store.set("runs", pickle.dumps(sync_data))
        dist.barrier()

        if self.world.is_master:
            # Calculate changes since last sync (this is what other ranks will see)
            new_runs = self.id_2_idx.keys() - self._last_synced_id_2_idx.keys()
            deleted_runs = self._last_synced_id_2_idx.keys() - self.id_2_idx.keys()
        else:
            sync_data: dict = pickle.loads(self.store.get("runs"))
            new_id_2_idx: dict[str, int] = sync_data["id_2_idx"]
            self.ready_to_update = sync_data["ready_to_update"]
            self.scaling_factors.copy_(sync_data["scaling_factors"])
            new_configs: dict[str, "OrchestratorConfig"] = sync_data["new_configs"]

            new_runs = new_id_2_idx.keys() - self.id_2_idx.keys()
            deleted_runs = self.id_2_idx.keys() - new_id_2_idx.keys()

            # Other ranks catch up with master's data state
            for deleted_run in deleted_runs:
                deleted_idx = self.id_2_idx[deleted_run]
                self._delete_run_data(deleted_run, deleted_idx)

            for new_run in new_runs:
                new_id = new_id_2_idx[new_run]
                self._create_run_data(new_run, new_id, new_configs[new_run])

        for new_run in new_runs:
            new_id = self.id_2_idx[new_run]
            self._create_run_hooks(new_id, new_run)

        # Update last synced state for master
        if self.world.is_master:
            self._last_synced_id_2_idx = self.id_2_idx.copy()

    @property
    def used_idxs(self):
        return sorted(self.idx_2_id.keys())

    @property
    def ready_to_update_idxs(self):
        return [idx for idx, ready in enumerate(self.ready_to_update) if ready]

    def run_dirs(self) -> list[Path]:
        return [self.output_dir / run_id for run_id in self.id_2_idx.keys()]

    def get_run_dir(self, idx: int) -> Path:
        return self.output_dir / self.idx_2_id[idx]

    def register_creation_hook(self, hook: Callable[[int, str], None]) -> None:
        """Register a hook to be called when a new run is created.

        Args:
            hook: A callable that takes (idx: int, run_id: str) as arguments.
                  Called when a new run is added to the system.
        """
        self._creation_hooks.append(hook)

    def register_module(self, prefix: str, module: "MultiLoRALinear") -> None:
        """Register a MultiLoRALinear module with its FQN prefix.

        This allows Runs to manage parameter access, reset, and state dict slicing
        for multi-adapter LoRA modules.

        Args:
            prefix: The module's fully qualified name in the model
                   (e.g., "model.layers.0.self_attn.q_proj")
            module: The MultiLoRALinear module to register
        """
        self._modules.append((prefix, module))

    def get_named_parameters_for_run(self, idx: int) -> list[tuple[str, torch.nn.Parameter]]:
        """Get named parameters for a specific run index.

        Args:
            idx: The run index to get parameters for

        Returns:
            List of (name, parameter) tuples for the specified run index
        """
        params = []
        for prefix, module in self._modules:
            for name, param in module.named_parameters_for_adapter(idx):
                params.append((f"{prefix}.{name}.weight", param))
        return params

    def get_state_dict_for_run(self, idx: int) -> dict[str, torch.Tensor]:
        """Get state dict for a specific run index.

        Args:
            idx: The run index to get state dict for

        Returns:
            State dict for the specified run index
        """
        state_dict = {}
        for prefix, module in self._modules:
            # Check if module has a custom state_dict_for_adapter method (e.g., MoE modules)
            # which returns vLLM-compatible per-expert format
            if hasattr(module, "state_dict_for_adapter"):
                for name, tensor in module.state_dict_for_adapter(idx).items():
                    state_dict[f"{prefix}.{name}"] = tensor.detach()
            else:
                # Default: use named_parameters_for_adapter
                for name, param in module.named_parameters_for_adapter(idx):
                    state_dict[f"{prefix}.{name}.weight"] = param.detach()
        return state_dict

    def reset_run_parameters(self, idx: int) -> None:
        """Reset parameters for a specific run index.

        Called when a new run is created to initialize fresh adapter weights.

        Args:
            idx: The run index to reset parameters for
        """
        for _, module in self._modules:
            module.reset_parameters(idx)

    def __repr__(self):
        return f"Runs(max={self.max_runs})[{self.idx_2_id.keys()}]"


# Singleton instance of Tenants
_RUNS: Runs | None = None


def get_runs() -> Runs:
    """Returns the World. If not initialized, it will initialize."""
    global _RUNS
    if _RUNS is None:
        raise RuntimeError("Runs not initialized. Please call `setup_runs` first.")
    return _RUNS


def setup_runs(output_dir: Path, max_runs: int, device: torch.device):
    global _RUNS
    _RUNS = Runs(output_dir, max_runs, device)
