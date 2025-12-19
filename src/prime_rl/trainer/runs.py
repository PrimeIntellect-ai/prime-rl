from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import tomli

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    import torch

    from prime_rl.orchestrator.config import OrchestratorConfig
    from prime_rl.trainer.models.layers.lora import MultiLoRALinear


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0


class Runs:
    """This class stores information about the runs in the system."""

    def __init__(self, output_dir: Path, max_runs: int):
        self.output_dir = output_dir
        self.max_runs = max_runs
        self.logger = get_logger()

        self.idx_2_id: dict[int, str] = {}
        self.id_2_idx: dict[str, int] = {}
        self.unused_idxs = {i for i in range(self.max_runs)}

        self.progress: dict[int, Progress] = {}
        self.config: dict[int, "OrchestratorConfig"] = {}
        self.ready_to_update = [False] * max_runs

        self._deletion_hooks: list[Callable[[int, str], None]] = []
        self._creation_hooks: list[Callable[[int, str], None]] = []

        # Store modules with their FQN prefixes for parameter management
        self._modules: list[tuple[str, "MultiLoRALinear"]] = []

    def get_orchestrator_config(self, run_id: str) -> Optional["OrchestratorConfig"]:
        # Load orchestrator config first to validate it
        config_path = self.output_dir / run_id / "configs" / "orch.toml"
        config_dir = config_path.parent
        error_path = config_dir / "error.txt"

        if not config_path.exists():
            # Skip run if no config exists
            if not error_path.exists():
                config_dir.mkdir(parents=True, exist_ok=True)
                with open(error_path, "w") as f:
                    f.write(f"Error: No orchestrator config found at {config_path}\n")
            self.logger.error(f"Error: No orchestrator config found at {config_path}")
            return None

        try:
            # Import here to avoid circular dependency

            with open(config_path, "rb") as f:
                config_dict = tomli.load(f)

            # Parse config with Pydantic validation
            from prime_rl.orchestrator.config import OrchestratorConfig

            config = OrchestratorConfig(**config_dict)

            # Remove error file if it exists (config is now valid)
            if error_path.exists():
                error_path.unlink()

        except Exception as e:
            # Write error to file and skip this run
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(error_path, "w") as f:
                f.write(f"Error parsing orchestrator config:\n{str(e)}\n")
            self.logger.error(f"Error parsing orchestrator config for run {run_id}: {e}")
            return None

        return config

    def _delete_run(self, deleted_run: str, deleted_idx: int) -> None:
        # Call deletion hooks
        for hook in self._deletion_hooks:
            hook(deleted_idx, deleted_run)

        del self.progress[deleted_idx]
        if deleted_idx in self.config:
            del self.config[deleted_idx]
        self.ready_to_update[deleted_idx] = False

        # Process mappings
        self.unused_idxs.add(deleted_idx)
        del self.idx_2_id[deleted_idx]
        del self.id_2_idx[deleted_run]

    def _create_run(self, new_run: str, new_id: int, config: "OrchestratorConfig") -> None:
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

        # Reset parameters for the new run
        self.reset_run_parameters(new_id)

        # Call creation hooks
        for hook in self._creation_hooks:
            hook(new_id, new_run)

    def check_for_changes(self) -> None:
        run_ids = {run_path.stem for run_path in self.output_dir.glob("run_*")}
        deleted_runs = self.id_2_idx.keys() - run_ids
        new_runs = run_ids - self.id_2_idx.keys()

        for deleted_run in deleted_runs:
            deleted_idx = self.id_2_idx[deleted_run]
            self._delete_run(deleted_run, deleted_idx)

        for new_run in new_runs:
            try:
                # Process mappings
                new_id = next(iter(self.unused_idxs))

                config = self.get_orchestrator_config(new_run)
                if config is None:
                    continue

                # Now that config is valid, proceed with run setup
                self._create_run(new_run, new_id, config)
            except StopIteration:
                continue

    @property
    def used_idxs(self):
        return self.idx_2_id.keys()

    def run_dirs(self) -> list[Path]:
        return [self.output_dir / run_id for run_id in self.id_2_idx.keys()]

    def get_run_dir(self, idx: int) -> Path:
        return self.output_dir / self.idx_2_id[idx]

    def register_deletion_hook(self, hook: Callable[[int, str], None]) -> None:
        """Register a hook to be called when a run is deleted.

        Args:
            hook: A callable that takes (idx: int, run_id: str) as arguments.
                  Called when a run is deleted from the system.
        """
        self._deletion_hooks.append(hook)

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

    def get_named_parameters_for_run(self, idx: int) -> list[tuple[str, "torch.nn.Parameter"]]:
        """Get named parameters for a specific run index.

        Args:
            idx: The run index to get parameters for

        Returns:
            List of (name, parameter) tuples for the specified run index
        """
        params = []
        for prefix, module in self._modules:
            params.extend(
                [
                    (f"{prefix}.lora_A.{idx}", module.lora_A[idx]),
                    (f"{prefix}.lora_B.{idx}", module.lora_B[idx]),
                ]
            )
        return params

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


def setup_runs(output_dir: Path, max_runs: int):
    global _RUNS
    _RUNS = Runs(output_dir, max_runs)
