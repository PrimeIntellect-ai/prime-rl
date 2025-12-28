# Runs

The `Runs` object is a singleton that manages multiple concurrent training runs within a single trainer process. It is the central coordination point for multi-run RL training, enabling a single trainer to serve multiple orchestrator experiments simultaneously with separate LoRA adapters, optimizers, and schedulers.

## Overview

When `max_concurrent_runs > 1`, the trainer can train multiple runs in parallel. Each run:
- Has its own LoRA adapter weights
- Has its own optimizer and scheduler
- Tracks its own training progress (step, tokens, samples)
- Loads its own orchestrator configuration

The `Runs` object provides:
- **Bidirectional mapping** between run IDs (e.g., `run_abc123`) and run indices (0, 1, 2, ...)
- **Progress tracking** per run (step count, total tokens, total samples)
- **Configuration management** for orchestrator configs
- **Distributed synchronization** across ranks via the PyTorch distributed store
- **LoRA module registration** for multi-adapter parameter management
- **Creation hooks** for initializing per-run resources (optimizers, schedulers)

## Initialization

The `Runs` singleton is set up at the start of training:

```python
from prime_rl.trainer.runs import setup_runs, get_runs

# Initialize with output directory and max concurrent runs
setup_runs(output_dir=Path("outputs/my-experiment"), max_runs=4)

# Get the singleton instance anywhere in the codebase
runs = get_runs()
```

## Run Discovery

Runs are discovered by scanning the output directory for directories matching the pattern `run_*`. Each run must contain a valid orchestrator config at `{run_dir}/configs/orch.toml`.

```python
# Master rank scans for new/deleted runs
runs.check_for_changes()

# All ranks synchronize state (must be called after check_for_changes)
runs.sync_runs()
```

The `check_for_changes()` method:
1. Scans the output directory for `run_*` directories
2. Detects new runs and deleted runs
3. Loads and validates the orchestrator config for each new run
4. Updates internal mappings and data structures

The `sync_runs()` method:
1. Master broadcasts run state to all ranks via the distributed store
2. All ranks execute creation hooks for new runs (e.g., optimizer setup)
3. All ranks reset LoRA parameters for new runs

## Key Properties and Methods

### Mappings

```python
runs.idx_2_id  # dict[int, str] - Index to run ID mapping
runs.id_2_idx  # dict[str, int] - Run ID to index mapping
runs.used_idxs  # Set of currently active run indices
runs.unused_idxs  # Set of available run indices
runs.max_runs  # Maximum number of concurrent runs
```

### Progress Tracking

```python
runs.progress[idx]  # Progress dataclass for run index
runs.progress[idx].step  # Current training step
runs.progress[idx].total_tokens  # Total tokens trained
runs.progress[idx].total_samples  # Total samples trained
```

### Run State

```python
runs.ready_to_update  # list[bool] - Which runs have data ready for training
runs.ready_to_update_idxs  # List of indices ready for training step
runs.config[idx]  # OrchestratorConfig for run index
```

### Paths

```python
runs.output_dir  # Base output directory
runs.get_run_dir(idx)  # Path to run's directory
runs.run_dirs()  # List of all run directory paths
```

## Component Integration

### MultiLoRAOptimizer

The `MultiLoRAOptimizer` creates and manages per-run optimizers. It registers a creation hook that is called when new runs are discovered:

```python
class MultiLoRAOptimizer:
    def __init__(self, config, device_mesh, model=None):
        self.runs = get_runs()
        self.optimizers: list[Optimizer | None] = [None] * self.runs.max_runs
        
        # Register hook for optimizer creation
        self.runs.register_creation_hook(self.optimizer_creation_hook)

    def optimizer_creation_hook(self, idx: int, run_id: str) -> None:
        # Get named parameters for this run from the Runs system
        named_params = self.runs.get_named_parameters_for_run(idx)
        self.optimizers[idx] = _setup_optimizer(self.config, named_params, self.device_mesh)

    def step(self):
        # Only step optimizers for runs with data
        for idx in self.runs.ready_to_update_idxs:
            self.optimizers[idx].step()
```

### MultiLoRAScheduler

Similar to `MultiLoRAOptimizer`, the `MultiLoRAScheduler` manages per-run learning rate schedulers:

```python
class MultiLoRAScheduler:
    def __init__(self, scheduler_config, max_steps, lr):
        self.runs = get_runs()
        self.schedulers: list[LRScheduler | None] = [None] * self.runs.max_runs

    def scheduler_creation_hook(self, optimizer: Optimizer, idx: int) -> None:
        self.schedulers[idx] = setup_scheduler(optimizer, self.scheduler_config, ...)

    def step(self) -> None:
        for idx in self.runs.ready_to_update_idxs:
            self.schedulers[idx].step()
```

### Packer

The `Packer` (on master rank) detects run changes and assembles training batches:

```python
class Packer:
    def __init__(self, ...):
        self.runs = get_runs()

    def get_batch(self) -> dict[int, TrainingBatch]:
        # Check for new/deleted runs
        self.runs.check_for_changes()
        batches = self.receiver.receive()
        return {batch.run_idx: batch for batch in batches}

    def pack(self):
        training_batches = self.get_batch()
        
        for idx, training_batch in training_batches.items():
            # Update progress for this run
            self.runs.progress[idx].step += 1
            self.runs.progress[idx].total_tokens += ...
            self.runs.progress[idx].total_samples += ...
            
            # Mark run as ready for training step
            self.runs.ready_to_update[idx] = True
```

### DataLoader

The `DataLoader` synchronizes runs before returning batches:

```python
class DataLoader:
    def __init__(self, ...):
        self.runs = get_runs()

    def wait_for_batch(self) -> None:
        if self.world.is_master:
            self.packer.pack()
        self.receiver.wait()
        # Sync run state across all ranks
        self.runs.sync_runs()
```

### LoRA Module Registration

LoRA modules register themselves with `Runs` for parameter management:

```python
# In apply_lora_to_model()
lora_module = MultiLoRALinear(
    base_layer=base_module,
    rank=config.rank,
    n_adapters=get_runs().max_runs,  # One adapter per run
    ...
)
lora_module.register_with_runs(get_runs(), module_name)
```

The `Runs` object then provides:

```python
# Get parameters for a specific run (used by optimizer)
runs.get_named_parameters_for_run(idx)

# Get state dict for a specific run (used for weight broadcast)
runs.get_state_dict_for_run(idx)

# Reset parameters for a new run
runs.reset_run_parameters(idx)
```

### Weight Broadcast

The `FileSystemWeightBroadcast` saves weights per run:

```python
class FileSystemWeightBroadcast:
    def broadcast_weights(self, model, step, adapter_only=False):
        for idx in self.runs.used_idxs:
            if not self.runs.ready_to_update[idx]:
                continue
            
            if adapter_only:
                # Get state dict for this specific run's adapter
                state_dict = self.runs.get_state_dict_for_run(idx)
            
            save_dir = get_step_path(
                get_broadcast_dir(self.runs.get_run_dir(idx)),
                self.runs.progress[idx].step
            )
            save_state_dict(state_dict, save_dir, ...)
```

## Training Loop Integration

In the training loop (`rl/train.py`), the `Runs` object coordinates all components:

```python
def train(config: RLTrainerConfig):
    # Setup runs singleton
    setup_runs(config.output_dir, config.max_concurrent_runs)
    runs = get_runs()
    
    # Setup multi-optimizer (registers creation hook)
    optimizer = setup_multi_optimizer(config.optim, device_mesh, model=None)
    
    # Setup multi-scheduler (also uses runs)
    scheduler = setup_multi_scheduler(config.scheduler, config.max_steps, config.optim.lr)
    optimizer.register_post_creation_callback(scheduler.scheduler_creation_hook)
    
    while True:
        # DataLoader.wait_for_batch() calls:
        #   - packer.pack() which calls runs.check_for_changes()
        #   - runs.sync_runs() to synchronize all ranks
        dataloader.wait_for_batch()
        
        # Forward/backward pass uses lora_num_tokens from batch
        # to route tokens to correct adapters
        
        # Step only active runs
        optimizer.step()  # Uses runs.ready_to_update_idxs
        scheduler.step()  # Uses runs.ready_to_update_idxs
        
        # Broadcast weights for runs that were updated
        weight_broadcast.broadcast_weights(model, step)
```

## Creation Hooks

Components can register hooks that are called when new runs are created:

```python
runs.register_creation_hook(callback)
```

The callback signature is:

```python
def callback(idx: int, run_id: str) -> None:
    """Called when a new run is created.
    
    Args:
        idx: The run's index (0 to max_runs-1)
        run_id: The run's ID (e.g., "run_abc123")
    """
    pass
```

Hooks are executed by all ranks during `sync_runs()`, ensuring consistent state.

## Distributed Synchronization

The `Runs` object uses PyTorch's distributed store for synchronization:

1. **Master rank** scans for changes via `check_for_changes()`
2. **Master rank** serializes state and writes to store
3. **All ranks** call `sync_runs()` which:
   - Non-master ranks read state from store
   - All ranks calculate new/deleted runs
   - All ranks execute creation hooks together
   - All ranks reset parameters for new runs

This ensures all ranks have consistent views of the run state.

## Single Run Mode

When `max_concurrent_runs == 1`, the system operates in a simplified mode:
- Uses standard single optimizers and schedulers
- Still requires run discovery for the initial run
- Training loop waits for optimizer creation before starting

```python
if config.max_concurrent_runs == 1:
    while optimizer.optimizers[0] is None:
        runs.check_for_changes()
        runs.sync_runs()
        time.sleep(1)
    scheduler = setup_scheduler(optimizer.optimizers[0], ...)
```

## File Structure

Each run's directory follows this structure:

```
{output_dir}/
├── run_abc123/
│   ├── configs/
│   │   └── orch.toml          # Orchestrator configuration
│   ├── checkpoints/
│   │   └── step_100/          # Training checkpoints
│   ├── weights/
│   │   └── step_100/          # Weight checkpoints
│   └── broadcast/
│       └── step_100/          # Broadcast weights for inference
├── run_def456/
│   └── ...
└── ...
```

## Summary

The `Runs` object is the coordination backbone for multi-run training. It:

| Responsibility | Description |
|---------------|-------------|
| **Discovery** | Scans for `run_*` directories and loads configs |
| **Mapping** | Provides bidirectional run ID ↔ index mapping |
| **Progress** | Tracks per-run training step, tokens, samples |
| **Synchronization** | Keeps all ranks in sync via distributed store |
| **Hooks** | Enables lazy initialization of per-run resources |
| **LoRA Management** | Registers modules for multi-adapter parameter access |
| **State Access** | Provides per-run parameters and state dicts |

This design enables efficient multi-tenant training where a single trainer can serve multiple experiments with independent adapter weights, optimizers, and learning rate schedules.

