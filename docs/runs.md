# Runs

The `Runs` object is a singleton that manages multiple concurrent training runs within a single trainer process. It is the central coordination point for multi-run RL training, enabling a single trainer to serve multiple orchestrator experiments simultaneously with separate LoRA adapters, optimizers, and schedulers.

## Overview

When `max_concurrent_runs > 1`, the trainer can train multiple runs in parallel. Each run:
- Has its own LoRA adapter weights
- Has its own optimizer and scheduler
- Tracks its own training progress (step, tokens, samples)
- Loads its own orchestrator configuration

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
