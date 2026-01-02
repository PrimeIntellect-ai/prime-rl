# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PRIME-RL is a framework for large-scale asynchronous reinforcement learning that scales to 1000+ GPUs. It implements an asynchronous pipeline-parallel RL training system with three independently-running processes that communicate asynchronously: orchestrator, trainer, and inference server.

## Common Commands

### Environment Setup

```bash
# Install dependencies (from lock file)
uv sync --all-extras

# Validate environment
uv run python -V  # Should show Python 3.12
uv run python -c "import flash_attn"

# Install pre-commit hooks (required for contributions)
uv run pre-commit install
```

### Testing

```bash
# Run full test suite
uv run pytest -v

# Run unit tests only
uv run pytest tests/unit -v

# Run integration tests only
uv run pytest tests/integration -v

# Run CPU-only tests (skip GPU tests)
uv run pytest -v -m "not gpu"

# Run nightly tests (long-running end-to-end tests)
uv run pytest tests/nightly -v
```

### Running Training Components

**Debug/Validation Commands** (single GPU):
```bash
# SFT trainer validation
uv run sft @ configs/debug/sft/train.toml

# RL trainer validation
uv run trainer @ configs/debug/rl/train.toml

# Inference server (keep running in background)
uv run inference @ configs/debug/infer.toml

# Orchestrator (against running inference server)
uv run orchestrator @ configs/debug/orch.toml

# Evaluation
uv run eval @ configs/debug/eval.toml
```

**Single-Node Multi-GPU RL Training**:
```bash
# Recommended: use the rl entrypoint to start all components
uv run rl \
    --trainer @ path/to/train.toml \
    --orchestrator @ path/to/orch.toml \
    --inference @ path/to/infer.toml \
    --inference-gpu-ids 0,1,2,3,4,5 \
    --trainer-gpu-ids 6,7 \
    --inference.parallel.dp 6
```

**Multi-GPU SFT Training**:
```bash
# Use torchrun for multi-GPU SFT
uv run torchrun \
    --local-rank-filter 0 \
    --nproc-per-node 8 \
    src/prime_rl/trainer/sft/train.py @ path/to/config.toml
```

**Multi-Node Training**:
```bash
# On head node (node 0)
uv run torchrun \
    --nnodes 2 \
    --node-rank 0 \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    --local-rank-filter 0 \
    --nproc-per-node 8 \
    src/prime_rl/trainer/sft/train.py @ path/to/config.toml

# On worker node (node 1)
uv run torchrun \
    --nnodes 2 \
    --node-rank 1 \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    --local-rank-filter 0 \
    --nproc-per-node 8 \
    src/prime_rl/trainer/sft/train.py @ path/to/config.toml
```

### Code Quality

```bash
# Run linting (ruff)
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Architecture

### Core Components

PRIME-RL implements asynchronous off-policy RL training by decoupling three normally sequential steps:

1. **Orchestrator** (`src/prime_rl/orchestrator/`): Lightweight CPU process that schedules rollout generation, collects completed rollouts, computes advantages, converts trajectories to training samples, and sends batches to the trainer. Manages policy updates and checkpoints.

2. **Trainer** (`src/prime_rl/trainer/`): GPU-intensive process that receives training batches, performs forward/backward passes with importance-weighted policy gradients, and broadcasts updated weights. Uses FSDP2 for distributed training.

3. **Inference Server** (`src/prime_rl/inference/`): High-throughput vLLM-based server that generates rollouts and receives weight updates from the trainer.

### Async Training Flow

```
ORCHESTRATOR                  INFERENCE SERVER         TRAINER
    │                               │                     │
    ├── Schedule rollouts ──────────>│                     │
    │                               │                     │
    │<─── Return completions ────────┤                     │
    │                               │                     │
    ├── Compute advantages           │                     │
    ├── Create training batch        │                     │
    │                               │                     │
    ├── Send TrainingBatch ──────────────────────────────>│
    │                               │                     │
    │                               │<─── Broadcast weights┤
    │                               │                     │
    ├── Update policy checkpoint     │                     │
    │                               │                     │
    └── Repeat                       └─────────────────────┘
```

**Key Async Concepts**:
- `async_level` = how many steps ahead rollout generation is from training
- `off_policy_steps` = how many training steps occurred since a rollout was generated
- `max_async_level`: Maximum allowed async level (default=1)
- `max_off_policy_steps`: Stale rollouts older than this are discarded

### Key Modules

**Orchestrator**:
- `Scheduler` (`orchestrator/scheduler.py`): AReal/PipelineRL-style async RL scheduling with continuous batching
- `Buffer` (`orchestrator/buffer.py`): Stores examples from environments with optional sampling ratios and curriculum learning
- `Trajectories` (`orchestrator/trajectories.py`): Converts multi-turn trajectories into training samples (interleaved or branching strategies)
- `Advantage` (`orchestrator/advantage.py`): Computes advantages from rewards using baseline subtraction

**Trainer**:
- `DataLoader` (`trainer/rl/data.py`): Receives training batches from orchestrator via transport layer
- `Packer` (`trainer/rl/packer.py`): Packs training samples into micro-batches with proper masking and padding
- `Loss` (`trainer/rl/loss.py`): Implements importance-weighted policy gradient loss with ratio clipping and KL penalty

**Transport** (`transport/`):
- `TrainingSample`: Individual training example with prompt/completion tokens and log probabilities
- `TrainingBatch`: Collection of training samples sent from orchestrator to trainer
- Implementations: FileSystemTransport (single-node) and ZMQTransport (multi-node)

### Project Scripts

All entrypoints are defined in `pyproject.toml`:
- `rl`: Main RL coordinator that spawns orchestrator, trainer, and inference processes
- `trainer`: RL trainer entrypoint
- `orchestrator`: Orchestrator entrypoint
- `inference`: vLLM-based inference server
- `sft`: Supervised fine-tuning trainer
- `eval`: Evaluation against verifiers environments
- `synthesize`: Synthetic data generation

## Configuration System

PRIME-RL uses `pydantic-settings` with custom functionality. Configuration sources in order of precedence:

1. **CLI arguments**: `--key.subkey value` (e.g., `--model.name Qwen/Qwen3-0.6B`)
2. **TOML config files**: `@ path/to/config.toml` (space required after `@`)
3. **Environment variables**: `PRIME_` prefix with `__` delimiter (e.g., `PRIME_MODEL__NAME=...`)
4. **Defaults**: Sensible defaults for most arguments

Example:
```bash
uv run rl \
    @ configs/base.toml \
    --model.name Qwen/Qwen3-4B \
    --trainer.model.seq-len 4096
```

**Important**: The `RLConfig` in `rl.py` unifies configurations for all three components. Auto-validators ensure consistency (same model across components, compatible sequence lengths, etc.).

## Development Guidelines

### SFT Training Requirements

For multi-turn SFT, tokenizer chat templates must satisfy a **prefix property**: tokenization of any conversation prefix must be a prefix of the full conversation tokenization. This is critical for proper loss masking. Example of a template that does NOT satisfy this: Qwen3's chat template (strips past think sections).

### LoRA Support

PRIME-RL supports LoRA fine-tuning across the entire stack. Enable with trainer config:
```toml
[trainer.lora]
enabled = true
r = 16
alpha = 32
```

The framework also supports multi-run training where multiple LoRA adapters are trained in parallel.

### Changelog Enforcement

**IMPORTANT**: Any PR that modifies configuration structures must update `CHANGELOG.md`. This includes changes to config fields (added, removed, renamed, moved, or default value changes) in:
- `src/prime_rl/*/config.py`
- `src/prime_rl/rl.py`
- `src/prime_rl/utils/config.py`

### Multi-Node Deployment

Multi-node RL training requires:
- Shared filesystem for all nodes
- Reachable IP address for inference server from orchestrator
- Environment variables: `OUTPUT_DIR`, `INFERENCE_SERVER_IP`, `INFERENCE_SERVER_API_KEY`
- NCCL/GLOO network interface configuration if nodes are not on same network

For multi-node SFT, configure `MASTER_ADDR`, `MASTER_PORT`, `GLOO_SOCKET_IFNAME`, and `NCCL_SOCKET_IFNAME`.

### Testing Strategy

- **Unit tests** (`tests/unit/`): Fast, isolated component tests
- **Integration tests** (`tests/integration/`): End-to-end tests for RL, SFT, eval, synthesize
- **Nightly tests** (`tests/nightly/`): Long-running full example tests (reverse_text, wordle, alphabet_sort, etc.)
- Use `@pytest.mark.gpu` for tests requiring GPU
- Use `@pytest.mark.slow` for long-running tests

### Verifiers Environments

PRIME-RL integrates with the `verifiers` library for RL environments. Environments can be:
- Installed from Environments Hub: `{env_org}/{env_id}` format
- Local custom environments
- Support single-turn, multi-turn, and tool-calling environments

### Code Style

- Ruff for linting and formatting (configured in `pyproject.toml`)
- Line length: 120 characters
- Ignore F722, F821 for jaxtyping compatibility
- Use pre-commit hooks to enforce style automatically

## Key Technologies

- **PyTorch 2.9+**: Base framework with native FSDP2
- **vLLM 0.12.0**: Inference engine with optimized kernels
- **transformers 4.56+**: Model loading and tokenization
- **verifiers**: Environment abstraction for multi-turn RL
- **torchtitan**: Inspiration for advanced parallelism (TP, CP, EP)
- **W&B**: Logging and experiment tracking
- **uv**: Fast Python package manager
- **ZMQ**: High-performance inter-process communication

## Important Notes

- Python 3.12 is required (`~=3.12.0`)
- Flash Attention is pre-installed via custom wheels; avoid running `uv sync` after manual FA3 installation
- Use `uv run --no-sync` or `uv sync --inexact` to preserve manually installed packages
- For Kubernetes deployments, see `k8s/README.md`
- Examples are in `examples/` directory with detailed READMEs (reverse_text, wordle, alphabet_sort, wiki_search)
