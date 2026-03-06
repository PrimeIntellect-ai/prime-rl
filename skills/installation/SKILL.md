---
name: installation
description: How to install prime-rl and its optional dependencies. Use when setting up the project, installing extras like deep-gemm for FP8 models, or troubleshooting dependency issues.
---

# Installation

## Basic install

```bash
uv sync
```

This installs all core dependencies defined in `pyproject.toml`.

## All extras at once

The recommended way to install for most users:

```bash
uv sync --all-extras
```

This installs all optional extras (flash-attn, flash-attn-cute, etc.) in one go.

## FP8 inference with deep-gemm

For certain models like GLM-5-FP8, you need `deep-gemm`. Install it via the `fp8-inference` dependency group:

```bash
uv sync --group fp8-inference
```

This installs the pre-built `deep-gemm` wheel. No CUDA build step is needed.

## Dev dependencies

```bash
uv sync --group dev
```

Installs pytest, ruff, pre-commit, and other development tools.

## Expert parallel kernels (DeepEP) for disaggregated inference

Disaggregated prefill/decode deployments with expert parallelism require DeepEP + NVSHMEM, which must be built from source:

```bash
bash scripts/install_ep_kernels.sh
```

This downloads NVSHMEM, clones and builds DeepEP, and installs it into the project venv. Auto-detects GPU architecture (Hopper/Blackwell).

For multi-node deployments, IBGDA drivers must also be configured (requires sudo + reboot):

```bash
bash scripts/install_ep_kernels.sh --configure-drivers
```

Verify installation:

```bash
uv run python -c "import deep_ep; print(deep_ep.__file__)"
```

## Key files

- `pyproject.toml` — all dependencies, extras, and dependency groups
- `uv.lock` — pinned lockfile (update with `uv sync --all-extras`)
- `scripts/install_ep_kernels.sh` — DeepEP + NVSHMEM installer for expert parallel
