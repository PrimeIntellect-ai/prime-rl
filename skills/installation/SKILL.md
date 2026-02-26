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

## Optional extras

Some features require optional extras:

```bash
# Flash Attention 2
uv sync --extra flash-attn

# Flash Attention 3
uv sync --extra flash-attn-3

# Flash Attention (cute variant)
uv sync --extra flash-attn-cute
```

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

## Key files

- `pyproject.toml` — all dependencies, extras, and dependency groups
- `uv.lock` — pinned lockfile (update with `uv sync --all-extras`)
