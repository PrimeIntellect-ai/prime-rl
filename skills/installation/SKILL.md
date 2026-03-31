---
name: installation
description: How to install prime-rl and its optional dependencies. Use when setting up the project, installing extras like deep-gemm for FP8 models, or troubleshooting dependency issues.
---

# Installation

## Basic
```bash
uv sync              # core dependencies only
uv sync --group dev  # dev tools: pytest, ruff, pre-commit
uv sync --all-extras # recommended: includes flash-attn, flash-attn-cute, etc.
```

## Advanced

### Mamba-SSM (NemotronH models)

For NemotronH (hybrid Mamba-Transformer-MoE) models, install `mamba-ssm` for Triton-based SSD kernels that match vLLM's precision:

```bash
CUDA_HOME=/usr/local/cuda uv pip install mamba-ssm
```

Requires `nvcc` (CUDA toolkit). Without `mamba-ssm`, NemotronH falls back to HF's pure-PyTorch implementation which computes softplus in bf16, causing ~0.4 KL divergence vs vLLM.

Note: do NOT install `causal-conv1d` unless your GPU architecture matches the compiled CUDA kernels. The code automatically falls back to PyTorch nn.Conv1d when it's absent.

### FP8 inference with deep-gemm

For certain models like GLM-5-FP8, you need `deep-gemm`. Install it via the `fp8-inference` dependency group:

```bash
uv sync --group fp8-inference
```

This installs the pre-built `deep-gemm` wheel. No CUDA build step is needed.

## Key files

- `pyproject.toml` — all dependencies, extras, and dependency groups
- `uv.lock` — pinned lockfile (update with `uv sync --all-extras`)
