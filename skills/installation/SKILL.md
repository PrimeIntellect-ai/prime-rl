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

### SGLang inference backend

SGLang is isolated in `packages/prime-rl-sglang` because its transitive dependencies conflict with vLLM and the root workspace pins Transformers 5. The launcher uses `uv run --frozen --project packages/prime-rl-sglang ...` so it uses that package's own locked environment instead of mutating the main `.venv` while trainer/orchestrator run. After dependency changes, refresh both locks through uv:

```bash
uv lock
uv lock --project packages/prime-rl-sglang
```

Use `inference.backend = "sglang"` for single-node filesystem-broadcast runs. NCCL broadcast and multi-node SGLang deployment are not wired yet.

The launcher also sets `FLASHINFER_WORKSPACE_BASE=packages/prime-rl-sglang` so FlashInfer JIT artifacts are tied to the SGLang project env. This avoids stale JIT cache entries that can contain include paths from uv temp environments.

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

## Trainer DeepEP backend

The trainer-side MoE `deepep` backend is optional and requires a local DeepEP build.

Install using the provided script, which auto-detects CUDA toolkit and GPU architecture:

```bash
bash scripts/install_ep_kernels.sh
```

The script downloads NVSHMEM, builds DeepEP from source, and places the wheel in `deps/`. It skips if DeepEP is already installed. Options:

- `--workspace DIR` — build directory (default: `./ep_kernels_workspace`)
- `--deepep-ref REF` — DeepEP commit hash (default: `73b6ea4`)
- `--nvshmem-ver VER` — NVSHMEM version (default: `3.3.24`)
- `--configure-drivers` — configure IBGDA drivers for multi-node (requires sudo + reboot)

Verify the install:

```bash
uv run python -c 'import deep_ep; print(deep_ep.__file__)'
```

## Dev dependencies

```bash
uv sync --group dev
```

Installs pytest, ruff, pre-commit, and other development tools.

## Key files

- `pyproject.toml` — all dependencies, extras, and dependency groups
- `uv.lock` — pinned lockfile (update with `uv sync --all-extras`)
- `packages/prime-rl-sglang/pyproject.toml` and `packages/prime-rl-sglang/uv.lock` — isolated SGLang runtime
