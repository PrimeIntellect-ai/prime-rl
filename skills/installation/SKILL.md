---
name: installation
description: How to install prime-rl and its optional dependencies. Use when setting up the project, installing extras like deep-gemm for FP8 models, or troubleshooting dependency issues.
---

# Installation

## Cloning

prime-rl is a monorepo with submodules. Clone the repo first, then initialize submodules explicitly:

```bash
git clone git@github.com:PrimeIntellect-ai/prime-rl.git
cd prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
```

For an existing clone, initialize the public submodules by passing explicit paths:

```bash
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
```

Do **not** use `git submodule update --init --recursive` without explicit paths — it will also attempt to clone `configs/private` (a private submodule) and fail with "Repository not found" for users without access. Git will retry the clone once internally before aborting, which is expected behavior.

The public submodules live under `deps/`: `deps/verifiers/`, `deps/renderers/`, `deps/research-environments/`, and `deps/pydantic-config/`. The uv workspace members among them apply source edits to the prime-rl venv without re-sync.

The install script (`scripts/install.sh`) handles the private submodule gracefully — it iterates each submodule individually so a failed `configs/private` init does not abort the rest.

## Basic
```bash
uv sync              # core dependencies only
uv sync --group dev  # dev tools: pytest, ruff, pre-commit
uv sync --all-extras # recommended: includes envs, flash-attn, flash-attn-cute, etc.
```

The `envs` extra installs the env packages listed in `pyproject.toml > [tool.uv.workspace] members`. Adding a new env means adding it there, in the `envs` extra, and in `[tool.uv.sources]`.

When intentionally bumping a package to a release newer than the workspace-wide `exclude-newer = "7 days"` window, add that package and any newly required transitive packages to `[tool.uv.exclude-newer-package]` before refreshing `uv.lock`.

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
