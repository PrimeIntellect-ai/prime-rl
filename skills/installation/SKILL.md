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

## Trainer DeepEP backend

The trainer-side MoE `deepep` backend is optional and requires a local DeepEP build.

Before installing DeepEP, make sure the CUDA toolkit matches `torch.version.cuda` from the project environment.
On our H200 nodes that means using `/usr/local/cuda-12.8`, not the newer default CUDA 13.x toolkit.

Example install for the `deepep` backend:

```bash
NVSHMEM_DIR=$(
  uv run python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("nvidia.nvshmem")
if not spec or not spec.submodule_search_locations:
    raise SystemExit("nvidia.nvshmem not found in .venv")
print(spec.submodule_search_locations[0])
PY
)

# Some NVSHMEM wheels only ship libnvshmem_host.so.3; DeepEP expects the unversioned name.
ln -sf "$NVSHMEM_DIR/lib/libnvshmem_host.so.3" "$NVSHMEM_DIR/lib/libnvshmem_host.so"

CUDA_HOME=/usr/local/cuda-12.8 \
CUDACXX=/usr/local/cuda-12.8/bin/nvcc \
NVSHMEM_DIR="$NVSHMEM_DIR" \
uv pip install --python .venv/bin/python git+https://github.com/deepseek-ai/DeepEP.git --no-build-isolation
```

Verify the install:

```bash
uv run python - <<'PY'
import deep_ep, deep_ep_cpp
print("deep_ep imports ok")
PY
```

## Dev dependencies

```bash
uv sync --group dev
```

Installs pytest, ruff, pre-commit, and other development tools.

## Key files

- `pyproject.toml` — all dependencies, extras, and dependency groups
- `uv.lock` — pinned lockfile (update with `uv sync --all-extras`)
