---
name: install
description: How to install prime-rl and its optional dependencies. Use when setting up the project, installing extras like deep-gemm for FP8 models, or troubleshooting dependency issues.
---

# Install

## Clone + submodules

prime-rl is a monorepo with submodules. Use the install script when bootstrapping a fresh machine:

```bash
bash scripts/install.sh   # clones, inits submodules, installs uv, runs `uv sync --all-extras`
```

For an existing clone, init submodules explicitly:

```bash
git submodule update --init --recursive
```

## Sync

```bash
uv sync                    # core only
uv sync --group dev        # + pytest, ruff, pre-commit
uv sync --all-extras       # recommended: envs, flash-attn, flash-attn-cute, etc.
```

The `envs` extra installs environments listed under `[project.optional-dependencies].envs`, resolved through `[tool.uv.sources]`. Adding a new env means adding its package name to `envs` and its editable source path to `[tool.uv.sources]`.

When bumping a package past the workspace-wide `exclude-newer = "7 days"` window, add it (and any newly-required transitives) to `[tool.uv.exclude-newer-package]` before refreshing `uv.lock`.

## Optional extras

### NemotronH (Mamba SSD kernels)

```bash
CUDA_HOME=/usr/local/cuda uv pip install mamba-ssm
```

Requires `nvcc`. Without `mamba-ssm`, NemotronH falls back to HF's pure-PyTorch SSD path, which computes softplus in bf16 and yields ~0.4 KL divergence vs vLLM. Do **not** install `causal-conv1d` unless your GPU arch matches the prebuilt kernels — the code falls back to `nn.Conv1d` when it's absent.

### FP8 inference (GLM-5-FP8, etc.)

```bash
uv sync --group fp8-inference   # installs the prebuilt deep-gemm wheel
```

### Trainer DeepEP backend

`scripts/install_ep_kernels.sh` auto-detects the CUDA toolkit matching torch and the GPU arch, builds NVSHMEM + DeepEP from source, and skips if `deep_ep` already imports.

```bash
bash scripts/install_ep_kernels.sh
```

Flags: `--workspace DIR`, `--deepep-ref REF` (default `73b6ea4`), `--nvshmem-ver VER` (default `3.3.24`), `--configure-drivers` (multi-node IBGDA; needs sudo + reboot).

Verify: `uv run python -c 'import deep_ep; print(deep_ep.__file__)'`.

### llm-d router backend

Multi-node / disaggregated deployments can route through the upstream llm-d Endpoint Picker instead of `vllm-router` (set `[...deployment.router] type = "llm-d"`). It needs three native binaries — install once:

```bash
bash scripts/install_llmd.sh   # builds epp + pd-sidecar from a pinned llm-d-router commit (vendored Go), fetches envoy
```

Binaries land in `third_party/llmd/bin/{epp,envoy,pd-sidecar}` (a shared path, so SLURM nodes see them). `epp` is pinned to the commit that includes the `vllmhttp-parser` (PR #1248) so prime-rl's renderer/TITO `/inference/v1/generate` path routes correctly. Override the pin with `LLMD_ROUTER_REF=<sha>`. The EPP + Envoy + endpoints configs are rendered from `templates/llmd/*.yaml.j2` (included into the SLURM script); only the per-node IPv4 addresses are filled in inline at launch time.

## Key files

- `pyproject.toml` — dependencies, extras, dependency groups
- `uv.lock` — pinned lockfile (refresh with `uv sync --all-extras`)
- `scripts/install.sh` — bootstrap installer
- `scripts/install_ep_kernels.sh` — DeepEP build script
