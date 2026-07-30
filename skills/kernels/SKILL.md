---
name: kernels
description: How prime-rl vendors, builds, and ships CUDA kernels (the `kernels/` tree and the `prime-kernels` wheel). Use when adding a kernel, building it locally, calling one from training code, or publishing prebuilt wheels.
---

# CUDA kernels

CUDA kernels live in `kernels/` and ship as one wheel, `prime-kernels`. One folder per
kernel under `kernels/prime_kernels/`, holding its Python surface and its sources (a pinned
submodule at `<kernel>/upstream/`); everything is declared in the single manifest
`kernels/prime_kernels/kernels.toml`. See [`kernels/README.md`](../../kernels/README.md).

`prime-rl` itself stays a pure-Python wheel — never add compiled extensions to it.

## Calling a kernel from prime-rl

Kernels are compiled for exact compute capabilities and may not be built at all, so always
gate. Never import `prime_kernels.<name>` directly in training code:

```python
import prime_kernels

if prime_kernels.is_available("flash_moe"):
    flash_moe = prime_kernels.load("flash_moe")
```

`prime_kernels.status()` maps every kernel to `"available"` or the reason it is not — log
it once at startup rather than failing a run halfway through.

## Building locally

```bash
git submodule update --init kernels   # sources (private repos, SSH)
uv sync --extra kernels               # or, while iterating:
uv pip install --no-build-isolation -e kernels
```

Requirements: `nvcc` on `CUDA_HOME` with the **same CUDA major as torch** (torch refuses to
build extensions otherwise), and new enough for the kernel's `min-cuda`.

Kernels whose sources or toolkit are missing are skipped with a message and reported
unavailable at runtime — the build still succeeds. `PRIME_KERNELS=a,b` builds a subset;
`PRIME_KERNELS_REQUIRE=1` turns any skip into an error.

## Adding a kernel

1. `git submodule add git@github.com:PrimeIntellect-ai/<repo>.git kernels/prime_kernels/<name>/upstream`
   (or commit sources straight into `kernels/prime_kernels/<name>/`).
2. Add a `[<name>]` table to `kernels/prime_kernels/kernels.toml` — sources, `arch`,
   `min-cuda`, `cxx-std`; paths are relative to the kernel folder.
3. `kernels/prime_kernels/<name>/__init__.py` — `from . import _C`, one wrapper and one
   `torch.library.register_fake` per op.
4. Nothing else: `setup.py` and the registry both read the manifest.

Rules the build assumes:

- The extension is always `prime_kernels.<name>._C`, so the C++ side defines
  `PYBIND11_MODULE(_C, m)` and registers ops with `TORCH_LIBRARY*`.
- `arch` matches the device **exactly** at runtime (`10.0a` runs only on sm_100a); no PTX is
  shipped to JIT from.
- Two packages registering the same `torch.ops` namespace collide. If a kernel's upstream
  package is also installed standalone (e.g. `prime_moe`), uninstall it.
- Vendored Python is ignored on purpose — copy what you need into the kernel's package so
  every kernel looks the same and upstream repos without Python work unchanged.

## Bumping a vendored kernel

```bash
git -C kernels/prime_kernels/<name>/upstream fetch
git -C kernels/prime_kernels/<name>/upstream checkout <sha>
git add kernels/prime_kernels/<name>/upstream
```

Rebuild and re-run whatever exercises the kernel — the ABI is not checked for you.

## Prebuilt wheels

[`build_kernels.yaml`](../../.github/workflows/build_kernels.yaml) builds the wheel for
x86_64 and aarch64 in the CUDA devel image (no GPU needed — nvcc cross compiles). It runs on
every change under `kernels/`, and `workflow_dispatch` with `release_tag: vX.Y.Z` attaches
the wheels to that release, alongside the deep-ep/deep-gemm/torchao wheels.

The wheel version carries the ABI it was built against, e.g.
`prime_kernels-0.1.0+cu128torch2.9.0-cp312-cp312-linux_x86_64.whl` — a wheel is only valid
for that torch and CUDA major.

Once wheels are published, point the extra at them instead of building from source:

```toml
[tool.uv.sources]
prime-kernels = [
    { url = ".../releases/download/vX.Y.Z/prime_kernels-...linux_x86_64.whl", marker = "platform_machine == 'x86_64'" },
    { url = ".../releases/download/vX.Y.Z/prime_kernels-...linux_aarch64.whl", marker = "platform_machine == 'aarch64'" },
]
```

CI needs `KERNELS_SUBMODULE_TOKEN` (read access to the private kernel repos) — the other
workflows only init the public submodules.

## Gotchas

- Changing anything about `prime-kernels` in the root `pyproject.toml` needs `uv lock`.
  Keep `[[tool.uv.dependency-metadata]] name = "prime-kernels"` in sync with
  `kernels/pyproject.toml`, or `uv lock` will build the kernels just to read metadata.
- `match-runtime = true` does not work for `prime-kernels`: uv cannot read a source tree's
  metadata without building it. The build gets the environment's torch through
  `no-build-isolation-package` instead.
- Ruff must not see vendored checkouts (upstream code, sometimes unformatted or broken) —
  `kernels/prime_kernels/*/upstream` is in `extend-exclude`, which is why the submodule
  directory is always named `upstream`.
