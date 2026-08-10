---
name: kernels
description: How prime-rl vendors, builds, and ships CUDA kernels (the `kernels/` tree and the `prime-kernels` wheel). Use when adding a kernel, building it locally, calling one from training code, or publishing prebuilt wheels.
---

# CUDA kernels

CUDA kernels live in `kernels/` and ship as one wheel, `prime-kernels`. `kernels/` is the
wheel root (`setup.py`, `pyproject.toml`); `kernels/prime_kernels/` inside it is the
importable package. One folder per kernel under it, holding the kernel's Python surface and
its C++/CUDA sources at `<kernel>/csrc/`; everything is declared in the single manifest
`kernels/prime_kernels/kernels.toml`. See [`kernels/README.md`](../../kernels/README.md).

Sources are committed here, not submoduled — a kernel is prime-rl code, so a clone builds it
and a change lands as one commit. Kernels developed in their own repo are copied in.

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

`uv sync --extra kernels` installs the prebuilt wheel (see "Pinning installs at the prebuilt
wheels"); building from source is for changing kernels. It is manual by design — no `uv sync`
may compile CUDA, so the extra resolves to release wheels, never to this source tree:

```bash
uv pip install --no-build-isolation -e kernels
```

Requirements: `nvcc` on `CUDA_HOME` with the **same CUDA major as torch** (torch refuses to
build extensions otherwise).

Kernels whose toolkit is unsuitable are skipped with a message and reported unavailable at
runtime — the build still succeeds. `PRIME_KERNELS=a,b` builds a subset;
`PRIME_KERNELS_REQUIRE=1` turns any skip into an error.

## Adding a kernel

1. Commit the sources into `kernels/prime_kernels/<name>/csrc/`.
2. Add a `[<name>]` table to `kernels/prime_kernels/kernels.toml` — sources, `arch`,
   `cxx-std`; paths are relative to the kernel folder.
3. `kernels/prime_kernels/<name>/__init__.py` — `from . import _C`, one wrapper and one
   `torch.library.register_fake` per op.
4. Nothing else: `setup.py` and the registry both read the manifest.

Rules the build assumes:

- The extension is always `prime_kernels.<name>._C`, so the C++ side defines
  `PYBIND11_MODULE(_C, m)` and registers ops with `TORCH_LIBRARY*`.
- `arch` matches the device **exactly** at runtime (`10.0a` runs only on sm_100a); no PTX is
  shipped to JIT from.
- Two packages registering the same `torch.ops` namespace collide. If a kernel's sources are
  also installed as a standalone package (e.g. `prime_moe`), uninstall it.
- Only the Python surface and the compiled `_C` ship in the wheel; `csrc/` is build input.

## Pulling in changes from an upstream repo

For a kernel copied in from a separate repo, copy the sources over and diff:

```bash
git clone <upstream> /tmp/<name> && git -C /tmp/<name> log --oneline -5
cp -r /tmp/<name>/<path>/csrc/. kernels/prime_kernels/<name>/csrc/
git diff --stat kernels/prime_kernels/<name>/csrc
```

Then, in order:

- Check `kernels.toml` still lists every source file, and record the copied commit in the
  commit message — without a submodule that is the only trace of what was copied.
- Read the upstream diff for **host-side contract changes**, not just kernel internals. A
  change to what the caller must pass (weight layout, scale packing, argument order) is
  silently wrong numbers, not a build error, and the Python surface here has to absorb it.
- Rebuild and re-run whatever exercises the kernel — the ABI is not checked for you.

Changes made here are the source of truth; port them back upstream if that repo is alive.

## Prebuilt wheels

[`build_kernels.yaml`](../../.github/workflows/build_kernels.yaml) builds the wheel for
x86_64 and aarch64 in the CUDA devel image (no GPU needed — nvcc cross compiles). It runs on
every change under `kernels/`, and attaches the wheels to a release when given a
`release_tag`, alongside the deep-ep/deep-gemm/torchao wheels.

Every release gets them: [`tag-and-release.yaml`](../../.github/workflows/tag-and-release.yaml)
calls this workflow after the tag is cut and **before** it promotes the draft, so a published
release always carries its wheels. To backfill a release that predates this, dispatch by hand:

```bash
gh workflow run build_kernels.yaml -f release_tag=vX.Y.Z -f ref=vX.Y.Z
```

The wheel version carries the ABI it was built against, e.g.
`prime_kernels-0.1.0+cu128torch2.11.0-cp312-cp312-linux_x86_64.whl` — it imports only under
that exact torch, so the build installs the torch pinned in `uv.lock`, not the newest one.

### Pinning installs at the prebuilt wheels

`uv sync --extra kernels` installs `prime-kernels` from the wheels named in
`[tool.uv.sources]` — the pattern deep-ep, deep-gemm and vllm already use, and the only form
the extra may take, since a sync must never compile:

```toml
[tool.uv.sources]
prime-kernels = [
    { url = "https://github.com/PrimeIntellect-ai/prime-rl/releases/download/v0.8.0/prime_kernels-0.1.0+cu128torch2.11.0-cp312-cp312-linux_x86_64.whl", marker = "platform_machine == 'x86_64'" },
    { url = "https://github.com/PrimeIntellect-ai/prime-rl/releases/download/v0.8.0/prime_kernels-0.1.0+cu128torch2.11.0-cp312-cp312-linux_aarch64.whl", marker = "platform_machine == 'aarch64'" },
]
```

To move the pin, the build prints both lines, ready to paste, in its job summary. Then
`uv lock`.

The pin necessarily trails by one release: a release's own assets do not exist until that
release is built, so `vX.Y.Z` can only point at wheels from an already published tag. Move it
whenever the kernels or the torch/CUDA pin change — a stale pin ships stale kernels, and a
pin whose torch no longer matches the lock fails at import, not at install.

## Gotchas

- `kernels/pyproject.toml` mirrors prime-rl's own `torch>=2.9.0`; keep the two identical, and
  build against the torch in `uv.lock` (`build_kernels.yaml` reads it) — the wheel imports
  only under the torch it was compiled against.
- Never reintroduce `prime-kernels` as a path source: uv cannot read a source tree's metadata
  without building it, so every `uv lock` would then need nvcc. A release-asset URL has static
  metadata and does not.
- Keep vendored-in Python out of the kernel folder — rewrite it as the kernel's own Python
  surface instead. Everything under `kernels/` is normal first-party code that ruff lints.
