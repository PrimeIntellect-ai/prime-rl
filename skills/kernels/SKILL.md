---
name: kernels
description: How prime-rl vendors, builds, and ships CUDA kernels (the `deps/prime-kernels` submodule and the `prime-kernels` wheel). Use when adding a kernel, building it locally, calling one from training code, or publishing prebuilt wheels.
---

# CUDA kernels

CUDA kernels live in their own monorepo,
[prime-kernels](https://github.com/PrimeIntellect-ai/prime-kernels), checked out here as the
git submodule `deps/prime-kernels`, alongside prime-rl's other submodules. That repo is the
wheel root (`setup.py`, `pyproject.toml`) and `prime_kernels/` inside it is the importable
package: one folder per kernel, holding the
kernel's Python surface and, for compiled kernels, its C++/CUDA sources under `csrc/`, all declared in the single
manifest `prime_kernels/kernels.toml`. See `deps/prime-kernels/README.md` once the submodule
is initialized.

Nothing about a kernel lives in prime-rl. prime-rl pins a prime-kernels commit, builds the
wheel from it, and installs the result — and stays a pure-Python wheel itself; never add
compiled extensions to it.

Living under `deps/` means `tool.ruff.extend-exclude = ["deps"]` in `pyproject.toml`
already covers it — prime-rl lints none of it.

## Calling a kernel from prime-rl

Kernels are compiled for exact compute capabilities and may not be built at all, so always
gate. Never import `prime_kernels.<name>` directly in training code:

```python
import prime_kernels

if prime_kernels.is_available("flash_moe"):
    flash_moe = prime_kernels.load("flash_moe")
```

`prime_kernels.status()` maps every kernel to `"available"` or the reason it is not — log
it once at startup rather than failing a run halfway through. `unavailable_reason(name)` is
the same answer for one kernel (`None` when it is usable), which is what a test's skip guard
wants; `is_available` is just that call compared to `None`.

`flash_moe` is a compiled fused MoE forward kernel (bf16 + mxfp8) on Blackwell
tcgen05. Its trainer integration is dormant. `mxfp8_moe` is a Python-only registered
kernel package for MXFP8 grouped GEMM and torch EP transport on SM100; it owns the
MoE-specific torchao-derived orchestration and exports explicit BF16 boundaries instead
of tensor-subclass interception.

What a kernel requires of its inputs — block sizes, alignments, shape constraints — belongs
to prime-kernels, which exports it: `flash_moe.BLOCK_M`, `flash_moe.MXFP8_SCALE_BLOCK`, and
`flash_moe.unsupported_shape_reason(dim, hidden_dim, mxfp8=...)`. When the trainer integration
is restored, call this once during setup so an unsupported model fails before training rather
than mid-step. Never hardcode a `128` on this side: then every requirement change is a change
in both repos.

## Building locally

`uv sync --extra kernels` installs the prebuilt wheel (see "Pinning installs at the prebuilt
wheels"); building from source is for changing kernels. It is manual by design — no `uv sync`
may compile CUDA, so the extra resolves to release wheels, never to this source tree:

```bash
git submodule update --init deps/prime-kernels
uv pip install --no-build-isolation -e deps/prime-kernels
```

Requirements: `nvcc` on `CUDA_HOME` with the **same CUDA major as torch** (torch refuses to
build extensions otherwise).

Kernels whose toolkit is unsuitable are skipped with a message and reported unavailable at
runtime — the build still succeeds. `PRIME_KERNELS=a,b` builds a subset;
`PRIME_KERNELS_REQUIRE=1` turns any skip into an error.

## Changing or adding a kernel

The work happens in the prime-kernels repo, not here. Inside `deps/prime-kernels/`:

1. Commit the sources under `prime_kernels/<name>/csrc/`.
2. Add a `[<name>]` table to `prime_kernels/kernels.toml` — `sources`, `include-dirs`,
   `arch`, `cxx-std`; paths are relative to the kernel folder. A vendored Python kernel
   uses `python-only = true`, may declare import checks in `requires`, and omits compiled
   extension fields.
3. Write `prime_kernels/<name>/__init__.py` — `from . import _C`, then per op a wrapper
   calling `torch.ops.<ns>.<op>` and a `torch.library.register_fake`. No
   `torch.library.custom_op` decorator: that defines a *Python* op, and `TORCH_LIBRARY`
   has already defined these C++ side — only the fake (meta) kernel is missing. A kernel
   used in training also needs `torch.library.register_autograd`, since a schema carries
   no backward. (`flash_moe` is forward only; prime-rl wraps it in an `autograd.Function`
   of its own.)
4. Nothing else: `setup.py` and the runtime registry both read the manifest.

Rules the build assumes:

- The extension is always `prime_kernels.<name>._C`, so the C++ side defines
  `PYBIND11_MODULE(_C, m)` and registers ops with `TORCH_LIBRARY*`.
- `arch` matches the device **exactly** at runtime (`10.0a` runs only on sm_100a); no PTX is
  shipped to JIT from.
- Two packages registering the same `torch.ops` namespace collide. If a kernel's sources are
  also installed as a standalone package (e.g. `prime_moe` from prime-flash-moe, where
  `flash_moe` originally came from), uninstall it.
- Only the Python surface and the compiled `_C` ship in the wheel; `csrc/` is build input.

Then land it in prime-rl as a submodule bump.

## Bumping the pin

Kernel sources are pinned by the submodule commit, so picking up any kernel change — yours
or someone else's — is a bump:

```bash
git -C deps/prime-kernels fetch origin
git -C deps/prime-kernels log --oneline HEAD..origin/main
git -C deps/prime-kernels checkout origin/main
git add deps/prime-kernels
```

Then, in order:

- Read the diff for **host-side contract changes**, not just kernel internals. A change to
  what the caller must pass (weight layout, scale packing, argument order) is silently wrong
  numbers, not a build error, and prime-rl's call sites have to absorb it.
- Rebuild and re-run whatever exercises the kernel — the ABI is not checked for you. For
  `flash_moe` that is `tests/unit/train/models/test_fused_moe.py`, which compares its
  forward and its hand-written backward against the grouped-mm expert path; it is
  `gpu`-marked and skips itself unless the kernel is available, so it only means anything
  on a machine the kernel was built for.
- The bump alone ships nothing: installs resolve `prime-kernels` from a release wheel, so the
  new code only reaches users once a release rebuilds the wheels and the pin below moves.

## Prebuilt wheels

[`build_kernels.yaml`](../../.github/workflows/build_kernels.yaml) builds the wheel for
x86_64 and aarch64 in the CUDA devel image (no GPU needed — nvcc cross compiles). It inits
the `deps/prime-kernels` submodule itself, runs on every bump of it, and attaches the wheels
to a release when given a `release_tag`, alongside the deep-ep/deep-gemm/torchao wheels.

Its `paths` trigger is `deps/prime-kernels` (the gitlink), **not** `deps/prime-kernels/**` —
no file under it is tracked by prime-rl, so a `**` pattern would match nothing and submodule
bumps would build no wheels.

Every release gets them: [`tag-and-release.yaml`](../../.github/workflows/tag-and-release.yaml)
calls this workflow after the tag is cut and **before** it promotes the draft, so a published
release always carries its wheels. To backfill a release that predates this, dispatch by hand:

```bash
gh workflow run build_kernels.yaml -f release_tag=vX.Y.Z -f ref=vX.Y.Z
```

The wheel version carries the ABI it was built against, e.g.
`prime_kernels-0.1.0+cu128torch2.11.0-cp312-cp312-linux_x86_64.whl` — it imports only under
that exact torch, so the build installs the torch pinned in `uv.lock`, not the newest one.
The base version comes from `deps/prime-kernels/pyproject.toml` in the prime-kernels repo.

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

The wheels are prime-rl release assets — the kernels live in their own repo, but they ship
with prime-rl's releases. To move the pin, the build prints both lines, ready to paste, in
its job summary. Then `uv lock`.

The pin necessarily trails by one release: a release's own assets do not exist until that
release is built, so `vX.Y.Z` can only point at wheels from an already published tag. Move it
whenever the kernels or the torch/CUDA pin change — a stale pin ships stale kernels, and a
pin whose torch no longer matches the lock fails at import, not at install.

## Gotchas

- prime-kernels' `pyproject.toml` mirrors prime-rl's own `torch>=2.9.0`; keep the two
  identical, and build against the torch in `uv.lock` (`build_kernels.yaml` reads it) — the
  wheel imports only under the torch it was compiled against.
- Never reintroduce `prime-kernels` as a path source: uv cannot read a source tree's metadata
  without building it, so every `uv lock` would then need nvcc. A release-asset URL has static
  metadata and does not.
- A fresh clone without the submodule still installs and trains — only the source build needs
  it. `build_kernels.yaml` inits it and sets `PRIME_KERNELS_REQUIRE=1`, so a wheel is never
  published with kernels silently skipped.
