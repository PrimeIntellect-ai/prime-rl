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

Nothing about a kernel lives in prime-rl. prime-rl pins a prime-kernels commit for local
source builds and a prime-kernels release for installs. prime-kernels builds and publishes
its own wheels. prime-rl stays a pure-Python wheel; never add compiled extensions to it.

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
of tensor-subclass interception. `indexed_attention` uses TileLang for token-indexed
grouped-query attention, Triton for QSA scoring, and a compiled persistent top-k on SM80,
SM90, and SM100.

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

1. Commit the implementation under `prime_kernels/<name>/`. Compiled kernels put their
   C++/CUDA sources in `csrc/`; Python-only kernels keep their Python implementation here.
2. Add a `[<name>]` table to `prime_kernels/kernels.toml` — `sources`, `include-dirs`,
   `arch`, `cxx-std`; paths are relative to the kernel folder. A vendored Python kernel
   uses `python-only = true`, may declare import checks in `requires`, and omits compiled
   extension fields.
3. For a compiled kernel, write `prime_kernels/<name>/__init__.py` with `from . import _C`,
   a wrapper calling `torch.ops.<ns>.<op>`, and a `torch.library.register_fake`. Do not use
   `torch.library.custom_op` for a compiled op because `TORCH_LIBRARY` already defines it.
   A Python-only implementation may define its dispatcher boundary with
   `torch.library.custom_op`; it still needs a fake implementation and, when used for
   training, registered autograd. `flash_moe` is forward only and currently has no trainer
   integration.
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

`prime_kernels.load` caches successful imports. Preload a Python-only kernel before a model
layer is passed to `torch.compile`; subsequent calls through `load` are compile-friendly
and still preserve the registry gate.

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
- Rebuild and re-run the kernel repository's numerical coverage plus every PrimeRL runtime
  path that calls the changed kernel; the ABI is not checked for you.
- The bump alone ships nothing: installs resolve `prime-kernels` from a release wheel, so the
  new code only reaches users once a release rebuilds the wheels and the pin below moves.

## Prebuilt wheels

The [prime-kernels](https://github.com/PrimeIntellect-ai/prime-kernels) repository owns
`build_kernels.yaml`. It builds `prime-kernels`, `deep-ep`, `deep-gemm`, and `torchao` for
x86_64 and aarch64. It publishes the wheels as prime-kernels release assets.

Run that workflow in prime-kernels for each torch or CUDA bump:

```bash
gh workflow run build_kernels.yaml \
  --repo PrimeIntellect-ai/prime-kernels \
  -f release_tag=vX.Y.Z
```

The wheel version carries the ABI it was built against, e.g.
`prime_kernels-0.1.0+cu130torch2.13.0-cp312-cp312-linux_x86_64.whl`. It imports only under
that exact torch and CUDA ABI. prime-kernels CI controls the build dependency pins.

### Pinning installs at the prebuilt wheels

`uv sync --extra kernels` installs `prime-kernels` from the wheels named in
`[tool.uv.sources]`. deep-ep, deep-gemm, torchao, and vLLM use the same pattern. A sync must
never compile these packages:

```toml
[tool.uv.sources]
prime-kernels = [
    { url = "https://github.com/PrimeIntellect-ai/prime-kernels/releases/download/vX.Y.Z/prime_kernels-0.1.0+cu130torch2.13.0-cp312-cp312-linux_x86_64.whl", marker = "platform_machine == 'x86_64'" },
    { url = "https://github.com/PrimeIntellect-ai/prime-kernels/releases/download/vX.Y.Z/prime_kernels-0.1.0+cu130torch2.13.0-cp312-cp312-linux_aarch64.whl", marker = "platform_machine == 'aarch64'" },
]
```

The wheels are prime-kernels release assets. Get the exact names before you move all pins:

```bash
gh release view vX.Y.Z \
  --repo PrimeIntellect-ai/prime-kernels \
  --json assets
```

Then run `uv lock`.

Move the pin when a kernel, torch, or CUDA pin changes. A stale pin ships stale kernels. A
torch ABI mismatch fails at import, not at install.

## Gotchas

- Keep the torch floor in prime-kernels aligned with prime-rl. The wheel imports only under
  the torch version it was compiled against.
- Never reintroduce `prime-kernels` as a path source: uv cannot read a source tree's metadata
  without building it, so every `uv lock` would then need nvcc. A release-asset URL has static
  metadata and does not.
- A fresh prime-rl clone without the submodule still installs and trains. Only a local source
  build needs it. prime-kernels CI sets `PRIME_KERNELS_REQUIRE=1`, so a release cannot omit
  kernels silently.
