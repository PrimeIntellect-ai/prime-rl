# prime-kernels

CUDA kernels vendored into prime-rl and shipped as one wheel, `prime-kernels`.

```
kernels/
├── setup.py                  # builds what the manifest declares; no edit needed to add a kernel
└── prime_kernels/
    ├── kernels.toml          # the manifest: one table per kernel
    ├── __init__.py           # registry: is_available / load / status
    ├── _spec.py              # manifest parser (build time + runtime)
    └── flash_moe/            # one folder per kernel
        ├── __init__.py       # Python surface: op wrappers, fake tensors
        ├── mxfp8.py
        └── upstream/         # git submodule, pinned: the C++/CUDA sources
```

A kernel folder holds its Python surface and its sources — either a vendored checkout under
`upstream/` or files committed here directly. Vendored Python is ignored on purpose: the
Python surface lives here so every kernel looks the same and upstream repos with no Python
at all work unchanged.

## Using a kernel

Kernels are compiled for specific compute capabilities and may not be built at all, so
never import one directly from application code:

```python
import prime_kernels

if prime_kernels.is_available("flash_moe"):
    flash_moe = prime_kernels.load("flash_moe")
    out = flash_moe.fused_moe_bf16(...)
```

`prime_kernels.status()` maps every kernel to `"available"` or the reason it is not.

## Installing

```bash
uv sync --extra kernels                          # from the pinned prebuilt wheel
uv pip install --no-build-isolation -e kernels   # build locally while iterating
```

A local build needs `nvcc` (`CUDA_HOME`) whose CUDA major matches torch's and which is new
enough for the kernel's `min-cuda`, plus the vendored submodule:

```bash
git submodule update --init kernels/prime_kernels/flash_moe
```

Kernels whose sources are missing or whose toolkit is unsuitable are skipped with a message
rather than failing the build; the registry then reports them unavailable. `PRIME_KERNELS=a,b`
builds a subset, `PRIME_KERNELS_REQUIRE=1` turns a skip into an error (the release workflow
sets it).

## Adding a kernel

1. `git submodule add <url> kernels/prime_kernels/<name>/upstream` (or just commit sources
   into `kernels/prime_kernels/<name>/`).
2. Add a table to `prime_kernels/kernels.toml`:

```toml
[<name>]
description = "..."
ops = "<torch.ops namespace the extension registers>"
upstream = "<url>"
sources = ["upstream/csrc/foo.cu", "upstream/csrc/torch_interface.cpp"]
include-dirs = ["upstream/csrc"]
arch = ["10.0a"]     # compute capabilities to compile for; exact match at runtime
min-cuda = "12.8"
cxx-std = 20
```

3. Add `prime_kernels/<name>/__init__.py`: `from . import _C` plus wrappers and
   `torch.library.register_fake` for each op.

The extension is always named `prime_kernels.<name>._C`, so the C++ side must define
`PYBIND11_MODULE(_C, m)` (ops themselves should be registered with `TORCH_LIBRARY*`).
Two installs registering the same `torch.ops` namespace collide — if a kernel's upstream
package is also installed standalone (e.g. `prime_moe`), uninstall it.
