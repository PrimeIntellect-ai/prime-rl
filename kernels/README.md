# prime-kernels

CUDA kernels living in prime-rl and shipped as one wheel, `prime-kernels`.

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
        └── csrc/             # the C++/CUDA sources
```

`kernels/` is the wheel: `setup.py` and `pyproject.toml` sit at its root, and
`prime_kernels/` inside it is the package you import. A kernel folder holds both halves of
one kernel — its Python surface and the sources compiled into `prime_kernels.<name>._C`.

Sources are committed here, not pulled in as a submodule: a kernel is prime-rl code, so a
clone builds it and a change to it is one commit and one review alongside the training code
that calls it. Kernels developed in their own repo get copied in.

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

`flash_moe` is used by the trainer's MoE layers under `model.moe_fused_kernel=true`, which
resolves the kernel during model setup so an unusable install fails before training starts.

## Installing

Building is manual and always explicit — no `uv sync` compiles CUDA:

```bash
uv pip install --no-build-isolation -e kernels
```

The build needs `nvcc` (`CUDA_HOME`) whose CUDA major matches torch's. Kernels whose toolkit
is unsuitable are skipped with a message rather than failing the build; the registry then
reports them unavailable. `PRIME_KERNELS=a,b` builds a subset, `PRIME_KERNELS_REQUIRE=1`
turns a skip into an error (the release workflow sets it).

Prebuilt wheels are attached to every release. Once one has them, they get pinned in the root
`[tool.uv.sources]` so installs download instead of compiling — see the `kernels` skill.

## Adding a kernel

1. Commit the sources into `kernels/prime_kernels/<name>/csrc/`.
2. Add a table to `prime_kernels/kernels.toml`:

```toml
[<name>]
description = "..."
ops = "<torch.ops namespace the extension registers>"
sources = ["csrc/foo.cu", "csrc/torch_interface.cpp"]
include-dirs = ["csrc"]
arch = ["10.0a"]       # compute capabilities to compile for; exact match at runtime
cxx-std = 20
```

3. Add `prime_kernels/<name>/__init__.py`: `from . import _C` plus wrappers and
   `torch.library.register_fake` for each op.

The extension is always named `prime_kernels.<name>._C`, so the C++ side must define
`PYBIND11_MODULE(_C, m)` (ops themselves should be registered with `TORCH_LIBRARY*`).
Two installs registering the same `torch.ops` namespace collide — if a kernel's sources are
also installed as a standalone package (e.g. `prime_moe`), uninstall it.
