import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import setuptools
import torch
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()
PACKAGE_DIR = ROOT / "prime_kernels"


def _import_spec_module():
    """Import `prime_kernels._spec` without importing the (not yet built) package."""
    spec = importlib.util.spec_from_file_location("prime_kernels_spec", PACKAGE_DIR / "_spec.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _nvcc_version() -> tuple[int, int] | None:
    if CUDA_HOME is None:
        return None
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc.exists():
        return None
    banner = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"release (\d+)\.(\d+)", banner)
    return (int(match[1]), int(match[2])) if match else None


def _skip_reason(kernel, cuda: tuple[int, int] | None) -> str | None:
    missing = [source for source in kernel.sources if not source.exists()]
    if missing:
        relative = ", ".join(str(source.relative_to(kernel.path)) for source in missing)
        return f"sources missing ({relative}) — kernels.toml and kernels/{kernel.path.relative_to(ROOT)} disagree"
    if cuda is None:
        return "no CUDA toolkit found (set CUDA_HOME)"
    # torch extensions must be compiled with the CUDA major torch itself was built with.
    torch_cuda = torch.version.cuda
    if torch_cuda is None:
        return "this torch build has no CUDA support"
    if int(torch_cuda.split(".")[0]) != cuda[0]:
        return f"CUDA toolkit {cuda[0]}.{cuda[1]} does not match torch's CUDA {torch_cuda}"
    return None


def _extension(kernel) -> CUDAExtension:
    std = [f"-std=c++{kernel.cxx_std}"]
    return CUDAExtension(
        name=f"{kernel.module}._C",
        # setuptools rejects absolute source paths; everything lives under kernels/.
        sources=[str(source.relative_to(ROOT)) for source in kernel.sources],
        include_dirs=[str(directory) for directory in kernel.include_dirs],
        extra_compile_args={
            "cxx": [*std, *kernel.cxx_flags],
            # Explicit -gencode per kernel: TORCH_CUDA_ARCH_LIST is process wide, and one
            # build can hold kernels targeting different architectures. Passing any arch
            # flag also stops torch from appending its own.
            "nvcc": [*std, *(arch.gencode() for arch in kernel.archs), *kernel.nvcc_flags],
        },
    )


kernels = _import_spec_module().load(PACKAGE_DIR)
selection = {name for name in os.environ.get("PRIME_KERNELS", "").split(",") if name}
require = os.environ.get("PRIME_KERNELS_REQUIRE") == "1"
cuda = _nvcc_version()

extensions, skipped = [], {}
for name, kernel in kernels.items():
    if selection and name not in selection:
        skipped[name] = "not selected by PRIME_KERNELS"
        continue
    reason = _skip_reason(kernel, cuda)
    if reason:
        skipped[name] = reason
        continue
    extensions.append(_extension(kernel))

for name, reason in skipped.items():
    print(f"prime-kernels: skipping {name}: {reason}", file=sys.stderr)
if require and skipped:
    raise SystemExit(f"PRIME_KERNELS_REQUIRE=1 but {len(skipped)} kernel(s) were skipped")
print(f"prime-kernels: building {', '.join(ext.name for ext in extensions) or '<nothing>'}", file=sys.stderr)

setuptools.setup(
    # Listed explicitly: the kernel folders carry C++/CUDA sources next to their Python, and
    # only the Python surface plus the compiled extension belongs in the wheel.
    packages=["prime_kernels", *(f"prime_kernels.{name}" for name in kernels)],
    package_data={"prime_kernels": ["kernels.toml"]},
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
)
