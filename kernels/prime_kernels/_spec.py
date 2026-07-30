"""Parser for `kernels.toml`, the manifest describing every kernel in this package.

Read at build time by `setup.py` (before anything is installed) and at runtime by the
registry, so this module must not import torch or any compiled extension.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

MANIFEST = "kernels.toml"

_ARCH = re.compile(r"^(\d+)\.(\d+)([af]?)$")


@dataclass(frozen=True)
class Arch:
    """A CUDA compute capability a kernel is compiled for, e.g. `10.0a` (sm_100a)."""

    major: int
    minor: int
    suffix: str  # "" (portable), "a" (architecture specific) or "f" (family specific)

    @classmethod
    def parse(cls, raw: str) -> Arch:
        match = _ARCH.match(raw)
        if not match:
            raise ValueError(f"invalid arch {raw!r}, expected e.g. '9.0', '10.0a' or '12.0f'")
        return cls(int(match[1]), int(match[2]), match[3])

    @property
    def capability(self) -> tuple[int, int]:
        return self.major, self.minor

    @property
    def sm(self) -> str:
        return f"sm_{self.major}{self.minor}{self.suffix}"

    def gencode(self) -> str:
        target = f"{self.major}{self.minor}{self.suffix}"
        return f"-gencode=arch=compute_{target},code=sm_{target}"

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}{self.suffix}"


@dataclass(frozen=True)
class KernelSpec:
    """Everything needed to build one kernel and to decide whether it can run here."""

    name: str
    path: Path  # the prime_kernels/<name> folder the paths below are relative to
    description: str
    ops: str  # torch.ops namespace the compiled extension registers into
    upstream: str | None  # where the sources were originally developed; provenance only
    sources: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    archs: tuple[Arch, ...]
    cxx_std: int
    min_cuda: tuple[int, int]
    cxx_flags: tuple[str, ...]
    nvcc_flags: tuple[str, ...]

    @property
    def module(self) -> str:
        return f"prime_kernels.{self.name}"

    @property
    def sm_list(self) -> str:
        return ", ".join(arch.sm for arch in self.archs)

    def supports(self, capability: tuple[int, int]) -> bool:
        """Whether a device of this compute capability can run the compiled code.

        Exact match only: architecture specific (`a`) cubins do not run on any other
        capability, and we do not ship PTX to JIT from.
        """
        return any(arch.capability == capability for arch in self.archs)


def load(package_dir: Path) -> dict[str, KernelSpec]:
    """Parse `<package_dir>/kernels.toml` into one spec per kernel, in manifest order.

    Wheels ship the manifest but not the sources, so at runtime this yields specs whose
    source paths simply do not exist.
    """
    package_dir = package_dir.resolve()
    manifest = tomllib.loads((package_dir / MANIFEST).read_text())
    return {name: _kernel(name, package_dir / name, table) for name, table in manifest.items()}


def _kernel(name: str, path: Path, table: dict) -> KernelSpec:
    major, minor = (int(part) for part in str(table["min-cuda"]).split("."))
    return KernelSpec(
        name=name,
        path=path,
        description=table["description"],
        ops=table["ops"],
        upstream=table.get("upstream"),
        sources=tuple(path / source for source in table["sources"]),
        include_dirs=tuple(path / directory for directory in table.get("include-dirs", [])),
        archs=tuple(Arch.parse(arch) for arch in table["arch"]),
        cxx_std=table.get("cxx-std", 20),
        min_cuda=(major, minor),
        cxx_flags=tuple(table.get("cxx-flags", [])),
        nvcc_flags=tuple(table.get("nvcc-flags", [])),
    )
