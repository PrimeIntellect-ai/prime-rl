from __future__ import annotations

import fcntl
import hashlib
import os
import platform
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.cpp_extension import LIB_EXT, _get_build_directory, load


def _content_addressed_name(
    base_name: str,
    *,
    fingerprint_files: Iterable[Path],
    fingerprint: Iterable[str],
) -> str:
    digest = hashlib.sha256()
    for value in (
        torch.__version__,
        torch.version.cuda or "cpu",
        platform.machine(),
        f"python-{sys.version_info.major}.{sys.version_info.minor}",
        *fingerprint,
    ):
        digest.update(value.encode())
        digest.update(b"\0")
    for path in sorted(path for path in fingerprint_files if path.is_file()):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"{base_name}_{digest.hexdigest()[:16]}"


def load_cuda_extension(
    *,
    base_name: str,
    sources: list[Path],
    fingerprint_files: Iterable[Path],
    fingerprint: Iterable[str] = (),
    extra_include_paths: list[Path] | None = None,
    extra_cflags: list[str] | None = None,
    extra_cuda_cflags: list[str] | None = None,
) -> None:
    """Build once per source/ABI and safely reuse the library across nodes.

    ``torch.utils.cpp_extension.load`` always invokes Ninja. A shared build
    directory is unsafe across hosts whose system-header mtimes differ: Ninja
    may rebuild an existing ``.so`` in place while another process has it
    mapped. Content-addressing prevents stale reuse, and the advisory lock plus
    ready marker lets every later process load the completed library directly.
    """

    verbose = os.environ.get("PRIME_RL_KERNELS_BUILD_VERBOSE") == "1"
    name = _content_addressed_name(
        base_name,
        fingerprint_files=fingerprint_files,
        fingerprint=(
            *(extra_cflags or ()),
            *(extra_cuda_cflags or ()),
            *fingerprint,
        ),
    )
    build_directory = Path(_get_build_directory(name, verbose))
    library_path = build_directory / f"{name}{LIB_EXT}"
    ready_path = build_directory / ".prime_rl_ready"
    lock_path = build_directory / ".prime_rl_build.lock"

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if ready_path.is_file() and library_path.is_file():
            try:
                torch.ops.load_library(str(library_path))
                return
            except (OSError, RuntimeError):
                ready_path.unlink(missing_ok=True)
                library_path.unlink(missing_ok=True)

        # A previous builder may have died after linking but before publishing
        # the ready marker. Removing the library forces Ninja to relink it.
        if not ready_path.is_file():
            library_path.unlink(missing_ok=True)
        # This is Torch's internal baton. No process using the content-addressed
        # directory can legitimately hold it while we own the outer flock.
        (build_directory / "lock").unlink(missing_ok=True)

        load(
            name=name,
            sources=[str(path) for path in sources],
            extra_include_paths=[
                str(path) for path in (extra_include_paths or ())
            ],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            with_cuda=True,
            is_python_module=False,
            build_directory=str(build_directory),
            verbose=verbose,
        )
        ready_path.write_text("ready\n")
