#!/usr/bin/env python3
"""Append a `.cuXX` suffix to a wheel's local version (PEP 440).

Repacks a built wheel with the same binary content but renamed:
    deep_gemm-2.3.0+477618c-cp312-cp312-linux_x86_64.whl
        --cu cu13 -->
    deep_gemm-2.3.0+477618c.cu13-cp312-cp312-linux_x86_64.whl

Updates METADATA (Version), dist-info dir name, RECORD hashes/paths, and the
output filename. Used by install_deep_gemm.sh / install_ep_kernels.sh so wheels
stamped with the build's CUDA major can coexist with prior cu12 builds in the
same release.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def _rehash(data: bytes) -> tuple[str, int]:
    h = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(h).rstrip(b"=").decode("ascii"), len(data)


def _strip_runpath(so_path: Path) -> None:
    """Strip DT_RUNPATH/DT_RPATH from a shared object using patchelf."""
    subprocess.run(["patchelf", "--remove-rpath", str(so_path)], check=True)


def _set_runpath(so_path: Path, runpath: str) -> None:
    """Overwrite DT_RUNPATH on a shared object using patchelf."""
    subprocess.run(["patchelf", "--set-rpath", runpath, str(so_path)], check=True)


def rebrand(
    src: Path,
    cu: str | None,
    out_dir: Path,
    strip_runpath: bool = False,
    set_runpath: str | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(src) as z:
        names = z.namelist()
        dist_info = next(n.split("/", 1)[0] for n in names if n.endswith(".dist-info/METADATA"))
        # "name-version" → split on the LAST hyphen so versions with hyphens survive
        pkg_name, old_ver = dist_info[: -len(".dist-info")].rsplit("-", 1)
        new_ver = f"{old_ver}.{cu}" if cu else old_ver
        new_dist_info = dist_info if cu is None else f"{pkg_name}-{new_ver}.dist-info"

        rewrites = {n: (new_dist_info + n[len(dist_info) :] if n.startswith(dist_info + "/") else n) for n in names}

        meta = z.read(dist_info + "/METADATA").decode()
        if cu is None:
            new_meta = meta.encode()
        else:
            new_meta = meta.replace(f"Version: {old_ver}\n", f"Version: {new_ver}\n", 1).encode()
            if b"Version: " + new_ver.encode() not in new_meta:
                sys.exit(f"failed to rewrite Version line in {src}")

        # Optionally rewrite RUNPATH on every .so. Patchelf needs a real file to
        # operate on, so extract the .so to a temp dir, modify it, and use the
        # modified bytes in both the repack and the RECORD hash recomputation.
        modified_so: dict[str, bytes] = {}
        if strip_runpath or set_runpath is not None:
            if strip_runpath and set_runpath is not None:
                sys.exit("--strip-runpath and --set-runpath are mutually exclusive")
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                for n in names:
                    if not (n.endswith(".so") or ".so." in n):
                        continue
                    so_path = tdp / Path(n).name
                    so_path.write_bytes(z.read(n))
                    try:
                        if strip_runpath:
                            _strip_runpath(so_path)
                        else:
                            _set_runpath(so_path, set_runpath)  # type: ignore[arg-type]
                    except subprocess.CalledProcessError:
                        # Not all .so files have RUNPATH; that's fine.
                        pass
                    modified_so[n] = so_path.read_bytes()

        record_lines: list[str] = []
        for line in z.read(dist_info + "/RECORD").decode().splitlines():
            if not line:
                continue
            path, _, rest = line.partition(",")
            new_path = rewrites.get(path, path)
            if path == dist_info + "/METADATA":
                h, sz = _rehash(new_meta)
                record_lines.append(f"{new_path},{h},{sz}")
            elif path == dist_info + "/RECORD":
                record_lines.append(f"{new_path},,")
            elif path in modified_so:
                h, sz = _rehash(modified_so[path])
                record_lines.append(f"{new_path},{h},{sz}")
            else:
                record_lines.append(f"{new_path},{rest}")
        new_record = ("\n".join(record_lines) + "\n").encode()

        # Filename: name-version-pyver-abi-plat.whl
        stem = src.stem
        head, sep, tail = stem.partition(f"-{old_ver}-")
        if not sep:
            sys.exit(f"can't split {stem} on -{old_ver}-")
        out_path = out_dir / f"{head}-{new_ver}-{tail}.whl"

        with ZipFile(out_path, "w", ZIP_DEFLATED) as zout:
            for n in names:
                new_n = rewrites[n]
                if n == dist_info + "/METADATA":
                    zout.writestr(new_n, new_meta)
                elif n == dist_info + "/RECORD":
                    zout.writestr(new_n, new_record)
                elif n in modified_so:
                    zout.writestr(new_n, modified_so[n])
                else:
                    zout.writestr(new_n, z.read(n))

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--cu", help="e.g. cu13 (omit to keep version unchanged; useful with --strip-runpath only)")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--replace", action="store_true", help="delete the source wheel after success")
    ap.add_argument(
        "--strip-runpath",
        action="store_true",
        help="strip DT_RUNPATH/RPATH from every .so so the wheel is portable",
    )
    ap.add_argument(
        "--set-runpath",
        metavar="PATH",
        help="overwrite DT_RUNPATH on every .so to PATH (e.g. /tmp/deepep_build/nvshmem/lib)",
    )
    args = ap.parse_args()
    out = rebrand(
        args.wheel,
        args.cu,
        args.out_dir,
        strip_runpath=args.strip_runpath,
        set_runpath=args.set_runpath,
    )
    print(out)
    if args.replace and out.resolve() != args.wheel.resolve():
        args.wheel.unlink()


if __name__ == "__main__":
    main()
