from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiskUsage:
    total_bytes: int
    used_bytes: int
    free_bytes: int


def _resolve_existing_path(path: Path) -> Path:
    """
    Resolve a path for disk-usage queries.

    shutil.disk_usage requires the path to exist. For a not-yet-created directory,
    we walk up to the first existing parent.
    """
    cur = path
    while not cur.exists():
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return cur


def get_disk_usage(path: Path) -> DiskUsage:
    existing = _resolve_existing_path(path)
    usage = shutil.disk_usage(str(existing))
    return DiskUsage(total_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free)


def format_bytes_binary(num_bytes: int, precision: int = 1) -> str:
    if num_bytes < 0:
        return f"-{format_bytes_binary(-num_bytes, precision=precision)}"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(num_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(value)} {units[unit_idx]}"
    return f"{value:.{precision}f} {units[unit_idx]}"


def format_percent(value: float, precision: int = 1) -> str:
    return f"{value * 100:.{precision}f}%"

