"""Small, CPU-only serialization helpers shared by trainer and inference diagnostics."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def code_provenance() -> dict:
    root = Path(__file__).resolve().parents[3]
    result = {"source_root": str(root), "packages": {}}
    for package in ("torch", "transformers", "vllm", "verifiers", "renderers", "prime-rl-configs"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    result["module_origins"] = {}
    for module in ("prime_rl", "transformers", "vllm", "verifiers", "renderers"):
        spec = importlib.util.find_spec(module)
        result["module_origins"][module] = spec.origin if spec is not None else None
    source_hash = hashlib.sha256()
    directories = [path for path in (root / "src", root / "packages" / "prime-rl-configs" / "src") if path.is_dir()]
    if not directories:
        directories = [Path(__file__).resolve().parents[1]]
    for directory in directories:
        for path in sorted(directory.rglob("*.py")):
            source_hash.update(str(path.relative_to(root)).encode())
            source_hash.update(path.read_bytes())
    result["source_tree_sha256"] = source_hash.hexdigest()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=False)
        if head.returncode == 0:
            status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=root, capture_output=True, check=True)
            diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, capture_output=True, check=True)
            branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, capture_output=True, check=True)
            result["git"] = {
                "commit": head.stdout.decode().strip(),
                "branch": branch.stdout.decode().strip(),
                "dirty": bool(status.stdout),
                "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
            }
        else:
            result["git"] = None
    except FileNotFoundError:
        result["git"] = None
    return result


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    result = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
    if value.numel():
        finite = value.float()[torch.isfinite(value)]
        result["nonfinite"] = value.numel() - finite.numel()
        if finite.numel():
            result.update(min=finite.min().item(), max=finite.max().item(), mean=finite.mean().item())
    return result


def image_summary(url: str) -> dict[str, Any]:
    header, separator, encoded = url.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        return {"inline": False, "url_sha256": hashlib.sha256(url.encode()).hexdigest()}
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as source:
        rgb = source.convert("RGB")
        return {
            "inline": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "size": list(source.size),
            "source_mode": source.mode,
        }


def tensor_tree_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_summary(value)
    if isinstance(value, Mapping):
        return {key: tensor_tree_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree_summary(item) for item in value]
    return {"type": type(value).__name__}


def payload_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (str, bytes)):
        return len(value.encode() if isinstance(value, str) else value)
    if isinstance(value, Mapping):
        return sum(payload_bytes(k) + payload_bytes(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sum(map(payload_bytes, value))
    return 16


def cpu_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, Mapping):
        return {key: cpu_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_payload(item) for item in value]
    return value


def save_capture(path: Path, payload: dict, max_bytes: int) -> dict:
    """Bound both tensor payload and final serialized size; never leave a partial artifact."""
    estimated = payload_bytes(payload)
    if estimated > max_bytes:
        return {"skipped": "payload_budget", "estimated_bytes": estimated}
    buffer = io.BytesIO()
    torch.save(cpu_payload(payload), buffer)
    if buffer.tell() > max_bytes:
        return {"skipped": "serialized_budget", "serialized_bytes": buffer.tell()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_bytes(buffer.getbuffer())
    temporary.replace(path)
    return {"path": str(path), "bytes": buffer.tell()}


def append_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
