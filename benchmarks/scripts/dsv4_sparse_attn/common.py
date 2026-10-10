"""Shared pieces of the DSv4 sparse attention benchmark: corpus access, inputs, FLOPs and provenance."""

import fnmatch
import hashlib
import importlib
import json
import pkgutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for import_root in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

DEFAULT_CORPUS_DIR = Path("~/tmp/dsv4-sparse-attn-bench/corpus").expanduser()
BASELINE_BACKEND = "tilelang"
HEADS = 64
DIM = 512
SM_SCALE = DIM**-0.5
DENSE_REFERENCE_MAX_LEN = 4096

PEAK_DENSE_BF16_TFLOPS = {"NVIDIA H200": 989.5}
PEAK_SOURCE = "https://www.nvidia.com/en-us/data-center/h200/ (H200 SXM BF16 1,979 TFLOPS with sparsity, halved)"


@dataclass(frozen=True)
class Item:
    id: str
    row: str
    composition: str
    total_len: int
    doc_lens: tuple[int, ...]
    layer_type: str
    cp: int
    cp_rank: int
    n_queries: int
    n_positions: int
    n_slots: int
    seed: int
    file: str
    indices_sha256: str

    @classmethod
    def from_json(cls, entry: dict) -> "Item":
        fields = {name: entry[name] for name in cls.__dataclass_fields__}
        fields["doc_lens"] = tuple(fields["doc_lens"])
        return cls(**fields)


def load_manifest(corpus_dir: Path) -> dict:
    return json.loads((corpus_dir / "manifest.json").read_text())


def select_items(manifest: dict, patterns: list[str] | None) -> list[Item]:
    """Manifest items whose id matches any glob in `patterns`, in manifest order; all of them if none."""
    items = [Item.from_json(entry) for entry in manifest["items"]]
    if not patterns:
        return items
    return [item for item in items if any(fnmatch.fnmatch(item.id, pattern) for pattern in patterns)]


def stream_items(manifest: dict) -> list[Item]:
    by_id = {entry["id"]: Item.from_json(entry) for entry in manifest["items"]}
    return [by_id[item_id] for item_id in manifest["stream"]]


def indices_sha256(indices: torch.Tensor) -> str:
    flat = indices.detach().to("cpu", torch.int32).contiguous()
    digest = hashlib.sha256(str(tuple(flat.shape)).encode())
    digest.update(flat.numpy().tobytes())
    return digest.hexdigest()


def load_indices(corpus_dir: Path, item: Item, device: str = "cuda") -> torch.Tensor:
    """`(1, n_queries, 1, n_slots)` int32 gather indices into a `(1, n_positions, 1, DIM)` KV buffer."""
    indices = torch.load(corpus_dir / item.file, weights_only=True)
    assert indices_sha256(indices) == item.indices_sha256, f"{item.id}: indices do not match the manifest hash"
    return indices.to(device)[None, :, None, :].contiguous()


def make_inputs(item: Item, requires_grad: bool) -> dict[str, torch.Tensor]:
    """`q`, `kv`, `sinks` and `grad_out`, regenerated from the item's seed at unit variance."""
    generator = torch.Generator(device="cuda").manual_seed(item.seed)

    def randn(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device="cuda", dtype=dtype)

    q = randn(1, item.n_queries, HEADS, DIM, dtype=torch.bfloat16)
    kv = randn(1, item.n_positions, 1, DIM, dtype=torch.bfloat16)
    sinks = randn(HEADS, dtype=torch.float32)
    grad_out = randn(1, item.n_queries, HEADS, DIM, dtype=torch.bfloat16)
    for leaf in (q, kv, sinks):
        leaf.requires_grad_(requires_grad)
    return {"q": q, "kv": kv, "sinks": sinks, "grad_out": grad_out}


def forward_backward(backend, q, kv, indices, sinks, scale, grad_out) -> tuple[torch.Tensor, ...]:
    """`(dq, dkv, dsink)` through `backend.fwd`, so the autograd glue and the torch-side sink gradient are timed."""
    out, _ = backend.fwd(q, kv, indices, sinks, scale)
    return torch.autograd.grad(out, (q, kv, sinks), grad_out)


def backend_names() -> list[str]:
    return sorted(module.name for module in pkgutil.iter_modules([str(SCRIPT_DIR / "backends")]))


def load_backend(name: str):
    return importlib.import_module(f"backends.{name}")


def slot_coverage(indices: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per query: `n_valid`, its count of valid slots, and `reach`, one past its last valid slot."""
    slots = indices[0, :, 0, :]
    is_valid = slots >= 0
    slot_numbers = torch.arange(1, slots.shape[-1] + 1, device=slots.device)
    return {
        "n_valid": is_valid.sum(-1),
        "reach": torch.where(is_valid, slot_numbers, 0).amax(-1),
    }


def executed_slots(coverage: dict[str, torch.Tensor], tile: int) -> int:
    """Slots a kernel touches when it reads each query's leading tiles of `tile` slots up to its reach."""
    return int(((coverage["reach"] + tile - 1) // tile * tile).sum())


def useful_flops(sum_valid: int) -> dict[str, int]:
    fwd = 4 * HEADS * DIM * sum_valid
    bwd = 10 * HEADS * DIM * sum_valid
    return {"fwd": fwd, "bwd": bwd, "fwd_bwd": fwd + bwd}


def _run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def git_state() -> dict:
    sha = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"]))
    return {"sha": sha, "dirty": dirty}


def gpu_provenance() -> dict:
    """SKU, driver, clocks and power limit of the current CUDA device, read by its UUID."""
    uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    query = "name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.limit,power.max_limit"
    values = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader", "-i", uuid])
    return {
        "uuid": uuid,
        "name": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "query": dict(zip(query.split(","), (value.strip() for value in values.split(",")))),
        "nvidia_smi_clock_power": _run(["nvidia-smi", "-q", "-d", "CLOCK,POWER", "-i", uuid]),
    }


def provenance(corpus_hash: str) -> dict:
    from importlib.metadata import version

    return {
        "hostname": socket.gethostname(),
        "git": git_state(),
        "corpus_hash": corpus_hash,
        "gpu": gpu_provenance(),
        "versions": {package: version(package) for package in ("torch", "tilelang")},
        "argv": sys.argv,
    }


def peak_dense_tflops() -> float | None:
    return PEAK_DENSE_BF16_TFLOPS.get(torch.cuda.get_device_name())
