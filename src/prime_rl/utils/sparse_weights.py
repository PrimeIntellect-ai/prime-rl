import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

SPARSE_WEIGHTS_FORMAT = "prime_rl_sparse_filesystem_v1"
SPARSE_WEIGHTS_MANIFEST = "sparse_manifest.json"


def read_sparse_manifest(weight_dir: Path) -> dict[str, Any] | None:
    manifest_path = weight_dir / SPARSE_WEIGHTS_MANIFEST
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def write_sparse_manifest(weight_dir: Path, manifest: dict[str, Any]) -> None:
    manifest = {"format": SPARSE_WEIGHTS_FORMAT, **manifest}
    manifest_path = weight_dir / SPARSE_WEIGHTS_MANIFEST
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(manifest_path)


def parse_step_from_dir(step_dir: Path) -> int:
    name = step_dir.name
    if not name.startswith("step_"):
        raise ValueError(f"Expected step directory name like step_<n>, got {step_dir}")
    return int(name.removeprefix("step_"))


def load_weight_map(weight_dir: Path) -> dict[str, str]:
    index_path = weight_dir / SAFE_WEIGHTS_INDEX_NAME
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        return dict(index["weight_map"])

    weight_map: dict[str, str] = {}
    for safetensors_path in sorted(weight_dir.glob("*.safetensors")):
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = safetensors_path.name
    return weight_map


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key).clone()
    return state_dict


def save_safetensors(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    contiguous_state_dict = {key: value.contiguous().clone() for key, value in state_dict.items()}
    save_file(contiguous_state_dict, path, metadata={"format": "pt"})


def apply_sparse_delta(delta_dir: Path, target_dir: Path) -> None:
    manifest = read_sparse_manifest(delta_dir)
    if manifest is None or manifest.get("type") != "delta":
        raise ValueError(f"{delta_dir} is not a sparse delta directory")

    weight_map = load_weight_map(target_dir)
    for patch in manifest.get("patch_files", []):
        patch_path = delta_dir / patch["file"]
        entries = patch.get("tensors", [])
        if not entries:
            continue

        with safe_open(patch_path, framework="pt", device="cpu") as patch_file:
            entries_by_shard: dict[str, list[dict[str, Any]]] = {}
            for entry in entries:
                name = entry["name"]
                if name not in weight_map:
                    raise KeyError(f"Sparse delta references unknown weight {name}")
                entries_by_shard.setdefault(weight_map[name], []).append(entry)

            for shard_file, shard_entries in entries_by_shard.items():
                shard_path = target_dir / shard_file
                shard_state = load_safetensors(shard_path)
                for entry in shard_entries:
                    name = entry["name"]
                    if name not in shard_state:
                        raise KeyError(f"Sparse delta references missing tensor {name} in {shard_path}")

                    tensor = shard_state[name].contiguous()
                    indices = patch_file.get_tensor(entry["indices_key"]).to(torch.long)
                    values = patch_file.get_tensor(entry["values_key"]).to(tensor.dtype)

                    flat = tensor.view(-1)
                    flat[indices] = values
                    shard_state[name] = tensor

                save_safetensors(shard_state, shard_path)
