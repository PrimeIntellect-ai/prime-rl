"""Pinned episode loading; hidden chemistry is kept out of actor observations."""

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import dm_alchemy
from dm_alchemy.encode import chemistries_proto_conversion
from dm_alchemy.types import graphs

_PACKAGED_ASSETS = Path(__file__).resolve().parent / "assets"
ROOT = Path(
    os.environ.get(
        "RIME_ASSET_ROOT", _PACKAGED_ASSETS if _PACKAGED_ASSETS.exists() else Path(__file__).resolve().parents[2]
    )
)
LEVEL = "perceptual_mapping_randomized_with_random_bottleneck"


def verify_sources() -> dict:
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    checkout = ROOT / "vendor/dm_alchemy"
    if (checkout / ".git").exists():
        commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if commit != lock["environment_commit"]:
            raise ValueError("Upstream environment commit differs from sources.lock.json")
    installed = Path(dm_alchemy.__file__).parent
    for relative, digest in lock["environment_python_sha256"].items():
        if hashlib.sha256((installed / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Installed Alchemy source differs from pin: {relative}")
    for local, metadata in lock["files"].items():
        if hashlib.sha256((ROOT / local).read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Source checksum mismatch: {local}")
    return lock


def training_streams() -> list[dict]:
    verify_sources()
    return [
        json.loads(line)
        for line in (ROOT / "data/alchemy_train_streams.jsonl").read_text().splitlines()
        if line.strip()
    ]


@lru_cache(maxsize=1)
def prebuilt_episodes() -> list:
    path = ROOT / "vendor/dm_alchemy/dm_alchemy/chemistries" / LEVEL / "chemistries"
    return chemistries_proto_conversion.load_chemistries_and_items(str(path))


def chemistry_fingerprint(chemistry) -> str:
    # This is evaluator-only metadata, never an actor/writer input.
    value = {
        "potion_dim_map": list(map(int, chemistry.potion_map.dim_map)),
        "potion_dir_map": list(map(int, chemistry.potion_map.dir_map)),
        "stone_map": list(map(int, chemistry.stone_map.latent_pos_dir)),
        "rotation": chemistry.rotation.tolist(),
        "graph": graphs.convert_graph_to_adj_mat(chemistry.graph).tolist(),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def edge_count(chemistry) -> int:
    return int((graphs.convert_graph_to_adj_mat(chemistry.graph) != graphs.NO_EDGE).sum() // 2)
