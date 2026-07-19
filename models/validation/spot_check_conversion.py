"""Spot-check prime->HF conversion of the real graft checkpoint against its HF source.

Samples a few LM layers (one of each block type + extras), the embeddings/norm/lm_head,
the full vision tower, and the projector from the pre-converted prime/ snapshot, runs
them through convert_nemotron_vl_prime_to_hf, and compares every produced tensor
bitwise against the HF shards in the graft root.

Known intentional non-identity: `gate.e_score_correction_bias` is shifted by its min
during HF->prime (bf16 representability; routing-invariant), and prime->HF does not
undo the shift. The check compares against `hf_value - hf_value.min()` for that key.

Run from the prime-rl repo root:
    uv run python models/validation/spot_check_conversion.py [--layers-per-type 2] [--seed 0]
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from safetensors import safe_open

from prime_rl.trainer.models.nemotron_vl.converting_nemotron_vl import convert_nemotron_vl_prime_to_hf

GRAFT_ROOT = Path(__file__).resolve().parent.parent / "Nemotron-3-Super-VL-graft"
PRIME_ROOT = GRAFT_ROOT / "prime"

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(mamba|self_attn|mlp)\.")


def load_index(root: Path) -> dict[str, str]:
    return json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]


def load_tensors(root: Path, index: dict[str, str], keys: list[str]) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index[key], []).append(key)
    tensors = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(root / shard, framework="pt") as f:
            for key in shard_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers-per-type", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    prime_index = load_index(PRIME_ROOT)
    hf_index = load_index(GRAFT_ROOT)

    layers_by_type: dict[str, set[int]] = {"mamba": set(), "self_attn": set(), "mlp": set()}
    for key in prime_index:
        m = _LAYER_RE.match(key)
        if m:
            layers_by_type[m.group(2)].add(int(m.group(1)))

    sampled_layers: set[int] = set()
    for block_type, layer_ids in layers_by_type.items():
        picked = rng.sample(sorted(layer_ids), min(args.layers_per_type, len(layer_ids)))
        print(f"sampled {block_type} layers: {picked}")
        sampled_layers.update(picked)

    def selected(key: str) -> bool:
        m = _LAYER_RE.match(key)
        if m:
            return int(m.group(1)) in sampled_layers
        # Everything that is not a decoder layer: embed/norm/lm_head/visual/mlp1 (+ layer norms).
        layer_prefix = key.startswith("model.language_model.layers.")
        return not layer_prefix

    # Per-layer norms of sampled layers live under layers.N.norm.weight; include them.
    prime_keys = [
        k
        for k in prime_index
        if selected(k)
        or (_m := re.match(r"^model\.language_model\.layers\.(\d+)\.", k)) is not None
        and int(_m.group(1)) in sampled_layers
    ]
    print(f"loading {len(prime_keys)} prime tensors ...")
    state_dict = load_tensors(PRIME_ROOT, prime_index, prime_keys)
    total_gb = sum(t.numel() * t.element_size() for t in state_dict.values()) / 1e9
    print(f"loaded {total_gb:.1f} GB, converting prime -> hf ...")

    convert_nemotron_vl_prime_to_hf(state_dict)

    missing, mismatched, checked = [], [], 0
    for key, tensor in sorted(state_dict.items()):
        if key not in hf_index:
            missing.append(key)
            continue
        expected = load_tensors(GRAFT_ROOT, hf_index, [key])[key]
        if key.endswith("gate.e_score_correction_bias"):
            expected = expected - expected.min()
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype or not torch.equal(tensor, expected):
            max_diff = (
                (tensor.float() - expected.float()).abs().max().item()
                if tensor.shape == expected.shape
                else float("nan")
            )
            mismatched.append((key, tuple(tensor.shape), tuple(expected.shape), max_diff))
        checked += 1

    print(f"\nchecked {checked} tensors bitwise against HF shards")
    if missing:
        print(f"FAIL: {len(missing)} converted keys missing from HF index, e.g. {missing[:5]}")
    if mismatched:
        print(f"FAIL: {len(mismatched)} tensors mismatch:")
        for key, got, want, max_diff in mismatched[:20]:
            print(f"  {key}: got {got} want {want} max|diff|={max_diff}")
    if not missing and not mismatched:
        print("PASS: all converted tensors bitwise-identical to HF source (bias shift accounted for)")


if __name__ == "__main__":
    main()
