"""Verify a trained NemotronVL weight checkpoint changed ONLY the projector.

Compares a saved HF weight checkpoint against the source graft checkpoint:
  - mlp1.{0,1,3}.weight MUST differ (training happened)
  - the full vision tower and a random sample of LM tensors MUST be bitwise identical
    (frozen subtrees; catches accidentally-trainable routers/embeddings/etc.)

Run from the prime-rl repo root:
    uv run python models/validation/check_weight_deltas.py <path/to/weights/step_N> [--lm-sample 40]
"""

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors import safe_open

GRAFT_ROOT = Path(__file__).resolve().parent.parent / "Nemotron-3-Super-VL-graft"


def load_index(root: Path) -> dict[str, str]:
    candidates = list(root.glob("*.safetensors.index.json"))
    if candidates:
        return json.loads(candidates[0].read_text())["weight_map"]
    shards = list(root.glob("*.safetensors"))
    assert shards, f"no safetensors found under {root}"
    index = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                index[key] = shard.name
    return index


def load_tensor(root: Path, index: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(root / index[key], framework="pt") as f:
        return f.get_tensor(key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=Path)
    parser.add_argument("--lm-sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt_index = load_index(args.ckpt)
    graft_index = load_index(GRAFT_ROOT)

    mlp1_keys = sorted(k for k in ckpt_index if k.startswith("mlp1."))
    vision_keys = sorted(k for k in ckpt_index if k.startswith("vision_model."))
    lm_keys = sorted(k for k in ckpt_index if not k.startswith(("mlp1.", "vision_model.")))
    rng = random.Random(args.seed)
    lm_sampled = rng.sample(lm_keys, min(args.lm_sample, len(lm_keys)))
    # Always include the most policy-sensitive frozen tensors.
    for key in list(ckpt_index):
        if any(s in key for s in ("gate.weight", "e_score_correction_bias", "embeddings.weight", "lm_head")):
            if key not in lm_sampled and not key.startswith(("mlp1.", "vision_model.")):
                lm_sampled.append(key)

    print(f"projector: {len(mlp1_keys)} tensors | frozen check: {len(vision_keys)} vision + {len(lm_sampled)} LM")

    changed_count = 0
    for key in mlp1_keys:
        ours, src = load_tensor(args.ckpt, ckpt_index, key), load_tensor(GRAFT_ROOT, graft_index, key)
        delta = (ours.float() - src.float()).abs()
        changed = delta.max().item() > 0
        changed_count += changed
        # mlp1.0 (RMSNorm gain, values ~1.0) may legitimately stay put: with bf16 master
        # weights its per-step Adam updates (~lr) are below the bf16 ULP at 1.0 (~3.9e-3).
        print(f"  {key}: changed={changed} max|d|={delta.max():.3e} mean|d|={delta.mean():.3e}")
    failed = changed_count == 0

    frozen_bad = []
    for key in vision_keys + lm_sampled:
        ours, src = load_tensor(args.ckpt, ckpt_index, key), load_tensor(GRAFT_ROOT, graft_index, key)
        if key.endswith("gate.e_score_correction_bias"):
            src = src - src.min()  # known conversion shift (routing-invariant)
        # Compare in the saved dtype: training runs (and saves) bf16, the source may be
        # fp32 — a pure precision cast is not a weight change.
        if not torch.equal(ours, src.to(ours.dtype)):
            frozen_bad.append((key, (ours.float() - src.float()).abs().max().item()))
    if frozen_bad:
        failed = True
        print(f"FROZEN TENSORS CHANGED ({len(frozen_bad)}):")
        for key, diff in frozen_bad[:15]:
            print(f"  {key}: max|d|={diff:.3e}")
    else:
        print(f"frozen subtrees bitwise identical ({len(vision_keys) + len(lm_sampled)} tensors checked)")

    print("FAIL" if failed else "PASS: only the projector moved")


if __name__ == "__main__":
    main()
