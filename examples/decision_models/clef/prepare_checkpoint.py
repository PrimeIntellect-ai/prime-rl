"""Prepare the checkpoint of `Qwen3_5Clef` from a Clef release: the backbone plus the joint head and the lexical copy.

    uv run python examples/decision_models/clef/prepare_checkpoint.py RELEASE OUT

OUT links every file of RELEASE (a local download of e.g. Cloudflare/clef-flash) except the weight index,
`config.json` and `joint_head.safetensors`, whose unprefixed keys prime-rl would read as model weights. It adds
`clef_heads.safetensors` with the joint head under `joint_head.` and a copy of the LM head's rows as
`lexical_embedding.weight`, an updated `model.safetensors.index.json`, and `config.json` with the head's sizes
(the release's `joint_head_config.json`) as `joint_head`. The head's three scalars are stored with one element each,
as the model class holds them.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX = "model.safetensors.index.json"
HEADS_FILE = "clef_heads.safetensors"
SKIPPED = {INDEX, "config.json", "joint_head.safetensors"}


def prepare(release: Path, out: Path) -> dict[str, torch.Tensor]:
    index = json.loads((release / INDEX).read_text())
    weight_map = index["weight_map"]
    heads = {
        f"joint_head.{name}": tensor.reshape(1) if tensor.ndim == 0 else tensor
        for name, tensor in load_file(release / "joint_head.safetensors").items()
    }
    with safe_open(release / weight_map["lm_head.weight"], framework="pt") as shard:
        heads["lexical_embedding.weight"] = shard.get_tensor("lm_head.weight")

    out.mkdir(parents=True)
    for path in sorted(release.iterdir()):
        if path.name not in SKIPPED and not path.name.startswith("."):
            (out / path.name).symlink_to(path.resolve())
    save_file(heads, out / HEADS_FILE, metadata={"format": "pt"})
    index["weight_map"] = {**weight_map, **dict.fromkeys(heads, HEADS_FILE)}
    metadata = index.setdefault("metadata", {})
    metadata["total_size"] = metadata.get("total_size", 0) + sum(t.numel() * t.element_size() for t in heads.values())
    (out / INDEX).write_text(json.dumps(index, indent=2) + "\n")
    config = json.loads((release / "config.json").read_text())
    config["joint_head"] = json.loads((release / "joint_head_config.json").read_text())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return heads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("release", type=Path, help="the Clef release directory")
    parser.add_argument("out", type=Path, help="the directory to create")
    args = parser.parse_args()
    heads = prepare(args.release, args.out)
    print(f"wrote {args.out} ({len(heads)} tensors in {HEADS_FILE})")


if __name__ == "__main__":
    main()
