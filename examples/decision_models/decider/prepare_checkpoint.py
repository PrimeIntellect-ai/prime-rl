"""Prepare the checkpoint of `Qwen3_5Decider`: a Decider release under prime-rl's weight names.

    uv run python examples/decision_models/decider/prepare_checkpoint.py RELEASE OUT

RELEASE is a local download of perplexity-ai/pplx-decider-flash-v1.2. It holds the backbone as a bare `Qwen3_5Model`
(`language_model.*`, `visual.*`) and the readout apart (`readout.safetensors`, key `weight`), and prime-rl loads every
parameter by its name in `Qwen3_5Decider` (`model.language_model.*`, `model.visual.*`, `readout.weight`). So OUT gets
every weight shard rewritten under the `model.` prefix (a shard stores the names of its tensors, so it cannot be
linked), `decider_readout.safetensors` with the readout as `readout.weight`, and an index that names both. Every other
file of RELEASE is linked: the config, the tokenizer with its chat template, and `decision_config.json`, whose answer
codes `DeciderEncoder` reads.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

INDEX = "model.safetensors.index.json"
READOUT_FILE = "decider_readout.safetensors"


def prime_rl_name(name: str) -> str:
    """A release weight's name in prime-rl's model."""
    if not name.startswith(("language_model.", "visual.")):
        raise ValueError(f"unexpected weight {name!r} in a Decider release")
    return f"model.{name}"


def prepare(release: Path, out: Path) -> dict[str, torch.Tensor]:
    index = json.loads((release / INDEX).read_text())
    readout = {"readout.weight": load_file(release / "readout.safetensors")["weight"]}

    out.mkdir(parents=True)
    weight_map = {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {prime_rl_name(name): tensor for name, tensor in load_file(release / shard).items()}
        save_file(tensors, out / shard, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(tensors, shard))
    save_file(readout, out / READOUT_FILE, metadata={"format": "pt"})
    weight_map.update(dict.fromkeys(readout, READOUT_FILE))
    total = index.get("metadata", {}).get("total_size", 0) + sum(t.numel() * t.element_size() for t in readout.values())
    (out / INDEX).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")

    for path in sorted(release.iterdir()):
        if path.name != INDEX and path.suffix != ".safetensors" and not path.name.startswith("."):
            (out / path.name).symlink_to(path.resolve())
    return readout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("release", type=Path, help="the Decider release directory")
    parser.add_argument("out", type=Path, help="the directory to create")
    args = parser.parse_args()
    readout = prepare(args.release, args.out)
    print(f"wrote {args.out} (readout.weight {tuple(readout['readout.weight'].shape)})")


if __name__ == "__main__":
    main()
