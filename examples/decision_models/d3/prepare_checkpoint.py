"""Prepare the checkpoint of `Qwen3_5D3` from a d3 release (vllm-sr/d3-nano).

    uv run python examples/decision_models/d3/prepare_checkpoint.py RELEASE OUT

RELEASE is the downloaded release directory. It stores the backbone under the names of transformers' `Qwen3_5Model`
(`language_model.*`, `visual.*`) and the readout apart, in `readout.safetensors`. OUT is one checkpoint of the whole
model under the names prime-rl's Qwen3.5 class uses:

- `model.safetensors`: the backbone under `model.language_model.*` and `model.visual.*`, and the readout as
  `readout.weight`;
- `config.json`: the release's, as `Qwen3_5ForConditionalGeneration`, with the `d3` entry the model class reads
  (attention mode, temperature, readout rows);
- links to the release's other files (tokenizer, chat template, `decision_config.json`, d3's code, licence).
"""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch import Tensor

WEIGHTS = "model.safetensors"
PREFIXES = {"language_model.": "model.language_model.", "visual.": "model.visual."}
FORMAT_ID = "d3-code-readout-v1"


def renamed(name: str) -> str:
    for old, new in PREFIXES.items():
        if name.startswith(old):
            return new + name.removeprefix(old)
    raise ValueError(f"unexpected tensor {name!r} in a d3 backbone")


def prepare(release: Path, out: Path) -> Tensor:
    decision = json.loads((release / "decision_config.json").read_text())
    if decision["format_id"] != FORMAT_ID or decision["pooling"] != "last":
        raise ValueError(
            f"expected a {FORMAT_ID} release with last-token pooling, got {decision['format_id']!r} with "
            f"{decision['pooling']!r} pooling"
        )
    tensors = {renamed(name): tensor for name, tensor in load_file(release / WEIGHTS).items()}
    readout = tensors["readout.weight"] = load_file(release / "readout.safetensors")["weight"]

    out.mkdir(parents=True)
    save_file(tensors, out / WEIGHTS, metadata={"format": "pt"})
    config = json.loads((release / "config.json").read_text())
    config = {key: value for key, value in config.items() if key not in ("auto_map", "custom_pipelines")}
    config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    config["d3"] = {
        "attention_mode": decision["attention_mode"],
        "temperature": decision["temperature"],
        "readout_rows": readout.shape[0],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # MODEL_MANIFEST.json holds the hashes of the release's files, which the output no longer has.
    rewritten = {WEIGHTS, "config.json", "readout.safetensors", "MODEL_MANIFEST.json"}
    for path in sorted(release.iterdir()):
        if path.is_file() and path.name not in rewritten and not path.name.startswith("."):
            (out / path.name).symlink_to(path.resolve())
    return readout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("release", type=Path, help="the d3 release directory")
    parser.add_argument("out", type=Path, help="the directory to create")
    args = parser.parse_args()
    readout = prepare(args.release, args.out)
    print(f"wrote {args.out} (readout.weight {tuple(readout.shape)})")


if __name__ == "__main__":
    main()
