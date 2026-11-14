"""Write the `config.json` of `Qwen3_5D3` for a d3 release (vllm-sr/d3-nano).

    uv run python examples/decision_models/d3/write_config.py RELEASE OUT

RELEASE is the downloaded release directory, which keeps the settings of the readout in `decision_config.json`. OUT
links the files of RELEASE, leaving out `MODEL_MANIFEST.json`, and holds the `config.json` this script writes: the
release's, as `Qwen3_5ForConditionalGeneration`, with the `d3` entry the model class reads (attention mode,
temperature, and readout rows, one per answer code). It reads and writes no weights: the trainer converts the linked
ones on the first load, into `OUT/prime`.
"""

import argparse
import json
from pathlib import Path

FORMAT_ID = "d3-code-readout-v1"


def write_config(release: Path, out: Path) -> dict:
    decision = json.loads((release / "decision_config.json").read_text())
    if decision["format_id"] != FORMAT_ID or decision["pooling"] != "last":
        raise ValueError(
            f"expected a {FORMAT_ID} release with last-token pooling, got {decision['format_id']!r} with "
            f"{decision['pooling']!r} pooling"
        )
    config = json.loads((release / "config.json").read_text())
    # A run's export copies this config.json, and d3's own loading code does not read what the export holds.
    config = {key: value for key, value in config.items() if key not in ("auto_map", "custom_pipelines")}
    config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    config["d3"] = {
        "attention_mode": decision["attention_mode"],
        "temperature": decision["temperature"],
        "readout_rows": len(decision["codes"]),
    }

    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # MODEL_MANIFEST.json holds the hash of the release's config.json.
    not_linked = {"config.json", "MODEL_MANIFEST.json"}
    for path in sorted(release.iterdir()):
        if path.is_file() and path.name not in not_linked and not path.name.startswith("."):
            (out / path.name).symlink_to(path.resolve())
    return config["d3"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("release", type=Path, help="the d3 release directory")
    parser.add_argument("out", type=Path, help="the directory to create")
    args = parser.parse_args()
    d3 = write_config(args.release, args.out)
    print(f"wrote {args.out} (d3 {d3})")


if __name__ == "__main__":
    main()
