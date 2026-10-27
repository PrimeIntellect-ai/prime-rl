"""Overlay a synthetic Qwen3.8 checkpoint with the config fields and processor files vLLM needs to load it.

The trainer's config class drops `text_config.output_gate_type` on save, and the synthetic checkpoint ships no
image processor configs; both are copied from the real Qwen3.8 snapshot. Weight files are symlinked unchanged.
"""

import argparse
import json
import shutil
from pathlib import Path

SNAPSHOT = Path(
    "/home/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
)
PROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")
RESTORED_TEXT_CONFIG_KEYS = ("output_gate_type",)
SKIPPED_FILES = ("config.json", "READY")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    args.output_dir.mkdir(parents=True)

    for path in source_dir.iterdir():
        if path.suffix == ".safetensors":
            (args.output_dir / path.name).symlink_to(path)
        elif path.name not in SKIPPED_FILES:
            shutil.copyfile(path, args.output_dir / path.name)
    for filename in PROCESSOR_FILES:
        shutil.copyfile(SNAPSHOT / filename, args.output_dir / filename)

    snapshot_text_config = json.loads((SNAPSHOT / "config.json").read_text())["text_config"]
    config = json.loads((source_dir / "config.json").read_text())
    for key in RESTORED_TEXT_CONFIG_KEYS:
        assert key not in config["text_config"], f"{key} is already in {source_dir}/config.json"
        config["text_config"][key] = snapshot_text_config[key]
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output_dir}: {', '.join(sorted(path.name for path in args.output_dir.iterdir()))}")


if __name__ == "__main__":
    main()
