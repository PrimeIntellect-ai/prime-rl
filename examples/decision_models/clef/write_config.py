"""Write the `config.json` of `Qwen3_5Clef` for a Clef release: the release's, plus the joint head's sizes.

    uv run python examples/decision_models/clef/write_config.py RELEASE OUT

OUT links every file of RELEASE (a local download of e.g. Cloudflare/clef-flash) except `config.json`, which it
writes with the head's sizes (the release's `joint_head_config.json`) as `joint_head`. RELEASE stays as downloaded,
and no weights are read or written: prime-rl loads the backbone through the links, and the model class reads the
head from `joint_head.safetensors`.
"""

import argparse
import json
from pathlib import Path


def write_config(release: Path, out: Path) -> dict:
    config = json.loads((release / "config.json").read_text())
    config["joint_head"] = json.loads((release / "joint_head_config.json").read_text())
    out.mkdir(parents=True)
    for path in sorted(release.iterdir()):
        if path.name != "config.json" and not path.name.startswith("."):
            (out / path.name).symlink_to(path.resolve())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config["joint_head"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("release", type=Path, help="the Clef release directory")
    parser.add_argument("out", type=Path, help="the directory to create")
    args = parser.parse_args()
    joint_head = write_config(args.release, args.out)
    print(f"wrote {args.out} (joint_head {joint_head})")


if __name__ == "__main__":
    main()
