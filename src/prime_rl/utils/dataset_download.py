"""Download dataset files independently of the trainer's offline model cache."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()

    commit = HfApi().dataset_info(args.repo_id, revision=args.revision or "main").sha
    # Keep each snapshot self-contained on the synchronized volume.
    destination = args.cache_dir / "repositories" / args.repo_id / commit
    snapshot = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=commit,
        local_dir=destination,
    )
    print(snapshot)


if __name__ == "__main__":
    main()
