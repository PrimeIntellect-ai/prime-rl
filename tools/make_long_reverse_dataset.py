"""Write deterministic long-string reverse-text inputs as a local Parquet taskset."""

import argparse
import random
import string
from pathlib import Path

from datasets import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rng = random.Random(42)
    prompts = ["".join(rng.choices(string.ascii_letters + " ", k=length)) for length in (1600, 5600) * 16]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"prompt": prompts}).to_parquet(args.output_dir / "train.parquet")
