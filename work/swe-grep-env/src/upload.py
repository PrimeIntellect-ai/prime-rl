"""Upload parquet files to Hugging Face Hub."""

import argparse
from datasets import Dataset
from huggingface_hub import login



ds_path = "/Users/christian/dev/my-prime/my-prime-rl/prime-rl/work/swe-grep-env/data/grep_dataset_2k_v3.parquet"


def upload_parquet(file_path: str, repo_id: str = "cdreetz/swe-grep-env-v3", split: str = "2k_v3"):
    """Upload a parquet file to Hugging Face Hub.

    Args:
        file_path: Path to the parquet file to upload
        repo_id: Hugging Face repository ID (default: cdreetz/swe-grep-env)
        split: Dataset split name (default: 5k_v3)
    """
    # Load the parquet file
    dataset = Dataset.from_parquet(file_path)

    print(f"Loaded dataset with {len(dataset)} rows")
    print(f"Columns: {dataset.column_names}")

    # Push to hub
    dataset.push_to_hub(
        repo_id=repo_id,
        split=split,
        private=False,
    )

    print(f"Successfully uploaded to {repo_id} (split: {split})")


if __name__ == "__main__":
    login()

    upload_parquet(file_path=ds_path)
