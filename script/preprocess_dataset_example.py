"""
This is just an example of how to preprocess a dataset in the genesys/prime format.

"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import upload_folder

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--push_to_hub", type=bool, default=False)
    args.add_argument("--hf_path", type=str, default="PrimeIntellect/reverse_text_dataset_debug")
    args.add_argument("--output_path", type=str, default="example_dataset")
    args = args.parse_args()

    dataset = load_dataset("agentlans/wikipedia-paragraphs", split="train").map(
        lambda x: {
            "prompt": f"Reverse the given text.{x['text']}",
            "verification_info": json.dumps({"ground_truth": x["text"][::-1]}),
            "task_type": "reverse_text",
        }
    )

    dataset.save_to_disk(args.output_path)

    readme_path = Path(args.output_path) / "README.md"
    with open(__file__, "r") as f:
        script_content = f.read()
    with open(readme_path, "w") as f:
        f.write(f"```\n{script_content}\n```")

    if args.push_to_hub:
        upload_folder(repo_id=args.hf_path, folder_path=args.output_path, repo_type="dataset")
