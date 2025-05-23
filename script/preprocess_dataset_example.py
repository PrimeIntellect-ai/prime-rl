"""
This is just an example of how to preprocess a dataset in the genesys/prime format.

"""

import argparse
import json

from datasets import load_dataset

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--push_to_hub", type=bool, default=False)
    args.add_argument("--hf_path", type=str, default="PrimeIntellect/reverse_text_dataset_debug")
    args = args.parse_args()

    dataset = load_dataset("agentlans/wikipedia-paragraphs", split="train").map(
        lambda x: {
            "prompt": f"Reverse the given text.{x['text'][0:50]}",
            "verification_info": json.dumps({"ground_truth": x["text"][0:50][::-1]}),
            "task_type": "reverse_text",
        }
    )

    if args.push_to_hub:
        dataset.push_to_hub(args.hf_path)
