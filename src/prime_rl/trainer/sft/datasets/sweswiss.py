import os
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from prime_rl.trainer.sft.config import LossMaskConfig
from prime_rl.trainer.sft.datasets.utils import messages_to_sample
import logging

logger = logging.getLogger(__name__)



def row_to_sample(index: int, tokenizer: PreTrainedTokenizer, data: dict, loss_mask_config: LossMaskConfig, with_text: bool = False) -> dict:
    messages = data['messages']
    return messages_to_sample(index, tokenizer, messages, loss_mask_config, with_text=with_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SWE-Swiss to a tokenized dataset")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--split", type=str, default="train", help="comma separated list of splits to use")
    parser.add_argument("--dataset", type=str, default="SWE-Swiss/SWESwiss-SFT-Merged-10K")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--peek", action="store_true", help="Peek at the dataset")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    splits = args.split.split(",")
    for split in splits:
        ds = load_dataset(args.dataset, split=split)
        if args.peek:
            print(row_to_sample(0, tokenizer, ds[0], LossMaskConfig(), with_text=args.peek))
            return
        num_proc = args.num_proc if args.num_proc is not None else max(1, os.cpu_count() - 1)
        ds = ds.map(
            lambda x, idx: row_to_sample(idx, tokenizer, x, LossMaskConfig()),
            with_indices=True,
            remove_columns=ds.column_names,
            num_proc=num_proc,
            desc="Tokenizing dataset",
        )
        ds.save_to_disk(os.path.join(args.output_path, split))


if __name__ == "__main__":
    main()