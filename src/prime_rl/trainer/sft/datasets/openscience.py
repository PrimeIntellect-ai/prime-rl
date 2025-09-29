import re
import datasets
import argparse
import os
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import cast
from asyncio import Semaphore
from datasets import Features, Sequence, Value
from prime_rl.trainer.sft.config import LossMaskConfig
from prime_rl.trainer.sft.datasets.utils import build_loss_mask, messages_to_sample
    
logger = logging.getLogger(__name__)

def row_to_sample(index: int, tokenizer: PreTrainedTokenizer, data: dict, loss_mask_config: LossMaskConfig, with_text: bool = False) -> dict:
    messages = [{"role": "user", "content": data["input"]}, {"role": "assistant", "content": data["output"]}]
    return messages_to_sample(index, tokenizer, messages, loss_mask_config, with_text=with_text)


def _extract_boxed_contents(text: str) -> list[str]:
    """
    Extract contents inside all occurrences of "\\boxed{...}" in a LaTeX-like string.

    - Supports nested braces: e.g., \\boxed{{answer with {curly} braces}}
    - Ignores escaped braces: "\\{" and "\\}" do not affect balancing
    - Allows optional whitespace after \\boxed
    """
    results: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start_index = match.end()  # position right after the opening '{'
        depth = 1
        index = start_index
        text_length = len(text)

        while index < text_length and depth > 0:
            char = text[index]

            # Treat escaped characters (e.g., '\{' or '\}') as literals
            if char == "\\":
                # Skip the backslash and the next character if present
                index += 1
                if index < text_length:
                    index += 1
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            index += 1

        # depth == 0 means we found a matching closing '}'
        if depth == 0:
            results.append(text[start_index:index - 1])

    return results


def verification(split: str):
    """verify that all assistant outputs are the right answer"""
    ds = datasets.load_dataset("nvidia/OpenScienceReasoning-2", split=split)
    correct = 0
    incorrect = 0
    total = len(ds)
    incorect_instances = []
    for d in tqdm(ds, desc="Verifying"):
        matches = _extract_boxed_contents(d["output"])  # robustly handle nested/escaped braces
        if len(matches) == 0:
            logger.info(f"bad format: Correct: {correct}, Incorrect: {incorrect}, Total: {total}")
            incorrect += 1
            continue
        answer = matches[-1]
        if answer == d["expected_answer"]:
            correct += 1
        else:
            incorect_instances.append(d)
            logger.info(f"incorrect answer: Correct: {correct}, Incorrect: {incorrect}, Total: {total}")
            incorrect += 1
    logger.info(f"Final: Correct: {correct}, Incorrect: {incorrect}, Total: {total}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert OpenScienceReasoning-2 to a tokenized dataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--peek", action="store_true", help="Peek at the dataset")
    parser.add_argument("--num_proc", type=int, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.verify:
        logger.info("Verifying dataset...")
        verification(args.split)
    else:
        logger.info("Converting dataset...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        ds = datasets.load_dataset("nvidia/OpenScienceReasoning-2", split=args.split)
        if args.peek:
            for sample in ds:
                print(row_to_sample(0, tokenizer, sample, LossMaskConfig(), with_text=True))
                break
            return
        num_proc = args.num_proc if args.num_proc is not None else max(1, os.cpu_count() - 1)
        tokenized_ds = ds.map(
            lambda x, idx: row_to_sample(idx, tokenizer, x, LossMaskConfig()),
            with_indices=True,
            remove_columns=ds.column_names,
            num_proc=num_proc,
            desc="Tokenizing dataset",
        )
        tokenized_ds.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
