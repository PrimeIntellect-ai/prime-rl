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

hf_datasets = {
    "taco": load_dataset("BAAI/TACO", trust_remote_code=True),
    "apps": load_dataset("codeparrot/apps", trust_remote_code=True),
    "code_contests": load_dataset("deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("open-r1/codeforces")
}


def get_question(ds_name, split, index):
    benchmark = hf_datasets[ds_name][split][int(index)]
    if ds_name == "code_contests":
        if not benchmark["description"]:
            return None
        return benchmark["description"]
    elif ds_name in ["taco", "apps"]:
        return benchmark["question"]
    elif ds_name == "open-r1/codeforces":
        if not benchmark["description"]:
            return None
        question = benchmark["description"]
        if benchmark["input_format"]:
            question += "\n\nInput\n\n" + benchmark["input_format"]
        if benchmark["output_format"]:
            question += "\n\nOutput\n\n" + benchmark["output_format"]
        if benchmark["examples"]:
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]
        return question

    return None


def row_to_sample(index: int, tokenizer: PreTrainedTokenizer, data: dict, loss_mask_config: LossMaskConfig, with_text: bool = False) -> dict:
    messages = data['messages']
    metadata = json.loads(data['metadata'])
    if metadata.get('dataset', None) in ["taco", "apps", "code_contests", "open-r1/codeforces"]:
        ds_name, ds_split, ds_index = metadata['dataset'], metadata['split'], int(metadata['index'])
        question = get_question(ds_name, ds_split, ds_index)
        assert question is not None
        assert messages[0]['role'] == "user"
        assert messages[0]['content'] == "-"
        messages = [{"role": "user", "content": question}] + messages[1:]
    return messages_to_sample(index, tokenizer, messages, loss_mask_config, tools=metadata.get('tools', None), with_text=with_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Nemotron to a tokenized dataset")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--split", type=str, default="code,math,tool_calling", help="comma separated list of splits to use")
    parser.add_argument("--dataset", type=str, default="nvidia/Nemotron-Post-Training-Dataset-v1")
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
    print(f"To run this script, you want to install an old version of datasets: pip install datasets==3.6.0")
    main()