import argparse
import datasets
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import cast, Any
from pathlib import Path
from tqdm import tqdm
import shutil
import json
import logging
import os

from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download, scan_cache_dir
from prime_rl.trainer.sft.config import LossMaskConfig
from prime_rl.trainer.sft.datasets.utils import build_loss_mask, messages_to_sample
from prime_rl.trainer.sft.datasets.dsdstill_cleaner import clean_and_load_dataset

logger = logging.getLogger(__name__)


class DatasetCleaner:
    """Some dataset, like this distillation dataset, has inconsistent schema that will prevent hf's dataset loading. We need to clean it first."""

    def __init__(self, dataset_path: str, cache_dir: str):
        self.dataset_path = dataset_path
        self.cache_dir = cache_dir
    
    def clean_conversations_data(self, conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean conversations by removing problematic null fields."""
        cleaned = []
        
        for conv in conversations:
            if not isinstance(conv, dict):
                continue
                
            cleaned_conv = {
                "from": conv.get("from", ""),
                "value": conv.get("value", "")
            }
            
            cleaned.append(cleaned_conv)
        
        return cleaned
    
    def clean_json_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Clean a single JSON record."""
        cleaned_record = {}
        
        for key, value in record.items():
            if key == "conversations" and isinstance(value, list):
                cleaned_record[key] = self.clean_conversations_data(value)
            elif value is not None:
                cleaned_record[key] = value
        
        return cleaned_record
    
    def process_jsonl_file(self, input_path: str, output_path: str) -> int:
        """
        Process a JSONL file and clean problematic fields.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output cleaned JSONL file
            
        Returns:
            Number of records processed
        """
        processed_count = 0
        
        logger.info(f"Processing {input_path} -> {output_path}")
        
        with open(input_path, "rb") as f:  # binary mode is a bit faster
            total_lines = sum(1 for _ in f)

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(tqdm(infile, desc="Processing lines", unit="lines", total=total_lines)):
                try:
                    if line.strip():  # Skip empty lines
                        record = json.loads(line.strip())
                        cleaned_record = self.clean_json_record(record)
                        outfile.write(json.dumps(cleaned_record, ensure_ascii=False) + '\n')
                        processed_count += 1
                        
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"  Warning: Skipping malformed JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"  Error processing line {line_num}: {e}")
        
        logger.info(f"Completed processing: {processed_count} records")
        return processed_count

    def clean(self):
        logger.info(f"Cleaning dataset: {self.dataset_path}")
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        total_processed = 0
        
        # Walk through all files in dataset path
        for root, dirs, files in os.walk(self.dataset_path):
            for file in tqdm(files, desc="Processing files", unit="files"):
                input_path = os.path.join(root, file)
                # Get relative path from dataset_path
                rel_path = os.path.relpath(input_path, self.dataset_path)
                output_path = os.path.join(self.cache_dir, rel_path)
                
                # Create output directory if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if file.endswith(('.jsonl', '.json')):
                    # Process JSON/JSONL files
                    count = self.process_jsonl_file(input_path, output_path)
                    total_processed += count
                else:
                    # Copy other files as-is
                    shutil.copy2(input_path, output_path)
                    logger.info(f"Copied: {rel_path}")
        
        logger.info(f"Dataset cleaning completed. Total records processed: {total_processed}")
        return self.cache_dir


def load_cleaned_dataset(dataset_name: str, split: str , cache_dir: str) -> Dataset:
    """
    Load the dataset using disk-based cleaning to handle null fields in nested structures.
    This bypasses HuggingFace's automatic schema inference that fails on mixed null/non-null fields.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to load
        cache_dir: Directory to store cleaned datasets
        
    Returns:
        Cleaned dataset
    """
    def _load_from_cache(cache_dir: str) -> Dataset:
        split_path = os.path.join(cache_dir, split)
        if not os.path.exists(split_path):
            split_path = cache_dir
        
        # Find data files
        data_files = []
        for file in os.listdir(split_path):
            if file.endswith(('.jsonl', '.json', '.parquet')):
                data_files.append(os.path.join(split_path, file))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found in {split_path}")
        
        logger.info(f"Found data files: {data_files}")
        
        # Load from the cleaned files
        if data_files[0].endswith('.parquet'):
            dataset = Dataset.from_parquet(data_files)
        else:
            dataset = Dataset.from_json(data_files)
        
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        return dataset

    if os.path.exists(cache_dir):
        return _load_from_cache(cache_dir)
    
    HF_HOME = os.getenv("HF_HOME")
    info = scan_cache_dir(os.path.join(HF_HOME, 'hub'))
    snapshot_path = [
        rev.snapshot_path
        for repo in info.repos
        if repo.repo_id == dataset_name and repo.repo_type == "dataset"
        for rev in repo.revisions
    ]
    if len(snapshot_path) == 0:
        logger.warning(f"Dataset {dataset_name} not found in cache, downloading...")
        snapshot_path = snapshot_download(dataset_name, revision=split, repo_type="dataset")
        return load_cleaned_dataset(snapshot_path, split, cache_dir)
    elif len(snapshot_path) > 1:
        logger.warning(f"Multiple snapshots found for dataset {dataset_name}, using the first one...")
    snapshot_path = snapshot_path[0]

    cleaner = DatasetCleaner(snapshot_path, cache_dir)
    cleaned_path = cleaner.clean()

    return _load_from_cache(cleaned_path)
    

def row_to_sample(index: int, tokenizer: PreTrainedTokenizer, data: dict, loss_mask_config: LossMaskConfig, with_text: bool = False) -> dict:
    DEFAULT_SYSTEM_PROMPT = r"You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    assert "conversations" in data and "system" in data, "conversations and system must be present in the data"
    role_mapping = {
        "human": "user",
        "assistant": "assistant",
        "system": "system",
    }
    messages = [{"role": role_mapping[d["from"]], "content": d["value"]} for d in data["conversations"]]
    messages = [{"role": "system", "content": data["system"] or DEFAULT_SYSTEM_PROMPT}] + messages

    return messages_to_sample(index, tokenizer, messages, loss_mask_config, with_text=with_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert AM-DeepSeek-R1-0528-Distilled to a tokenized dataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--clean-only", action="store_true", help="Clean the dataset only")
    parser.add_argument("--peek", type=int, default=0)
    parser.add_argument("--num_proc", type=int, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info("Clean and Converting dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_cleaned_dataset("a-m-team/AM-DeepSeek-R1-0528-Distilled", split=args.split, cache_dir=os.path.expanduser("~/cache/datasets"))
    if args.clean_only:
        logger.info("Dataset cleaned, exiting...")
        return 
    if args.peek:
        logger.info("Peeking at the dataset...")
        for i in range(args.peek):
            print(row_to_sample(i, tokenizer, ds[i], LossMaskConfig(), with_text=True))
        return
    num_proc = args.num_proc if args.num_proc is not None else max(1, os.cpu_count() - 1)
    ds = ds.map(
        lambda x, idx: row_to_sample(idx, tokenizer, x, LossMaskConfig()),
        with_indices=True,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing dataset",
    )
    ds.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()