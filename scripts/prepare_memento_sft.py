"""Prepare Qwen3-0.6B + OpenMementos for a Memento block masking SFT run.

Adds 4 block masking special tokens to the tokenizer, resizes model embeddings,
and converts OpenMementos into prime-rl SFT format (messages column).

Usage:
    python scripts/prepare_memento_sft.py --output-dir /tmp/memento-sft-prep
    python scripts/prepare_memento_sft.py --output-dir /tmp/memento-sft-prep --max-examples 1000
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BLOCK_MASKING_TOKENS = [
    "<|block_start|>",
    "<|block_end|>",
    "<|summary_start|>",
    "<|summary_end|>",
]


def main():
    parser = argparse.ArgumentParser(description="Prepare model + dataset for Memento SFT")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--model-only", action="store_true", help="Only prepare model, skip dataset")
    args = parser.parse_args()

    output = Path(args.output_dir)
    model_dir = output / "model"
    data_dir = output / "data"

    # --- 1. Add special tokens to tokenizer + resize model embeddings ---
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="bfloat16")

    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in BLOCK_MASKING_TOKENS if t not in existing]

    if new_tokens:
        added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added {added} special tokens, resized embeddings to {len(tokenizer)}")
    else:
        print("All block masking tokens already in vocabulary")

    for tok in BLOCK_MASKING_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        assert tid != tokenizer.unk_token_id, f"Token {tok} not found after adding"
        print(f"  {tok} -> {tid}")

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Saved model + tokenizer to {model_dir}")

    if args.model_only:
        print("--model-only: skipping dataset preparation")
        return

    # --- 2. Convert OpenMementos to messages format ---
    print(f"Loading microsoft/OpenMementos (first {args.max_examples} examples)...")
    ds = load_dataset("microsoft/OpenMementos", split=f"train[:{args.max_examples}]")

    def to_messages(example):
        return {
            "messages": [
                {"role": "user", "content": example["problem"]},
                {"role": "assistant", "content": example["response"]},
            ]
        }

    ds = ds.map(to_messages, remove_columns=ds.column_names)
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(train_dir / "data.parquet"))
    print(f"Saved {len(ds)} examples to {train_dir}/data.parquet")

    # --- 3. Print token IDs for the block masking config ---
    print("\nBlock masking config token IDs (for inference):")
    for tok in BLOCK_MASKING_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        print(f"  {tok}: {tid}")


if __name__ == "__main__":
    main()
