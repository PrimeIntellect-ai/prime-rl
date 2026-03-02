"""Convert HuggingFace weights to PrimeRL format.

Usage:
    uv run python scripts/convert_hf_to_prime.py --model-path <hf_repo_or_local_path> --output-path <output_dir>
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from prime_rl.trainer.models import AutoModelForCausalLMPrimeRL, supports_custom_impl
from prime_rl.trainer.weights import load_state_dict, load_state_dict_keys, save_state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace weights to PrimeRL format")
    parser.add_argument("--model-path", type=str, required=True, help="HuggingFace repo ID or local path to model")
    args = parser.parse_args()

    model_path = args.model_path

    if not Path(model_path).exists():
        print(f"Downloading snapshot from {model_path}...")
        snapshot_path = Path(snapshot_download(repo_id=model_path, repo_type="model"))
    else:
        print(f"Using local model path: {model_path}")
        snapshot_path = Path(model_path)

    model_config = AutoConfig.from_pretrained(str(snapshot_path), trust_remote_code=True)
    if not supports_custom_impl(model_config):
        raise ValueError(
            f"Model type {type(model_config).__name__} does not have a custom PrimeRL implementation. "
            f"No conversion needed."
        )

    print("Loading model on meta device...")
    with torch.device("meta"):
        model = AutoModelForCausalLMPrimeRL.from_config(model_config, dtype=torch.bfloat16)

    snapshot_keys = dict.fromkeys(load_state_dict_keys(snapshot_path))
    model_keys = dict.fromkeys(model.state_dict().keys())

    if not model.is_hf_state_dict(snapshot_keys):
        raise ValueError("Snapshot is not in HuggingFace format — nothing to convert.")

    if not model.is_prime_state_dict(model_keys):
        raise ValueError("Model does not use PrimeRL format — unexpected state.")

    output_path = snapshot_path / "prime"

    print(f"Loading state dict from {snapshot_path}...")
    state_dict = load_state_dict(snapshot_path)

    print("Converting to PrimeRL format...")
    model.convert_to_prime(state_dict)

    print(f"Saving converted weights to {output_path}...")
    save_state_dict(state_dict, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
