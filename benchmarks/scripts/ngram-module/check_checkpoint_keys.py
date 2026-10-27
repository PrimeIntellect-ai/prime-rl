"""Check that an HF checkpoint converts to exactly this worktree's PrimeRL state dict: no missing or unexpected keys."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

from prime_rl.trainer.models import get_custom_vlm_cls
from prime_rl.utils.weights import load_state_dict, load_state_dict_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.checkpoint, attn_implementation="flash_attention_2")
    model_cls = get_custom_vlm_cls(config)
    with torch.device("meta"):
        model = model_cls(config)
    expected = model.state_dict()

    checkpoint_keys = dict.fromkeys(load_state_dict_keys(args.checkpoint))
    is_hf = model.is_hf_state_dict(checkpoint_keys) and not model.is_prime_state_dict(checkpoint_keys)
    model_is_prime = model.is_prime_state_dict(dict.fromkeys(expected))

    state_dict = load_state_dict(args.checkpoint)
    model.convert_to_prime(state_dict)
    missing = sorted(expected.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected.keys())
    mismatched = sorted(
        name for name in expected.keys() & state_dict.keys() if expected[name].shape != state_dict[name].shape
    )

    print(f"model class: {model_cls.__module__}.{model_cls.__name__}")
    print(f"checkpoint detected as HF: {is_hf}, model detected as PrimeRL: {model_is_prime}")
    print(f"model keys: {len(expected)}, converted checkpoint keys: {len(state_dict)}")
    for label, names in (("missing", missing), ("unexpected", unexpected), ("shape mismatch", mismatched)):
        print(f"{label}: {len(names)}" + "".join(f"\n  {name}" for name in names[:20]))
    sys.exit(0 if is_hf and model_is_prime and not (missing or unexpected or mismatched) else 1)


if __name__ == "__main__":
    main()
