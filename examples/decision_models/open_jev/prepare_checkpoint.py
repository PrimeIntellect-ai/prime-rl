"""Prepare the checkpoint of `Qwen3_5Decision`: the base model plus the decision head's weights.

    uv run python examples/decision_models/open_jev/prepare_checkpoint.py BASE OUT

OUT links every file of BASE (a local Qwen3.5 checkpoint: safetensors shards with an index), adds
`decision_head.safetensors` with weight = lm_head[Yes] - lm_head[No] and bias = 0, and gets an updated
`model.safetensors.index.json`. The untrained head then scores a candidate as logit(Yes) - logit(No), a zero-shot
judge, which is how Open-Jev initialises it.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

INDEX = "model.safetensors.index.json"
HEAD_FILE = "decision_head.safetensors"


def token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} must be a single token for the Yes/No initialisation, got {ids}")
    return ids[0]


def output_embedding_key(weight_map: dict[str, str]) -> str:
    """The LM head's weight, or with tied embeddings the language model's input embedding."""
    if "lm_head.weight" in weight_map:
        return "lm_head.weight"
    keys = [key for key in weight_map if key.endswith("embed_tokens.weight") and not key.startswith("mtp.")]
    if len(keys) != 1:
        raise ValueError(f"cannot identify the output embedding among {keys}")
    return keys[0]


def prepare(base: Path, out: Path) -> dict[str, torch.Tensor]:
    index = json.loads((base / INDEX).read_text())
    weight_map = index["weight_map"]
    if any(key.startswith("decision_head.") for key in weight_map):
        raise ValueError(f"{base} already has a decision head")

    tokenizer = AutoTokenizer.from_pretrained(base)
    yes, no = token_id(tokenizer, "Yes"), token_id(tokenizer, "No")
    key = output_embedding_key(weight_map)
    with safe_open(base / weight_map[key], framework="pt") as shard:
        rows = shard.get_slice(key)
        weight = (rows[yes : yes + 1].float() - rows[no : no + 1].float()).contiguous()
    head = {"decision_head.weight": weight, "decision_head.bias": torch.zeros(1)}

    out.mkdir(parents=True)
    for path in sorted(base.iterdir()):
        if path.name != INDEX and not path.name.startswith("."):
            (out / path.name).symlink_to(path.resolve())
    save_file(head, out / HEAD_FILE, metadata={"format": "pt"})
    index["weight_map"] = {**weight_map, **dict.fromkeys(head, HEAD_FILE)}
    metadata = index.setdefault("metadata", {})
    metadata["total_size"] = metadata.get("total_size", 0) + sum(t.numel() * t.element_size() for t in head.values())
    (out / INDEX).write_text(json.dumps(index, indent=2) + "\n")
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="the base Qwen3.5 checkpoint directory")
    parser.add_argument("out", type=Path, help="the directory to create")
    args = parser.parse_args()
    head = prepare(args.base, args.out)
    print(f"wrote {args.out} (decision_head.weight norm {head['decision_head.weight'].norm():.4f})")


if __name__ == "__main__":
    main()
