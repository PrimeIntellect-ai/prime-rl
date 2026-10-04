"""Untie a checkpoint's LM head from its input embeddings.

The trainer rejects tied checkpoints (``tie_word_embeddings: true``). This writes a copy whose
``lm_head.weight`` is a clone of the input embedding and whose config sets
``tie_word_embeddings: false``. The model is unchanged; only its storage is. Tokenizer, chat
template and other non-weight files are copied as-is.

Usage (from the prime-rl repo):
    uv run python tools/untie_word_embeddings.py <model_dir_or_hub_id> <output_dir>
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

# Embedding key -> LM head key, for text models and for VLMs that nest the text model.
EMBEDDING_TO_LM_HEAD = {
    "model.embed_tokens.weight": "lm_head.weight",
    "model.language_model.embed_tokens.weight": "lm_head.weight",
}


def untie(source: Path, output: Path) -> None:
    config = json.loads((source / "config.json").read_text())
    text_config = config.get("text_config", {})
    if not (config.get("tie_word_embeddings") or text_config.get("tie_word_embeddings")):
        raise ValueError(f"{source} does not tie its word embeddings")

    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.suffix != ".safetensors" and path.name not in ("config.json", "model.safetensors.index.json"):
            if path.is_file():
                shutil.copyfile(path, output / path.name)

    shards = sorted(source.glob("*.safetensors"))
    weight_map: dict[str, str] = {}
    lm_head_added = False
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            tensors = {key: f.get_tensor(key) for key in f.keys()}
            metadata = f.metadata()
        # Some tied checkpoints also store a (possibly stale) lm_head; a tied model ignores it.
        tensors.pop("lm_head.weight", None)
        for embedding_key, lm_head_key in EMBEDDING_TO_LM_HEAD.items():
            if embedding_key in tensors:
                tensors[lm_head_key] = tensors[embedding_key].clone()
                lm_head_added = True
        save_file(tensors, output / shard.name, metadata=metadata)
        weight_map.update(dict.fromkeys(tensors, shard.name))
    if not lm_head_added:
        raise ValueError(f"No input embedding ({', '.join(EMBEDDING_TO_LM_HEAD)}) found in {source}")

    if (source / "model.safetensors.index.json").exists():
        index = json.loads((source / "model.safetensors.index.json").read_text())
        index["weight_map"] = weight_map
        (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    config["tie_word_embeddings"] = False
    if text_config:
        text_config["tie_word_embeddings"] = False
    (output / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Wrote untied checkpoint to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Local checkpoint directory or HuggingFace Hub repo id")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source = Path(args.model) if Path(args.model).is_dir() else Path(snapshot_download(args.model))
    untie(source, args.output_dir)


if __name__ == "__main__":
    main()
