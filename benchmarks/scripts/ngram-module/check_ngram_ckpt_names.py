"""Check n-gram table names in a step's DCP checkpoint and in its HF export against the source HF checkpoint."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader

from prime_rl.utils.weights import load_state_dict_keys

HASH_BUFFERS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")


def read_tensors(checkpoint: Path, names: list[str]) -> dict[str, torch.Tensor]:
    tensors = {}
    for path in checkpoint.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as file:
            tensors |= {name: file.get_tensor(name) for name in file.keys() if name in names}
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_dir", type=Path, help="<run>/checkpoints/step_<n>")
    parser.add_argument("hf_dir", type=Path, help="HF export of that step")
    parser.add_argument("source", type=Path, help="HF checkpoint the run started from")
    args = parser.parse_args()
    failures = []

    dcp_keys = list(FileSystemReader(args.step_dir / "trainer").read_metadata().state_dict_metadata)
    ngram_keys = sorted(key for key in dcp_keys if ".ngram." in key)
    print(f"DCP: {len(dcp_keys)} keys, {len(ngram_keys)} contain '.ngram.':")
    print("".join(f"  {key}\n" for key in ngram_keys), end="")
    table_indices = sorted(
        {int(m.group(1)) for key in ngram_keys if (m := re.search(r"\.ngram\.tables\.(\d+)\.", key))}
    )
    print(f"DCP table indices: {table_indices}")

    text_config = json.loads((args.source / "config.json").read_text())["text_config"]
    layer_indices = [layer_id - 1 for layer_id in text_config["ple_layer_ids"]]
    shard_count = text_config["split_ngram_parts"]
    if table_indices != layer_indices:
        failures.append(f"DCP table indices {table_indices} != PLE layer indices {layer_indices}")

    hf_keys = set(load_state_dict_keys(args.hf_dir))
    source_keys = set(load_state_dict_keys(args.source))
    print(f"HF export: {len(hf_keys)} keys, source: {len(source_keys)} keys")
    if hf_keys != source_keys:
        failures.append(f"HF export keys differ from source: {sorted(hf_keys ^ source_keys)[:20]}")

    buffer_names = []
    for layer_index in layer_indices:
        prefix = f"model.language_model.layers.{layer_index}.ple.ple_embedding"
        expected = {f"{prefix}.ngram_embedding.shard_{shard}.weight" for shard in range(shard_count)}
        buffer_names += [f"{prefix}.{buffer}" for buffer in HASH_BUFFERS]
        expected |= set(buffer_names[-len(HASH_BUFFERS) :])
        present = sorted(expected & hf_keys)
        print(f"layer {layer_index}: {len(present)}/{len(expected)} expected HF n-gram keys present")
        if present != sorted(expected):
            failures.append(f"layer {layer_index} missing {sorted(expected - hf_keys)[:10]}")
        if f"{prefix}.ngram_embedding.shard_{shard_count}.weight" in hf_keys:
            failures.append(f"layer {layer_index} has an extra shard_{shard_count}")

    exported = read_tensors(args.hf_dir, buffer_names)
    original = read_tensors(args.source, buffer_names)
    for name in buffer_names:
        same = torch.equal(exported[name], original[name])
        print(f"{name}: dtype={exported[name].dtype} equal to source: {same}")
        if not same:
            failures.append(f"{name} differs from source")

    first_shard = f"model.language_model.layers.{layer_indices[0]}.ple.ple_embedding.ngram_embedding.shard_0.weight"
    trained, initial = (
        read_tensors(args.hf_dir, [first_shard])[first_shard],
        read_tensors(args.source, [first_shard])[first_shard],
    )
    print(f"{first_shard}: max abs change from source {(trained.float() - initial.float()).abs().max().item():.3e}")

    print("FAIL:\n  " + "\n  ".join(failures) if failures else "OK")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
