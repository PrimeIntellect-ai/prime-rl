"""Convert a PrimeRL conversion cache back to an HF-layout checkpoint directory.

The trainer writes ``<snapshot>/prime/`` the first time it loads a published checkpoint whose
naming differs from its own module tree (`src/prime_rl/trainer/model.py`), dequantizing on the
way in. That cache is the trainer's weights in the trainer's spelling, with MoE experts fused
into single ``[E, ...]`` tensors. This plays the same conversion chain the weight broadcast uses
in reverse, so the result is a directory an inference engine can boot: the published key names,
per-expert weights, an index, the tokenizer assets, and a ``config.json`` with the quantization
fields removed, since the weights are no longer quantized.

For DeepSeek V4 that turns the fp8 + MXFP4 release into a servable bf16 checkpoint without a
569 GB download: the cache is already bf16 (with fp32 mHC, sink, bias and int64 router tensors).

``--num-layers N`` keeps only the first ``N`` layers, which is what makes a cheap reduced-depth
tier possible; vLLM raises ``KeyError`` on weights for layers beyond ``num_hidden_layers``.

Usage (from the prime-rl repo):
    uv run python tools/convert_prime_to_hf.py <snapshot>/prime <snapshot> <output_dir>
    uv run python tools/convert_prime_to_hf.py <snapshot>/prime <snapshot> <output_dir> --num-layers 4
"""

import argparse
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ASSET_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")
DROP_CONFIG_FIELDS = ("quantization_config", "expert_dtype")


def truncate_config(config, num_layers: int):
    """Rebuild the config at ``num_layers``, through the constructor.

    A config's per-layer schedules (``layer_types``, ``compress_ratios``, ``mlp_layer_types``) are
    derived inside ``__init__``, and ``from_pretrained(..., num_hidden_layers=N)`` assigns the
    override afterwards, leaving those lists at full depth.
    """
    fields = {k: v for k, v in config.to_dict().items() if not k.startswith("_")}
    for field in (*DROP_CONFIG_FIELDS, "layer_types"):
        fields.pop(field, None)
    fields["num_hidden_layers"] = num_layers
    return type(config)(**fields)


def materialize(tensor: torch.Tensor) -> torch.Tensor:
    """Unstacking fused experts yields views that would pin the whole ``[E, ...]`` parent."""
    tensor = tensor.contiguous()
    if tensor.untyped_storage().size() != tensor.numel() * tensor.element_size():
        tensor = tensor.clone()
    return tensor


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.shards: list[list[str]] = []
        self.weight_bytes: dict[str, int] = {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        tensor = materialize(tensor)
        self.buffer[name] = tensor
        self.weight_bytes[name] = tensor.numel() * tensor.element_size()
        self.buffer_bytes += self.weight_bytes[name]
        if self.buffer_bytes >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        save_file(self.buffer, str(self.out_dir / f"tmp-{len(self.shards):05d}.safetensors"), metadata={"format": "pt"})
        self.shards.append(sorted(self.buffer))
        self.buffer.clear()
        self.buffer_bytes = 0

    def finalize(self) -> dict[str, str]:
        self.flush()
        total = len(self.shards)
        weight_map: dict[str, str] = {}
        for i, names in enumerate(self.shards):
            shard = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
            (self.out_dir / f"tmp-{i:05d}.safetensors").rename(self.out_dir / shard)
            weight_map.update({name: shard for name in names})
        index = {
            "metadata": {"total_size": sum(self.weight_bytes.values())},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (self.out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        return weight_map


def read_group(prime_dir: Path, weight_map: dict[str, str], keys: list[str]) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)
    group: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(prime_dir / shard), framework="pt", device="cpu") as f:
            group.update({key: f.get_tensor(key) for key in shard_keys})
    return group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prime_dir", type=Path, help="the PrimeRL conversion cache (a `prime` directory)")
    parser.add_argument("source_dir", type=Path, help="the published snapshot, for config.json and the tokenizer")
    parser.add_argument("output_dir", type=Path, help="where to write the HF-layout checkpoint")
    parser.add_argument("--num-layers", type=int, default=None, help="keep only layers 0..N-1")
    parser.add_argument("--shard-size", type=int, default=5_000_000_000, help="target shard size in bytes")
    args = parser.parse_args()

    from transformers import AutoConfig

    from prime_rl.trainer.models import get_custom_causal_lm_cls
    from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
    from prime_rl.utils.vlm import get_layer_prefix

    source_config = json.loads((args.source_dir / "config.json").read_text())
    config = AutoConfig.from_pretrained(args.source_dir)
    prefix = get_layer_prefix(config)

    weight_map = json.loads((args.prime_dir / "model.safetensors.index.json").read_text())["weight_map"]
    layer_re = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    depth = max(int(m.group(1)) for m in map(layer_re.match, weight_map) if m) + 1
    num_layers = depth if args.num_layers is None else min(args.num_layers, depth)
    print(f"{args.prime_dir}: {len(weight_map)} tensors, {depth} layers -> exporting {num_layers}")

    chain = get_custom_causal_lm_cls(config).conversion_chain(truncate_config(config, num_layers))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(args.output_dir, args.shard_size)

    # One layer group at a time, so peak memory is a layer rather than the whole checkpoint.
    groups: list[tuple[str, list[str]]] = [("non-layer", [k for k in weight_map if not k.startswith(prefix)])]
    groups += [(f"layer {i}", [k for k in weight_map if k.startswith(f"{prefix}{i}.")]) for i in range(num_layers)]

    start = time.perf_counter()
    for label, keys in groups:
        group = read_group(args.prime_dir, weight_map, keys)
        apply_prime_to_hf(group, chain)
        for name in sorted(group):
            writer.add(name, group.pop(name))
        written = sum(writer.weight_bytes.values())
        print(f"  {label}: {len(keys)} keys, {written / 1e9:.1f} GB written, {time.perf_counter() - start:.0f}s")

    weight_map_out = writer.finalize()

    output_config = {k: v for k, v in source_config.items() if k not in DROP_CONFIG_FIELDS}
    output_config["num_hidden_layers"] = num_layers
    (args.output_dir / "config.json").write_text(json.dumps(output_config, indent=2))
    for asset in ASSET_FILES:
        if (args.source_dir / asset).exists():
            shutil.copyfile(args.source_dir / asset, args.output_dir / asset)

    total = sum(writer.weight_bytes.values())
    print(f"\n{len(weight_map_out)} tensors in {len(writer.shards)} shards, {total / 1e9:.1f} GB")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
