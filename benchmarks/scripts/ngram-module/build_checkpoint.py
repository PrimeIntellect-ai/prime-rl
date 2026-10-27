"""Build the small synthetic Qwen3.8-Flash-Next HF checkpoint that both arms of the n-gram module A/B load."""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE
from prime_rl.trainer.models.qwen3_8_flash_next import (
    NGramEmbedding,
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextForCausalLM,
)
from prime_rl.utils.weights import load_state_dict_keys, save_state_dict

SNAPSHOT = Path(
    "/home/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")
HASH_BUFFERS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
NUM_LAYERS = 8
PLE_LAYER_IDS = [2, 6]
TEXT_CONFIG_OVERRIDES = {
    "num_hidden_layers": NUM_LAYERS,
    "layer_types": ["linear_attention"] * NUM_LAYERS,
    "hidden_size": 1024,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "hc_count": 4,
    "hc_lowrank": 128,
    "num_experts": 32,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "ple_layer_ids": PLE_LAYER_IDS,
    "ple_embed_dim": 2560,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 80_000,
    "make_ngram_vocab_size_divisible_by": 128,
    "split_ngram_parts": 128,
    "vocab_size": 248320,
    "eos_token_id": 248044,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parameter_group(name: str) -> str:
    return re.sub(r"\.layers\.\d+\.", ".layers.*.", name)


def print_init_stats(model: Qwen3_8FlashNextForCausalLM) -> None:
    groups: dict[str, dict] = {}
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if not tensor.is_floating_point():
            continue
        values = tensor.detach().double()
        assert torch.isfinite(values).all(), f"{name} has non-finite values"
        group = groups.setdefault(
            parameter_group(name),
            {"numel": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf")},
        )
        group["numel"] += values.numel()
        group["sum"] += values.sum().item()
        group["sum_sq"] += values.square().sum().item()
        group["min"] = min(group["min"], values.min().item())
        group["max"] = max(group["max"], values.max().item())

    width = max(len(group) for group in groups)
    for name, group in groups.items():
        mean = group["sum"] / group["numel"]
        std = max(group["sum_sq"] / group["numel"] - mean**2, 0.0) ** 0.5
        print(
            f"{name:<{width}}  numel={group['numel']:>12,}  min={group['min']:+.4e}  "
            f"max={group['max']:+.4e}  mean={mean:+.4e}  std={std:.4e}"
        )


def print_parameter_counts(model: Qwen3_8FlashNextForCausalLM) -> None:
    def count(modules) -> int:
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    tables = count(module.ngram_embedding for module in model.modules() if isinstance(module, NGramEmbedding))
    experts = count(module for module in model.modules() if isinstance(module, GroupedExperts))
    embeddings = count([model.get_input_embeddings(), model.lm_head])
    total = count([model])
    for name, value in [
        ("n-gram tables", tables),
        ("routed experts", experts),
        ("embeddings + lm_head", embeddings),
        ("rest", total - tables - experts - embeddings),
        ("total", total),
    ]:
        print(f"{name:<22} {value:>14,}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True)

    raw_config = json.loads((SNAPSHOT / "config.json").read_text())
    raw_config["text_config"] = raw_config["text_config"] | TEXT_CONFIG_OVERRIDES
    raw_config["vision_config"]["out_hidden_size"] = TEXT_CONFIG_OVERRIDES["hidden_size"]
    config = Qwen3_8FlashNextConfig.from_dict(raw_config)
    text_config = config.text_config

    torch.manual_seed(args.seed)
    model = Qwen3_8FlashNextForCausalLM(config)
    for module in model.modules():
        if isinstance(module, MoE):
            module.init_weights(text_config.initializer_range, torch.device("cpu"))
    model.init_buffers_post_meta()

    for module in model.modules():
        if isinstance(module, NGramEmbedding):
            print(
                f"n-gram table {module.ngram_layer_index}: rows={module.ngram_embedding.num_embeddings:,} "
                f"head_dim={module.head_dim} hashed_rows={module.ngram_heads_vocab_sizes.sum().item():,}"
            )
    print_parameter_counts(model)
    print_init_stats(model)

    state_dict = {
        name: tensor.detach().to(torch.bfloat16) if tensor.is_floating_point() else tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    hash_buffers = {name: tensor for name, tensor in state_dict.items() if name.endswith(HASH_BUFFERS)}
    model.convert_to_hf(state_dict)
    assert model.is_hf_state_dict(state_dict) and not model.is_prime_state_dict(state_dict)
    converted_keys = set(state_dict)
    save_state_dict(state_dict, args.output_dir)

    config.save_pretrained(args.output_dir)
    saved_config_path = args.output_dir / "config.json"
    saved_config = json.loads(saved_config_path.read_text())
    saved_config = saved_config | {key: value for key, value in raw_config.items() if key not in saved_config}
    saved_config_path.write_text(json.dumps(saved_config, indent=2, sort_keys=True) + "\n")
    for filename in TOKENIZER_FILES:
        shutil.copyfile(SNAPSHOT / filename, args.output_dir / filename)

    saved_keys = set(load_state_dict_keys(args.output_dir))
    assert saved_keys == converted_keys, sorted(saved_keys ^ converted_keys)
    shard_count = text_config.split_ngram_parts
    for layer_index in (layer_id - 1 for layer_id in PLE_LAYER_IDS):
        prefix = f"model.language_model.layers.{layer_index}.ple.ple_embedding"
        expected = {f"{prefix}.ngram_embedding.shard_{shard}.weight" for shard in range(shard_count)}
        expected |= {f"{prefix}.{buffer}" for buffer in HASH_BUFFERS}
        assert expected <= saved_keys, sorted(expected - saved_keys)
        assert f"{prefix}.ngram_embedding.shard_{shard_count}.weight" not in saved_keys
        assert f"{prefix}.ngram_embedding.weight" not in saved_keys

    saved_hash_buffers = {}
    for safetensors_path in args.output_dir.glob("*.safetensors"):
        with safe_open(safetensors_path, framework="pt", device="cpu") as file:
            saved_hash_buffers |= {name: file.get_tensor(name) for name in file.keys() if name in hash_buffers}
    assert saved_hash_buffers.keys() == hash_buffers.keys()
    for name, saved_value in sorted(saved_hash_buffers.items()):
        assert saved_value.dtype == torch.int64, (name, saved_value.dtype)
        assert torch.equal(saved_value, hash_buffers[name]), name
        print(f"{name}: {saved_value.tolist()}")

    reloaded_config = Qwen3_8FlashNextConfig.from_pretrained(args.output_dir)
    assert reloaded_config.to_dict() == config.to_dict()
    size_bytes = sum(path.stat().st_size for path in args.output_dir.iterdir())
    print(f"saved {len(saved_keys)} tensors, {size_bytes / 1e9:.2f} GB, to {args.output_dir}")


if __name__ == "__main__":
    main()
