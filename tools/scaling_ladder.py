"""Generate a pretraining scaling ladder: one SFT recipe at several widths.

Every rung trains the same recipe at a fixed number of tokens per active (non-embedding)
parameter. The model shape, batch, steps, LR, Adam beta2/epsilon and the WSD schedule are scaled
per rung (``prime_rl.utils.scaling.Heuristic``). Per rung the tool writes a from-scratch HF model
config (``<output_dir>/<rung>/model``) and an overlay config (``<output_dir>/<rung>/sft.toml``),
plus ``<output_dir>/ladder.json`` for ``tools/fit_scaling_law.py``. See docs/scaling-ladder.md.

Usage:
    uv run python tools/scaling_ladder.py <recipe.toml> <output_dir> --widths 512 768 1024 2048
    uv run sft @ <recipe.toml> @ <output_dir>/<rung>/sft.toml
"""

import argparse
import copy
import json
import tomllib
from pathlib import Path

from transformers import AutoConfig, PretrainedConfig

from prime_rl.utils.flops import forward_flops
from prime_rl.utils.scaling import Heuristic

SCALED_INTERMEDIATE_KEYS = ("intermediate_size", "moe_intermediate_size", "shared_expert_intermediate_size")


def scale_config(base: PretrainedConfig, hidden: int, layers: int | None = None) -> PretrainedConfig:
    """``base`` at width ``hidden``: same head dim and GQA ratio, heads and MLP sizes scaled with width.
    Depth scales with width too unless ``layers`` is given."""
    ratio = hidden / base.hidden_size
    heads = max(1, round(base.num_attention_heads * ratio))
    layers = layers or max(1, round(base.num_hidden_layers * ratio))
    updates = {
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": max(1, heads * base.num_key_value_heads // base.num_attention_heads),
        "num_hidden_layers": layers,
        # Scratch init never ties embeddings.
        "tie_word_embeddings": False,
    }
    for key in SCALED_INTERMEDIATE_KEYS:
        if getattr(base, key, None):
            updates[key] = 128 * max(1, round(getattr(base, key) * ratio / 128))
    if getattr(base, "layer_types", None):
        if len(set(base.layer_types)) != 1:
            raise ValueError("cannot rescale the depth of a config with mixed layer_types")
        updates["layer_types"] = base.layer_types[:1] * layers
    config = copy.deepcopy(base)
    config.update(updates)
    return config


def total_params(config: PretrainedConfig) -> int:
    """All matmul parameters plus the (untied) input embedding: the active count with every expert routed."""
    dense = copy.deepcopy(config)
    if getattr(config, "num_experts_per_tok", None):
        dense.num_experts_per_tok = getattr(config, "n_routed_experts", None) or config.num_experts
    return forward_flops(dense)[0] // 2 + config.vocab_size * config.hidden_size


def toml_overlay(values: dict[str, object]) -> str:
    return "".join(f"{key} = {json.dumps(value)}\n" for key, value in values.items())


def human(n: float) -> str:
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if n >= scale:
            return f"{n / scale:.3g}{unit}"
    return f"{n:.3g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path, help="SFT recipe TOML; model.name is the base shape")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", required=True, help="hidden sizes, one rung each")
    parser.add_argument("--layers", type=int, nargs="+", help="depth per width; defaults to the base depth/width ratio")
    parser.add_argument(
        "--batch-tokens", type=int, nargs="+", help="tokens per batch per width; defaults to the heuristic"
    )
    parser.add_argument("--tokens-per-param", type=float, default=20.0, help="tokens per active parameter")
    parser.add_argument("--name", help="ladder name (W&B group, rung name prefix); defaults to the recipe stem")
    args = parser.parse_args()
    for option in ("layers", "batch_tokens"):
        if getattr(args, option) is not None and len(getattr(args, option)) != len(args.widths):
            raise SystemExit(f"--{option.replace('_', '-')} needs one value per width")

    recipe = tomllib.loads(args.recipe.read_text())
    model, data = recipe.get("model", {}), recipe.get("data", {})
    base_name = model.get("name", "Qwen/Qwen3-0.6B")
    seq_len = data.get("seq_len", 128)
    name = args.name or args.recipe.stem
    base = AutoConfig.from_pretrained(base_name, trust_remote_code=model.get("trust_remote_code", False))
    heuristic = Heuristic()

    rungs = []
    for i, hidden in enumerate(args.widths):
        config = scale_config(base, hidden, args.layers[i] if args.layers else None)
        linear, quadratic = forward_flops(config)
        active = linear // 2 - config.vocab_size * hidden
        if args.batch_tokens:
            batch_size = args.batch_tokens[i] // seq_len
        else:
            batch_size = heuristic.batch_size(args.tokens_per_param * active, seq_len)
        batch_tokens = batch_size * seq_len
        steps = max(1, round(args.tokens_per_param * active / batch_tokens))
        tokens = steps * batch_tokens
        lr = heuristic.lr(tokens, hidden, batch_tokens)
        rung = f"{name}-d{hidden}"
        rung_dir = args.output_dir / rung
        config.save_pretrained(rung_dir / "model")
        overlay = {
            "model.name": str((rung_dir / "model").resolve()),
            "model.init": "scratch",
            "tokenizer.name": recipe.get("tokenizer", {}).get("name") or base_name,
            "data.batch_size": batch_size,
            "max_steps": steps,
            "optim.lr": lr,
            "optim.betas2": heuristic.beta2(batch_tokens),
            "optim.eps": heuristic.eps(tokens, batch_tokens),
            "scheduler.type": "linear",
            "scheduler.warmup_steps": round(heuristic.warmup_fraction * steps),
            "scheduler.decay_steps": round(heuristic.decay_fraction * steps),
            "scheduler.min_lr": heuristic.min_lr_ratio * lr,
            "run.name": rung,
            "monitors.wandb.group": name,
        }
        (rung_dir / "sft.toml").write_text(toml_overlay(overlay))
        rungs.append(
            {
                "name": rung,
                "hidden": hidden,
                "layers": config.num_hidden_layers,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "steps": steps,
                "tokens": tokens,
                "active_params": active,
                "total_params": total_params(config),
                # Training FLOPs as in the trainer's MFU: 3x forward, full (non-causal) attention.
                "flops_per_token": 3 * (linear + quadratic * seq_len),
                "flops": 3 * (linear + quadratic * seq_len) * tokens,
                "lr": lr,
            }
        )
    (args.output_dir / "ladder.json").write_text(json.dumps(rungs, indent=2))

    print(
        f"{'size':>8} {'layers':>6} {'batch':>6} {'steps':>8} {'tokens':>7} {'active':>7} {'total':>7} {'FLOPs':>8} {'lr':>8}"
    )
    for r in rungs:
        print(
            f"{'d' + str(r['hidden']):>8} {r['layers']:>6} {r['batch_size']:>6} {r['steps']:>8} {human(r['tokens']):>7} "
            f"{human(r['active_params']):>7} {human(r['total_params']):>7} {r['flops']:>8.2e} {r['lr']:>8.2e}"
        )
    print(f"\nLaunch a rung: uv run sft @ {args.recipe} @ {args.output_dir}/<rung>/sft.toml")


if __name__ == "__main__":
    main()
