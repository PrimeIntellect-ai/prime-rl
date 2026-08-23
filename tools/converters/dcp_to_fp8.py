"""Convert a DCP trainer checkpoint into blockwise-FP8 HF weights.

Same gather as ``dcp_to_bf16``, but each rank quantizes its slice on GPU and
only the fp8 shards are written — no intermediate bf16 export on disk.

Usage (from the prime-rl repo; more ranks = faster gathers, quantization and writes):
    uv run python tools/converters/dcp_to_fp8.py <run>/checkpoints/step_{n} [output_dir]
    uv run torchrun --nproc-per-node 8 tools/converters/dcp_to_fp8.py \
        <run>/checkpoints/step_{n} [output_dir]

Writes to ``<ckpt_dir>/weights-FP8`` by default.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from bf16_to_fp8 import quantization_config, quantize_state_dict
from dcp_to_bf16 import load_and_convert, save_model_assets

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.weights import save_state_dict_parallel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "ckpt_dir", type=Path, help="the DCP checkpoint (<run>/checkpoints/step_{n} or .../step_{n}/trainer)"
    )
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <ckpt_dir>/weights-FP8")
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    model, model_config, tokenizer_config, state_dict, step_dir = load_and_convert(args.ckpt_dir)
    output_dir = args.output_dir if args.output_dir is not None else step_dir / "weights-FP8"
    logger = get_logger()
    world = get_world()

    logger.info("Quantizing weights to blockwise fp8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict, modules_to_not_convert = quantize_state_dict(state_dict, args.block_size, device)

    logger.info(f"Writing fp8 weights to {output_dir}")
    save_state_dict_parallel(state_dict, output_dir)

    all_modules: list[list[str] | None] = [None] * world.world_size
    dist.all_gather_object(all_modules, modules_to_not_convert)
    dist.destroy_process_group()

    if world.is_master:
        save_model_assets(model, model_config, tokenizer_config, output_dir)
        config_path = output_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["quantization_config"] = quantization_config(
            [module for modules in all_modules for module in modules or []], args.block_size
        )
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        logger.info(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
