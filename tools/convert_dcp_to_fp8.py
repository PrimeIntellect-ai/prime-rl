"""Export DCP checkpoints as layer-parallel blockwise FP8 HF weights on CPU."""

import argparse
import json
import tempfile
from pathlib import Path

from convert_bf16_to_fp8 import quantization_config, quantize_state_dict
from convert_dcp_to_bf16 import (
    open_checkpoint,
    publish_export,
    save_model_assets,
    write_weights,
)

from prime_rl.utils.logger import get_logger, setup_logger

logger = get_logger()


def convert(ckpt_dir: Path, output_dir: Path | None = None, block_size: int = 128, *, workers: int = 1) -> Path:
    """Export blockwise-FP8 weights with CPU streaming only."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    checkpoint = open_checkpoint(ckpt_dir)
    output_dir = Path(output_dir) if output_dir is not None else checkpoint.dcp_dir.parent / "weights-FP8"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    modules_to_not_convert: list[str] = []

    def quantize(shard):
        quantized, kept_in_bf16 = quantize_state_dict(shard, block_size, "cpu")
        modules_to_not_convert.extend(kept_in_bf16)
        return quantized

    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as directory:
        staging_dir = Path(directory)
        logger.info(f"Saving model assets to {staging_dir}")
        save_model_assets(checkpoint.model, checkpoint.model_config, checkpoint.tokenizer_config, staging_dir)
        logger.info(f"Reading and quantizing model tensors from {checkpoint.dcp_dir} on CPU")
        weight_map = write_weights(checkpoint, staging_dir, workers, transform=quantize)
        num_scales = sum(name.endswith("_scale_inv") for name in weight_map)
        if num_scales == 0:
            raise ValueError(f"No quantizable weights found in {checkpoint.dcp_dir} - is this a bf16 checkpoint?")
        logger.info(f"Quantized {num_scales} weights to blockwise fp8")
        config_path = staging_dir / "config.json"
        exported_config = json.loads(config_path.read_text())
        exported_config["quantization_config"] = quantization_config(modules_to_not_convert, block_size)
        config_path.write_text(json.dumps(exported_config, indent=2) + "\n")
        publish_export(staging_dir, output_dir)
    logger.info(f"Done: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "ckpt_dir", type=Path, help="the DCP checkpoint (<run>/checkpoints/step_{n} or .../step_{n}/trainer)"
    )
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <step_dir>/weights-FP8")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1, help="decoder layers loaded and written at once (default: 1)")
    args = parser.parse_args()
    setup_logger("info")
    convert(args.ckpt_dir, args.output_dir, args.block_size, workers=args.workers)


if __name__ == "__main__":
    main()
