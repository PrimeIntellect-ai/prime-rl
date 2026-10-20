"""Convert a DCP trainer checkpoint into blockwise-FP8 HF weights offline.

Same CPU streaming export as ``dcp_to_bf16``, but each layer group is quantized
on the fly with the blockwise-FP8 scheme of ``tools/convert_bf16_to_fp8.py``
(e4m3 weights, per-128x128-block fp32 scales under ``<name>.weight_scale_inv``,
same skip-module defaults), so only fp8 shards are written - no intermediate
bf16 export on disk.

Usage:
    uv run python tools/convert_dcp_to_fp8.py <run>/checkpoints/step_{n} [output_dir] [--block-size 128]
    uv run python tools/convert_dcp_to_fp8.py <checkpoint> [output_dir] --device cuda

CPU streaming is the default; CUDA gathers and quantizes each rank's slice on GPU.
Writes to <step_dir>/weights-FP8 by default. CPU output must not already exist.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from convert_bf16_to_fp8 import quantization_config, quantize_state_dict

from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards
from prime_rl.utils.export_assets import prepare_export_config, read_model_config, save_model_assets
from prime_rl.utils.export_config import resolve_dcp_dir, resolve_run_configs
from prime_rl.utils.logger import get_logger, setup_logger

logger = get_logger()


def convert(ckpt_dir: Path, output_dir: Path | None = None, block_size: int = 128, *, device: str = "cpu") -> Path:
    """Export blockwise-FP8 weights using CPU streaming or the parallel CUDA gather."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if device == "cuda":
        from convert_dcp_to_bf16_gpu import convert_fp8

        return convert_fp8(ckpt_dir, output_dir, block_size)
    if device != "cpu":
        raise ValueError(f"Unsupported export device: {device}")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("CPU export requires one process; run without torchrun (WORLD_SIZE must be 1)")
    ckpt_dir = Path(ckpt_dir)
    dcp_dir = resolve_dcp_dir(ckpt_dir)
    step_dir = dcp_dir.parent
    model, tokenizer = resolve_run_configs(step_dir)
    model_config, tokenizer_config = model.model_dump(mode="json"), tokenizer.model_dump(mode="json")
    config = prepare_export_config(read_model_config(model_config), model_config)
    output_dir = Path(output_dir) if output_dir is not None else step_dir / "weights-FP8"
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"CPU export output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        logger.info(f"Saving model assets to {staging_dir}")
        save_model_assets(config, model_config, tokenizer_config, staging_dir)
        logger.info(f"Reading and quantizing model tensors from {dcp_dir} on CPU")
        modules_to_not_convert: list[str] = []
        num_scales = 0

        def quantized_groups():
            nonlocal num_scales
            for group in iter_cpu_hf_groups(dcp_dir, config, model_config=model_config):
                group, kept_in_bf16 = quantize_state_dict(group, block_size, "cpu")
                modules_to_not_convert.extend(kept_in_bf16)
                num_scales += sum(name.endswith("_scale_inv") for name in group)
                yield group

        write_hf_shards(quantized_groups(), staging_dir)
        if num_scales == 0:
            raise ValueError(f"No quantizable weights found in {dcp_dir} - is this a bf16 checkpoint?")
        logger.info(f"Quantized {num_scales} weights to blockwise fp8")
        # One canonical config file: assets wrote config.json, quantization metadata is added in place.
        config_path = staging_dir / "config.json"
        exported_config = json.loads(config_path.read_text())
        exported_config["quantization_config"] = quantization_config(modules_to_not_convert, block_size)
        config_path.write_text(json.dumps(exported_config, indent=2) + "\n")
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"CPU export output already exists: {output_dir}")
        staging_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir)
        raise
    logger.info(f"Done: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "ckpt_dir", type=Path, help="the DCP checkpoint (<run>/checkpoints/step_{n} or .../step_{n}/trainer)"
    )
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <step_dir>/weights-FP8")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="export device (default: cpu)")
    args = parser.parse_args()
    setup_logger("info")
    convert(args.ckpt_dir, args.output_dir, args.block_size, device=args.device)


if __name__ == "__main__":
    main()
