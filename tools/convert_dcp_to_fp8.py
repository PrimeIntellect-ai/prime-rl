"""Convert a DCP trainer checkpoint into blockwise-FP8 HF weights offline.

Same CPU streaming export as ``dcp_to_bf16``, but each layer group is quantized
on the fly with the blockwise-FP8 scheme of ``tools/convert_bf16_to_fp8.py``
(e4m3 weights, per-128x128-block fp32 scales under ``<name>.weight_scale_inv``,
same skip-module defaults), so only fp8 shards are written - no intermediate
bf16 export on disk.

Usage:
    uv run python tools/convert_dcp_to_fp8.py <run>/checkpoints/step_{n} [output_dir] [--block-size 128]

CPU streaming is the only supported export path. Use ``--workers`` to
parallelize independent CPU groups.
WORLD_SIZE must be 1.
Writes to <step_dir>/weights-FP8 by default. Existing output files are overwritten.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from convert_bf16_to_fp8 import quantization_config, quantize_state_dict

from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_cpu_hf_groups, write_hf_shards
from prime_rl.utils.export_assets import prepare_export_config, publish_export, read_model_config, save_model_assets
from prime_rl.utils.export_config import resolve_dcp_dir, resolve_run_configs
from prime_rl.utils.logger import get_logger, setup_logger

logger = get_logger()
SINGLE_PROCESS_ERROR = (
    "CPU-only DCP fp8 export does not support WORLD_SIZE>1. "
    "Run it as a single process and use --workers for CPU parallelism."
)


def _require_single_process_export() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(SINGLE_PROCESS_ERROR)


def convert(ckpt_dir: Path, output_dir: Path | None = None, block_size: int = 128, *, workers: int = 1) -> Path:
    """Export blockwise-FP8 weights with CPU streaming only."""
    _require_single_process_export()
    if workers <= 0:
        raise ValueError("workers must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    ckpt_dir = Path(ckpt_dir)
    dcp_dir = resolve_dcp_dir(ckpt_dir)
    step_dir = dcp_dir.parent
    model, tokenizer = resolve_run_configs(step_dir)
    model_config, tokenizer_config = model.model_dump(mode="json"), tokenizer.model_dump(mode="json")
    config = prepare_export_config(read_model_config(model_config), model_config)
    output_dir = Path(output_dir) if output_dir is not None else step_dir / "weights-FP8"
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

        if workers == 1:
            write_hf_shards(quantized_groups(), staging_dir)
        else:
            num_scales, modules_to_not_convert = write_cpu_hf_groups(
                dcp_dir,
                config,
                staging_dir,
                model_config=model_config,
                workers=workers,
                transform=lambda group: quantize_state_dict(group, block_size, "cpu"),
            )
        if num_scales == 0:
            raise ValueError(f"No quantizable weights found in {dcp_dir} - is this a bf16 checkpoint?")
        logger.info(f"Quantized {num_scales} weights to blockwise fp8")
        config_path = staging_dir / "config.json"
        exported_config = json.loads(config_path.read_text())
        exported_config["quantization_config"] = quantization_config(modules_to_not_convert, block_size)
        config_path.write_text(json.dumps(exported_config, indent=2) + "\n")
        publish_export(staging_dir, output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
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
    parser.add_argument("--workers", type=int, default=1, help="CPU export threads (default: 1)")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error(SINGLE_PROCESS_ERROR)
    setup_logger("info")
    convert(args.ckpt_dir, args.output_dir, args.block_size, workers=args.workers)


if __name__ == "__main__":
    main()
