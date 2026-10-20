"""Convert a DCP trainer checkpoint into bf16 HF-format weights offline.

Export uses a single CPU process in the normal prime-rl environment without
loading base-model weights. Custom PrimeRL layouts use their model's conversion chain;
other HF-compatible layouts are validated against a meta-device HF model.

Usage:
    uv run python tools/convert_dcp_to_bf16.py <run>/checkpoints/step_{n} [output_dir]
    uv run python tools/convert_dcp_to_bf16.py <checkpoint> [output_dir] --device cuda

CPU streaming is the default; --workers parallelizes independent CPU groups.
CUDA uses the parallel GPU gather path.
Multi-rank torchrun automatically selects CUDA for compatibility.
Writes to <step_dir>/weights by default. Existing output files are overwritten.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_cpu_hf_groups, write_hf_shards
from prime_rl.utils.export_assets import prepare_export_config, publish_export, read_model_config, save_model_assets
from prime_rl.utils.export_config import resolve_dcp_dir, resolve_run_configs
from prime_rl.utils.logger import get_logger, setup_logger

logger = get_logger()


def convert(ckpt_dir: Path, output_dir: Path | None = None, *, device: str = "cpu", workers: int = 1) -> Path:
    """Export a full trainer checkpoint using CPU streaming or the parallel CUDA gather."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported export device: {device}")
    if device == "cuda" or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from convert_dcp_to_bf16_gpu import convert as convert_cuda

        return convert_cuda(ckpt_dir, output_dir)
    ckpt_dir = Path(ckpt_dir)
    dcp_dir = resolve_dcp_dir(ckpt_dir)
    step_dir = dcp_dir.parent
    model, tokenizer = resolve_run_configs(step_dir)
    model_config, tokenizer_config = model.model_dump(mode="json"), tokenizer.model_dump(mode="json")
    config = read_model_config(model_config)
    config = prepare_export_config(config, model_config)
    output_dir = Path(output_dir) if output_dir is not None else step_dir / "weights"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        logger.info(f"Saving model assets to {staging_dir}")
        save_model_assets(config, model_config, tokenizer_config, staging_dir)
        logger.info(f"Reading model tensors from {dcp_dir} on CPU")
        if workers == 1:
            write_hf_shards(iter_cpu_hf_groups(dcp_dir, config, model_config=model_config), staging_dir)
        else:
            write_cpu_hf_groups(dcp_dir, config, staging_dir, model_config=model_config, workers=workers)
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
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <step_dir>/weights")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="export device (default: cpu)")
    parser.add_argument("--workers", type=int, default=1, help="CPU export threads (default: 1)")
    args = parser.parse_args()
    setup_logger("info")
    convert(args.ckpt_dir, args.output_dir, device=args.device, workers=args.workers)


if __name__ == "__main__":
    main()
