"""Convert a DCP trainer checkpoint into bf16 HF-format weights offline.

Export uses a single CPU process in the normal prime-rl environment without
loading base-model weights. Custom PrimeRL layouts use their model's conversion chain;
other HF-compatible layouts are validated against a meta-device HF model.

Usage:
    uv run python tools/convert_dcp_to_bf16.py <run>/checkpoints/step_{n} [output_dir]

Writes to <step_dir>/weights by default. CPU output must not already exist.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from prime_rl.utils.dcp_export import iter_cpu_hf_groups, write_hf_shards
from prime_rl.utils.export_assets import prepare_export_config, read_model_config, save_model_assets
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_config_dir

logger = get_logger()
RUN_CONFIG_NAMES = ("trainer.json", "sft.json")


def resolve_dcp_dir(ckpt_dir: Path) -> Path:
    """Find the DCP checkpoint in a step directory or its trainer subdirectory."""
    if (ckpt_dir / "trainer" / ".metadata").exists():
        return ckpt_dir / "trainer"
    if (ckpt_dir / ".metadata").exists():
        return ckpt_dir
    raise FileNotFoundError(f"No DCP checkpoint found at {ckpt_dir} (expected {ckpt_dir}/trainer/.metadata)")


def resolve_run_configs(step_dir: Path) -> tuple[dict, dict]:
    """Read the model and tokenizer config for this run."""
    config_dir = get_config_dir(step_dir.parent.parent)
    for name in RUN_CONFIG_NAMES:
        path = config_dir / name
        if path.exists():
            logger.info(f"Reading model config from {path}")
            run_config = json.loads(path.read_text())
            return run_config["model"], run_config["tokenizer"]
    raise FileNotFoundError(
        f"No resolved run config ({' or '.join(RUN_CONFIG_NAMES)}) found under {config_dir} "
        "- the checkpoint must live in its run directory"
    )


def convert(ckpt_dir: Path, output_dir: Path | None = None) -> Path:
    """Export a full trainer checkpoint on CPU without loading base-model weights."""
    ckpt_dir = Path(ckpt_dir)
    dcp_dir = resolve_dcp_dir(ckpt_dir)
    step_dir = dcp_dir.parent
    model_config, tokenizer_config = resolve_run_configs(step_dir)
    config = read_model_config(model_config)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("CPU export requires one process; run without torchrun (WORLD_SIZE must be 1)")
    config = prepare_export_config(config, model_config)
    output_dir = Path(output_dir) if output_dir is not None else step_dir / "weights"
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"CPU export output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        logger.info(f"Saving model assets to {staging_dir}")
        save_model_assets(config, model_config, tokenizer_config, staging_dir)
        logger.info(f"Reading model tensors from {dcp_dir} on CPU")
        write_hf_shards(iter_cpu_hf_groups(dcp_dir, config, model_config=model_config), staging_dir)
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
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <step_dir>/weights")
    args = parser.parse_args()
    setup_logger("info")
    convert(args.ckpt_dir, args.output_dir)


if __name__ == "__main__":
    main()
