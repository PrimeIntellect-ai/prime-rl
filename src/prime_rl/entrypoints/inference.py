import subprocess
import sys
from pathlib import Path

import tomli_w

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import get_config_dir
from prime_rl.utils.pydantic_config import parse_argv

INFERENCE_TOML = "inference.toml"
INFERENCE_SBATCH = "inference.sbatch"


def write_config(config: InferenceConfig, output_dir: Path, exclude: set[str] | None = None) -> Path:
    """Write resolved config to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / INFERENCE_TOML
    with open(config_path, "wb") as f:
        tomli_w.dump(config.model_dump(exclude=exclude, exclude_none=True, mode="json"), f)
    return config_path


def write_slurm_script(config: InferenceConfig, config_path: Path, script_path: Path) -> None:
    """Write the SLURM script to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    script = template.render(
        config_path=config_path,
        output_dir=config.output_dir,
        job_name=config.slurm.job_name,
        project_dir=config.slurm.project_dir,
        gpus_per_node=config.deployment.gpus_per_node,
        partition=config.slurm.partition,
        num_nodes=config.deployment.num_nodes if config.deployment.type == "multi_node" else 1,
        port=config.server.port,
    )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def inference_slurm(config: InferenceConfig):
    """Run inference via SLURM."""
    assert config.slurm is not None

    logger = setup_logger("info")

    config_dir = get_config_dir(config.output_dir)
    exclude = {"deployment", "slurm", "dry_run"} if config.deployment.type == "multi_node" else {"slurm", "dry_run"}
    config_path = write_config(config, config_dir, exclude=exclude)
    logger.info(f"Wrote config to {config_path}")

    script_path = config.output_dir / INFERENCE_SBATCH
    write_slurm_script(config, config_path, script_path)
    logger.info(f"Wrote SLURM script to {script_path}")

    log_message = (
        f"Logs:\n"
        f"  Job:        tail -F {config.output_dir}/job_*.log\n"
        f"  Inference:  tail -F {config.output_dir}/slurm/latest_infer_node_rank_*.log\n"
    )

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def inference_local(config: InferenceConfig):
    """Run inference locally."""
    from prime_rl.inference.server import setup_vllm_env

    logger = setup_logger("info")

    config_dir = get_config_dir(config.output_dir)
    config_path = write_config(config, config_dir)
    logger.info(f"Wrote config to {config_path}")

    if config.dry_run:
        logger.success("Dry run complete. To start inference locally, remove --dry-run from your command.")
        return

    host = config.server.host or "0.0.0.0"
    port = config.server.port
    logger.info(f"Starting inference on http://{host}:{port}/v1\n")

    setup_vllm_env(config)

    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_args=config.get_unknown_args())


def inference(config: InferenceConfig):
    if config.slurm is not None:
        inference_slurm(config)
    else:
        inference_local(config)


def main():
    config = parse_argv(InferenceConfig, allow_extras=True)
    inference(config)


if __name__ == "__main__":
    main()
