import os
import subprocess
import sys
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import tomli_w

from prime_rl.configs.sft import SFTConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import validate_output_dir
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import get_free_port


def write_trainer_config(config: SFTConfig, output_dir: Path) -> None:
    """Write resolved trainer config to disk, excluding launcher-only fields."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_data = config.model_dump(exclude={"deployment"}, exclude_none=True, mode="json")
    with open(output_dir / "trainer.toml", "wb") as f:
        tomli_w.dump(trainer_data, f)


def render_slurm_script(config: SFTConfig, config_dir: Path) -> tuple[str, str]:
    """Render the SLURM script template. Returns (script, log_message)."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    if config.deployment.type == "single_node":
        config_path = config_dir / "sft.toml"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "wb") as f:
            tomli_w.dump(config.model_dump(exclude={"slurm"}, exclude_none=True, mode="json"), f)

        script = template.render(
            config_path=config_path,
            output_dir=config.output_dir,
            job_name=config.slurm.job_name,
            project_dir=config.slurm.project_dir,
            gpus_per_node=config.deployment.gpus_per_node,
            partition=config.slurm.partition,
        )
        log_dir = config.output_dir / "logs"
        log_message = f"Logs:\n  Trainer:  tail -F {log_dir}/trainer/rank_0.log"
    else:
        script = template.render(
            config_dir=config_dir,
            output_dir=config.output_dir,
            job_name=config.slurm.job_name,
            project_dir=config.slurm.project_dir,
            num_nodes=config.deployment.num_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            partition=config.slurm.partition,
        )
        log_message = f"Logs:\n  Trainer:  tail -F {config.output_dir}/slurm/latest_train_node_rank_0.log"

    return script, log_message


def sft_slurm(config: SFTConfig):
    """Run SFT training via SLURM."""
    assert config.slurm is not None

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = config.output_dir / "configs"

    if config.deployment.type == "multi_node":
        write_trainer_config(config, config_dir)
        logger.info(f"Wrote trainer config to {config_dir}")

    script, log_message = render_slurm_script(config, config_dir)
    script_path = config.output_dir / "sft.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    logger.info(f"Wrote SLURM script to {script_path}")

    if config.slurm.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def sft_local(config: SFTConfig):
    """Run SFT training locally with process monitoring and cleanup."""
    assert config.deployment.type == "single_node"

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = Path(".pydantic_config") / uuid.uuid4().hex
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "trainer.toml"
    with open(config_path, "wb") as f:
        tomli_w.dump(config.model_dump(exclude={"deployment"}, exclude_none=True, mode="json"), f)

    log_dir = config.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer_cmd = [
        "uv",
        "run",
        "env",
        "PYTHONUNBUFFERED=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "torchrun",
        f"--rdzv-endpoint=localhost:{get_free_port()}",
        f"--rdzv-id={uuid.uuid4().hex}",
        f"--log-dir={config.output_dir / 'torchrun'}",
        "--local-ranks-filter=0",
        "--redirect=3",
        "--tee=3",
        f"--nproc-per-node={config.deployment.num_gpus}",
        "-m",
        "prime_rl.trainer.sft.train",
        "@",
        config_path.as_posix(),
    ]

    logger.info(f"Starting SFT trainer with {config.deployment.num_gpus} GPU(s)")
    logger.debug(f"Trainer command: {' '.join(trainer_cmd)}")

    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []

    try:
        with open(log_dir / "trainer.stdout", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                },
                stdout=log_file,
                stderr=log_file,
            )
        processes.append(trainer_process)

        stop_event = Event()
        monitor_thread = Thread(
            target=monitor_process,
            args=(trainer_process, stop_event, error_queue, "trainer"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        logger.success("Startup complete. Showing trainer logs...")
        tail_process = Popen(["tail", "-F", str(log_dir / "trainer.stdout")])
        processes.append(tail_process)

        stop_event.wait()

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("SFT training finished!")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise


def sft(config: SFTConfig):
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    clean = config.clean_output_dir and not os.environ.get("NEVER_CLEAN_OUTPUT_DIR")
    validate_output_dir(config.output_dir, resuming=resuming, clean=clean)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.slurm is not None:
        sft_slurm(config)
    else:
        sft_local(config)


def main():
    sft(parse_argv(SFTConfig))


if __name__ == "__main__":
    main()
