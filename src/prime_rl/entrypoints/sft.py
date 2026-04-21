import os
import subprocess
import sys
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import tomli_w

from prime_rl.configs.sft import SFTConfig
from prime_rl.trainer.model import pre_download_model
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import format_log_message, get_config_dir, get_log_dir, validate_output_dir
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process, set_proc_title
from prime_rl.utils.utils import get_free_port

SFT_TOML = "sft.toml"
SFT_SBATCH = "sft.sbatch"


def write_config(config: SFTConfig, config_path: Path, exclude: set[str] | None = None) -> None:
    """Write resolved config to disk, excluding launcher-only fields."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = config.model_dump(exclude=exclude, exclude_none=True, mode="json")
    with open(config_path, "wb") as f:
        tomli_w.dump(config_dict, f)


def write_slurm_script(config: SFTConfig, config_path: Path, script_path: Path) -> None:
    """Write the SLURM script to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    if config.deployment.type == "single_node":
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_path,
            output_dir=config.output_dir,
            gpus_per_node=config.deployment.gpus_per_node,
        )
    else:
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_path,
            output_dir=config.output_dir,
            num_nodes=config.deployment.num_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            ranks_filter=",".join(map(str, config.log.ranks_filter)),
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def sft_slurm(config: SFTConfig):
    """Run SFT training via SLURM."""
    assert config.slurm is not None

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = get_config_dir(config.output_dir)
    config_path = config_dir / SFT_TOML
    exclude = (
        {"deployment", "slurm", "dry_run", "clean_output_dir"}
        if config.deployment.type == "multi_node"
        else {"slurm", "dry_run", "clean_output_dir"}
    )
    write_config(config, config_path, exclude=exclude)
    logger.info(f"Wrote config to {config_path}")

    script_path = config.output_dir / SFT_SBATCH
    write_slurm_script(config, config_path, script_path)
    logger.info(f"Wrote SLURM script to {script_path}")

    log_dir = get_log_dir(config.output_dir)
    num_nodes = config.deployment.num_nodes if config.deployment.type == "multi_node" else 1
    log_message = format_log_message(log_dir=log_dir, trainer=True, num_train_nodes=num_nodes)

    if config.dry_run:
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

    config_dir = get_config_dir(config.output_dir)
    config_path = config_dir / SFT_TOML
    write_config(config, config_path)
    logger.info(f"Wrote config to {config_path}")

    if config.dry_run:
        logger.success("Dry run complete. To start an SFT run locally, remove --dry-run from your command.")
        return

    log_dir = config.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer_cmd = [
        "torchrun",
        "--role=trainer",
        f"--rdzv-endpoint=localhost:{get_free_port()}",
        f"--rdzv-id={uuid.uuid4().hex}",
        f"--log-dir={config.output_dir / 'logs' / 'trainer' / 'torchrun'}",
        f"--local-ranks-filter={','.join(map(str, config.log.ranks_filter))}",
        "--redirect=3",
        "--tee=3",
        f"--nproc-per-node={config.deployment.num_gpus}",
        "-m",
        "prime_rl.trainer.sft.train",
        "@",
        (config_dir / SFT_TOML).as_posix(),
    ]

    logger.info(f"Starting SFT trainer with {config.deployment.num_gpus} GPU(s)")
    logger.debug(f"Trainer command: {' '.join(trainer_cmd)}")

    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []

    try:
        with open(log_dir / "trainer.log", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
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
        tail_process = Popen(
            f"tail -F '{log_dir / 'trainer.log'}' | sed -u 's/^\\[[a-zA-Z]*[0-9]*\\]://'",
            shell=True,
        )
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

    if not config.dry_run:
        pre_download_model(config.model.name)

    if config.slurm is not None:
        sft_slurm(config)
    else:
        sft_local(config)


def main():
    set_proc_title("SFT")
    sft(cli(SFTConfig))


if __name__ == "__main__":
    main()
