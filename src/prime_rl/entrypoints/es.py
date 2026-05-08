import os
import subprocess
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import tomli_w

import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive import
from prime_rl.configs.es import ESConfig
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import get_config_dir, validate_output_dir
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process, set_proc_title
from prime_rl.utils.utils import get_free_port

ES_TOML = "es.toml"


def write_config(config: ESConfig, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "wb") as f:
        tomli_w.dump(config.model_dump(exclude_none=True, mode="json"), f)


def es_local(config: ESConfig) -> None:
    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = get_config_dir(config.output_dir)
    config_path = config_dir / ES_TOML
    write_config(config, config_path)
    logger.info(f"Wrote config to {config_path}")

    if config.dry_run:
        logger.success("Dry run complete. To start ES training locally, remove --dry-run from your command.")
        return

    log_dir = config.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer_cmd = [
        "torchrun",
        "--role=trainer",
        f"--rdzv-endpoint=localhost:{get_free_port()}",
        f"--rdzv-id={uuid.uuid4().hex}",
        f"--log-dir={log_dir / 'trainer' / 'torchrun'}",
        f"--local-ranks-filter={','.join(map(str, config.log.ranks_filter))}",
        "--redirect=3",
        "--tee=3",
        f"--nproc-per-node={config.deployment.num_gpus}",
        "-m",
        "prime_rl.trainer.es.train",
        "@",
        config_path.as_posix(),
    ]

    logger.info(f"Starting ES trainer with {config.deployment.num_gpus} GPU(s)")
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
            args=(trainer_process, stop_event, error_queue, "es-trainer"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        stop_event.wait()

        if error_queue:
            raise error_queue[0]

        if trainer_process.returncode != 0:
            raise subprocess.CalledProcessError(trainer_process.returncode, trainer_cmd)

        logger.success("ES training completed")
    except KeyboardInterrupt:
        logger.info("Interrupted, cleaning up")
        raise
    finally:
        cleanup_processes(processes)
        cleanup_threads(monitor_threads)


def es(config: ESConfig) -> None:
    if config.slurm is not None:
        raise ValueError("ES SLURM launch is not implemented yet.")
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    clean = config.clean_output_dir and not os.environ.get("NEVER_CLEAN_OUTPUT_DIR")
    validate_output_dir(config.output_dir, resuming=resuming, clean=clean)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    es_local(config)


def main():
    set_proc_title("ES")
    es(cli(ESConfig))


if __name__ == "__main__":
    main()
