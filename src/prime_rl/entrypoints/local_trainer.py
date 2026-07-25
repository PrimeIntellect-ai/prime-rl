import os
import sys
import uuid
from subprocess import Popen
from threading import Event, Thread

import tomli_w

from prime_rl.configs.sft import SFTConfig
from prime_rl.utils.config import to_toml_dict
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import get_config_dir
from prime_rl.utils.process import (
    DEFAULT_COMMON_ENV_VARS,
    DEFAULT_TRAINER_ENV_VARS,
    cleanup_processes,
    cleanup_threads,
    monitor_process,
)


def launch_local_trainer(
    config: SFTConfig,
    *,
    config_filename: str,
    trainer_module: str,
    display_name: str,
) -> None:
    assert config.deployment.type == "single_node"

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)
    config_dir = get_config_dir(config.output_dir)
    config_path = config_dir / config_filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("wb") as file:
        tomli_w.dump(to_toml_dict(config), file)
    logger.info(f"Wrote config to {config_path}")

    if config.dry_run:
        logger.success(f"Dry run complete. To start {display_name} locally, remove --dry-run from your command.")
        return

    log_dir = config.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    from prime_rl.utils.utils import get_free_port

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
        trainer_module,
        "@",
        config_path.as_posix(),
    ]

    logger.info(f"Starting {display_name} trainer with {config.deployment.num_gpus} GPU(s)")
    logger.debug(f"Trainer command: {' '.join(trainer_cmd)}")

    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    try:
        with (log_dir / "trainer.log").open("w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    **DEFAULT_COMMON_ENV_VARS,
                    **DEFAULT_TRAINER_ENV_VARS,
                    **config.env_vars,
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

        logger.success(f"{display_name} training finished!")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)
    except Exception as error:
        logger.error(f"Error occurred: {error}")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise
