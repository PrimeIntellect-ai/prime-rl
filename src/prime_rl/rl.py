import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread
from typing import Annotated

import tomli_w
import torch
from loguru import logger as loguru_logger
from loguru._logger import Logger
from pydantic import Field, model_validator

from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.trainer.config import CheckpointConfig, FakeDataLoaderConfig, TrainerConfig
from prime_rl.utils.config import WandbMonitorConfig
from prime_rl.utils.logger import format_message, format_time, get_logger, set_logger, setup_handlers
from prime_rl.utils.pydantic_config import BaseSettings, get_temp_toml_file, parse_argv
from prime_rl.utils.utils import get_cuda_visible_devices, get_free_port


class LogConfig(BaseSettings):
    """Configures the logging for the RL run."""

    path: Annotated[Path, Field(description="The path to the logs directory.")] = Path("logs")

    level: Annotated[str, Field(description="The log level to use.")] = "info"

    utc: Annotated[
        bool,
        Field(
            description="Whether to use UTC time in the logger. If False, it will default to the local time. If the local time is wrong, you can set it by setting the `TZ` environment variable. For example, `TZ=America/Los_Angeles` will set the local time to SF time."
        ),
    ] = False


class RLConfig(BaseSettings):
    """Configures an RL training run."""

    trainer: TrainerConfig
    orchestrator: OrchestratorConfig
    inference: Annotated[
        InferenceConfig | None,
        Field(
            description="The inference config. If None, will not start an inference process. Only viable, if an inference server was started manually."
        ),
    ] = None

    log: LogConfig = LogConfig()

    exp_id: Annotated[
        str | None,
        Field(
            description="The experiment ID. If set, will be used to identify shared resources, like log files, weight and rollout directories, etc."
        ),
    ] = "rl"  # This value has to match the `DEFAULT_EXPERIMENT_ID` in `tmux.sh`

    trainer_gpus: Annotated[int, Field(description="The number of GPUs to use for trainer.")] = 1

    inference_gpus: Annotated[int, Field(description="The number of GPUs to use for inference.")] = 1

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the trainer and orchestrator to benchmark mode and, if present, configure the W&B project by suffixing the project with `-bench`.",
        ),
    ] = False

    clean: Annotated[
        bool,
        Field(
            description="Whether to clean the rollouts, checkpoint, checkpoint weights and logs directories at the beginning of the run. If True, will forceably, and irreversibly, delete all directories.",
        ),
    ] = True

    @model_validator(mode="after")
    def validate_device(self):
        available_gpus = torch.cuda.device_count()
        if self.trainer_gpus + self.inference_gpus > available_gpus:
            raise ValueError(
                f"Total number of GPUs ({self.trainer_gpus + self.inference_gpus}) exceeds available GPUs ({available_gpus})"
            )
        if self.inference and self.inference_gpus != self.inference.parallel.dp * self.inference.parallel.tp:
            raise ValueError(
                f"Total number of inference GPUs ({self.inference_gpus}) does not match the local sharding strategy ({self.inference.parallel.dp} DP + {self.inference.parallel.tp} TP)"
            )
        return self

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            # Set trainer and orchestrator to benchmark mode
            self.trainer.bench = True
            self.orchestrator.bench = True

            # Configure the trainer fake data to match the orchestrator config
            self.trainer.data.fake = FakeDataLoaderConfig(
                micro_batch_size=self.orchestrator.micro_batch_size,
                batch_size=self.orchestrator.batch_size,
                seq_len=self.orchestrator.seq_len,
            )

        return self

    @model_validator(mode="after")
    def auto_setup_logs(self):
        # Copy log level
        self.trainer.log.level = self.log.level
        self.orchestrator.log.level = self.log.level

        return self

    @model_validator(mode="after")
    def auto_setup_wandb(self):
        # Automatically use same W&B project for orchestrator and trainer
        if self.orchestrator and self.trainer.monitor.wandb:
            if not self.orchestrator.monitor.wandb:
                self.orchestrator.monitor.wandb = WandbMonitorConfig()
            self.orchestrator.monitor.wandb.project = self.trainer.monitor.wandb.project

            # If name is set on trainer, copy it and add suffixes
            if self.trainer.monitor.wandb.name:
                run_name = deepcopy(self.trainer.monitor.wandb.name)
                self.trainer.monitor.wandb.name = f"{run_name}-trainer"
                self.orchestrator.monitor.wandb.name = f"{run_name}-orchestrator"
        return self

    @model_validator(mode="after")
    def auto_setup_model(self):
        # Use trainer model on orchestrator and inference
        self.orchestrator.model.name = self.trainer.model.name
        if self.inference:
            self.inference.model.name = self.trainer.model.name
        return self

    @model_validator(mode="after")
    def auto_setup_orchestrator_log_level(self):
        # Use trainer log level on orchestrator
        self.orchestrator.log.level = self.trainer.log.level
        return self

    @model_validator(mode="after")
    def auto_setup_max_step(self):
        # Use trainer max steps on orchestrator
        if self.trainer.max_steps is not None:
            self.orchestrator.max_steps = self.trainer.max_steps
        return self

    @model_validator(mode="after")
    def auto_setup_async_level(self):
        # Use trainer async level on orchestrator
        self.orchestrator.async_level = self.trainer.async_level
        return self

    @model_validator(mode="after")
    def auto_setup_exp(self):
        if self.exp_id:
            self.log.path = self.log.path / self.exp_id
            # Will be shared to orchestrator via `auto_setup_paths`
            self.trainer.data.path = self.trainer.data.path / self.exp_id
            self.trainer.weights.path = self.trainer.weights.path / self.exp_id
            # Will be shared to orchestrator via `auto_setup_ckpt`
            if self.trainer.ckpt:
                self.trainer.ckpt.path = self.trainer.ckpt.path / self.exp_id
        return self

    @model_validator(mode="after")
    def auto_setup_paths(self):
        # Ensure trainer and orchestrator use the same paths for communicating data and weights
        self.orchestrator.rollout_path = self.trainer.data.path
        self.orchestrator.weights_path = self.trainer.weights.path
        return self

    @model_validator(mode="after")
    def auto_setup_ckpt(self):
        # Ensures that trainer and orchestrator checkpoints are synchronized
        if self.trainer.ckpt:
            self.orchestrator.ckpt = CheckpointConfig()
            self.orchestrator.ckpt.path = self.trainer.ckpt.path
            self.orchestrator.ckpt.interval = self.trainer.ckpt.interval

            # If resuming training, ensure orchestrator resumes from the same step
            if self.trainer.ckpt.resume_step:
                self.orchestrator.ckpt.resume_step = self.trainer.ckpt.resume_step
        return self

    @model_validator(mode="after")
    def warn_wandb_resume_id_missing(self):
        if self.trainer.ckpt and self.trainer.ckpt.resume_step:
            if self.trainer.monitor.wandb and not self.trainer.monitor.wandb.id:
                warnings.warn(
                    "W&B run ID is not set for trainer even though resuming training. The current run will be created as a new run."
                )
        if self.orchestrator.ckpt and self.orchestrator.ckpt.resume_step:
            if self.orchestrator.monitor.wandb and not self.orchestrator.monitor.wandb.id:
                warnings.warn(
                    "W&B run ID is not set for orchestrator even though resuming training. The current run will be created as a new run."
                )
        return self

    @model_validator(mode="after")
    def auto_setup_num_train_workers(self):
        if self.trainer_gpus > 1:
            self.orchestrator.num_train_workers = self.trainer_gpus
        return self

    @model_validator(mode="after")
    def auto_setup_seq_len(self):
        if self.inference:
            self.inference.model.max_model_len = self.orchestrator.seq_len
        return self


def setup_logger(log_config: LogConfig) -> Logger:
    if get_logger():
        raise RuntimeError("Logger already setup. Call reset_logger first.")

    # Setup the logger handlers
    format = format_time(log_config) + format_message()
    logger = setup_handlers(loguru_logger, format, log_config, rank=0)
    set_logger(logger)

    return logger


def cleanup_threads(threads: list[Thread]):
    for thread in threads:
        thread.join(timeout=5)


def cleanup_processes(processes: list[Popen]):
    for process in processes:
        if process.poll() is None:  # Process is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def monitor_process(process: Popen, stop_event: Event, error_queue: list, process_name: str):
    """Monitor a subprocess and signal errors via shared queue"""
    try:
        # Wait for process to complete
        process.wait()

        if process.returncode != 0:
            err_msg = f"{process_name.capitalize()} failed with exit code {process.returncode}"
            if process.stderr:
                err_msg += f"\n{process.stderr.read().decode('utf-8')}"
            error_queue.append(RuntimeError(err_msg))
        stop_event.set()
    except Exception as e:
        error_queue.append(RuntimeError(f"Error monitoring {process_name}: {e}"))
        stop_event.set()


def rl(config: RLConfig):
    # Setup logger
    logger = setup_logger(config.log)
    start_command = sys.argv
    logger.info("Starting RL run")
    logger.debug(f"RL start command: {' '.join(start_command)}")

    # Prepare paths to communicate with the trainer
    if config.clean:
        logger.info("Cleaning checkpoint, logs, checkpoint weights and rollout directories")

        # Cleaning logs
        logger.info(f"Cleaning logs ({config.log.path})")
        shutil.rmtree(config.log.path, ignore_errors=True)
        config.log.path.mkdir(parents=True, exist_ok=True)

        # Cleaning checkpoints
        if config.trainer.ckpt and not config.trainer.ckpt.resume_step:  # Only clean if we don't resume
            logger.info(f"Cleaning trainer checkpoint path ({config.trainer.ckpt.path})")
            shutil.rmtree(config.trainer.ckpt.path, ignore_errors=True)

        if config.orchestrator.ckpt and not config.orchestrator.ckpt.resume_step:  # Only clean if we don't resume
            logger.info(f"Cleaning orchestrator checkpoint path ({config.orchestrator.ckpt.path})")
            shutil.rmtree(config.orchestrator.ckpt.path, ignore_errors=True)

        if not (config.orchestrator.ckpt and config.orchestrator.ckpt.resume_step):  # Only clean if we don't resume
            logger.info(f"Cleaning checkpoint weights path ({config.orchestrator.weights_path})")
            shutil.rmtree(config.orchestrator.weights_path, ignore_errors=True)

        logger.info(f"Cleaning rollout path ({config.trainer.data.path})")
        shutil.rmtree(config.trainer.data.path, ignore_errors=True)

    # Start processes
    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}
    all_devices = get_cuda_visible_devices()
    devices = all_devices[: config.trainer_gpus + config.inference_gpus]
    logger.info(f"Available GPUs: {', '.join(map(str, all_devices))} (using: {', '.join(map(str, devices))})")

    try:
        # Optionally, start inference process
        if config.inference:
            inference_file = get_temp_toml_file()
            with open(inference_file, "wb") as f:
                tomli_w.dump(config.inference.model_dump(exclude_none=True, mode="json"), f)

            inference_cmd = ["uv", "run", "inference", "@", inference_file.as_posix()]
            inference_gpu_ids = devices[: config.inference_gpus]
            logger.info(f"Starting inference process on GPU(s) {' '.join(map(str, inference_gpu_ids))}")
            logger.debug(f"Inference start command: {' '.join(inference_cmd)}")
            # If we don't log stdout, the server hangs
            with open(config.log.path / "inference.log", "w") as log_file:
                inference_process = Popen(
                    inference_cmd,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, inference_gpu_ids))},
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(inference_process)

            # Start monitoring thread
            stop_event = Event()
            stop_events["inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(inference_process, stop_event, error_queue, "inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        else:
            logger.warning(
                "No inference config specified, skipping starting inference server. Is your inference server running?"
            )

        # Start orchestrator process
        orchestrator_file = get_temp_toml_file()
        with open(orchestrator_file, "wb") as f:
            tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)

        orchestrator_cmd = [
            "uv",
            "run",
            "orchestrator",
            "@",
            orchestrator_file.as_posix(),
        ]
        logger.info("Starting orchestrator process")
        logger.debug(f"Orchestrator start command: {' '.join(orchestrator_cmd)}")
        with open(config.log.path / "orchestrator.log", "w") as log_file:
            orchestrator_process = Popen(
                orchestrator_cmd,
                stdout=log_file,
                stderr=log_file,
                env={
                    **os.environ,
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
            )
        processes.append(orchestrator_process)

        # Start monitoring thread
        stop_event = Event()
        stop_events["orchestrator"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(orchestrator_process, stop_event, error_queue, "orchestrator"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # Start training process
        trainer_file = get_temp_toml_file()
        with open(trainer_file, "wb") as f:
            tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)

        trainer_cmd = [
            "uv",
            "run",
            "torchrun",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            f"--rdzv-id={config.exp_id}",
            "--nproc-per-node",
            str(config.trainer_gpus),
            "src/prime_rl/trainer/train.py",
            "@",
            trainer_file.as_posix(),
        ]
        train_gpu_ids = devices[config.inference_gpus :]
        logger.info(f"Starting trainer process on GPU(s) {' '.join(map(str, train_gpu_ids))}")
        logger.debug(f"Training start command: {' '.join(trainer_cmd)}")
        with open(config.log.path / "trainer.log", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, train_gpu_ids)),
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
                stdout=log_file,
                stderr=log_file,
            )
        processes.append(trainer_process)

        # Start monitoring thread
        stop_event = Event()
        stop_events["trainer"] = stop_event
        monitor_thread = Thread(
            target=monitor_process, args=(trainer_process, stop_event, error_queue, "trainer"), daemon=True
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # Monitor all processes for failures
        logger.success("Startup complete. Showing trainer logs...")

        tail_process = Popen(["tail", "-F", config.log.path / "trainer.log"])
        processes.append(tail_process)

        # Check for errors from monitor threads
        while not (stop_events["orchestrator"].is_set() and stop_events["trainer"].is_set()):
            if error_queue:
                error = error_queue[0]
                logger.error(f"Error: {error}")
                logger.error("Terminating all processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)

            # Small delay to avoid busy waiting
            time.sleep(1)

        logger.success("RL training finished!")

        # Cleanup threads and processes
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


def main():
    rl(parse_argv(RLConfig))


if __name__ == "__main__":
    main()
