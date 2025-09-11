import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import warnings
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
from prime_rl.orchestrator.config import CheckpointConfig as OrchestratorCheckpointConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.trainer.config import CheckpointConfig as TrainerCheckpointConfig
from prime_rl.trainer.rl.config import FakeDataLoaderConfig
from prime_rl.trainer.rl.config import RLTrainerConfig as TrainerConfig
from prime_rl.utils.config import WandbMonitorConfig
from prime_rl.utils.logger import format_message, format_time, set_logger, setup_handlers
from prime_rl.utils.pydantic_config import BaseSettings, get_temp_toml_file, parse_argv
from prime_rl.utils.utils import (
    get_ckpt_dir,
    get_cuda_visible_devices,
    get_free_port,
    get_log_dir,
    get_rollout_dir,
    get_weights_dir,
)
from prime_rl.utils.validation import (
    validate_shared_async_level,
    validate_shared_ckpt_config,
    validate_shared_max_model_len,
    validate_shared_max_steps,
    validate_shared_model_name,
    validate_shared_output_dir,
    validate_shared_wandb_config,
)


class LogConfig(BaseSettings):
    """Configures shared logging."""

    level: Annotated[str | None, Field(description="The log level to use.")] = "info"

    utc: Annotated[
        bool | None,
        Field(
            description="Whether to use UTC time in the logger. If False, it will default to the local time. If the local time is wrong, you can set it by setting the `TZ` environment variable. For example, `TZ=America/Los_Angeles` will set the local time to SF time."
        ),
    ] = False


class WandbConfig(BaseSettings):
    """Configures shared W&B configs."""

    project: Annotated[str | None, Field(description="The W&B project to use.")] = "prime-rl"

    name: Annotated[str | None, Field(description="The W&B run name to use.")] = None

    offline: Annotated[bool | None, Field(description="Whether to run W&B in offline mode.")] = False


class CheckpointConfig(BaseSettings):
    """Configures shared checkpoint configs."""

    interval: Annotated[int | None, Field(description="The interval at which to save checkpoints.")] = 50

    resume_step: Annotated[
        int | None, Field(description="The step to resume from. If None, will not resume from a checkpoint.")
    ] = None

    keep: Annotated[
        int | None,
        Field(
            ge=1,
            description="Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints.",
        ),
    ] = None


class ModelConfig(BaseSettings):
    """Configures shared model settings."""

    name: Annotated[
        str,
        Field(description="The name of the model to use."),
    ] = "Qwen/Qwen3-0.6B"


class RLConfig(BaseSettings):
    """Configures an RL training run."""

    ### Submodule configurations

    trainer: TrainerConfig
    orchestrator: OrchestratorConfig
    inference: Annotated[
        InferenceConfig | None,
        Field(
            description="The inference config. If None, will not start an inference process. Only viable, if an inference server was started manually."
        ),
    ] = None

    ### Top-level configurations

    log: Annotated[
        LogConfig,
        Field(
            description="Shared log configs. If None, will fallback to the log configs specified on submodule configs."
        ),
    ] = LogConfig()

    clean: Annotated[
        bool,
        Field(
            description="Whether to clean the rollouts, checkpoint, checkpoint weights and logs directories at the beginning of the run. If True, will forceably, and irreversibly, delete all directories.",
        ),
    ] = True

    trainer_gpus: Annotated[int, Field(description="The number of GPUs to use for trainer.")] = 1

    inference_gpus: Annotated[int, Field(description="The number of GPUs to use for inference.")] = 1

    ### Shared configurations

    output_dir: Annotated[
        Path,
        Field(description="The directory to store the outputs. Should typically be set to an experiment identifier."),
    ] = Path("outputs")  # NOTE: Must match `OUTPUT_DIR` in `tmux.sh` to see logs

    ckpt: Annotated[
        CheckpointConfig | None,
        Field(
            description="Shared checkpoint configs. If None, will fallback to the checkpoint configs specified on submodule configs."
        ),
    ] = None

    wandb: Annotated[
        WandbConfig | None,
        Field(
            description="Shared W&B configs. If None, will fallback to the W&B configs specified on submodule configs."
        ),
    ] = None

    model: Annotated[
        ModelConfig | None,
        Field(
            description="Shared model configs. If None, will fallback to the model configs specified on submodule configs."
        ),
    ] = None

    max_steps: Annotated[
        int | None,
        Field(
            description="The maximum number of steps to train for. If None, will fallback to the max steps specified on submodule configs."
        ),
    ] = None

    max_model_len: Annotated[
        int | None,
        Field(
            description="The maximum model length to use. If None, will fallback to the max model length specified on submodule configs."
        ),
    ] = None

    async_level: Annotated[
        int | None,
        Field(
            description="The async level to use. If None, will fallback to the async level specified on submodule configs."
        ),
    ] = None

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the trainer and orchestrator to benchmark mode and, if present, configure the W&B project by suffixing the project with `-bench`.",
        ),
    ] = False

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
    def auto_setup_num_train_workers(self):
        if self.trainer_gpus > 1:
            self.orchestrator.num_train_workers = self.trainer_gpus
        return self

    @model_validator(mode="after")
    def auto_setup_logs(self):
        # Copy log level
        if self.log and self.log.level:
            self.trainer.log.level = self.log.level
            self.orchestrator.log.level = self.log.level

        return self

    ### Setup and validate shared configs

    @model_validator(mode="after")
    def auto_setup_ckpt(self):
        # If specified, automatically setup checkpoint configs for trainer and orchestrator
        if self.ckpt:
            # Create checkpoint configs if not specified
            if not self.trainer.ckpt:
                self.trainer.ckpt = TrainerCheckpointConfig()
            if not self.orchestrator.ckpt:
                self.orchestrator.ckpt = OrchestratorCheckpointConfig()

            # If specified, use the same ckpt interval
            if self.ckpt.interval:
                self.trainer.ckpt.interval = self.ckpt.interval
                self.orchestrator.ckpt.interval = self.ckpt.interval

            # If resuming training, ensure orchestrator resume from the same step
            if self.ckpt.resume_step:
                self.trainer.ckpt.resume_step = self.ckpt.resume_step
                self.orchestrator.ckpt.resume_step = self.ckpt.resume_step

            # If specified, propagate keep policy
            if self.ckpt.keep:
                self.trainer.ckpt.keep = self.ckpt.keep
                self.orchestrator.ckpt.keep = self.ckpt.keep

        validate_shared_ckpt_config(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_wandb(self):
        # If specified, automatically use shared W&B project for orchestrator and trainer
        if self.wandb:
            if not self.trainer.wandb:
                self.trainer.wandb = WandbMonitorConfig()
            if not self.orchestrator.wandb:
                self.orchestrator.wandb = WandbMonitorConfig()

            if self.wandb.project:
                self.trainer.wandb.project = self.wandb.project
                self.orchestrator.wandb.project = self.wandb.project

            # If specified, automatically use shared W&B name for orchestrator and trainer with suffixes
            if self.wandb.name:
                self.trainer.wandb.name = f"{self.wandb.name}-trainer"
                self.orchestrator.wandb.name = f"{self.wandb.name}-orchestrator"

            # If specified, automatically use shared W&B offline mode for orchestrator and trainer
            if self.wandb.offline:
                self.trainer.wandb.offline = self.wandb.offline
                self.orchestrator.wandb.offline = self.wandb.offline

        validate_shared_wandb_config(self.trainer, self.orchestrator)

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

        if self.trainer.bench != self.orchestrator.bench:
            raise ValueError(
                f"Trainer benchmark mode ({self.trainer.bench}) and orchestrator benchmark mode ({self.orchestrator.bench}) are not the same. Please specify the same benchmark mode for both."
            )

        return self

    @model_validator(mode="after")
    def auto_setup_model(self):
        # Use the same model for trainer, orchestrator and inference
        if self.model is not None and self.model.name:
            self.trainer.model.name = self.model.name
            self.orchestrator.model.name = self.model.name
            if self.inference:
                self.inference.model.name = self.model.name

        validate_shared_model_name(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def auto_setup_max_steps(self):
        # If specified, use the same max steps for trainer and orchestrator
        if self.max_steps:
            self.trainer.max_steps = self.max_steps
            self.orchestrator.max_steps = self.max_steps

        validate_shared_max_steps(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_max_model_len(self):
        if self.max_model_len:
            self.orchestrator.seq_len = self.max_model_len
            if self.inference:
                self.inference.model.max_model_len = self.max_model_len

        validate_shared_max_model_len(self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def auto_setup_async_level(self):
        # If specified, use the same async level for trainer and orchestrator
        if self.async_level:
            self.trainer.async_level = self.async_level
            self.orchestrator.async_level = self.async_level

        validate_shared_async_level(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_output_dir(self):
        # If specified, use the same outputs directory for trainer and orchestrator
        if self.output_dir:
            self.trainer.output_dir = self.output_dir
            self.orchestrator.output_dir = self.output_dir

        validate_shared_output_dir(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def warn_wandb_resume_id_missing(self):
        if self.trainer.ckpt and self.trainer.ckpt.resume_step:
            if self.trainer.wandb and not self.trainer.wandb.id:
                warnings.warn(
                    "W&B run ID is not set for trainer even though resuming training. The current run will be created as a new run."
                )
        if self.orchestrator.ckpt and self.orchestrator.ckpt.resume_step:
            if self.orchestrator.wandb and not self.orchestrator.wandb.id:
                warnings.warn(
                    "W&B run ID is not set for orchestrator even though resuming training. The current run will be created as a new run."
                )
        return self


def setup_logger(log_config: LogConfig) -> Logger:
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
    log_dir = get_log_dir(config.output_dir)
    ckpt_dir = get_ckpt_dir(config.output_dir)
    weights_dir = get_weights_dir(config.output_dir)
    rollout_dir = get_rollout_dir(config.output_dir)

    # Clean up directories if specified
    if config.clean:
        logger.info("Cleaning checkpoint, logs, weights and rollout directories")

        # Cleaning logs
        logger.info(f"Cleaning log dir ({log_dir})")
        shutil.rmtree(log_dir, ignore_errors=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Cleaning checkpoints and weights, unless resuming
        do_resume = config.trainer.ckpt and config.trainer.ckpt.resume_step
        if not do_resume:  # Only clean if we don't resume
            logger.info(f"Cleaning checkpoint directory ({ckpt_dir})")
            shutil.rmtree(ckpt_dir, ignore_errors=True)

            logger.info(f"Cleaning checkpoint weights directory ({weights_dir})")
            shutil.rmtree(weights_dir, ignore_errors=True)

        # Cleaning rollouts
        logger.info(f"Cleaning rollout dir ({rollout_dir})")
        shutil.rmtree(rollout_dir, ignore_errors=True)

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
            with open(log_dir / "inference.log", "w") as log_file:
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
        with open(log_dir / "orchestrator.log", "w") as log_file:
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
            f"--rdzv-id={uuid.uuid4().hex}",
            "--nproc-per-node",
            str(config.trainer_gpus),
            "-m",
            "prime_rl.trainer.rl.train",
            "@",
            trainer_file.as_posix(),
        ]
        train_gpu_ids = devices[config.inference_gpus :]
        logger.info(f"Starting trainer process on GPU(s) {' '.join(map(str, train_gpu_ids))}")
        logger.debug(f"Training start command: {' '.join(trainer_cmd)}")
        with open(log_dir / "trainer.log", "w") as log_file:
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

        tail_process = Popen(["tail", "-F", log_dir / "trainer.log"])
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
