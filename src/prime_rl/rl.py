import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from subprocess import Popen
from threading import Event, Thread
from typing import Annotated, Literal

import tomli_w
from pydantic import Field, model_validator

from prime_rl.config import BaseRLLauncherConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pydantic_config import BaseSettings, get_temp_toml_file, parse_argv
from prime_rl.utils.utils import (
    get_ckpt_dir,
    get_cuda_visible_devices,
    get_free_port,
    get_log_dir,
    get_rollout_dir,
    get_weights_dir,
)


class LogConfig(BaseSettings):
    """Configures shared logging."""

    level: Annotated[str | None, Field(description="The log level to use.")] = "info"

    file: Annotated[bool | None, Field(description="Whether to log to a file.")] = True


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


class WeightBroadcastConfig(BaseSettings):
    """Configures shared weight broadcast settings."""

    type: Annotated[Literal["nccl", "filesystem"], Field(description="The type of weight broadcast to use.")] = (
        "filesystem"
    )


class RLConfig(BaseRLLauncherConfig):
    """Configures an RL training run."""

    inference_gpu_ids: Annotated[list[int], Field(description="The GPU IDs to use for inference.")] = [0]
    trainer_gpu_ids: Annotated[list[int], Field(description="The GPU IDs to use for trainer.")] = [1]

    @model_validator(mode="after")
    def validate_device(self):
        available_gpu_ids = get_cuda_visible_devices()
        # If no CUDA devices are available (e.g., in CPU-only test environments), skip GPU validation
        if len(available_gpu_ids) == 0:
            return self
        requested_gpu_ids = sorted(set(self.trainer_gpu_ids + self.inference_gpu_ids))
        if len(requested_gpu_ids) > len(available_gpu_ids):
            raise ValueError(
                f"The number of requested GPUs ({len(requested_gpu_ids)}) exceeds available GPUs ({len(available_gpu_ids)})"
            )
        if any(not (gpu_id in available_gpu_ids) for gpu_id in requested_gpu_ids):
            raise ValueError(
                f"Some requested GPU IDs are not available. Available GPUs: {available_gpu_ids}, Requested GPUs: {requested_gpu_ids}"
            )
        if self.inference and len(self.inference_gpu_ids) != self.inference.parallel.dp * self.inference.parallel.tp:
            assert len(self.inference_gpu_ids) % self.inference.parallel.tp == 0, (
                "Number of inference GPUs must be divisible by the tensor parallel size"
            )
            self.inference.parallel.dp = len(self.inference_gpu_ids) // self.inference.parallel.tp
        return self

    @model_validator(mode="after")
    def auto_setup_num_train_workers(self):
        if len(self.trainer_gpu_ids) > 1:
            self.orchestrator.num_train_workers = len(self.trainer_gpu_ids)
        return self

    @model_validator(mode="after")
    def validate_enough_devices_for_nccl(self):
        if self.trainer.weight_broadcast.type == "nccl":
            num_gpus = len(set(self.trainer_gpu_ids + self.inference_gpu_ids))
            if num_gpus < 2:
                raise ValueError("NCCL weight broadcast requires at least 2 GPUs to build the broadcast process group.")
        return self


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
    logger = setup_logger(
        config.log.level or "info", log_file=config.output_dir / "logs" / "rl.log" if config.log.file else None
    )
    start_command = sys.argv
    logger.info("Starting RL run")
    logger.debug(f"RL start command: {' '.join(start_command)}")

    # Install any environments given in user/env-id format
    env_ids_to_install = set()

    # Collect training environment IDs
    for env_config in config.orchestrator.env:
        if "/" in env_config.id:
            env_ids_to_install.add(env_config.id)

    # Collect evaluation environment IDs
    if config.orchestrator.eval:
        for eval_env_config in config.orchestrator.eval.env:
            if "/" in eval_env_config.id:
                env_ids_to_install.add(eval_env_config.id)

    # Install each environment
    for env_id in env_ids_to_install:
        logger.info(f"Installing environment: {env_id}")
        install_cmd = ["uv", "run", "--no-sync", "prime", "env", "install", env_id]
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to install environment {env_id}: {result.stderr}")
            raise RuntimeError(f"Failed to install environment {env_id}")
        logger.info(f"Successfully installed environment: {env_id}")

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

    try:
        # Optionally, start inference process
        if config.inference:
            inference_file = get_temp_toml_file()
            with open(inference_file, "wb") as f:
                tomli_w.dump(config.inference.model_dump(exclude_none=True, mode="json"), f)

            inference_cmd = ["uv", "run", "inference", "@", inference_file.as_posix()]
            logger.info(f"Starting inference process on GPU(s) {' '.join(map(str, config.inference_gpu_ids))}")
            logger.debug(f"Inference start command: {' '.join(inference_cmd)}")
            # If we don't log stdout, the server hangs
            with open(log_dir / "inference.stdout", "w") as log_file:
                inference_process = Popen(
                    inference_cmd,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.inference_gpu_ids))},
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
        with open(log_dir / "orchestrator.stdout", "w") as log_file:
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
            "env",
            "PYTHONUNBUFFERED=1",
            "torchrun",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            f"--rdzv-id={uuid.uuid4().hex}",
            # Pipe all logs to file, and only master rank logs to stdout
            f"--log-dir={config.output_dir / 'torchrun'}",
            "--local-ranks-filter=0",
            "--redirect=3",
            "--tee=3",
            f"--nproc-per-node={len(config.trainer_gpu_ids)}",
            "-m",
            "prime_rl.trainer.rl.train",
            "@",
            trainer_file.as_posix(),
        ]
        logger.info(f"Starting trainer process on GPU(s) {' '.join(map(str, config.trainer_gpu_ids))}")
        logger.debug(f"Training start command: {' '.join(trainer_cmd)}")
        with open(log_dir / "trainer.stdout", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.trainer_gpu_ids)),
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

        tail_process = Popen(["tail", "-F", log_dir / "trainer.stdout"])
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
