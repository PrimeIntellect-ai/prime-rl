import os
import subprocess
import sys
import warnings
from itertools import chain
from pathlib import Path
from subprocess import Popen
from typing import Annotated

from loguru import logger as loguru_logger
from loguru._logger import Logger
from pydantic import Field, model_validator
from rich import print as rprint

from zeroband.inference.config import InferenceConfig
from zeroband.orchestrator.config import OrchestratorConfig
from zeroband.trainer.config import CheckpointConfig, TrainerConfig
from zeroband.utils.config import LogConfig, WandbMonitorConfig
from zeroband.utils.logger import format_message, format_time, get_logger, set_logger, setup_handlers
from zeroband.utils.pydantic_config import BaseSettings, parse_argv

TRAINER_LOGS = Path("logs/trainer.log")
ORCHESTRATOR_LOGS = Path("logs/orchestrator.log")


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

    @model_validator(mode="after")
    def auto_setup_wandb(self):
        # Automatically use same W&B project for orchestrator and trainer
        if self.orchestrator and self.trainer.monitor.wandb:
            if not self.orchestrator.monitor.wandb:
                self.orchestrator.monitor.wandb = WandbMonitorConfig()
            self.orchestrator.monitor.wandb.project = self.trainer.monitor.wandb.project

            # If group is set, use it and auto-generate run names
            if self.trainer.monitor.wandb.group:
                self.orchestrator.monitor.wandb.group = self.trainer.monitor.wandb.group

                self.trainer.monitor.wandb.name = f"{self.trainer.monitor.wandb.group}-train"
                self.orchestrator.monitor.wandb.name = f"{self.trainer.monitor.wandb.group}-orchestrator"
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


def setup_logger() -> Logger:
    if get_logger():
        raise RuntimeError("Logger already setup. Call reset_logger first.")

    # Setup the logger handlers
    log_config = LogConfig(level="info", path=None)
    format = format_time(log_config) + format_message()
    logger = setup_handlers(loguru_logger, format, log_config, rank=0)
    set_logger(logger)

    return logger


def cleanup(processes: list[Popen]):
    for process in processes:
        if process.poll() is None:  # Process is still running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def to_cli(prefix, d):
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from to_cli(path, v)
        else:
            if isinstance(v, bool):
                if v:
                    yield (f"--{path}",)
            else:
                yield f"--{path}", str(v)


def rl(config: RLConfig):
    # Setup logger
    logger = setup_logger()
    logger.info("Starting RL run")

    # Cleaning up old logs
    TRAINER_LOGS.unlink(missing_ok=True)
    ORCHESTRATOR_LOGS.unlink(missing_ok=True)

    # Start processes
    processes: list[Popen] = []
    try:
        # Start inference process
        if config.inference:
            logger.info("Starting inference process")
            inference_args = list(chain.from_iterable(to_cli("", config.inference.model_dump())))
            inference_cmd = ["uv", "run", "inference", *inference_args]
            logger.info("Starting inference process")
            rprint(f"{' '.join(inference_cmd)}")
            inference_process = subprocess.Popen(
                inference_cmd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            processes.append(inference_process)
        else:
            logger.warning(
                "No inference config specified, skipping starting inference server. Is your inference server running?"
            )

        # Start orchestrator process
        orchestrator_args = list(chain.from_iterable(to_cli("", config.orchestrator.model_dump())))
        orchestrator_cmd = [
            "uv",
            "run",
            "orchestrator",
            *orchestrator_args,
            "--log.path",
            ORCHESTRATOR_LOGS.with_suffix("").as_posix(),
        ]
        logger.info("Starting orchestrator process")
        orchestrator_process = subprocess.Popen(
            orchestrator_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(orchestrator_process)
        rprint(f"{' '.join(orchestrator_cmd)}")

        # Start training process
        train_args = list(chain.from_iterable(to_cli("", config.trainer.model_dump())))
        training_cmd = ["uv", "run", "trainer", *train_args, "--log.path", TRAINER_LOGS.with_suffix("").as_posix()]
        logger.info("Starting training process")
        rprint(f"{' '.join(training_cmd)}")
        training_process = subprocess.Popen(
            training_cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(training_process)

        # Wait for all processes to complete
        logger.info("Waiting for training to complete...")
        orchestrator_process.wait()
        training_process.wait()
        logger.info("Done!")
        cleanup(processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup(processes)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup(processes)
        raise


def main():
    rl(parse_argv(RLConfig))


if __name__ == "__main__":
    main()
