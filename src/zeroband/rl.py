import os
import subprocess
import sys
import warnings
from itertools import chain
from subprocess import Popen
from typing import Annotated

from pydantic import Field, model_validator

from zeroband.inference.config import InferenceConfig
from zeroband.orchestrator.config import OrchestratorConfig
from zeroband.trainer.config import CheckpointConfig, TrainerConfig
from zeroband.utils.config import WandbMonitorConfig
from zeroband.utils.pydantic_config import BaseSettings, parse_argv


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
    processes: list[Popen] = []
    try:
        print("Starting processes...")
        # Start inference process
        if config.inference:
            inference_args = list(chain.from_iterable(to_cli("", config.inference.model_dump())))
            inference_cmd = ["uv", "run", "inference", *inference_args]
            print(f"Starting inference process with command: {' '.join(inference_cmd)}")
            inference_process = subprocess.Popen(
                inference_cmd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            processes.append(inference_process)
        else:
            print("No inference process specified, skipping...")

        # Start orchestrator process
        orchestrator_args = list(chain.from_iterable(to_cli("", config.orchestrator.model_dump())))
        orchestrator_cmd = ["uv", "run", "orchestrator", *orchestrator_args]
        print(f"Starting orchestrator process with command: {' '.join(orchestrator_cmd)}")
        orchestrator_process = subprocess.Popen(
            orchestrator_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(orchestrator_process)

        # Start training process
        train_args = list(chain.from_iterable(to_cli("", config.trainer.model_dump())))
        training_cmd = ["uv", "run", "trainer", *train_args]
        print(f"Starting training process with command: {' '.join(training_cmd)}")
        training_process = subprocess.Popen(
            training_cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(training_process)

        # Wait for all processes to complete
        print("Waiting for all processes to complete...")
        orchestrator_process.wait()
        training_process.wait()
        print("Done!")
        cleanup(processes)

    except KeyboardInterrupt:
        print("\nReceived interrupt signal, terminating all processes...")
        cleanup(processes)
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred: {e}")
        cleanup(processes)
        raise


def main():
    rl(parse_argv(RLConfig))


if __name__ == "__main__":
    main()
