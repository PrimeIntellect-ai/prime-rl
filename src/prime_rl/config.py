from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from prime_rl.inference.config import InferenceConfig
from prime_rl.inference.config import WeightBroadcastConfig as InferenceWeightBroadcastConfig
from prime_rl.orchestrator.config import CheckpointConfig as OrchestratorCheckpointConfig
from prime_rl.orchestrator.config import FileSystemWeightBroadcastConfig as OrchestratorFileSystemWeightBroadcastConfig
from prime_rl.orchestrator.config import NCCLWeightBroadcastConfig as OrchestratorNCCLWeightBroadcastConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.trainer.config import CheckpointConfig as TrainerCheckpointConfig
from prime_rl.trainer.rl.config import FakeDataLoaderConfig
from prime_rl.trainer.rl.config import FileSystemWeightBroadcastConfig as TrainerFileSystemWeightBroadcastConfig
from prime_rl.trainer.rl.config import NCCLWeightBroadcastConfig as TrainerNCCLWeightBroadcastConfig
from prime_rl.trainer.rl.config import RLTrainerConfig as TrainerConfig
from prime_rl.utils.config import WandbMonitorConfig
from prime_rl.utils.pydantic_config import BaseSettings
from prime_rl.utils.validation import (
    validate_shared_async_level,
    validate_shared_ckpt_config,
    validate_shared_max_steps,
    validate_shared_model_name,
    validate_shared_output_dir,
    validate_shared_wandb_config,
    validate_shared_weight_broadcast,
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


class BaseRLLauncherConfig(BaseSettings):
    """Configures a base launcher config."""

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

    max_async_level: Annotated[
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

    weight_broadcast: Annotated[WeightBroadcastConfig | None, Field(description="The weight broadcast config.")] = None

    @model_validator(mode="after")
    def auto_setup_logs(self):
        # Copy log level
        if self.log is not None:
            if self.log.level is not None:
                self.trainer.log.level = self.log.level
                self.orchestrator.log.level = self.log.level
            if self.log.file is not None:
                self.trainer.log.file = self.log.file
                self.orchestrator.log.file = self.log.file

        return self

    ### Setup and validate shared configs

    @model_validator(mode="after")
    def auto_setup_ckpt(self):
        # If specified, automatically setup checkpoint configs for trainer and orchestrator
        if self.ckpt is not None:
            # Create checkpoint configs if not specified
            if self.trainer.ckpt is None:
                self.trainer.ckpt = TrainerCheckpointConfig()
            if self.orchestrator.ckpt is None:
                self.orchestrator.ckpt = OrchestratorCheckpointConfig()

            # If specified, use the same ckpt interval
            if self.ckpt.interval is not None:
                self.trainer.ckpt.interval = self.ckpt.interval
                self.orchestrator.ckpt.interval = self.ckpt.interval

            # If resuming training, ensure orchestrator resume from the same step
            if self.ckpt.resume_step is not None:
                self.trainer.ckpt.resume_step = self.ckpt.resume_step
                self.orchestrator.ckpt.resume_step = self.ckpt.resume_step

            # If specified, propagate keep policy
            if self.ckpt.keep is not None:
                self.trainer.ckpt.keep = self.ckpt.keep
                self.orchestrator.ckpt.keep = self.ckpt.keep

        validate_shared_ckpt_config(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_wandb(self):
        # If specified, automatically use shared W&B project for orchestrator and trainer
        if self.wandb is not None:
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
        if self.model is not None:
            self.trainer.model.name = self.model.name
            self.orchestrator.model.name = self.model.name
            if self.inference is not None:
                self.inference.model.name = self.model.name

        validate_shared_model_name(self.trainer, self.orchestrator, self.inference)

        return self

    @model_validator(mode="after")
    def auto_setup_max_steps(self):
        # If specified, use the same max steps for trainer and orchestrator
        if self.max_steps is not None:
            self.trainer.max_steps = self.max_steps
            self.orchestrator.max_steps = self.max_steps

        validate_shared_max_steps(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_async_level(self):
        # If specified, use the same async level for trainer and orchestrator
        if self.max_async_level is not None:
            self.trainer.max_async_level = self.max_async_level
            self.orchestrator.max_async_level = self.max_async_level

        validate_shared_async_level(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_output_dir(self):
        # If specified, use the same outputs directory for trainer and orchestrator
        if self.output_dir is not None:
            self.trainer.output_dir = self.output_dir
            self.orchestrator.output_dir = self.output_dir

        validate_shared_output_dir(self.trainer, self.orchestrator)

        return self

    @model_validator(mode="after")
    def auto_setup_weight_broadcast(self):
        if self.weight_broadcast is not None:
            if self.weight_broadcast.type == "nccl":
                inference_world_size = self.inference.parallel.dp * self.inference.parallel.tp if self.inference else 1
                self.trainer.weight_broadcast = TrainerNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type, inference_world_size=inference_world_size
                )
                self.orchestrator.weight_broadcast = OrchestratorNCCLWeightBroadcastConfig(
                    type=self.weight_broadcast.type
                )
            elif self.weight_broadcast.type == "filesystem":
                self.trainer.weight_broadcast = TrainerFileSystemWeightBroadcastConfig()
                self.orchestrator.weight_broadcast = OrchestratorFileSystemWeightBroadcastConfig()
            if self.inference is not None:
                self.inference.weight_broadcast = InferenceWeightBroadcastConfig(type=self.weight_broadcast.type)

        validate_shared_weight_broadcast(self.trainer, self.orchestrator, self.inference)

        return self
