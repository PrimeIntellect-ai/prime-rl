from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from prime_rl.trainer.config import (
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    WeightCheckpointConfig,
)
from prime_rl.utils.config import LogConfig, MultiMonitorConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class DataConfig(BaseConfig):
    """Configures the data used for training."""

    name: Annotated[str, Field(description="Name or path of the HF dataset to use.")] = (
        "PrimeIntellect/Reverse-Text-SFT"
    )
    split: Annotated[str, Field(description="Split to use from the HF dataset.")] = "train"
    collate_mode: Annotated[Literal["padding", "packing"], Field(description="Collate mode to use.")] = "packing"
    micro_batch_size: Annotated[int, Field(ge=1)] = 8
    batch_size: Annotated[int, Field(ge=1)] = 128
    seq_len: Annotated[int, Field(ge=1)] = 128
    shuffle: Annotated[bool, Field(description="Whether to shuffle the dataset at the beginning of each epoch.")] = True

    fake: Annotated[bool, Field(description="Whether to use a fake dataset.")] = False

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self


class SFTTrainerConfig(BaseSettings):
    """Configures the SFT trainer"""

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The data configuration
    data: DataConfig = DataConfig()

    # The optimizer configuration
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()

    # The learning rate scheduler configuration
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = ConstantSchedulerConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    # The weight checkpoint configuration
    weights: WeightCheckpointConfig = WeightCheckpointConfig()

    # The logging configuration
    log: LogConfig = LogConfig()

    # The monitor configuration
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    outputs_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs")

    max_steps: Annotated[
        int | None,
        Field(description="Maximum number of steps to run training for. If None, will run indefinitely."),
    ] = None

    profile_path: Annotated[Path | None, Field(description="Path to write memory profile to.")] = None

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 5 and use fake data.",
        ),
    ] = False

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.max_steps = 4  # 1 Warmup + 3 Benchmark
            if not self.data.fake:
                self.data.fake = True
        return self

    @model_validator(mode="after")
    def validate_scheduler(self):
        # Constant scheduler does not require any validation/ setup
        if self.scheduler.type == "constant":
            return self

        # Must specify max_steps when using a scheduler other than `constant`
        if self.max_steps is None:
            raise ValueError("Must specify max_steps when using a scheduler other than `constant`")

        # If decay_steps is not specified, use remaining steps after warmup
        if self.scheduler.decay_steps is None:
            if not (self.warmup_steps <= self.max_steps):
                raise ValueError("config.scheduler.warmup_steps must be less than or equal to config.max_steps")

            self.scheduler.decay_steps = self.max_steps - self.scheduler.warmup_steps
            assert self.scheduler.decay_steps >= 0, "config.scheduler.decay_steps must be positive"

        # If decay_steps is specified, validate it
        else:
            if not (self.scheduler.warmup_steps + self.scheduler.decay_steps <= self.max_steps):
                raise ValueError(
                    "config.scheduler.warmup_steps + config.scheduler.decay_steps must be less than or equal to config.max_steps"
                )

        return self

    @model_validator(mode="after")
    def disable_logging_wandb_samples(self):
        if self.monitor.wandb and self.monitor.wandb.log_extras:
            self.monitor.wandb.log_extras.samples = False
        return self
