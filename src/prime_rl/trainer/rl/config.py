from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

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


class BaseLossConfig(BaseModel):
    """Base config for loss."""


class ClippingLossConfig(BaseLossConfig):
    """Configures the clipping loss."""

    type: Literal["clip"] = "clip"
    epsilon_low: Annotated[float, Field(ge=0)] = 0.2
    epsilon_high: Annotated[float, Field(ge=0)] = 0.2
    clip_ratio: Annotated[float, Field(ge=0)] = 4.0


class RatioLossConfig(BaseLossConfig):
    """Configures the ratio loss."""

    type: Literal["ratio"] = "ratio"
    clip_ratio: Annotated[float, Field(ge=0)] = 8.0


LossConfigType: TypeAlias = ClippingLossConfig | RatioLossConfig


class FakeDataLoaderConfig(BaseConfig):
    """Configures a fake data loader sampling random micro batches for debugging."""

    micro_batch_size: Annotated[int, Field(ge=1)] = 8
    batch_size: Annotated[int, Field(ge=1)] = 8
    seq_len: Annotated[int, Field(ge=1)] = 128

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self


class DataLoaderConfig(BaseConfig):
    """Configures the data loader used for training."""

    fake: Annotated[FakeDataLoaderConfig | None, Field(description="Whether to use a fake data loader.")] = None


class RLTrainerConfig(BaseSettings):
    """Configures the RL trainer"""

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The data configuration
    data: DataLoaderConfig = DataLoaderConfig()

    # The loss configuration
    loss: Annotated[LossConfigType, Field(discriminator="type")] = RatioLossConfig()

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
            description="Directory to write outputs to. Will be populated with checkpoints, weights, rollouts and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs")

    max_steps: Annotated[
        int | None,
        Field(
            description="Maximum number of steps to run training for. If None, will run indefinitely.",
        ),
    ] = None

    async_level: Annotated[
        int,
        Field(
            ge=0,
            description="Maximum number of steps that inference can be ahead of training. Determines how 'off-policy' the inference engines can be. Higher values yield better throughput through async execution, but may yield lower powerofrmance. If 0, will be fully synchronous.",
        ),
    ] = 2

    profile_path: Annotated[Path | None, Field(description="Path to write memory profile to.")] = None

    recompute_logprobs: Annotated[
        bool,
        Field(
            description="Whether to recompute the logprobs. If True, will always recompute logprobs and overwrite those found in the training batch.",
        ),
    ] = False

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
                self.data.fake = FakeDataLoaderConfig()
            if self.monitor.wandb:  # Do not log extras
                self.monitor.wandb.log_extras = None
            if self.ckpt:  # Do not checkpoint
                self.ckpt = None
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
            if not (self.scheduler.warmup_steps <= self.max_steps):
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
