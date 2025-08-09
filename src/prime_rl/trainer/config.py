from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.utils.config import LogConfig, MultiMonitorConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings

AttnImplementation: TypeAlias = Literal["sdpa", "flash_attention_2"]


class ModelConfig(BaseConfig):
    """Configures the model for training."""

    name: Annotated[
        str,
        Field(
            description="Name or path of the HF model to use.",
        ),
    ] = "Qwen/Qwen3-0.6B"

    attn: Annotated[AttnImplementation, Field(description="The attention implementation to use.")] = "flash_attention_2"

    compile: Annotated[
        bool,
        Field(
            description="Whether to compile the model using `torch.compile`. Currently discouraged because it was found to destabilize training.",
        ),
    ] = False

    ac: Annotated[
        bool,
        Field(
            description="Whether to apply activation checkpointing to the model.",
        ),
    ] = False

    reshard_after_forward: Annotated[
        bool, Field(description="Whether to reshard the model after each forward pass.")
    ] = True


class ConstantSchedulerConfig(BaseModel):
    """Configuration for constant learning rate scheduler."""

    type: Literal["constant"] = "constant"


class LinearSchedulerConfig(BaseModel):
    """Configuration for linear learning rate scheduler."""

    type: Literal["linear"] = "linear"

    warmup_steps: Annotated[int, Field(ge=0, description="Number of warmup steps for the learning rate scheduler.")] = 0

    decay_steps: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of steps to decay the learning rate during the final portion of training. If None, will use remaining steps after warmup.",
        ),
    ] = None


class CosineSchedulerConfig(BaseModel):
    """Configuration for cosine learning rate scheduler."""

    type: Literal["cosine"] = "cosine"

    warmup_steps: Annotated[int, Field(ge=0, description="Number of warmup steps for the learning rate scheduler.")] = 0

    decay_steps: Annotated[
        int | None,
        Field(
            ge=1,
            description="Number of steps to decay the learning rate during the final portion of training. If None, will use remaining steps after warmup.",
        ),
    ] = None


SchedulerConfig: TypeAlias = ConstantSchedulerConfig | LinearSchedulerConfig | CosineSchedulerConfig


class OptimizerConfig(BaseConfig):
    """Configures the Adam optimizer and learning rate scheduler."""

    lr: Annotated[float, Field(ge=0)] = 4e-4
    weight_decay: Annotated[float, Field(ge=0)] = 0.01
    betas1: Annotated[float, Field(ge=0)] = 0.9
    betas2: Annotated[float, Field(ge=0)] = 0.999

    # LR Scheduler configuration
    scheduler: SchedulerConfig = Field(discriminator="type", default=ConstantSchedulerConfig())


class CheckpointConfig(BaseConfig):
    """Configures checkpointing the full model, optimizer and training state for resuming training."""

    interval: Annotated[int, Field(ge=1, description="Interval at which to save the checkpoint.")] = 50

    save_async: Annotated[
        bool,
        Field(
            description="Whether to save the checkpoint asynchronously.",
        ),
    ] = False

    resume_step: Annotated[
        int | None,
        Field(
            ge=1,
            description="Step to resume training from. If None, will start from scratch.",
        ),
    ] = None


class WeightCheckpointConfig(BaseConfig):
    """Configures checkpointing the model weights for updating the inference engines."""

    interval: Annotated[
        int | None,
        Field(
            description="Interval of checkpoints to save. If None, will automatically delete weight checkpoints that are more than `async_level` steps old. This is useful to keep some weight-only checkpoints for online evals.",
        ),
    ] = None

    save_async: Annotated[
        bool,
        Field(
            description="Whether to save the weights asynchronously.",
        ),
    ] = True


class BaseLossConfig(BaseModel):
    """Base config for loss."""

    max_norm: Annotated[float, Field(ge=0, description="Maximum gradient norm to clip.")] = 1.0


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


LossConfig: TypeAlias = ClippingLossConfig | RatioLossConfig


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


class TrainerConfig(BaseSettings):
    """Configures the trainer"""

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The data configuration
    data: DataLoaderConfig = DataLoaderConfig()

    # The optimizer configuration
    optim: OptimizerConfig = OptimizerConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    # The weight checkpoint configuration
    weights: WeightCheckpointConfig = WeightCheckpointConfig()

    # The loss configuration
    loss: LossConfig = Field(discriminator="type", default=RatioLossConfig())

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
    ] = True

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
        return self

    @model_validator(mode="after")
    def validate_scheduler(self):
        # Constant scheduler does not require any validation/ setup
        if self.optim.scheduler.type == "constant":
            return self

        # Must specify max_steps when using a scheduler other than `constant`
        if self.max_steps is None:
            raise ValueError("Must specify max_steps when using a scheduler other than `constant`")

        # If decay_steps is not specified, use remaining steps after warmup
        if self.optim.scheduler.decay_steps is None:
            if not (self.optim.scheduler.warmup_steps <= self.max_steps):
                raise ValueError("config.optim.scheduler.warmup_steps must be less than or equal to config.max_steps")

            self.optim.scheduler.decay_steps = self.max_steps - self.optim.scheduler.warmup_steps
            assert self.optim.scheduler.decay_steps >= 0, "config.optim.scheduler.decay_steps must be positive"

        # If decay_steps is specified, validate it
        else:
            if not (self.optim.scheduler.warmup_steps + self.optim.scheduler.decay_steps <= self.max_steps):
                raise ValueError(
                    "config.optim.scheduler.warmup_steps + config.optim.scheduler.decay_steps must be less than or equal to config.max_steps"
                )

        return self

    @model_validator(mode="after")
    def disable_logging_wandb_samples(self):
        if self.monitor.wandb and self.monitor.wandb.log_extras:
            self.monitor.wandb.log_extras.samples = False
        return self
