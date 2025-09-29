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
from prime_rl.utils.config import LogConfig, WandbMonitorConfig
from prime_rl.utils.pydantic_config import BaseSettings


class BaseDataConfig(BaseModel):
    """Base config for SFT data."""

    micro_batch_size: Annotated[int, Field(ge=1)] = 8
    batch_size: Annotated[int, Field(ge=1)] = 128
    seq_len: Annotated[int, Field(ge=1)] = 128
    num_examples: Annotated[
        int | None, Field(description="Number of examples to use from the dataset. If None, will use all examples.")
    ] = None
    pack_function: Literal["cat", "stack"] = "cat"

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")
        return self


class FakeDataConfig(BaseDataConfig):
    """Configures fake data used for debugging."""

    type: Literal["fake"] = "fake"

    length: Literal["fixed", "variable"] = "fixed"
    input_ids: Literal["increasing", "random"] = "increasing"


class LossMaskConfig(BaseModel):
    """Configures which message types contribute to the loss. If True, the loss_mask will be True and the message type will contribute to the loss."""

    system: Annotated[bool, Field(description="Whether system messages contribute to the loss.")] = False
    user: Annotated[bool, Field(description="Whether user messages contribute to the loss.")] = False
    assistant: Annotated[bool, Field(description="Whether assistant messages contribute to the loss.")] = True
    tool: Annotated[bool, Field(description="Whether tool messages contribute to the loss.")] = False


class SFTDataConfig(BaseDataConfig):
    """Configures the data used for training."""

    type: Literal["sft"] = "sft"

    name: Annotated[str, Field(description="Name or path of the HF dataset to use.")] = (
        "PrimeIntellect/Reverse-Text-SFT"
    )
    splits: Annotated[list[str], Field(description="Splits to use from the HF dataset.")] = ["train"]
    shuffle: Annotated[bool, Field(description="Whether to shuffle the dataset at the beginning of each epoch.")] = True
    seed: Annotated[int, Field(description="Random seed to use for shuffling the dataset. We also shuffle at the end of each epoch by adding epoch count to the seed.")] = 0

    # Configuring
    loss_mask: LossMaskConfig = LossMaskConfig()


DataConfigType: TypeAlias = FakeDataConfig | SFTDataConfig


class SFTTrainerConfig(BaseSettings):
    """Configures the SFT trainer"""

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The data configuration
    data: Annotated[DataConfigType, Field(discriminator="type")] = SFTDataConfig()

    # The optimizer configuration
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()

    # The learning rate scheduler configuration
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = ConstantSchedulerConfig()

    # The checkpoint configuration
    ckpt: CheckpointConfig | None = None

    # The weight checkpoint configuration
    weights: WeightCheckpointConfig | None = None

    # The logging configuration
    log: LogConfig = LogConfig()

    # The wandb configuration
    wandb: WandbMonitorConfig | None = None

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs")

    max_steps: Annotated[
        int | None,
        Field(description="Maximum number of steps to run training for. If None, will run indefinitely."),
    ] = None

    memory_profiler_path: Annotated[Path | None, Field(description="Path to write memory profile to.")] = None

    bench: Annotated[
        bool,
        Field(
            description="Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 5 and use fake data.",
        ),
    ] = False

    trace_path: Annotated[Path | None, Field(description="Path to write pytorch profiler trace to.")] = None

    dist_timeout_seconds: Annotated[
        int,
        Field(
            description="Timeout in seconds for torch distributed ops. Defaults to 600 seconds.",
        ),
    ] = 600

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench:
            self.max_steps = 4  # 1 Warmup + 3 Benchmark
            if self.wandb:  # Do not log extras
                self.wandb.log_extras = None
            if self.ckpt:  # Do not checkpoint
                self.ckpt = None
        return self

    @model_validator(mode="after")
    def disable_logging_wandb_samples(self):
        if self.wandb and self.wandb.log_extras:
            self.wandb.log_extras.samples = False
        return self

    @model_validator(mode="after")
    def validate_pack_function(self):
        if self.model.cp > 1 and self.data.pack_function != "stack":
            raise ValueError("Packing function must be 'stack' when CP is enabled")
        return self

    @model_validator(mode="after")
    def validate_seq_len(self):
        if self.data.pack_function == "stack":
            if self.data.seq_len % 256 != 0:
                raise ValueError("The sequence length must be divisible by 256 when using pack function stack")
        return self

    @model_validator(mode="after")
    def dont_do_massive_traces(self):
        if self.trace_path:
            if self.max_steps is None:
                raise ValueError("Must specify max_steps when tracing")
            if self.max_steps >= 10:
                raise ValueError(
                    "Tracing more than 10 steps is not recommended as your trace will be massive. Remove this line if you really want to trace more steps."
                )
        return self

    @model_validator(mode="after")
    def validate_ckpt_managers(self):
        # Ensures that we save a weight checkpoint with every full checkpoint as well
        if self.ckpt is not None:
            if self.weights is None:
                self.weights = WeightCheckpointConfig()
            # If not interval is specified, use the same interval as the full checkpoint
            if self.ckpt.interval is not None and self.weights.interval is None:
                self.weights.interval = self.ckpt.interval
            # If an interval is specified, ensure that the weight checkpoint interval is a multiple of the full checkpoint interval
            if (
                self.ckpt.interval is not None
                and self.weights.interval is not None
                and self.ckpt.interval % self.weights.interval != 0
            ):
                raise ValueError(
                    "Use a weight checkpoint interval that ensures that a weight checkpoint is saved with every full checkpoint"
                )
        return self
