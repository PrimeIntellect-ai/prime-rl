from pathlib import Path
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field, model_validator

from zeroband.utils.config import MultiMonitorConfig
from zeroband.utils.pydantic_config import BaseConfig, BaseSettings

AttnImplementation: TypeAlias = Literal["sdpa", "flash_attention_2"]


class ModelConfig(BaseConfig):
    """Configures the model for training."""

    name: Annotated[
        str,
        Field(
            default="Qwen/Qwen3-0.6B",
            description="Name or path of the HF model to use.",
        ),
    ]

    attn: Annotated[
        AttnImplementation, Field(default="flash_attention_2", description="The attention implementation to use.")
    ]

    compile: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to compile the model using `torch.compile`. Currently discouraged because it was found to destabilize training.",
        ),
    ]

    ac: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to apply activation checkpointing to the model.",
        ),
    ]

    reshard_after_forward: Annotated[
        bool, Field(default=True, alias="reshard", description="Whether to reshard the model after each forward pass.")
    ]


class OptimizerConfig(BaseConfig):
    """Configures the Adam optimizer."""

    lr: Annotated[float, Field(default=4e-4, ge=0)]
    weight_decay: Annotated[float, Field(default=0.01, ge=0)]
    betas1: Annotated[float, Field(default=0.9, ge=0)]
    betas2: Annotated[float, Field(default=0.99, ge=0)]


class CheckpointConfig(BaseConfig):
    """Configures checkpointing the full model, optimizer and training state for resuming training."""

    path: Annotated[Path, Field(default=Path("checkpoints"))]

    clean: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to clean the path at the beginning of the run. If True, will delete the entire directory.",
        ),
    ]

    interval: Annotated[int, Field(default=50, ge=1, description="Interval at which to save the checkpoint.")]

    resume_path: Annotated[
        Path | None,
        Field(
            default=None,
            description="Checkpoint path to resume training from. If None, will start from scratch.",
        ),
    ]


class WeightCheckpointConfig(BaseConfig):
    """Configures checkpointing the model weights for updating the inference engines."""

    path: Annotated[
        Path,
        Field(
            default=Path("weights"),
            description="Path to write weights to. Will write to `{path}/step_{step}` at every training step, which will be read by the orchestrator to update the inference engines.",
        ),
    ]

    interval: Annotated[
        int | None,
        Field(
            default=None,
            description="Interval of checkpoints to save. If None, will automatically delete weight checkpoints that are more than `max_async_level` steps old. This is useful to keep some weight-only checkpoints for online evals.",
        ),
    ]

    save_async: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to save the checkpoint asynchronously.",
        ),
    ]


class BaseGRPOVariantConfig(BaseConfig):
    """Base config class for GRPO variants."""

    highest_entropy_ratio_loss: Annotated[float, Field(default=1.0)]


class ClippingConfig(BaseGRPOVariantConfig):
    """Configures the clipping loss."""

    type: Annotated[Literal["clip"], Field(default="clip")]
    epsilon_low: Annotated[float, Field(default=0.2)]
    epsilon_high: Annotated[float, Field(default=0.2)]
    clip_ratio: Annotated[float, Field(default=4.0)]


class RatioConfig(BaseGRPOVariantConfig):
    """Configures the ratio loss."""

    type: Annotated[Literal["ratio"], Field(default="ratio")]
    clip_ratio: Annotated[float, Field(default=8.0)]


GRPOVariantsConfig: TypeAlias = Union[ClippingConfig, RatioConfig]


class GRPOLossConfig(BaseConfig):
    """Configures the GRPO loss."""

    variant: Annotated[GRPOVariantsConfig, Field(default=RatioConfig())]

    max_norm: Annotated[float, Field(default=1.0, ge=0, description="Maximum gradient norm to clip.")]

    normalize_to_token_count: Annotated[
        bool, Field(default=True, description="Whether to normalize the batch to token count.")
    ]


class FakeDataLoaderConfig(BaseConfig):
    """Configures a fake data loader sampling random micro batches for debugging."""

    micro_batch_size: Annotated[int, Field(default=8, ge=1)]
    batch_size: Annotated[int, Field(default=8, ge=1)]
    seq_len: Annotated[int, Field(default=128, ge=1)]

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("Batch size must be divisible by micro batch size")
        if self.batch_size < self.micro_batch_size:
            raise ValueError("Batch size must be greater than or equal to micro batch size")


class DataLoaderConfig(BaseConfig):
    """Configures the data loader used for training."""

    path: Annotated[Path, Field(default=Path("rollouts"))]

    fake: Annotated[FakeDataLoaderConfig | None, Field(default=None)]


class LogConfig(BaseConfig):
    """Configures the training logger."""

    level: Annotated[
        Literal["debug", "info"],
        Field(
            default="info",
            description="Logging level for the inference run. Will determine the logging verbosity and format.",
        ),
    ]

    all_ranks: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to log from all DP ranks. If False, will only log from the main rank (DP rank 0).",
        ),
    ]

    utc: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to use UTC time in the logger. If False, it will default to the local time. If the local time is wrong, you can set it by setting the `TZ` environment variable. For example, `TZ=America/Los_Angeles` will set the local time to SF time.",
        ),
    ]


class Config(BaseSettings):
    """Configures training"""

    # The model configuration
    model: Annotated[ModelConfig, Field(default=ModelConfig())]

    # The data configuration
    data: Annotated[DataLoaderConfig, Field(default=DataLoaderConfig())]

    # The optimizer configuration
    optim: Annotated[OptimizerConfig, Field(default=OptimizerConfig())]

    # The checkpoint configuration
    ckpt: Annotated[CheckpointConfig, Field(default=CheckpointConfig(path="checkpoints", clean=True))]

    # The weight checkpoint configuration
    weights: Annotated[WeightCheckpointConfig, Field(default=WeightCheckpointConfig())]

    # The loss configuration
    loss: Annotated[GRPOLossConfig, Field(default=GRPOLossConfig())]

    # The logging configuration
    log: LogConfig = LogConfig()

    # The monitor configuration
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    max_steps: Annotated[
        int | None,
        Field(
            default=None,
            description="Maximum number of steps to run training for. If None, will run indefinitely.",
        ),
    ]

    async_level: Annotated[
        int,
        Field(
            default=2,
            ge=0,
            description="Maximum number of steps that inference can be ahead of training. Determines how 'off-policy' the inference engines can be. Higher values yield better throughput through async execution, but may yield lower powerofrmance. If 0, will be fully synchronous.",
        ),
    ]

    profile_path: Annotated[Path | None, Field(default=None, description="Path to write memory profile to.")]

    recompute_logprobs: Annotated[
        bool,
        Field(
            default=True,
            description="Whether to recompute the logprobs. If True, will always recompute logprobs and overwrite those potentially found in the training batch.",
        ),
    ]
