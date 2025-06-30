from pathlib import Path
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field

from zeroband.utils.config import ModelConfig as BaseModelConfig
from zeroband.utils.config import MultiMonitorConfig, PathConfig
from zeroband.utils.models import AttnImpl
from zeroband.utils.pydantic_config import BaseConfig, BaseSettings


class ModelConfig(BaseModelConfig):
    """Configures the model for training."""

    attn: Annotated[AttnImpl, Field(default="flash_attention_2")]

    compile: Annotated[bool, Field(default=False, description="Whether to compile the model using `torch.compile`.")]


class OptimizerConfig(BaseConfig):
    """Configures the Adam optimizer."""

    lr: Annotated[float, Field(default=4e-4, ge=0)]
    weight_decay: Annotated[float, Field(default=0.01, ge=0)]
    betas1: Annotated[float, Field(default=0.9, ge=0)]
    betas2: Annotated[float, Field(default=0.99, ge=0)]


class CkptPathConfig(PathConfig):
    """Configures a checkpoint path."""

    interval: Annotated[int, Field(default=100, ge=0, description="Interval at which to save the checkpoint.")]

    save_async: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to save the checkpoint asynchronously.",
        ),
    ]


class CheckpointConfig(PathConfig):
    """Configures checkpointing the full model, optimizer and training state for resuming training."""

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

    path: Annotated[Path, Field(default=Path("weights"), description="Path to write weights to. Will write ")]

    interval: Annotated[int, Field(default=1, ge=1, description="Interval at which to save the checkpoint.")]

    save_async: Annotated[
        bool,
        Field(
            default=True,
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

    # The GRPO variant configuration
    variant: GRPOVariantsConfig = RatioConfig()


class FakeDataLoaderConfig(BaseConfig):
    """Configures the fake data loader used for training."""

    batch_size: Annotated[int, Field(default=128)]
    micro_batch_size: Annotated[int, Field(default=128)]
    seq_len: Annotated[int, Field(default=1024)]


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

    max_async_level: Annotated[
        int,
        Field(
            default=1,
            ge=0,
            description="Maximum number of steps that inference can be ahead of training. Determines how 'off-policy' the inference engines can be. Higher values yield better throughput through async execution, but may yield lower powerofrmance. If 0, will be fully synchronous. .",
        ),
    ]

    normalize_batch_to_token_count: Annotated[bool, Field(default=True)]

    grad_norm_clip: Annotated[float, Field(default=1.0)]

    recompute_logprobs: Annotated[bool, Field(default=True)]

    ac_ckpt: Annotated[bool | int, Field(default=False)]

    reshard_after_forward: Annotated[bool, Field(default=True)]

    memory_profile: Annotated[str | None, Field(default=None)]
