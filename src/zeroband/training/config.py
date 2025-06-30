from pathlib import Path
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field, model_validator

from zeroband.utils.config import ModelConfig, MultiMonitorConfig, PathConfig
from zeroband.utils.models import AttnImpl
from zeroband.utils.pydantic_config import BaseConfig, BaseSettings


class OptimConfig(BaseConfig):
    """Configures the optimizer."""

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


class CkptConfig(BaseConfig):
    """Configures checkpointing the full model for resuming training."""

    path: Annotated[
        CkptPathConfig,
        Field(
            default=CkptPathConfig(path="checkpoints", interval=50, save_async=True),
            description="Path to write checkpoints to.",
        ),
    ]

    resume_path: Annotated[
        Path | None,
        Field(
            default=None,
            description="Checkpoint path to resume training from. If None, will start from scratch.",
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


CollateMode: TypeAlias = Literal["packing", "padding", "balancing"]


class DataConfig(BaseConfig):
    path: Annotated[Path, Field(default=Path("rollouts"))]

    fake: Annotated[bool, Field(default=False)]


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
    data: Annotated[DataConfig, Field(default=DataConfig())]

    # The optimizer configuration
    optim: Annotated[OptimConfig, Field(default=OptimConfig())]

    # The checkpoint configuration
    ckpt: Annotated[CkptConfig | None, Field(default=None)]

    # The loss configuration
    loss: Annotated[GRPOLossConfig, Field(default=GRPOLossConfig())]

    # The logging configuration
    log: LogConfig = LogConfig()

    # The monitor configuration
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    weights: Annotated[
        CkptPathConfig,
        Field(
            default=CkptPathConfig(path="weights", interval=1, save_async=True),
            description="Configures the path to write updated model weights to. Will be read by the orchestrator to notify the inference engines.",
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

    start_step: Annotated[int, Field(default=0, ge=0, description="Step to start training from.")]

    start_total_samples: Annotated[int | None, Field(default=None)]

    stop_after_steps: Annotated[int | None, Field(default=None)]

    normalize_batch_to_token_count: Annotated[bool, Field(default=True)]

    batch_size: Annotated[int, Field(default=512)]

    grad_norm_clip: Annotated[float, Field(default=1.0)]

    recompute_logprobs: Annotated[bool, Field(default=True)]

    micro_bs: Annotated[int, Field(default=1)]

    ac_ckpt: Annotated[bool | int, Field(default=False)]

    reshard_after_forward: Annotated[bool, Field(default=True)]

    memory_profile: Annotated[str | None, Field(default=None)]

    torch_compile: Annotated[bool, Field(default=False)]  # Disabled bc too unstable atm

    liger_qwen: Annotated[bool, Field(default=False)]

    attn_impl: Annotated[AttnImpl, Field(default="flash_attention_2")]

    @model_validator(mode="after")
    def check_liger(self):
        if self.train.liger_qwen:
            assert "Qwen" in self.model.name, "train.liger_qwen can only be applied to Qwen2 models."
        return self
