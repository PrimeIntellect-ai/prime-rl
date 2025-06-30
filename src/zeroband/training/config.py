from pathlib import Path
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field, model_validator

from zeroband.utils.config import MultiMonitorConfig
from zeroband.utils.models import AttnImpl
from zeroband.utils.pydantic_config import BaseConfig, BaseSettings


class AdamConfig(BaseConfig):
    """Configures the Adam optimizer."""

    type: Annotated[Literal["adam"], Field(default="adam")]
    lr: Annotated[float, Field(default=4e-4, ge=0)]
    weight_decay: Annotated[float, Field(default=0.01, ge=0)]
    betas1: Annotated[float, Field(default=0.9, ge=0)]
    betas2: Annotated[float, Field(default=0.99, ge=0)]


class OptimConfig(BaseConfig):
    """Configures the optimizer."""

    # The optimizer configuration
    optim: AdamConfig = AdamConfig()

    batch_size: Annotated[int, Field(default=512)]
    grad_norm_clip: Annotated[float, Field(default=1.0)]


class TrainConfig(BaseConfig):
    """Configures general training parameters."""

    micro_bs: Annotated[int, Field(default=1)]
    ac_ckpt: Annotated[bool | int, Field(default=False)]
    reshard_after_forward: Annotated[bool, Field(default=True)]
    memory_profile: Annotated[str | None, Field(default=None)]
    torch_compile: Annotated[bool, Field(default=False)]  # Disabled bc too unstable atm
    liger_qwen: Annotated[bool, Field(default=False)]
    attn_impl: Annotated[AttnImpl, Field(default="flash_attention_2")]


class CkptConfig(BaseConfig):
    """Configures checkpointing"""

    path: Annotated[str | None, Field(default=None)]
    interval: Annotated[int | None, Field(default=None)]
    interval_rollout: Annotated[int | None, Field(default=None)]
    resume: Annotated[str | None, Field(default=None)]

    rollout_path: Annotated[str | None, Field(default=None)]
    clean_rollout_path: Annotated[bool, Field(default=False)]
    async_save: Annotated[
        bool,
        Field(
            default=False,
            description="Enable async checkpointing. Checkpoint will first be move from GPU to CPU and then write to disk in a async way.",
        ),
    ]

    @model_validator(mode="after")
    def check_path_and_interval(self):
        if (self.path is None) != (self.interval is None):
            raise ValueError("path and interval must be either both None or both not None")
        return self


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
    off_policy: GRPOVariantsConfig = RatioConfig()

    kl_coef: Annotated[float | None, Field(default=None)]


class ModelConfig(BaseConfig):
    """Configures the model to be used for training."""

    name: Annotated[
        str,
        Field(
            default="Qwen/Qwen3-0.6B",
            description="Name or path of the HF model to use.",
        ),
    ]


CollateMode: TypeAlias = Literal["packing", "padding", "balancing"]


class DataConfig(BaseConfig):
    path: Annotated[Path, Field(default=Path("rollouts"))]
    seq_length: Annotated[int, Field(default=1024)]
    fake: Annotated[bool, Field(default=False)]


class PathConfig(BaseConfig):
    """Configures a path used for input/ output operations"""

    path: Annotated[Path, Field(description="Path to write to.")]

    clean: Annotated[
        bool,
        Field(
            default=True,
            description="Whether to clean the at the beginning of the run.",
        ),
    ]


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
    model: ModelConfig = ModelConfig()

    # The training configuration
    train: TrainConfig

    # The optimizer configuration
    optim: OptimConfig = OptimConfig()

    # The checkpoint configuration
    ckpt: CkptConfig = CkptConfig()

    # The data configuration
    data: DataConfig = DataConfig()

    # The GRPO loss configuration
    grpo: GRPOLossConfig = GRPOLossConfig()

    # The logging configuration
    log: LogConfig = LogConfig()

    # The monitor configuration
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    gpus_ids: Annotated[list[int] | None, Field(default=None)]

    max_async_level: Annotated[int, Field(default=2, ge=1)]

    collate_mode: Annotated[CollateMode, Field(default="padding")]

    start_step: Annotated[int, Field(default=0, ge=0, description="Step to start training from.")]

    start_total_samples: Annotated[int | None, Field(default=None)]

    stop_after_steps: Annotated[int | None, Field(default=None)]

    normalize_batch_to_token_count: Annotated[bool, Field(default=True)]

    recompute_logprobs: Annotated[bool, Field(default=True)]

    @model_validator(mode="after")
    def check_liger(self):
        if self.train.liger_qwen:
            assert "Qwen" in self.model.name, "train.liger_qwen can only be applied to Qwen2 models."
        return self
