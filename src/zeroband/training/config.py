from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource

from zeroband.training.data import CollateMode, DataConfig
from zeroband.utils.config import BaseConfig
from zeroband.utils.models import AttnImpl
from zeroband.utils.monitor import MultiMonitorConfig

# These are two somewhat hacky workarounds inspired by https://github.com/pydantic/pydantic-settings/issues/259 to ensure backwards compatibility with our old CLI system `pydantic_config`
TOML_PATHS: list[str] = []


def set_toml_paths(toml_paths: list[str]) -> None:
    global TOML_PATHS
    TOML_PATHS = toml_paths


class AdamConfig(BaseConfig):
    """Configures the Adam optimizer."""

    type: Annotated[Literal["adam"], Field(default="adam", description="The type of optimizer to use.")]
    lr: Annotated[float, Field(default=4e-4, ge=0, description="The learning rate.")]
    weight_decay: Annotated[float, Field(default=0.01, ge=0, description="The weight decay.")]
    betas1: Annotated[float, Field(default=0.9, ge=0, description="The first beta parameter for the Adam optimizer.")]
    betas2: Annotated[float, Field(default=0.99, ge=0, description="The second beta parameter for the Adam optimizer.")]


class OptimConfig(BaseConfig):
    """Configures the optimizer."""

    # The optimizer configuration
    optim: AdamConfig = AdamConfig()

    sched_type: Annotated[Literal["cosine", "linear", "wsd-sqrt"], Field(default="linear", description="The type of scheduler to use.")]
    warmup_steps: Annotated[int, Field(default=1000, description="The number of warmup steps.")]
    stable_steps: Annotated[int, Field(default=80_000, description="The number of stable steps.")]
    total_steps: Annotated[int, Field(default=88_000, description="The total number of steps.")]
    batch_size: Annotated[int, Field(default=512, description="The batch size.")]
    grad_norm_clip: Annotated[float, Field(default=1.0, description="The gradient norm clip.")]
    step_per_rollout: Annotated[int, Field(default=1, description="The number of steps per rollout.")]


class TrainConfig(BaseConfig):
    """Configures general training parameters."""

    micro_bs: Annotated[int, Field(default=1, description="The micro batch size.")]
    ac_ckpt: Annotated[bool | int, Field(default=False, description="Whether to use AC checkpointing.")]
    reshard_after_forward: Annotated[bool, Field(default=True, description="Whether to reshard after forward.")]
    memory_profile: Annotated[str | None, Field(default=None, description="The path to the memory profile.")]
    torch_compile: Annotated[bool, Field(default=False, description="Whether to use torch compile.")]  # Disabled bc too unstable atm
    liger_qwen: Annotated[bool, Field(default=False, description="Whether to use Liger Qwen.")]
    attn_impl: Annotated[AttnImpl, Field(default="flash_attention_2", description="The attention implementation.")]


class CkptConfig(BaseConfig):
    """Configures checkpointing"""

    path: Annotated[str | None, Field(default=None, description="The path to the checkpoint.")]
    interval: Annotated[int | None, Field(default=None, description="The interval at which to save the checkpoint.")]
    resume: Annotated[str | None, Field(default=None, description="The path to the checkpoint to resume from.")]

    rollout_path: Annotated[str | None, Field(default=None, description="The path to the rollout.")]
    clean_rollout_path: Annotated[bool, Field(default=False, description="Whether to clean the rollout path.")]

    @model_validator(mode="after")
    def check_path_and_interval(self):
        if (self.path is None) != (self.interval is None):
            raise ValueError("path and interval must be either both None or both not None")
        return self


class BaseGRPOVariantConfig(BaseConfig):
    """Base config class for GRPO variants."""

    highest_entropy_ratio_loss: Annotated[float, Field(default=1.0, description="The highest entropy ratio loss.")]


class KlCovConfig(BaseGRPOVariantConfig):
    """Configures the KL-Covariance loss."""

    type: Annotated[Literal["kl_cov"], Field(default="kl_cov", description="The type of KL-Covariance loss.")]
    kl_coef: Annotated[float, Field(default=1.0, description="The KL-Covariance loss coefficient.")]
    k_percent: Annotated[float, Field(default=0.2, description="The K percent.")]


class ClippingConfig(BaseGRPOVariantConfig):
    """Configures the clipping loss."""

    type: Annotated[Literal["clip"], Field(default="clip", description="The type of clipping loss.")]
    epsilon_low: Annotated[float, Field(default=0.2, description="The epsilon low.")]
    epsilon_high: Annotated[float, Field(default=0.2, description="The epsilon high.")]
    clip_ratio: Annotated[float, Field(default=4.0, description="The clip ratio.")]


class RatioConfig(BaseGRPOVariantConfig):
    """Configures the ratio loss."""

    type: Annotated[Literal["ratio"], Field(default="ratio", description="The type of ratio loss.")]
    clip_ratio: Annotated[float, Field(default=8.0, description="The clip ratio.")]


GRPOVariantsConfig: TypeAlias = Annotated[Union[ClippingConfig, KlCovConfig, RatioConfig], Field(discriminator="type")]


class GRPOLossConfig(BaseConfig):
    """Configures the GRPO loss."""

    # The GRPO variant configuration
    off_policy: GRPOVariantsConfig = ClippingConfig()

    kl_coef: Annotated[float | None, Field(default=None, description="The KL-Covariance loss coefficient.")]
    entropy_loss_coeff: Annotated[float, Field(default=0.001, description="The entropy loss coefficient.")]


class ModelConfig(BaseConfig):
    """Configures the model to be used for training."""

    name: Annotated[str, Field(default="Qwen/Qwen3-0.6B", description="Name or path of the HF model to use.")]


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

    # The monitor configuration
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    # W&B configurations
    wandb: Annotated[bool, Field(default=True, description="Whether to enable W&B.")]
    project: Annotated[str, Field(default="prime_simple", description="The project name for W&B.")]
    wandb_run_name: Annotated[str | None, Field(default=None, description="The run name for W&B.")]

    gpus_ids: Annotated[list[int] | None, Field(default=None, description="The GPU IDs to use.")]

    temperature: Annotated[float, Field(default=0.6, ge=0, description="The temperature for the logprobs.")]

    async_level: Annotated[int, Field(default=2, ge=1, description="The amount of rollout checkpoints to keep.")]

    collate_mode: Annotated[CollateMode, Field(default="padding", description="The collate mode to use for batching.")]

    start_step: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Training step to start from.",
        ),
    ]

    start_total_samples: Annotated[
        int | None,
        Field(
            default=None,
            description="Total number of samples seen at the start of training (non-zero if resuming from a checkpoint).",
        ),
    ]

    start_rollout_step: Annotated[
        int | None,
        Field(
            default=None,
            description="Rollout step to start from.",
        ),
    ]

    stop_after_steps: Annotated[
        int | None,
        Field(
            default=None,
            description="Stop training after this number of training steps.",
        ),
    ]

    normalize_batch_to_token_count: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to normalize the batch to the token count.",
        ),
    ]

    recompute_logprobs: Annotated[
        bool,
        Field(
            default=True,
            description="Whether to recompute the logprobs. If false, it relies on the logprobs reported from the inference worker.",
        ),
    ]

    @model_validator(mode="after")
    def check_liger(self):
        if self.train.liger_qwen:
            assert "Qwen" in self.model.name, "train.liger_qwen can only be applied to Qwen2 models."
        return self

    @model_validator(mode="after")
    def check_ckpt_interval(self):
        if self.ckpt.interval is not None:
            assert self.ckpt.interval % self.optim.step_per_rollout == 0, "ckpt.interval must be divisible by train.step_per_rollout"
        return self

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_prefix="PRIME_",
        env_nested_delimiter="__",
        # By default, we do not parse CLI. To activate, set `_cli_parse_args` to true or a list of arguments at init time.
        cli_parse_args=False,
        cli_kebab_case=True,
        cli_avoid_json=True,
        cli_implicit_flags=True,
        cli_use_class_docs_for_groups=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # This is a hacky way to dynamically load TOML file paths from CLI
        # https://github.com/pydantic/pydantic-settings/issues/259
        global TOML_PATHS
        return (
            TomlConfigSettingsSource(settings_cls, toml_file=TOML_PATHS),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
