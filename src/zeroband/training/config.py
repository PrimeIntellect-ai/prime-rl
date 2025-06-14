from typing import Annotated, Literal, TypeAlias, Union

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource

from zeroband.utils.config import BaseConfig, MultiMonitorConfig
from zeroband.utils.logger import get_logger
from zeroband.utils.models import AttnImpl

# These are two somewhat hacky workarounds inspired by https://github.com/pydantic/pydantic-settings/issues/259 to ensure backwards compatibility with our old CLI system `pydantic_config`
TOML_PATHS: list[str] = []


def set_toml_paths(toml_paths: list[str]) -> None:
    global TOML_PATHS
    TOML_PATHS = toml_paths


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
    step_per_rollout: Annotated[int, Field(default=1)]

    @model_validator(mode="after")
    def warn_step_per_rollout(self):
        if self.step_per_rollout > 1:
            get_logger("TRAIN").info(
                f"step_per_rollout is set to {self.step_per_rollout}. The recommended value is 1, any other value should be either to run a legacy run or a experiment.."
            )
        return self


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
    resume: Annotated[str | None, Field(default=None)]

    rollout_path: Annotated[str | None, Field(default=None)]
    clean_rollout_path: Annotated[bool, Field(default=False)]

    @model_validator(mode="after")
    def check_path_and_interval(self):
        if (self.path is None) != (self.interval is None):
            raise ValueError("path and interval must be either both None or both not None")
        return self


class BaseGRPOVariantConfig(BaseConfig):
    """Base config class for GRPO variants."""

    highest_entropy_ratio_loss: Annotated[float, Field(default=1.0)]


class KlCovConfig(BaseGRPOVariantConfig):
    """Configures the KL-Covariance loss."""

    type: Annotated[Literal["kl_cov"], Field(default="kl_cov")]
    kl_coef: Annotated[float, Field(default=1.0)]
    k_percent: Annotated[float, Field(default=0.2)]


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


GRPOVariantsConfig: TypeAlias = Annotated[Union[ClippingConfig, KlCovConfig, RatioConfig], Field(discriminator="type")]


class GRPOLossConfig(BaseConfig):
    """Configures the GRPO loss."""

    # The GRPO variant configuration
    off_policy: GRPOVariantsConfig = ClippingConfig()

    kl_coef: Annotated[float | None, Field(default=None)]
    entropy_loss_coeff: Annotated[float, Field(default=0)]


class ModelConfig(BaseConfig):
    """Configures the model to be used for training."""

    name: Annotated[str, Field(default="Qwen/Qwen3-0.6B", description="Name or path of the HF model to use.")]


CollateMode: TypeAlias = Literal["packing", "padding", "balancing"]


class DataConfig(BaseConfig):
    path: Annotated[str, Field(default="datasets/fineweb-edu")]
    seq_length: Annotated[int, Field(default=1024)]
    fake: Annotated[bool, Field(default=False)]
    num_workers: Annotated[int, Field(default=1)]
    timeout: Annotated[float, Field(default=3600)]

    local_dir: Annotated[str, Field(default="/dev/shm/zeroband/data")]  # only used if path is gcp

    ignore_zero_advantages: Annotated[bool, Field(default=False)]  # don't use in local setup


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
    wandb: Annotated[bool, Field(default=True)]

    project: Annotated[str, Field(default="prime_simple")]

    wandb_run_name: Annotated[str | None, Field(default=None)]

    gpus_ids: Annotated[list[int] | None, Field(default=None)]

    temperature: Annotated[float, Field(default=0.6, ge=0)]

    async_level: Annotated[int, Field(default=2, ge=1)]

    collate_mode: Annotated[CollateMode, Field(default="padding")]

    start_step: Annotated[int, Field(default=0, ge=0)]

    start_total_samples: Annotated[int | None, Field(default=None)]

    start_rollout_step: Annotated[int | None, Field(default=None)]

    stop_after_steps: Annotated[int | None, Field(default=None)]

    normalize_batch_to_token_count: Annotated[bool, Field(default=True)]

    recompute_logprobs: Annotated[bool, Field(default=True)]

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
