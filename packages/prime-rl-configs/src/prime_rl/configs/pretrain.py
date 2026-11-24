from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.supervised import SupervisedTrainerConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    ConstantSchedulerConfig,
    CosineSchedulerConfig,
    InitConfig,
    LinearSchedulerConfig,
    ModelConfig,
    MuonConfig,
    ScratchInit,
    SignSGDConfig,
)


class PretrainAdamWConfig(AdamWConfig):
    lr: float = Field(1e-3, ge=0)
    """Peak learning rate. Defaults to PyTorch's AdamW default."""


class PretrainMuonConfig(MuonConfig):
    lr: float = Field(1e-3, ge=0)
    """Peak learning rate. Defaults to 1e-3, as for AdamW."""


PretrainOptimizerConfig: TypeAlias = Annotated[
    PretrainAdamWConfig | PretrainMuonConfig | SignSGDConfig, Field(discriminator="type")
]


class PretrainLinearSchedulerConfig(LinearSchedulerConfig):
    decay_steps: int = Field(0, ge=0)
    """Steps to decay the learning rate during the final portion of training. Defaults to 0: a decay needs
    ``max_steps``."""


PretrainSchedulerConfig: TypeAlias = Annotated[
    ConstantSchedulerConfig | PretrainLinearSchedulerConfig | CosineSchedulerConfig, Field(discriminator="type")
]


class PretrainModelConfig(ModelConfig):
    init: InitConfig = ScratchInit()


class PretrainConfig(SupervisedTrainerConfig):
    """Pretraining: next-token loss on tokenized shards, with the weights initialized from scratch by default."""

    model: PretrainModelConfig = PretrainModelConfig()

    mode: Literal["pretrain"] | None = None
    """Run kind of this config. ``uv run launch`` picks the entrypoint from it; ``uv run pretrain`` accepts only
    "pretrain" (or no mode)."""

    optim: PretrainOptimizerConfig = PretrainAdamWConfig()

    scheduler: PretrainSchedulerConfig = PretrainLinearSchedulerConfig()
    """Learning rate schedule. Defaults to a linear warmup over 10 steps, then constant: from a random init, Adam's
    first updates at the full learning rate are unstable."""

    @model_validator(mode="before")
    @classmethod
    def default_data_type(cls, values: Any) -> Any:
        if isinstance(values, dict) and isinstance(values.get("data"), dict):
            values["data"].setdefault("type", "documents")
        return values
