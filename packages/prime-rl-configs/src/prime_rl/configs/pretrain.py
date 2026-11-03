from typing import Annotated, TypeAlias

from pydantic import Field

from prime_rl.configs.sft import SFTConfig, TokensDataConfig
from prime_rl.configs.trainer import AdamWConfig, ModelConfig, MuonConfig, SignSGDConfig


class PretrainModelConfig(ModelConfig):
    from_scratch: bool = True


class PretrainAdamWConfig(AdamWConfig):
    lr: float = Field(1e-3, ge=0)
    """Peak learning rate. Defaults to PyTorch's AdamW default."""


class PretrainMuonConfig(MuonConfig):
    lr: float = Field(1e-3, ge=0)
    """Peak learning rate. Defaults to PyTorch's Muon default."""


PretrainOptimizerConfig: TypeAlias = Annotated[
    PretrainAdamWConfig | PretrainMuonConfig | SignSGDConfig, Field(discriminator="type")
]


class PretrainConfig(SFTConfig):
    """SFT on pre-tokenized data, with the weights initialized from scratch. The defaults are SFT's, except the
    learning rate, which is PyTorch's default instead of a fine-tuning value."""

    model: PretrainModelConfig = PretrainModelConfig()

    data: TokensDataConfig

    optim: PretrainOptimizerConfig = PretrainAdamWConfig()
