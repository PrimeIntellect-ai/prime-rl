from typing import Literal

from pydantic import Field, model_validator
from renderers import AutoRendererConfig, RendererConfig

from prime_rl.configs.sft import SFTConfig, SingleNodeDeploymentConfig
from prime_rl.configs.trainer import AdamWConfig, ConstantSchedulerConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from prime_rl.utils.config import BaseConfig


class BradleyTerryDataConfig(BaseConfig):
    type: Literal["bradley_terry"] = "bradley_terry"
    name: str
    """Hugging Face dataset name or local JSON/JSONL path."""

    split: str = "train"
    batch_size: int = Field(128, ge=1)
    """Global number of Bradley-Terry pairs per optimizer step."""

    micro_batch_size: int = Field(1, ge=1)
    """Bradley-Terry pairs per device forward pass."""

    seq_len: int = Field(2048, ge=1)
    shuffle: bool = True
    seed: int = 0

    @model_validator(mode="after")
    def validate_batch_size(self):
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        return self


class RewardModelValConfig(BaseConfig):
    interval: int = Field(50, ge=1)
    eval_on_start: bool = False
    data: BradleyTerryDataConfig
    """Validation data. ``micro_batch_size`` controls memory; the full split is evaluated."""


class RewardModelConfig(SFTConfig):
    model: ModelConfig = ModelConfig(impl="hf", attn="flash_attention_2")
    renderer: RendererConfig = AutoRendererConfig()
    data: BradleyTerryDataConfig
    val: RewardModelValConfig | None = None
    optim: OptimizerConfig = AdamWConfig()
    scheduler: SchedulerConfig = ConstantSchedulerConfig()
    deployment: SingleNodeDeploymentConfig = SingleNodeDeploymentConfig()

    @model_validator(mode="after")
    def validate_reward_model(self):
        if self.slurm is not None:
            raise ValueError("Reward-model SLURM launch is not implemented; omit the slurm configuration.")
        if self.model.impl != "hf":
            raise ValueError("Reward-model training currently requires model.impl='hf'.")
        if self.model.vlm is not None:
            raise ValueError("Reward-model training currently supports text-only models.")
        if self.model.cp != 1:
            raise ValueError("Reward-model training currently requires model.cp=1.")
        if self.model.ep not in (1, "auto"):
            raise ValueError("Reward-model training currently requires model.ep=1 or 'auto'.")
        if self.ckpt is not None and self.ckpt.resume_step is not None:
            raise ValueError("Reward-model checkpoint resume is not supported yet.")
        if self.model.lora is not None:
            raise ValueError("Reward-model LoRA is not supported yet because the scalar head must remain trainable.")
        return self
