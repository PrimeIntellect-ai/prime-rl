from typing import Literal

from pydantic import model_validator
from pydantic_config import BaseConfig

from zeroband.training.data import CollateMode, DataConfig
from zeroband.utils.models import AttnImpl, ModelName


class AdamConfig(BaseConfig):
    type: Literal["adam"] = "adam"
    lr: float = 4e-4
    weight_decay: float = 0.01
    betas1: float = 0.9
    betas2: float = 0.99


class OptimConfig(BaseConfig):
    optim: AdamConfig = AdamConfig()
    sched_type: Literal["cosine", "linear", "wsd-sqrt"] = "cosine"
    warmup_steps: int = 1000
    stable_steps: int = 80_000
    total_steps: int = 88_000
    batch_size: int = 512
    grad_norm_clip: float = 1.0

    step_per_rollout: int = 1


class TrainConfig(BaseConfig):
    micro_bs: int = 1
    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard
    memory_profile: str | None = None
    torch_compile: bool = False  #  disabling torch compile because its too unstable for RL
    liger_qwen: bool = False

    attn_impl: AttnImpl = "flash_attention_2"


class CkptConfig(BaseConfig):
    path: str | None = None
    interval: int | None = None
    resume: str | None = None

    rollout_path: str | None = None  # if rollout path is set we saved at each step
    clean_rollout_path: bool = False  # if true, the rollout path will be cleaned up before running the training

    @model_validator(mode="after")
    def check_path_and_interval(self):
        if (self.path is None) != (self.interval is None):
            raise ValueError("path and interval must be either both None or both not None")
        return self


class Config(BaseConfig):
    model_name: ModelName

    ckpt: CkptConfig = CkptConfig()

    project: str = "prime_simple"
    wandb: bool = True

    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig

    gpus_ids: list[int] | None = None

    temperature: float = 0.6  # todo remove this and add this to the data

    grpo_epsilon_low: float = 0.2
    grpo_epsilon_high: float = 0.2
    entropy_loss_coeff: float = 0.001
    clamp_log_prob_coef: float = 4.0

    max_async_level: int = 2  # the amount of rollout checkpoints to keep

    collate_mode: CollateMode = "padding"

    kl_coef: float | None = None

    start_step: int = 0
    start_total_samples: int | None = None
    start_rollout_step: int | None = None

    @model_validator(mode="after")
    def check_liger(self):
        if self.train.liger_qwen:
            assert "Qwen" in self.model_name, "train.liger_qwen can only be applied to Qwen2 models."
        return self

    @model_validator(mode="after")
    def check_ckpt_interval(self):
        if self.ckpt.interval is not None:
            assert self.ckpt.interval % self.optim.step_per_rollout == 0, "ckpt.interval must be divisible by train.step_per_rollout"
        return self
