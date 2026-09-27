"""Configuration for the separate PPO value trainer."""

from pathlib import Path

from pydantic import Field, model_validator

from prime_rl.configs.shared import EnvVars, ResumeConfig, TransportConfig, ZMQTransportConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from prime_rl.utils.config import BaseConfig, default_output_dir


class ValueConfig(BaseConfig):
    model: ModelConfig = ModelConfig()
    optim: OptimizerConfig = AdamWConfig(lr=5e-6)
    scheduler: SchedulerConfig = ConstantSchedulerConfig()
    ckpt: CheckpointConfig | None = None
    resume: ResumeConfig | None = None
    rollout_transport: TransportConfig = ZMQTransportConfig()
    output_dir: Path = Field(default_factory=default_output_dir)
    rollout_dir: Path = Field(default_factory=default_output_dir)
    max_steps: int | None = None
    head_warmup_steps: int = Field(10, ge=0)
    updates_per_step: int = Field(2, ge=1)
    freeze_attention: bool = False
    pretrain_data: Path | None = None
    pretrain_steps: int = Field(0, ge=0)
    service_host: str = "127.0.0.1"
    service_port: int = Field(8123, ge=1, le=65535)
    policy_sync_interval: int | None = Field(None, ge=1)
    policy_sync_retune_updates: int = Field(2, ge=1)
    policy_sync_dir: Path | None = None
    env_vars: EnvVars = {}

    @model_validator(mode="after")
    def validate_pretraining(self):
        if self.pretrain_steps and self.pretrain_data is None:
            raise ValueError("value.pretrain_steps requires value.pretrain_data")
        return self
