from typing import Literal

from prime_rl.configs.supervised import ChatDataConfig, DataConfig, SupervisedTrainerConfig


class SFTConfig(SupervisedTrainerConfig):
    mode: Literal["sft"] | None = None
    """Run kind of this config. ``uv run launch`` picks the entrypoint from it; ``uv run sft`` accepts only
    "sft" (or no mode)."""

    data: DataConfig = ChatDataConfig()
