from typing import Literal

from prime_rl.configs.supervised import ChatDataConfig, DataConfig, SupervisedTrainerConfig


class SFTConfig(SupervisedTrainerConfig):
    """``uv run sft``: the supervised trainer with fine-tuning defaults: ``chat`` data, the weights of ``model.name``
    and a constant learning rate of 1e-6."""

    mode: Literal["sft"] | None = None
    """Run kind of this config. ``uv run launch`` picks the entrypoint from it; ``uv run sft`` accepts only
    "sft" (or no mode)."""

    data: DataConfig = ChatDataConfig()
    """Training data. Defaults to ``type = "chat"``."""
