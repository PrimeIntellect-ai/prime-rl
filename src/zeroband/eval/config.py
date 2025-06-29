from typing import Annotated

from pydantic import Field

from zeroband.training.orchestrator.config import (
    ClientConfig,
    EvalConfig,
    LogConfig,
    SamplingConfig,
)
from zeroband.utils.config import ModelConfig, MultiMonitorConfig
from zeroband.utils.pydantic_config import BaseSettings


class Config(BaseSettings):
    """Configures evaluation."""

    # The client configuration
    client: Annotated[ClientConfig, Field(default=ClientConfig())]

    # The model configuration
    model: Annotated[ModelConfig, Field(default=ModelConfig())]

    # The sampling configuration
    sampling: Annotated[SamplingConfig, Field(default=SamplingConfig())]

    # The evaluation configuration
    eval: Annotated[EvalConfig, Field(default=EvalConfig())]

    # The monitor configuration
    monitor: Annotated[MultiMonitorConfig, Field(default=MultiMonitorConfig())]

    # The logging configuration
    log: Annotated[LogConfig, Field(default=LogConfig())]

    use_tqdm: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to use tqdm to display progress bars during model generation.",
        ),
    ]
