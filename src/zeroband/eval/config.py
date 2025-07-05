from pathlib import Path
from typing import Annotated

from pydantic import Field

from zeroband.training.orchestrator.config import (
    ClientConfig,
    LogConfig,
    SamplingConfig,
)
from zeroband.training.orchestrator.config import (
    EvalConfig as OrchestratorEvalConfig,
)
from zeroband.utils.config import ModelConfig, MultiMonitorConfig
from zeroband.utils.pydantic_config import BaseSettings


class EvalConfig(BaseSettings):
    """Configures evaluation."""

    # The client configuration
    client: ClientConfig = ClientConfig()

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The sampling configuration
    sampling: SamplingConfig = SamplingConfig()

    # The evaluation configuration
    eval: OrchestratorEvalConfig = OrchestratorEvalConfig()

    # The monitor configuration
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    # The logging configuration
    log: LogConfig = LogConfig(path=Path("logs/eval"))

    use_tqdm: Annotated[
        bool,
        Field(
            description="Whether to use tqdm to display progress bars during model generation.",
        ),
    ] = False
