from typing import Annotated

from pydantic import Field

from prime_rl.orchestrator.config import ClientConfig, EvalConfig
from prime_rl.utils.config import LogConfig, ModelConfig, MultiMonitorConfig
from prime_rl.utils.pydantic_config import BaseSettings


class OfflineEvalConfig(EvalConfig, BaseSettings):
    """Configures evaluation."""

    # The client configuration
    client: ClientConfig = ClientConfig()

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The monitor configuration
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    # The logging configuration
    log: LogConfig = LogConfig()

    use_tqdm: Annotated[
        bool,
        Field(
            description="Whether to use tqdm to display progress bars during model generation.",
        ),
    ] = False
