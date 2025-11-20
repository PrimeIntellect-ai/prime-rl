from pathlib import Path
from typing import Annotated

from pydantic import Field

from prime_rl.orchestrator.config import EvalConfig
from prime_rl.utils.config import ClientConfig, LogConfig, ModelConfig
from prime_rl.utils.pydantic_config import BaseSettings


class SynthesizeConfig(EvalConfig, BaseSettings):
    """Configures synthetic data generation."""

    # The client configuration
    client: ClientConfig = ClientConfig(timeout=36000)

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The logging configuration
    log: LogConfig = LogConfig()

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with artifacts such as reports and HF datasets as subdirectories. Should be set to a persistent directory with enough disk space."
        ),
    ] = Path("outputs")

    max_concurrent: Annotated[
        int | None,
        Field(
            description="Maximum number of concurrent rollouts to generate and score. Will create a global semaphore and pass to verifiers Environment. If None, will not limit concurrency.",
        ),
    ] = None
