from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_config import BaseConfig

from prime_rl.orchestrator.config import EvalConfig
from prime_rl.utils.config import ClientConfig, LogConfig, ModelConfig


class SynthesizeConfig(EvalConfig, BaseConfig):
    """Configures synthetic data generation."""

    # The client configuration
    client: ClientConfig = ClientConfig(
        timeout=36000, base_url=["https://api.openai.com/v1"], api_key_var="OPENAI_API_KEY"
    )

    # The model configuration
    model: ModelConfig = ModelConfig(name="gpt-4.1-mini")

    # The logging configuration
    log: LogConfig = LogConfig()

    reasoning_field: Annotated[
        str,
        Field(
            description="The field in the raw model response that contains the reasoning content. Defaults to 'reasoning_content', which is the default for vLLM when serving a model with a reasoning parser. Other APIs (e.g. DeepSeek, GLM, etc.) may use different field names.",
        ),
    ] = "reasoning_content"

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
