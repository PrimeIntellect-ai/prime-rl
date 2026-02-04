from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from prime_rl.utils.config import ClientConfig, LogConfig, ModelConfig, WandbWithExtrasConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class EvalSamplingConfig(BaseConfig):
    """Sampling configuration for evaluation."""

    temperature: Annotated[
        float | None,
        Field(description="Temperature for sampling. If None, uses server default."),
    ] = None

    max_tokens: Annotated[
        int | None,
        Field(description="Maximum tokens to generate. If None, uses server default."),
    ] = None

    top_p: Annotated[
        float | None,
        Field(description="Top-p sampling. If None, uses server default."),
    ] = None

    top_k: Annotated[
        int | None,
        Field(description="Top-k sampling. If None, uses server default."),
    ] = None

    min_p: Annotated[
        float | None,
        Field(description="Min-p sampling. If None, uses server default."),
    ] = None

    repetition_penalty: Annotated[
        float | None,
        Field(description="Repetition penalty. If None, uses server default."),
    ] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class EvalEnvConfig(BaseConfig):
    """Configuration for a single evaluation environment."""

    env_id: Annotated[str, Field(description="Environment ID (e.g., 'gsm8k', 'math-python').")] = "reverse-text"

    env_args: Annotated[
        dict[str, Any],
        Field(description="Arguments to pass to the environment constructor."),
    ] = {}

    name: Annotated[
        str | None,
        Field(description="Display name for the environment. If None, uses env_id."),
    ] = None

    num_examples: Annotated[
        int | None,
        Field(description="Number of examples to evaluate. If None, uses global default."),
    ] = None

    rollouts_per_example: Annotated[
        int | None,
        Field(description="Number of rollouts per example. If None, uses global default."),
    ] = None


class WatcherConfig(BaseConfig):
    """Configuration for checkpoint watching mode."""

    enabled: Annotated[
        bool,
        Field(description="Whether to enable watcher mode."),
    ] = False

    weights_dir: Annotated[
        Path | None,
        Field(description="Directory to watch for checkpoints. Required if enabled."),
    ] = None

    poll_interval: Annotated[
        int,
        Field(ge=1, description="Seconds between checkpoint scans."),
    ] = 10


class OfflineEvalConfig(BaseSettings):
    """Configuration for offline evaluation with verifiers."""

    # Model configuration
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Client configuration (supports multiple URLs for load balancing)
    client: ClientConfig = Field(default_factory=ClientConfig)

    # Environment configurations
    env: Annotated[
        list[EvalEnvConfig],
        Field(description="List of environments to evaluate."),
    ] = [EvalEnvConfig()]

    # Sampling configuration
    sampling: EvalSamplingConfig = Field(default_factory=EvalSamplingConfig)

    # Evaluation parameters
    num_examples: Annotated[
        int,
        Field(description="Default number of examples per environment. -1 for all."),
    ] = -1

    rollouts_per_example: Annotated[
        int,
        Field(ge=1, description="Default number of rollouts per example."),
    ] = 1

    max_concurrent: Annotated[
        int,
        Field(description="Maximum concurrent requests. -1 for unlimited."),
    ] = 32

    # Watcher configuration
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)

    # Logging configuration
    wandb: Annotated[
        WandbWithExtrasConfig | None,
        Field(description="Weights & Biases configuration. If None, wandb is disabled."),
    ] = None

    log: LogConfig = Field(default_factory=LogConfig)

    # Output configuration
    output_dir: Annotated[
        Path,
        Field(description="Directory for output files."),
    ] = Path("outputs")
