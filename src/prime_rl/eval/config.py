from pathlib import Path
from typing import Annotated

from pydantic import Field

from prime_rl.eval.registry import Benchmark
from prime_rl.orchestrator.config import (
    ClientConfig,
    LogConfig,
)
from prime_rl.utils.config import ModelConfig, MultiMonitorConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class SamplingConfig(BaseConfig):
    """Configures how tokens are sampled from the model. Largely follows the vLLM sampling parameters."""

    temperature: Annotated[
        float,
        Field(
            ge=0,
            description="Scales the output probability distribution. Lower values => more deterministic, higher values => more random. If 0, will sample greedily.",
        ),
    ] = 1.0

    top_p: Annotated[
        float,
        Field(
            gt=0,
            le=1,
            description="Cumulative probability of the top tokens to consider. If 1, all tokens are considered.",
        ),
    ] = 1

    top_k: Annotated[
        int,
        Field(
            ge=-1,
            description="Number of top tokens to consider. If -1, all tokens are considered.",
        ),
    ] = -1

    min_p: Annotated[
        float,
        Field(
            ge=0,
            description="Minimum probability for a token to be considered, relative to the probability of the most likely token. If 0, all tokens are considered.",
        ),
    ] = 0.0

    max_tokens: Annotated[
        int | None,
        Field(
            description="Maximum number of output tokens to generate per turn. If None, will generate until maximum context length or EOS token is hit.",
        ),
    ] = None

    min_tokens: Annotated[
        int,
        Field(
            ge=0,
            description="Minimum number of output tokens to generate per sequence.",
        ),
    ] = 0

    seed: Annotated[
        int | None,
        Field(
            description="Random seed to use for sampling. If None, no seeding is used.",
        ),
    ] = None


class OnlineEvalConfig(BaseConfig):
    """Configures online evaluation."""

    ckpt_path: Annotated[
        Path,
        Field(
            description="Path to read checkpoints from when doing online evaluation. Expects subdirectories named 'step_x' within the directory.",
        ),
    ] = Path("checkpoints")

    interval: Annotated[
        int,
        Field(
            ge=0,
            description="Interval at which to evaluate the model.",
        ),
    ] = 100

    max_steps: Annotated[
        int | None,
        Field(
            description="Maximum number of steps to run online evaluation for. If None, will run indefinitely.",
        ),
    ] = None


class EvalConfig(BaseSettings):
    """Configures evaluation."""

    # The client configuration
    client: ClientConfig = ClientConfig()

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The sampling configuration
    sampling: SamplingConfig = SamplingConfig()

    benchmarks: Annotated[
        list[Benchmark],
        Field(
            description="Benchmarks to evaluate on. By default, it will evaluate only on the MATH-500 benchmark.",
        ),
    ] = ["math500"]

    rollouts_per_prompt: Annotated[
        list[int],
        Field(
            description="Number of samples to generate for each benchmark.",
        ),
    ] = [1]

    online: Annotated[OnlineEvalConfig | None, Field(description="Whether to do online evaluation.")] = None

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
