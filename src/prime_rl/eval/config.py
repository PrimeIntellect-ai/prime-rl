from pathlib import Path
from typing import Annotated

from pydantic import Field

from prime_rl.eval.registry import Benchmark
from prime_rl.orchestrator.config import (
    ClientConfig,
    LogConfig,
    SamplingConfig,
)
from prime_rl.utils.config import ModelConfig, MultiMonitorConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


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
