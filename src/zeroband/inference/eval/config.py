from pathlib import Path
from typing import Annotated, Literal

# Import environment before any other imports
# ruff: noqa
from zeroband.inference import envs

from pydantic import Field

from zeroband.inference.config import LogConfig, ModelConfig, SamplingConfig as InferenceSamplingConfig
from zeroband.inference.eval.registry import Benchmark
from zeroband.utils.config import MultiMonitorConfig
from zeroband.utils.pydantic_config import BaseConfig, BaseSettings


class SamplingConfig(InferenceSamplingConfig):
    """Configures sampling parameters for evaluation."""

    n: Annotated[
        int,
        Field(default=1, description="Number of completions to generate for each prompt. This is essentially the pass@k rate (where k=n)."),
    ]


class ParallelConfig(BaseConfig):
    """Configures multi-node and multi-GPU setups through different types of parallelism (TP, DP, PP)."""

    tp: Annotated[
        int | Literal["auto"],
        Field(
            default=1,
            description="Number of local GPUs to use for tensor parallelism. It is directly passed to vLLM. If 'auto', will be set to all available local GPUs.",
        ),
    ]


class OnlineEvalConfig(BaseConfig):
    """Configures online evaluation."""

    ckpt_path: Annotated[
        Path,
        Field(
            default=Path("checkpoints"),
            description="Path to read checkpoints from when doing online evaluation. Expects subdirectories named 'step_x' within the directory.",
        ),
    ]

    interval: Annotated[
        int,
        Field(
            default=100,
            ge=0,
            description="Interval at which to evaluate the model.",
        ),
    ]


class EvalConfig(BaseConfig):
    """Configures evaluation."""

    benchmarks: Annotated[
        list[Benchmark],
        Field(
            default=["math500"],
            description="Benchmarks to evaluate on. By default, it will evaluate only on the MATH-500 benchmark.",
        ),
    ]

    online: Annotated[OnlineEvalConfig | None, Field(default=None)]


class Config(BaseSettings):
    """Configures offline evaluation."""

    # The model configuration
    model: Annotated[ModelConfig, Field(default=ModelConfig())]

    # The sampling configuration
    sampling: Annotated[SamplingConfig, Field(default=SamplingConfig())]

    # The parallel configuration
    parallel: Annotated[ParallelConfig, Field(default=ParallelConfig())]

    # The monitor configuration
    monitor: Annotated[MultiMonitorConfig, Field(default=MultiMonitorConfig())]

    # The logging configuration
    log: Annotated[LogConfig, Field(default=LogConfig())]

    # The evaluation configuration
    eval: Annotated[EvalConfig, Field(default=EvalConfig())]

    use_tqdm: Annotated[bool, Field(default=False, description="Whether to use tqdm to display progress bars during model generation.")]

    seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed used across inference components. If None, no seeding is used.",
        ),
    ]
