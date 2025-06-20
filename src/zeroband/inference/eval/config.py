from pathlib import Path
from typing import Annotated, Literal

# Import environment before any other imports
# ruff: noqa
from zeroband.inference import envs

from pydantic import Field

from zeroband.inference.config import LogConfig, ModelConfig, SamplingConfig as InferenceSamplingConfig
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


class DataConfig(BaseConfig):
    """Configures the data to be used for inference."""

    name: Annotated[
        str,
        Field(
            default="PrimeIntellect/INTELLECT-2-RL-Dataset",
            description="Name of the HF dataset to use.",
        ),
    ]

    split: Annotated[str, Field(default="train", description="Split of the dataset to use.")]


class Config(BaseSettings):
    """Configures evaluation."""

    # The model configuration
    model: Annotated[ModelConfig, Field(default=ModelConfig())]

    # The sampling configuration
    sampling: Annotated[SamplingConfig, Field(default=SamplingConfig())]

    # The data configuration
    data: Annotated[DataConfig, Field(default=DataConfig())]

    # The parallel configuration
    parallel: Annotated[ParallelConfig, Field(default=ParallelConfig())]

    # The monitor configuration
    monitor: Annotated[MultiMonitorConfig, Field(default=MultiMonitorConfig())]

    # The logging configuration
    log: Annotated[LogConfig, Field(default=LogConfig())]

    ckpt_path: Annotated[
        Path | None,
        Field(
            default=None,
            description="Path to read checkpoints from when doing online evaluation. Expects subdirectories named 'step_x' within the directory.",
        ),
    ]

    seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed used across inference components. If None, no seeding is used.",
        ),
    ]
