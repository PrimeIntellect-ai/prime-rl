from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import ClientConfig
from prime_rl.utils.config import LogConfig, MultiMonitorConfig, ModelConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class EvaluateConfig(BaseConfig):
    """Evaluation settings for GEPA prompt scoring."""

    benchmark: Annotated[str, Field(description="Benchmark/dataset to evaluate on.")] = "math500"
    rollouts_per_prompt: Annotated[int, Field(ge=1)] = 1
    pareto_size: Annotated[int, Field(ge=1, description="Number of instances in D_pareto.")] = 64
    feedback_pool_size: Annotated[int, Field(ge=1, description="Number of instances in D_feedback pool.")] = 256
    max_tokens: Annotated[int | None, Field(description="Max generated tokens per completion.")] = None
    min_tokens: Annotated[int, Field(ge=0)] = 0


class OperatorsConfig(BaseConfig):
    """Mutation/crossover operator rates and constraints."""

    mutation_rate: Annotated[float, Field(ge=0, le=1)] = 0.7
    crossover_rate: Annotated[float, Field(ge=0, le=1)] = 0.3
    max_prompt_chars: Annotated[int, Field(ge=1)] = 4000
    enforce_diversity: Annotated[bool, Field(description="Avoid near-duplicate prompts via distance checks.")] = True
    min_levenshtein_distance: Annotated[int, Field(ge=0)] = 40


class SelectionConfig(BaseConfig):
    """Selection policy settings."""

    strategy: Annotated[Literal["top-k", "tournament"], Field()] = "top-k"
    k: Annotated[int, Field(ge=1)] = 8
    tournament_size: Annotated[int, Field(ge=1)] = 3
    keep_elite: Annotated[int, Field(ge=0)] = 2


class GEPAConfig(BaseSettings):
    """Top-level configuration for the GEPA optimizer."""

    # Model/Client
    model: ModelConfig = ModelConfig()

    # Evaluation
    evaluate: EvaluateConfig = EvaluateConfig()

    # Evolution
    population_size: Annotated[int, Field(ge=2)] = 16
    generations: Annotated[int, Field(ge=1)] = 10
    seed: Annotated[int | None, Field(description="Random seed for reproducibility.")] = 42

    # Operators & selection
    operators: OperatorsConfig = OperatorsConfig()
    selection: SelectionConfig = SelectionConfig()

    # Budget & minibatch
    budget_rollouts: Annotated[int, Field(ge=1, description="Total rollout budget for evolution.")] = 200
    minibatch_size: Annotated[int, Field(ge=1, description="Minibatch size b for acceptance test.")] = 8

    # Logging/monitoring
    log: LogConfig = LogConfig()
    monitor: MultiMonitorConfig = MultiMonitorConfig()

    # IO
    outputs_dir: Annotated[Path, Field(description="Directory for GEPA outputs.")] = Path("outputs")
    run_name: Annotated[str | None, Field(description="Optional run name for outputs/W&B.")] = None

    # Dev
    dry_run: Annotated[bool, Field(description="If True, skip real inference and fabricate scores.")] = False

    # Inference server (optional auto-spawn) and client
    inference: InferenceConfig | None = None
    client: ClientConfig = ClientConfig()


