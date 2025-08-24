from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

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

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with artifacts such as reports and HF datasets as subdirectories. Should be set to a persistent directory with enough disk space."
        ),
    ] = Path("outputs")

    weights_dir: Annotated[
        Path | None,
        Field(
            description="Directory to load weight checkpoints (searches for `{weights_dir}/step_{x}`) generated during a training run (RL/ SFT). If set, will automatically eval all checkpoints found, including the base model. If None, will only eval the base model.",
        ),
    ] = None

    eval_base: Annotated[
        bool,
        Field(
            description="Whether to evaluate the base model. If True, will evaluate the base model before evaluating the checkpoints.",
        ),
    ] = True

    use_tqdm: Annotated[
        bool,
        Field(
            description="Whether to use tqdm to display progress bars during model generation.",
        ),
    ] = False

    @model_validator(mode="after")
    def validate_eval_base(self):
        if self.weights_dir is None and not self.eval_base:
            raise ValueError(
                "You should either evaluate the base model and/or checkpoints. Set `--eval-base` or specify a weight checkpoint directory with `--weights-dir`."
            )
        return self
