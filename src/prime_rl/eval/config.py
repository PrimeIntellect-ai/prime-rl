from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from prime_rl.orchestrator.config import ClientConfig, EvalConfig
from prime_rl.utils.config import LogConfig, ModelConfig, WandbMonitorConfig
from prime_rl.utils.pydantic_config import BaseSettings


class OfflineEvalConfig(EvalConfig, BaseSettings):
    """Configures evaluation."""

    # The client configuration
    client: ClientConfig = ClientConfig(timeout=36000)

    # The model configuration
    model: ModelConfig = ModelConfig()

    # The wandb configuration
    wandb: WandbMonitorConfig | None = None

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

    steps: Annotated[
        list[int] | None,
        Field(
            description="Steps to evaluate. If None, will evaluate all steps found in the weights directory. If set, will only evaluate the specified steps. If any of the specified steps are not found in the weights directory, will raise an error.",
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
    def validate_steps(self):
        if self.steps is not None and self.weights_dir is not None:
            ckpt_steps = sorted([int(step_path.name.split("_")[-1]) for step_path in self.weights_dir.glob("step_*")])
            for step in self.steps:
                if step not in ckpt_steps:
                    raise ValueError(f"Step {step} not found in weights directory {self.weights_dir}")
        return self

    @model_validator(mode="after")
    def validate_eval_base(self):
        if self.weights_dir is None and not self.eval_base:
            raise ValueError(
                "You should either evaluate the base model and/or checkpoints. Set `--eval-base` or specify a weight checkpoint directory with `--weights-dir`."
            )
        return self
