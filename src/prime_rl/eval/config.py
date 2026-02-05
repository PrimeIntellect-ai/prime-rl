from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from prime_rl.orchestrator.config import EvalConfig
from prime_rl.utils.config import ClientConfig, LogConfig, ModelConfig, WandbConfig
from prime_rl.utils.pydantic_config import BaseSettings


class OfflineEvalConfig(EvalConfig, BaseSettings):
    """Configures evaluation."""

    # The client configuration
    client: ClientConfig = ClientConfig(timeout=36000, base_url=["https://api.openai.com/v1"])

    # The model configuration
    model: ModelConfig = ModelConfig(name="gpt-4.1-mini")

    # The wandb configuration
    wandb: WandbConfig | None = None

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

    max_concurrent: Annotated[
        int | None,
        Field(
            description="Maximum number of concurrent rollouts to generate and score. Will create a global semaphore and pass to verifiers Environment. If None, will not limit concurrency.",
        ),
    ] = None

    resume_path: Annotated[
        Path | None,
        Field(
            description="Path to the directory containing results.jsonl to resume from. When set, will read existing example_ids from the results.jsonl file, filter the dataset to exclude already-evaluated examples, and append new results to the existing file. Can be an absolute path or relative to the current working directory.",
        ),
    ] = None

    watcher: Annotated[
        bool,
        Field(
            description=(
                "If True, watch `weights_dir` for newly-created stable checkpoints (folders named `step_{x}` that contain a `STABLE` file) "
                "and evaluate them as they appear, instead of immediately iterating over all existing checkpoints."
            ),
        ),
    ] = False

    @model_validator(mode="after")
    def validate_eval_base(self):
        if self.weights_dir is None and not self.eval_base:
            raise ValueError(
                "You should either evaluate the base model and/or checkpoints. Set `--eval-base` or specify a weight checkpoint directory with `--weights-dir`."
            )
        return self

    @model_validator(mode="after")
    def validate_resume_stream(self):
        """Enforce per_rollout when resume_path or save.stream is enabled, as both require per-rollout scheduling."""
        if self.resume_path is not None:
            if not self.save.stream:
                raise ValueError(
                    "resume_path requires save.stream. Streaming saves are required for resume functionality."
                )
        return self

    @model_validator(mode="after")
    def validate_resume_multiple_environments(self):
        """Prevent resume_path when multiple environments are configured, as they would share the same results.jsonl file."""
        if self.resume_path is not None and len(self.env) > 1:
            raise ValueError(
                f"resume_path cannot be used when evaluating multiple environments ({len(self.env)} environments configured). "
                f"All environments will read from and write to the same file ({self.resume_path / 'results.jsonl'}), "
                f"which may cause environments to incorrectly skip rollouts based on other environments' data. "
                f"Consider evaluating environments separately."
            )
        return self

    @model_validator(mode="after")
    def validate_resume_multiple_checkpoints(self):
        """Prevent resume_path when multiple checkpoints will be evaluated, as they would share the same results.jsonl file."""
        if self.resume_path is not None and self.weights_dir is not None:
            # When weights_dir is set, multiple checkpoint evaluations will occur (at least 1 checkpoint,
            # and possibly the base model if eval_base=True). All would write to the same results.jsonl file.
            raise ValueError(
                f"resume_path cannot be used when evaluating multiple checkpoints (weights_dir is set). "
                f"When resume_path is set, all checkpoint evaluations (including base model if eval_base=True) "
                f"would write to the same results.jsonl file ({self.resume_path / 'results.jsonl'}), "
                f"causing later checkpoints to incorrectly skip examples that were already evaluated for earlier checkpoints. "
                f"Consider evaluating checkpoints separately, each with its own resume_path, or remove resume_path to use checkpoint-specific output directories."
            )
        return self
