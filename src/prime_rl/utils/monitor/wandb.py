import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import WandbConfig, WandbWithExtrasConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor
from prime_rl.utils.pydantic_config import BaseSettings


class WandbMonitor(Monitor):
    """Logs to Weights and Biases."""

    def __init__(
        self,
        config: WandbConfig | WandbWithExtrasConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseSettings | None = None,
    ):
        self.config = config
        self.logger = get_logger()
        self._history: list[dict[str, Any]] = []
        self.output_dir = output_dir

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0
        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return
        assert config is not None
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")
        self._maybe_overwrite_wandb_command()
        self.wandb = wandb.init(
            project=config.project,
            name=config.name,
            id=config.id,
            dir=output_dir,
            resume="allow",
            config=run_config.model_dump() if run_config else None,
            mode="offline" if config.offline else None,
        )

        # Optionally, initialize sample logging attributes
        if config is not None and isinstance(config, WandbWithExtrasConfig) and config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.samples_cols = [
                    "step",
                    "tag",
                    "problem_id",
                    "sample_id",
                    "num_input_tokens",
                    "num_output_tokens",
                    "input_tokens",
                    "output_tokens",
                    "prompt",
                    "completion",
                    "reward",
                    "advantage",
                ]
                self.samples_table = wandb.Table(
                    columns=self.samples_cols,
                    log_mode="INCREMENTAL",
                )
                self.tokenizer = tokenizer
                self.samples = []

            if config.log_extras.distributions:
                self.last_log_distributions_step = -1
                # Incremental table is initialized dynamically in `log_distributions`
                self.distributions_table = None
                self.distributions = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Returns the history of logged metrics."""
        return self._history

    def _maybe_overwrite_wandb_command(self) -> None:
        """Overwrites sys.argv with the start command if it is set in the environment variables."""
        wandb_args = os.environ.get("WANDB_ARGS", None)
        if wandb_args:
            self.logger.debug(f"Found WANDB_ARGS in environment variables {wandb_args}")
            sys.argv = json.loads(wandb_args)

    def log(self, metrics: dict[str, Any]) -> None:
        self._history.append(metrics)
        if not self.is_master:
            return
        if not self.enabled:
            return
        wandb.log(metrics, step=metrics.get("step", None))

    def log_samples(
        self,
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rewards: list[float],
        advantages: list[float],
        rollouts_per_problem: int,
        step: int,
    ) -> None:
        """Log prompt/response samples to W&B table.

        Args:
            input_tokens: List of input token sequences
            output_tokens: List of output token sequences
            rewards: List of rewards for each sample
            advantages: List of advantages for each sample
            rollouts_per_problem: Number of rollouts per problem
            step: Current training step
        """
        if not self.is_master:
            return
        if not Monitor.should_log_samples(self.config, step, self.last_log_samples_step):
            return
        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.logger is not None, "Logger is required for sample logging"
        self.logger.info(f"Logging samples to W&B table at step {step}")
        start_time = time.perf_counter()

        # Use shared sample selection logic
        problem_ids = Monitor.select_sample_problems(input_tokens, output_tokens, rollouts_per_problem)
        self.logger.debug(f"Logging samples for problems: {problem_ids}")

        # Log samples for selected problems
        for tag, problem_id in problem_ids.items():
            start_idx = problem_id * rollouts_per_problem
            for sample_id in range(start_idx, start_idx + rollouts_per_problem):
                sample = {
                    "step": step,
                    "tag": tag,
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "num_input_tokens": len(input_tokens[sample_id]),
                    "num_output_tokens": len(output_tokens[sample_id]),
                    "input_tokens": str(input_tokens[sample_id]),
                    "output_tokens": str(output_tokens[sample_id]),
                    "prompt": self.tokenizer.decode(input_tokens[sample_id]),
                    "completion": self.tokenizer.decode(output_tokens[sample_id]),
                    "reward": float(rewards[sample_id]),
                    "advantage": float(advantages[sample_id]),
                }
                assert list(sample.keys()) == self.samples_cols, (
                    "Order of columns in the table must be the same as order of the keys here"
                )
                self.samples_table.add_data(*sample.values())
                self.samples.append(sample)
        wandb.log({"samples": self.samples_table}, step=step)
        self.last_log_samples_step = step
        self.logger.debug(f"Logged samples at step {step} to W&B table in {time.perf_counter() - start_time:.2f}s")

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if not self.is_master:
            return
        if not Monitor.should_log_distributions(self.config, step, self.last_log_distributions_step):
            return
        assert self.logger is not None
        self.logger.info(f"Logging distributions for keys {list(distributions.keys())} to W&B table at step {step}")

        # Initialize incremental table if not already done
        if self.distributions_table is None:
            self.distributions_cols = list(distributions.keys())
            self.distributions_table = wandb.Table(
                columns=["step"] + self.distributions_cols,
                log_mode="INCREMENTAL",
            )
        assert self.distributions_cols == list(distributions.keys()), (
            "Columns in the table must be the same across all steps"
        )

        # Append to distributions
        start_time = time.perf_counter()
        row = {"step": step, **distributions}
        self.distributions.append(row)
        self.distributions_table.add_data(*row.values())
        wandb.log({"distributions": self.distributions_table}, step=step)
        self.last_log_distributions_step = step
        self.logger.debug(
            f"Logged distributions at step {step} to W&B table in {time.perf_counter() - start_time:.2f}s"
        )

    def log_final_samples(self) -> None:
        """Log final samples to W&B table."""
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
        ):
            return
        self.logger.info("Logging final samples to W&B table")
        df = pd.DataFrame(self.samples)
        table = wandb.Table(dataframe=df)
        wandb.log({"final-samples": table})

    def log_final_distributions(self) -> None:
        """Log final distributions to W&B table."""
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.distributions
        ):
            return
        self.logger.info("Logging final distributions to W&B table")
        df = pd.DataFrame(self.distributions)
        table = wandb.Table(dataframe=df)
        wandb.log({"final-distributions": table})

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        """Save final summary to W&B table."""
        if not self.is_master or not self.enabled:
            return
        self.logger.info("Saving final summary to file")
        assert self.output_dir is not None, "Output directory is required for saving final summary"
        dir_path = self.output_dir / f"run-{self.wandb.id}"
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / filename, "w") as f:
            json.dump(wandb.summary._as_dict(), f)
