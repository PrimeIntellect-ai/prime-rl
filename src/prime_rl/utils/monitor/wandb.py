import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import verifiers as vf
import wandb
from transformers.tokenization_utils import PreTrainedTokenizer
from wandb.errors import CommError

from prime_rl.configs.shared import WandbConfig, WandbWithExtrasConfig
from prime_rl.utils.chat_template import deserialize_tool_calls
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor, sample_items_for_logging


class WandbMonitor(Monitor):
    """Logs to Weights and Biases."""

    def __init__(
        self,
        config: WandbConfig | WandbWithExtrasConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseConfig | None = None,
    ):
        self.config = config
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
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

        shared_mode = os.environ.get("WANDB_SHARED_MODE") == "1"
        if shared_mode:
            run_id = os.environ.get("WANDB_SHARED_RUN_ID")
            label = os.environ.get("WANDB_SHARED_LABEL")
            primary = label == "orchestrator"
            settings = wandb.Settings(
                mode="shared",
                x_label=label,
                x_primary=primary,
                x_update_finish_state=primary,
            )
            self.logger.info(
                f"Using shared W&B mode ({label=}, {primary=}). "
                "This is an experimental feature. Disable with --wandb.shared False"
            )
        else:
            run_id = None
            primary = False
            settings = wandb.Settings(
                mode="offline" if config.offline else "online",
            )

        def init_wandb(max_retries: int):
            for attempt in range(max_retries):
                try:
                    return wandb.init(
                        id=run_id,
                        project=config.project,
                        name=config.name,
                        dir=output_dir,
                        config=run_config.model_dump() if run_config else None,
                        settings=settings,
                    )
                except CommError:
                    if attempt + 1 == max_retries:
                        raise
                    self.logger.info(
                        f"Shared W&B run not yet created by primary, retrying in 10s ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(10)

        max_retries = 1 if not shared_mode or primary else 30
        self.wandb = init_wandb(max_retries)

        wandb.define_metric("*", step_metric="step")

        # Optionally, initialize sample logging attributes
        if config is not None and isinstance(config, WandbWithExtrasConfig) and config.log_extras:
            if config.log_extras.samples:
                self.last_log_samples_step = -1
                self.samples_cols = ["step", "env_name", "task", "example_id", "messages", "input_ids", "reward"]
                self.samples_table = wandb.Table(
                    columns=self.samples_cols,
                    log_mode="INCREMENTAL",
                )
                self.tokenizer = tokenizer
                self.samples = []
                self.eval_samples_cols = ["step", "env", "task", "example_id", "completion", "reward"]
                self.eval_samples_table = wandb.Table(
                    columns=self.eval_samples_cols,
                    log_mode="INCREMENTAL",
                )

    def _maybe_overwrite_wandb_command(self) -> None:
        """Overwrites sys.argv with the start command if it is set in the environment variables."""
        wandb_args = os.environ.get("WANDB_ARGS", None)
        if wandb_args:
            self.logger.debug(f"Found WANDB_ARGS in environment variables {wandb_args}")
            sys.argv = json.loads(wandb_args)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.history.append(metrics)
        if not self.is_master:
            return
        if not self.enabled:
            return
        wandb.log({**metrics, "step": step})

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        """Logs rollouts to W&B table."""
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            # Do not log samples if not enabled or not log interval step
            return

        rollouts = sample_items_for_logging(
            rollouts,
            self.config.log_extras.sample_ratio,
        )
        if not rollouts:
            return

        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.last_log_samples_step <= step, "Step must be greater than last logged step"
        assert self.logger is not None, "Logger is required for sample logging"

        self.logger.info(f"Logging {len(rollouts)} samples to W&B table at step {step}")
        start_time = time.perf_counter()

        for rollout in rollouts:
            trajectory = rollout["trajectory"]
            if not trajectory:
                continue
            last_step = trajectory[-1]
            tokens = last_step["tokens"]
            full_ids = tokens["prompt_ids"] + tokens["completion_ids"]
            messages_text = self.tokenizer.decode(full_ids)
            sample = {
                "step": step,
                "env_name": rollout.get("env_name"),
                "task": rollout.get("task"),
                "example_id": rollout["example_id"],
                "messages": messages_text,
                "input_ids": str(full_ids),
                "reward": rollout["reward"],
            }
            assert list(sample.keys()) == self.samples_cols, (
                "Order of columns in the table must be the same as order of the keys here"
            )
            self.samples_table.add_data(*sample.values())
            self.samples.append(sample)

        wandb.log({"samples": self.samples_table, "step": step})
        self.last_log_samples_step = step
        self.logger.debug(f"Logged samples at step {step} to W&B table in {time.perf_counter() - start_time:.2f}s")

    def log_eval_samples(self, rollouts: list[vf.RolloutOutput], env_name: str, step: int) -> None:
        """Logs eval rollouts to a separate W&B table."""
        if not self.is_master:
            return
        if (
            not self.config
            or not isinstance(self.config, WandbWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
        ):
            return

        for rollout in rollouts:
            completion = rollout.get("completion")
            if not completion:
                continue
            if isinstance(completion, list):
                try:
                    completion = self.tokenizer.apply_chat_template(deserialize_tool_calls(completion), tokenize=False)
                except Exception:
                    completion = str(completion)
            sample = {
                "step": step,
                "env": env_name,
                "task": rollout.get("task"),
                "example_id": rollout["example_id"],
                "completion": completion,
                "reward": rollout["reward"],
            }
            self.eval_samples_table.add_data(*sample.values())

        wandb.log({"eval/samples": self.eval_samples_table, "step": step})

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

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        """Log distributions (no-op for W&B)."""
        pass

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
