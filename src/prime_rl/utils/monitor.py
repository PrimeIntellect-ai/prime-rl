import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import wandb
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import WandbMonitorConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pydantic_config import BaseSettings


class WandbMonitor:
    """Logs to Weights and Biases."""

    def __init__(
        self,
        config: WandbMonitorConfig | None,
        output_dir: Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        run_config: BaseSettings | None = None,
    ):
        self.config = config
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self.output_dir = output_dir

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        self.enabled = self.config is not None
        self.is_master = rank == 0

        self.platform_enabled = False
        self.platform_run_id = None
        self.platform_api_url = None
        self.platform_api_key = None
        self.platform_config = None
        self.platform_upload_interval = 1
        self.platform_max_rollouts = 10
        self.run_config = run_config

        if config and config.platform and config.platform.enabled:
            self.platform_enabled = True
            self.platform_config = config.platform
            self.platform_api_url = config.platform.api_url.rstrip("/")
            self.platform_api_key = config.platform.api_key or os.getenv("PRIME_API_KEY")
            self.platform_upload_interval = config.platform.upload_interval
            self.platform_max_rollouts = config.platform.max_rollouts_per_step

            if not self.platform_api_key:
                self.logger.warning(
                    "Platform upload enabled but no API key provided. Set PRIME_API_KEY env var or config.platform.api_key"
                )
                self.platform_enabled = False

        if self.platform_enabled and self.is_master:
            if self.output_dir:
                run_id_file = self.output_dir / "platform_run_id.txt"

                if run_id_file.exists():
                    try:
                        file_age = time.time() - run_id_file.stat().st_mtime
                        existing_run_id = run_id_file.read_text().strip()

                        if existing_run_id and file_age < 120:
                            self.platform_run_id = existing_run_id
                            self.logger.info(
                                f"Using existing platform run_id from orchestrator: {self.platform_run_id}"
                            )
                        else:
                            if file_age >= 120:
                                self.logger.info(f"Stale platform_run_id.txt (age: {file_age:.0f}s) - creating new run")
                                run_id_file.unlink()
                            else:
                                self.logger.info("Empty platform_run_id.txt - registering new run")
                            self._register_platform_run()
                    except Exception as e:
                        self.logger.warning(f"Failed to read platform_run_id file: {e}")
                        self._register_platform_run()
                else:
                    self.logger.info("No platform_run_id.txt found - registering new run (orchestrator)")
                    self._register_platform_run()
            else:
                self._register_platform_run()

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
        if config is not None and config.log_extras:
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

            if config is not None and config.log_extras.distributions:
                self.last_log_distributions_step = -1
                # Incremental table is initialized dynamically in `log_distributions`
                self.distributions_table = None
                self.distributions = []

    def _maybe_overwrite_wandb_command(self) -> None:
        """Overwrites sys.argv with the start command if it is set in the environment variables."""
        wandb_args = os.environ.get("WANDB_ARGS", None)
        if wandb_args:
            self.logger.debug(f"Found WANDB_ARGS in environment variables {wandb_args}")
            sys.argv = json.loads(wandb_args)

    def log(self, metrics: dict[str, Any]) -> None:
        self.history.append(metrics)

        if self.platform_enabled and self.platform_run_id and self.is_master:
            step = metrics.get("step")
            if step is not None and step % self.platform_upload_interval == 0:
                self._upload_metrics_to_platform(metrics)

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
        tasks: list[str] | None = None,
    ) -> None:
        """Log prompt/response samples to W&B table.

        Args:
            input_tokens: List of input token sequences
            output_tokens: List of output token sequences
            rewards: List of rewards for each sample
            task_rewards: Optional list of task-specific rewards
            step: Current training step
        """
        if not self.is_master:
            return
        if (
            not self.config
            or not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            # Do not log samples if not enabled or not log interval step
            return
        assert self.tokenizer is not None, "Tokenizer is required for sample logging"
        assert self.last_log_samples_step <= step, "Step must be greater than last logged step"
        assert self.logger is not None, "Logger is required for sample logging"
        self.logger.info(f"Logging samples to W&B table at step {step}")
        start_time = time.time()
        batch_size = len(input_tokens)
        num_problems = batch_size // rollouts_per_problem

        # Compute per-problem statistics
        per_problem_tokens = defaultdict(list)
        tokens = [input_tokens[i] + output_tokens[i] for i in range(batch_size)]
        for i, t in enumerate(tokens):
            problem_id = i // rollouts_per_problem
            per_problem_tokens[problem_id].append(t)
        assert len(per_problem_tokens) == num_problems
        assert list(per_problem_tokens.keys()) == list(range(num_problems))

        per_problem_seq_len = {
            problem_id: sum(len(t) for t in tokens) / len(tokens) for problem_id, tokens in per_problem_tokens.items()
        }
        self.logger.debug(f"Per-problem seq len: {per_problem_seq_len}")
        min_len_problem_id = min(per_problem_seq_len.items(), key=lambda kv: kv[1])[0]
        max_len_problem_id = max(per_problem_seq_len.items(), key=lambda kv: kv[1])[0]
        random_problem_id = random.choice(list(range(num_problems)))
        problem_ids = {
            "min_len": min_len_problem_id,
            "max_len": max_len_problem_id,
            "random": random_problem_id,
        }
        self.logger.debug(f"Logging samples for problems: {problem_ids}")

        # Randomly select and log samples
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
        self.logger.debug(f"Logged samples at step {step} to W&B table in {time.time() - start_time:.2f}s")

        if (
            self.platform_enabled
            and self.platform_run_id
            and self.platform_config
            and self.platform_config.upload_rollouts
        ):
            self._upload_rollouts_to_platform(input_tokens, output_tokens, rewards, advantages, step, tasks)

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if not self.is_master:
            return
        if (
            not self.config
            or not self.config.log_extras
            or not self.config.log_extras.distributions
            or step % self.config.log_extras.interval != 0
        ):
            return
        assert self.last_log_distributions_step <= step, "Step must be greater than last logged step"
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
        start_time = time.time()
        row = {"step": step, **distributions}
        self.distributions.append(row)
        self.distributions_table.add_data(*row.values())
        wandb.log({"distributions": self.distributions_table}, step=step)
        self.last_log_distributions_step = step
        self.logger.debug(f"Logged distributions at step {step} to W&B table in {time.time() - start_time:.2f}s")

    def log_final_samples(self) -> None:
        """Log final samples to W&B table."""
        if not self.is_master:
            return
        if not self.config or not self.config.log_extras or not self.config.log_extras.samples:
            return
        self.logger.info("Logging final samples to W&B table")
        df = pd.DataFrame(self.samples)
        table = wandb.Table(dataframe=df)
        wandb.log({"final-samples": table})

    def log_final_distributions(self) -> None:
        """Log final distributions to W&B table."""
        if not self.is_master:
            return
        if not self.config or not self.config.log_extras or not self.config.log_extras.distributions:
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

        if self.platform_enabled and self.platform_run_id:
            self._mark_platform_run_completed()

    def _register_platform_run(self) -> None:
        """Register this run with the Prime platform."""
        if not self.platform_enabled or not self.platform_api_key:
            return

        try:
            user_id = os.getenv("PRIME_USER_ID")
            if not user_id:
                self.logger.warning("PRIME_USER_ID not set, skipping platform registration")
                self.platform_enabled = False
                return

            wandb_url = None
            if hasattr(self, "wandb") and self.wandb and self.config:
                wandb_url = f"https://wandb.ai/{self.config.project}/{self.wandb.id}"

            config_dict = {}
            if self.run_config:
                try:
                    config_dict = json.loads(json.dumps(self.run_config.model_dump(), default=str))
                except Exception:
                    config_dict = {
                        "model": self.run_config.model.name if hasattr(self.run_config, "model") else "unknown"
                    }

            run_data = {
                "user_id": user_id,
                "team_id": os.getenv("PRIME_TEAM_ID"),
                "name": self.config.name if self.config else None,
                "env_id": os.getenv("PRIME_ENV_ID", "unknown"),
                "env_name": os.getenv("PRIME_ENV_NAME"),
                "model_name": self.run_config.model.name
                if self.run_config and hasattr(self.run_config, "model")
                else "unknown",
                "config": config_dict,
                "wandb_url": wandb_url,
            }

            endpoint = f"{self.platform_api_url}/rl-runs"

            self.logger.info("Registering run with platform...")

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    endpoint,
                    json=run_data,
                    headers={
                        "Authorization": f"Bearer {self.platform_api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if response.status_code == 201:
                result = response.json()
                self.platform_run_id = result.get("run_id")

                platform_url = f"https://app.primeintellect.ai/dashboard/rl/{self.platform_run_id}"
                if "localhost" in self.platform_api_url:
                    platform_url = f"http://localhost:3000/dashboard/rl/{self.platform_run_id}"

                self.logger.info(f"✓ Registered RL run with platform: {self.platform_run_id}")
                self.logger.info(f"→ View run at: {platform_url}")

                # Save run_id to file so trainer can use it
                if self.output_dir:
                    run_id_file = self.output_dir / "platform_run_id.txt"
                    run_id_file.parent.mkdir(parents=True, exist_ok=True)
                    run_id_file.write_text(self.platform_run_id)
            else:
                self.logger.error(f"Failed to register run with platform: {response.status_code} - {response.text}")
                self.platform_enabled = False

        except Exception as e:
            self.logger.warning(f"Error registering run with platform: {e}")
            self.platform_enabled = False

    def _upload_metrics_to_platform(self, metrics: dict[str, Any]) -> None:
        """Upload metrics batch to the Prime platform."""
        if not self.platform_enabled or not self.platform_run_id:
            return

        try:
            step = metrics.get("step")
            if step is None:
                return

            metrics_data = {
                "run_id": self.platform_run_id,
                "step": step,
                "rewards": [float(metrics["reward/mean"])] if "reward/mean" in metrics else [],
                "entropies": [float(metrics["entropy/mean"])] if "entropy/mean" in metrics else [],
                "seq_lengths": [int(metrics["seq_len/mean"])] if "seq_len/mean" in metrics else [],
                "grad_norm": float(metrics.get("optim/grad_norm", 0.0)),
                "kl_mismatch": float(metrics.get("mismatch_kl/mean", 0.0)),
                "time_per_step": float(metrics.get("time/step", 0.0)),
                "total_tokens": int(metrics.get("progress/total_tokens", 0)),
                "total_samples": int(metrics.get("progress/total_samples", 0)),
                "total_examples": int(metrics.get("progress/total_problems", 0)),
                "additional_stats": {},
            }

            metrics_endpoint = f"{self.platform_api_url}/rl-runs/{self.platform_run_id}/metrics"

            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    metrics_endpoint,
                    json=metrics_data,
                    headers={
                        "Authorization": f"Bearer {self.platform_api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if response.status_code != 201:
                self.logger.warning(
                    f"Failed to upload metrics for step {step}: {response.status_code} - {response.text}"
                )

        except httpx.TimeoutException:
            self.logger.warning(f"Timeout uploading metrics for step {step} - backend may be slow or unresponsive")
        except Exception as e:
            self.logger.warning(f"Error uploading metrics for step {step}: {e}")

    def _upload_rollouts_to_platform(
        self,
        input_tokens: list[list[int]],
        output_tokens: list[list[int]],
        rewards: list[float],
        advantages: list[float],
        step: int,
        tasks: list[str] | None = None,
    ) -> None:
        """Upload sample rollouts to platform."""
        if not self.platform_enabled or not self.platform_run_id or not self.tokenizer:
            return

        try:
            num_rollouts = min(len(input_tokens), self.platform_max_rollouts)

            rollout_samples = []
            for i in range(num_rollouts):
                rollout_samples.append(
                    {
                        "example_id": i,
                        "prompt": self.tokenizer.decode(input_tokens[i]),
                        "completion": self.tokenizer.decode(output_tokens[i]),
                        "reward": float(rewards[i]),
                        "advantage": float(advantages[i]) if i < len(advantages) else None,
                        "prompt_tokens": input_tokens[i],
                        "completion_tokens": output_tokens[i],
                        "task": tasks[i] if tasks and i < len(tasks) else None,
                        "is_truncated": False,
                    }
                )

            rollouts_data = {
                "run_id": self.platform_run_id,
                "step": step,
                "rollouts": rollout_samples,
            }

            rollouts_endpoint = f"{self.platform_api_url}/rl-runs/{self.platform_run_id}/rollouts"

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    rollouts_endpoint,
                    json=rollouts_data,
                    headers={
                        "Authorization": f"Bearer {self.platform_api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if response.status_code != 201:
                self.logger.warning(
                    f"Failed to upload rollouts for step {step}: {response.status_code} - {response.text}"
                )

        except Exception as e:
            self.logger.warning(f"Error uploading rollouts to platform: {e}")

    def _mark_platform_run_completed(self) -> None:
        """Mark the run as completed on the platform."""
        if not self.platform_enabled or not self.platform_run_id:
            return

        try:
            update_endpoint = f"{self.platform_api_url}/rl-runs/{self.platform_run_id}"

            with httpx.Client(timeout=30.0) as client:
                response = client.put(
                    update_endpoint,
                    json={"status": "completed"},
                    headers={
                        "Authorization": f"Bearer {self.platform_api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if response.status_code == 200:
                self.logger.info("Marked platform run as completed")
            else:
                self.logger.warning(f"Failed to mark run as completed: {response.status_code} - {response.text}")

        except Exception as e:
            self.logger.warning(f"Error marking run as completed: {e}")


_MONITOR: WandbMonitor | None = None


def get_monitor() -> WandbMonitor:
    """Returns the global monitor."""
    global _MONITOR
    if _MONITOR is None:
        raise RuntimeError("WandbMonitor not initialized. Please call `setup_monitor` first.")
    return _MONITOR


def setup_monitor(
    config: WandbMonitorConfig | None,
    output_dir: Path | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    run_config: BaseSettings | None = None,
) -> WandbMonitor:
    """Sets up a monitor to log metrics to W&B."""
    global _MONITOR
    if _MONITOR is not None:
        raise RuntimeError("WandbMonitor already initialized. Please call `setup_monitor` only once.")
    _MONITOR = WandbMonitor(config=config, output_dir=output_dir, tokenizer=tokenizer, run_config=run_config)
    return _MONITOR
