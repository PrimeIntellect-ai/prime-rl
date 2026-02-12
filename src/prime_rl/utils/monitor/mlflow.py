import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import verifiers as vf
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.utils.config import MLflowConfig, MLflowWithExtrasConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor
from prime_rl.utils.pydantic_config import BaseSettings


class MLflowMonitor(Monitor):
    """Logs to MLflow."""

    def __init__(
        self,
        config: MLflowConfig | MLflowWithExtrasConfig | None,
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
        if not self.enabled or not self.is_master:
            if not self.is_master:
                self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return

        assert config is not None
        self.logger.info(f"Initializing {self.__class__.__name__} ({config})")

        import mlflow

        self._mlflow = mlflow

        mlflow.set_tracking_uri(config.tracking_uri)

        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(config.experiment_name)
        if experiment is None and config.artifact_location:
            experiment_id = client.create_experiment(config.experiment_name, artifact_location=config.artifact_location)
            mlflow.set_experiment(experiment_id=experiment_id)
        else:
            mlflow.set_experiment(config.experiment_name)

        self.run = mlflow.start_run(run_name=config.run_name)

        if run_config is not None:
            params = run_config.model_dump()
            # MLflow has a 500-param limit per batch; flatten and truncate values
            flat_params = _flatten_dict(params)
            # MLflow param values are limited to 500 chars
            flat_params = {k: str(v)[:500] for k, v in flat_params.items()}
            mlflow.log_params(flat_params)

        if isinstance(config, MLflowWithExtrasConfig) and config.log_extras:
            if config.log_extras.samples:
                self.tokenizer = tokenizer
                self.samples: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.history.append(metrics)
        if not self.is_master or not self.enabled:
            return

        if step is None and "step" in metrics:
            step = int(metrics["step"])

        numeric_metrics = {}
        for k, v in metrics.items():
            if k == "step":
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_metrics[k] = v

        if numeric_metrics:
            self._mlflow.log_metrics(numeric_metrics, step=step)

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        if not self.is_master or not self.enabled:
            return
        if (
            not self.config
            or not isinstance(self.config, MLflowWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
            or step % self.config.log_extras.interval != 0
        ):
            return

        assert self.tokenizer is not None, "Tokenizer is required for sample logging"

        self.logger.info(f"Logging samples to MLflow at step {step}")
        start_time = time.perf_counter()

        rows = []
        for rollout in rollouts:
            trajectory = rollout["trajectory"]
            if not trajectory:
                continue
            last_step = trajectory[-1]
            tokens = last_step["tokens"]
            full_ids = tokens["prompt_ids"] + tokens["completion_ids"]
            messages_text = self.tokenizer.decode(full_ids)
            row = {
                "step": step,
                "task": rollout.get("task"),
                "example_id": rollout["example_id"],
                "messages": messages_text,
                "reward": rollout["reward"],
            }
            rows.append(row)
            self.samples.append(row)

        if rows and self.config.log_artifacts:
            df = pd.DataFrame(rows)
            self._mlflow.log_table(df, artifact_file=f"samples/step_{step}.json")

        self.logger.debug(f"Logged samples at step {step} to MLflow in {time.perf_counter() - start_time:.2f}s")

    def log_final_samples(self) -> None:
        if not self.is_master or not self.enabled:
            return
        if (
            not self.config
            or not isinstance(self.config, MLflowWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.samples
        ):
            return

        if not self.config.log_artifacts:
            return

        self.logger.info("Logging final samples to MLflow")
        if self.samples:
            df = pd.DataFrame(self.samples)
            self._mlflow.log_table(df, artifact_file="final_samples.json")

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if not self.is_master or not self.enabled:
            return
        if (
            not self.config
            or not isinstance(self.config, MLflowWithExtrasConfig)
            or not self.config.log_extras
            or not self.config.log_extras.distributions
            or step % self.config.log_extras.interval != 0
        ):
            return

        dist_metrics = {}
        for name, values in distributions.items():
            if not values:
                continue
            dist_metrics[f"distributions/{name}/mean"] = statistics.mean(values)
            dist_metrics[f"distributions/{name}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            dist_metrics[f"distributions/{name}/min"] = min(values)
            dist_metrics[f"distributions/{name}/max"] = max(values)
            dist_metrics[f"distributions/{name}/median"] = statistics.median(values)

        if dist_metrics:
            self._mlflow.log_metrics(dist_metrics, step=step)

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        if not self.is_master or not self.enabled:
            return
        if not self.config.log_artifacts:
            return

        self.logger.info("Saving final summary via MLflow")
        assert self.output_dir is not None, "Output directory is required for saving final summary"

        run_id = self.run.info.run_id
        dir_path = self.output_dir / f"mlflow-{run_id}"
        dir_path.mkdir(parents=True, exist_ok=True)

        summary = self.history[-1] if self.history else {}
        summary_path = dir_path / filename
        with open(summary_path, "w") as f:
            json.dump(summary, f)

        self._mlflow.log_artifact(str(summary_path))

    def close(self) -> None:
        if not self.is_master or not self.enabled:
            return
        self._mlflow.end_run()


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
