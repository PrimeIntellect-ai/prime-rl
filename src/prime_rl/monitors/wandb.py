from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import wandb
from wandb.errors import CommError
from wandb.sdk.mailbox.mailbox_handle import ServerResponseError

from prime_rl.configs.shared import WandbConfig
from prime_rl.monitors.base import Monitor
from prime_rl.monitors.wandb_overview import ensure_overview_view
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger


class WandbMonitor(Monitor):
    """Logs metrics to Weights and Biases."""

    def __init__(
        self,
        config: WandbConfig,
        output_dir: Path,
        run_config: BaseConfig | None = None,
        train_env_names: list[str] | None = None,
        eval_env_names: list[str] | None = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.run_config = run_config
        self.train_env_names = train_env_names or []
        self.eval_env_names = eval_env_names or []
        self.logger = get_logger()

    def init(self) -> None:
        # W&B reads the start command off sys.argv; the launcher passes the original
        # command to subprocesses via $WANDB_ARGS.
        wandb_args = os.environ.get("WANDB_ARGS")
        if wandb_args:
            self.logger.debug(f"Found WANDB_ARGS in environment variables {wandb_args}")
            sys.argv = json.loads(wandb_args)

        # WANDB_MODE=disabled/offline takes precedence over shared mode — shared mode
        # requires a server connection and can't work offline.
        _wandb_mode = os.environ.get("WANDB_MODE")
        shared_mode = os.environ.get("WANDB_SHARED_MODE") == "1" and _wandb_mode not in ("disabled", "offline")
        if shared_mode:
            # W&B's native run-id var, set by the launcher to $PRL_RUN_ID.
            run_id = os.environ.get("WANDB_RUN_ID")
            label = os.environ.get("WANDB_SHARED_LABEL")
            primary = label == "orchestrator"
            settings = wandb.Settings(
                mode="shared",
                x_label=label,
                x_primary=primary,
                x_update_finish_state=primary,
            )
            self.logger.info(f"Using shared W&B mode ({label=}, {primary=})")
            is_online = True
        else:
            run_id = None
            primary = False
            mode = os.environ.get("WANDB_MODE", "offline" if self.config.offline else "online")
            settings = wandb.Settings(mode=mode)
            is_online = mode == "online"

        retryable_errors = (CommError, ServerResponseError) if shared_mode else (CommError,)

        def init_wandb(max_retries: int):
            for attempt in range(max_retries):
                try:
                    return wandb.init(
                        id=run_id,
                        resume="allow" if run_id else None,
                        project=self.config.project,
                        entity=self.config.entity,
                        name=self.config.name,
                        group=self.config.group,
                        tags=self.config.tags,
                        dir=self.output_dir,
                        config=self.run_config.model_dump() if self.run_config else None,
                        settings=settings,
                    )
                except retryable_errors as e:
                    if attempt + 1 == max_retries:
                        raise
                    if shared_mode and not primary:
                        msg = (
                            f"Shared W&B run not yet created by primary - retrying in 10s ({attempt + 1}/{max_retries})"
                        )
                    else:
                        msg = f"Transient W&B init error ({e}) - retrying in 10s ({attempt + 1}/{max_retries})"
                    self.logger.info(msg)
                    # A failed wandb.init leaves the run_id registered in the local
                    # wandb-core StreamMux, causing the next attempt to fail with
                    # "run ID ... is in use". Tear down the service so the retry
                    # starts from a clean state.
                    wandb.teardown()
                    time.sleep(10)

        # Non-primary processes in shared mode wait for the primary to create the run.
        # Everyone else still retries to absorb transient W&B server errors (e.g. 404 on upsertBucket).
        max_retries = 30 if shared_mode and not primary else 5
        self.wandb = init_wandb(max_retries)
        self.run_id = self.wandb.id

        wandb.define_metric("*", step_metric="step")

        # Provision the curated "overview" saved view once per project (the run's primary process
        # in shared mode, else the single master). Best-effort: a workspaces/API failure must never
        # take down training.
        if is_online and (primary if shared_mode else True):
            try:
                url = ensure_overview_view(
                    self.wandb.entity,
                    self.wandb.project,
                    train_envs=self.train_env_names,
                    eval_envs=self.eval_env_names,
                )
                if url:
                    self.logger.info(f"Created W&B overview view - {url}")
            except Exception as e:
                self.logger.warning(f"Failed to create W&B overview view - {e}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        wandb.log({**metrics, "step": step})

    def save_final_summary(self) -> None:
        dir_path = self.output_dir / f"run-{self.wandb.id}"
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / "final_summary.json", "w") as f:
            json.dump(wandb.summary._as_dict(), f)
