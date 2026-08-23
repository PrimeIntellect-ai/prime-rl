from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import wandb
from wandb.errors import CommError
from wandb.sdk.mailbox.mailbox_handle import ServerResponseError

from prime_rl.configs.monitors import WandbMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.monitors.wandb.overview import ensure_overview_view
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import format_time

if TYPE_CHECKING:
    import verifiers.v1 as vf


class WandbMonitor(Monitor):
    """Logs metrics to Weights and Biases."""

    config: WandbMonitorConfig

    async def init(
        self,
        output_dir: Path,
        config: BaseConfig | None = None,
        train_env_names: list[str] | None = None,
        eval_env_names: list[str] | None = None,
        overview_flavor: Literal["rl", "sft"] = "rl",
    ) -> None:
        # W&B reads the start command off sys.argv; the launcher passes the original
        # command to subprocesses via $WANDB_ARGS.
        wandb_args = os.environ.get("WANDB_ARGS")
        if wandb_args:
            self.logger.debug(f"Found WANDB_ARGS in environment variables {wandb_args}")
            sys.argv = json.loads(wandb_args)

        # WANDB_MODE=disabled/offline takes precedence over shared mode — shared mode
        # requires a server connection and can't work offline.
        shared_mode = os.environ.get("WANDB_SHARED_MODE") == "1" and os.environ.get("WANDB_MODE") not in (
            "disabled",
            "offline",
        )
        if shared_mode:
            # W&B's native run-id var, set by the launcher to $PRL_RUN_ID.
            run_id = os.environ.get("WANDB_RUN_ID")
            label = os.environ.get("WANDB_SHARED_LABEL")
            primary_label = os.environ.get("WANDB_SHARED_PRIMARY", "orchestrator")
            primary = label == primary_label
            # The primary creates the run; the finisher writes its final state. They can
            # differ when the run's creator is not its last-alive writer (e.g. the SFT
            # trainer creates the run, the evals process outlives it and finalizes).
            finisher = label == os.environ.get("WANDB_SHARED_FINISHER", primary_label)
            settings = wandb.Settings(
                mode="shared",
                x_label=label,
                x_primary=primary,
                x_update_finish_state=finisher,
            )
            self.logger.debug(f"Using shared W&B mode ({label=}, {primary=}, {finisher=})")
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
                        dir=output_dir,
                        config=config.model_dump() if config else None,
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

        wandb.define_metric("*", step_metric="step")
        # inference metrics are sampled on wall time, not the training step
        wandb.define_metric("inference/*", step_metric="_timestamp")

        # Provision the curated "overview" saved view once per project (the run's primary process
        # in shared mode, else the single master). Best-effort: a workspaces/API failure must never
        # take down training.
        if is_online and (primary if shared_mode else True):
            try:
                url = ensure_overview_view(
                    self.wandb.entity,
                    self.wandb.project,
                    train_envs=train_env_names or [],
                    eval_envs=eval_env_names or [],
                    flavor=overview_flavor,
                )
                if url:
                    self.logger.info(f"Created W&B overview view - {url}")
            except Exception as e:
                self.logger.warning(f"Failed to create W&B overview view - {e}")

        self.logger.info(f"Logging metrics to W&B ({self.wandb.url})")

    async def log_metrics(self, metrics: dict[str, Any], step: int | None) -> None:
        wandb.log(metrics if step is None else {**metrics, "step": step})

    async def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None:
        pass

    async def finalize(self) -> None:
        self.logger.info(f"Finalizing W&B run {self.wandb.id}")
        t0 = time.perf_counter()
        # Explicit finish: in (experimental) shared mode the SDK's atexit finish does
        # not land the run state - without this, even clean runs decay to "crashed".
        await asyncio.to_thread(wandb.finish, exit_code=0)
        self.logger.debug(f"Finalized W&B run in {format_time(time.perf_counter() - t0)}")
