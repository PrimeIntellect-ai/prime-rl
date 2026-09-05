from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

import prime_runs as pr

from prime_rl.configs.monitors import PrimeMonitorConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    import verifiers.v1 as vf

BASE_URL_VAR = "PRIME_API_BASE"
# How long finish() and the SDK's atexit crash hook let queued uploads drain. The SDK
# default (300 s) is sized for eval sample batches; a crashed training process should
# not linger that long, and a clean finish rarely has more than the last step queued.
FINISH_TIMEOUT = 60.0


def _base_url() -> str | None:
    """$PRIME_API_BASE historically points at the RFT API root (``.../api/v1/rft``);
    the SDK takes the platform base URL. Unset means the SDK resolves it."""
    base = os.getenv(BASE_URL_VAR)
    return base.rstrip("/").removesuffix("/rft") if base else None


class PrimeMonitor(Monitor):
    """Logs metrics and episodes to the Prime platform through ``prime_runs``.

    The run handle owns what ``TrainRun`` used to do by hand: the RFT
    lifecycle (register or attach, finalize), the per-step metrics POSTs, the
    every-10th-step Parquet sample uploads (presign -> PUT -> confirm), and
    the terminal status — a process that exits without finalizing is reported
    crashed by the SDK's atexit hook, replacing the old ``_mark_failed`` one.

    ``init``/``finish`` do network I/O and run in worker threads; the log
    calls are queue puts onto the SDK's uploader thread, which owns retries
    and backpressure, so they never stall the loop.
    """

    config: PrimeMonitorConfig
    run: pr.Run

    async def init(self, config: BaseConfig | None = None) -> None:
        init_kwargs: dict[str, Any]
        if run_id := os.getenv("RUN_ID"):
            # A managed launch pre-created the platform run and injected its id -
            # attach instead of registering a duplicate. The backend owns the run's
            # failure marking then; a clean finish() still marks it completed.
            init_kwargs = {"id": run_id}
        elif config is not None:
            init_kwargs = dict(
                name=self.config.name,
                model=config.model.name,
                environments=[env.env_id for env in config.train.source],
                training=pr.TrainingSpec(
                    max_steps=config.max_steps or 0,
                    batch_size=config.batch_size,
                    rollouts_per_example=config.group_size,
                    seq_len=config.seq_len,
                    wandb_project=config.monitors.wandb.project if config.monitors.wandb else None,
                ),
                config=config.model_dump(exclude_none=True, mode="json"),
            )
        else:
            # The RFT API requires a base model; "unknown" is what the
            # pre-SDK register sent for a config-less init.
            init_kwargs = {"name": self.config.name, "model": "unknown"}

        # A configured monitor must work (see monitors.setup), so the default is
        # mode="online": a missing key or a team outside the external-runs allowlist
        # raises here instead of training silently untracked. $PRIME_RUNS_MODE=disabled
        # stays the explicit opt-out.
        self.run = await asyncio.to_thread(
            pr.init,
            kind="train",
            mode=os.getenv(pr.MODE_ENV) or "online",
            base_url=_base_url(),
            finish_timeout=FINISH_TIMEOUT,
            **init_kwargs,
        )
        if self.run.url:
            attached = " (attached via $RUN_ID)" if self.run.attached else ""
            self.logger.info(f"Logging metrics and episodes to platform run {self.run.id} ({self.run.url}){attached}")
        else:
            self.logger.info(f"Platform run disabled ({pr.MODE_ENV}=disabled)")

    async def log_metrics(self, metrics: dict[str, Any], step: int | None) -> None:
        # The SDK also drops non-finite values, but silently; sanitize first so
        # the dropped paths are named in the log.
        metrics, dropped = sanitize(metrics)
        if dropped:
            self.logger.warning(f"Dropping {len(dropped)} non-finite metric value(s): {', '.join(dropped[:5])}")
        # A queue put that can block briefly under backpressure - off the loop. The SDK
        # stamps `_timestamp` on every row, so step=None rows keep a time anchor.
        await asyncio.to_thread(self.run.log_metrics, metrics, step=step)

    async def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None:
        """Only the trained cohort ships to the platform. The upload cadence
        (every 10th step) and the Parquet encoding live in the SDK's training
        samples sink, which reads each episode's dispatch step off ``run.work``
        - the ``TrainRunInfo`` the dispatcher stamps at emit time."""
        if kind != "train" or subset != "effective" or not episodes:
            return
        # A queue put that can block briefly under backpressure - off the loop.
        await asyncio.to_thread(self.run.log_episodes, episodes)

    async def finalize(self) -> None:
        # Drains queued uploads so the final step's metrics and episodes land,
        # then finalizes (idempotent on the platform side); an attached run's
        # failure marking stays with the launcher.
        await asyncio.to_thread(self.run.finish)
