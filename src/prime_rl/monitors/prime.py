from __future__ import annotations

import asyncio
import atexit
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import prime_runs as pr
from prime_sandboxes import Config as PrimeConfig

from prime_rl.configs.eval import EvalConfig, SFTOnlineEvalConfig
from prime_rl.configs.monitors import PrimeMonitorConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.monitors.base import Kind, Monitor
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.links import ATTEMPT_VAR, PRIME_LINKS, publish_link, register_links
from prime_rl.utils.pathing import get_platform_run_path
from prime_rl.utils.process import RUN_FINISHER_VAR
from prime_rl.utils.record import read_record, update_record, write_record
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    import verifiers.v1 as vf

BASE_URL_VAR = "PRIME_API_BASE"
EVAL_ID_VAR = "PRIME_RUNS_EVAL_ID"
# How long finish() and the SDK's atexit crash hook let queued uploads drain. The SDK
# default (300 s) is sized for eval sample batches; a crashed training process should
# not linger that long, and a clean finish rarely has more than the last step queued.
FINISH_TIMEOUT = 60.0
# How long an SFT online-eval process waits for the trainer to register the run: the
# trainer starts with it, but registers only after its imports and config setup.
TRAIN_RUN_TIMEOUT_S = 1200.0
TRAIN_RUN_POLL_S = 1.0
OFF_HINT = "turn the monitor off with --no-monitors.prime"


def write_platform_record(output_dir: Path, record: dict[str, Any]) -> None:
    """Leave the run's Prime identity in the run directory for the dashboard's
    "View on Prime" link."""
    write_record(get_platform_run_path(output_dir), record)


def read_platform_record(output_dir: Path) -> dict[str, Any] | None:
    return read_record(get_platform_run_path(output_dir))


def _merge_platform_record(output_dir: Path, merge: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """Locked read-modify-write of the platform record. Best-effort, see ``update_record``."""
    update_record(get_platform_run_path(output_dir), merge)


def update_platform_record(output_dir: Path, update: dict[str, Any]) -> None:
    """Overlay ``update`` on the platform record, keeping the keys it doesn't set."""
    _merge_platform_record(output_dir, lambda record: {**record, **update})


def prime_mode() -> str:
    """$PRIME_RUNS_MODE when set, else online. Like the W&B monitor, a configured
    monitor without credentials fails the run instead of leaving it untracked."""
    if mode := os.getenv(pr.MODE_ENV):
        return mode
    if not (os.getenv("PRIME_API_KEY") or PrimeConfig().api_key):
        raise RuntimeError(
            f"The prime monitor found no Prime API key. Authenticate with `prime login` (or set PRIME_API_KEY), or {OFF_HINT}"
        )
    return "online"


def _base_url() -> str | None:
    """$PRIME_API_BASE historically points at the RFT API root (``.../api/v1/rft``);
    the SDK takes the platform base URL. Unset means the SDK resolves it."""
    base = os.getenv(BASE_URL_VAR)
    return base.rstrip("/").removesuffix("/rft") if base else None


def _registration(name: str | None, config: BaseConfig | None) -> dict[str, Any]:
    """``pr.init`` arguments that register a new training run for ``config``."""
    if isinstance(config, SFTConfig):
        # SFT trains on a dataset, not on environments, and has no rollouts.
        return dict(
            name=name,
            model=config.model.name,
            environments=[],
            training=pr.TrainingSpec(
                max_steps=config.max_steps or 0,
                batch_size=config.data.batch_size,
                seq_len=config.data.seq_len,
                wandb_project=config.monitors.wandb.project if config.monitors.wandb else None,
            ),
            config=config.model_dump(exclude_none=True, mode="json"),
        )
    if config is not None:
        return dict(
            name=name,
            model=config.model.name,
            environments=[env.env_id for env in config.train.source],
            training=pr.TrainingSpec(
                max_steps=config.max_steps or 0,
                batch_size=config.batch_size,
                rollouts_per_example=max(
                    (source.group_size for source in config.train.source), default=config.train.group_size
                ),
                seq_len=config.seq_len,
                wandb_project=config.monitors.wandb.project if config.monitors.wandb else None,
            ),
            config=config.model_dump(exclude_none=True, mode="json"),
        )
    # The RFT API requires a base model.
    return {"name": name, "model": "unknown"}


def wait_for_train_run(output_dir: Path, timeout: float = TRAIN_RUN_TIMEOUT_S) -> str:
    """The id of the training run this attempt's trainer registered, from the platform
    record it leaves in the shared run dir."""
    attempt = os.getenv(ATTEMPT_VAR)
    deadline = time.monotonic() + timeout
    while True:
        try:
            record = read_platform_record(output_dir) or {}
        except (OSError, ValueError):
            record = {}
        if record.get("kind") == "train" and record.get("attempt") == attempt and record.get("id"):
            return record["id"]
        if time.monotonic() >= deadline:
            raise RuntimeError(f"The trainer registered no Prime run within {timeout:.0f}s; {OFF_HINT}")
        time.sleep(TRAIN_RUN_POLL_S)


class PrimeMonitor(Monitor):
    """Logs a run to Prime through ``prime_runs``.

    A training run's processes (the rl orchestrator, the sft trainer and its online-eval
    process) log to one Prime training run: metrics, and every episode as it arrives.
    The process with its own config registers the run, or attaches to the one a
    hosted launch created ($RUN_ID); an SFT online-eval process attaches to its
    trainer's. ``$PRL_RUN_FINISHER`` names the process that finishes a shared run.

    A standalone eval opens one Prime evaluation per env and epoch when the epoch
    starts (``log_eval_plan``), streams its episodes into it, and finishes it with the
    epoch's aggregates (``log_eval_epoch``).

    Delivery is best-effort: the SDK's uploader thread owns retries and backpressure,
    and its atexit hook reports a process that exits without finishing. An abrupt kill
    skips the hook, and the platform owns the terminal state of attached runs.
    """

    config: PrimeMonitorConfig

    async def init(self) -> None:
        config = self.context.run_config
        self.mode = prime_mode()
        self.run: pr.Run | None = None
        # (env, step) -> its open evaluation; None once opening it failed, so the epoch's
        # episodes do not retry Prime on every arrival
        self.evaluations: dict[tuple[str, int], pr.Run | None] = {}
        if isinstance(config, EvalConfig):
            await self._init_evaluations(config)
            return
        online_eval = isinstance(config, SFTOnlineEvalConfig)
        self.run = await asyncio.to_thread(self._open_run, config, online_eval)
        if not self.run.url:
            self.logger.info("Prime run disabled")
        else:
            attached = " (attached)" if self.run.attached else ""
            self.logger.info(f"Logging metrics and episodes to Prime run {self.run.id} ({self.run.url}){attached}")
        if not online_eval:
            # Off the event loop: the flock retry must not stall it.
            await asyncio.to_thread(self._record_run)

    def _open_run(self, config: BaseConfig | None, online_eval: bool) -> pr.Run:
        """Register the training run, or attach to it. Blocking: runs in a worker thread."""
        run_id = os.getenv("RUN_ID")
        if online_eval and not run_id and self.mode == "online":
            run_id = wait_for_train_run(self.context.output_dir)
        kwargs = {"id": run_id} if run_id else _registration(self.config.name, config)
        try:
            return pr.init(kind="train", mode=self.mode, base_url=_base_url(), finish_timeout=FINISH_TIMEOUT, **kwargs)
        except (pr.ConfigurationError, pr.APIError) as e:
            # Registering fails without a team, or for a team outside the external-runs
            # allowlist. An attached run's error is the launcher's to explain.
            if run_id:
                raise
            raise RuntimeError(
                f"Failed to register the Prime run ({type(e).__name__}: {e}). Select a team with "
                f"`prime switch` (or set PRIME_TEAM_ID), or {OFF_HINT}"
            ) from e

    def _record_run(self) -> None:
        output_dir = self.context.output_dir
        register_links(output_dir, PRIME_LINKS, 1 if self.run.url else 0)
        if self.run.url:
            update_platform_record(
                output_dir,
                {"kind": "train", "id": self.run.id, "url": self.run.url, "attempt": os.getenv(ATTEMPT_VAR)},
            )
            publish_link(output_dir, PRIME_LINKS, "run", self.run.url)

    async def _init_evaluations(self, config: EvalConfig) -> None:
        self.model = config.model
        self.sources = {source.resolved_name: source for source in config.source}
        self.evaluation_id = os.getenv(EVAL_ID_VAR)
        if self.evaluation_id and len(self.sources) != 1:
            raise ValueError(
                f"${EVAL_ID_VAR} names one Prime evaluation, so the run needs "
                f"exactly one eval source (got {len(self.sources)})"
            )
        online = self.mode == "online"
        self.logger.info("Streaming eval epochs to Prime" if online else "Prime evaluations disabled")

        def record() -> None:
            if online:
                write_platform_record(self.context.output_dir, {"kind": "eval", "evaluations": {}})
            register_links(self.context.output_dir, PRIME_LINKS, len(self.sources) if online else 0)

        await asyncio.to_thread(record)

    async def log_metrics(self, metrics: dict[str, Any], step: int | None) -> None:
        """Onto the training run. A standalone eval's evaluations carry their own
        aggregates through ``log_eval_epoch``."""
        if self.run is None:
            return
        # The SDK also drops non-finite values, but silently; sanitize first so the
        # dropped paths are named in the log.
        metrics, dropped = sanitize(metrics)
        if dropped:
            self.logger.warning(f"Dropping {len(dropped)} non-finite metric value(s): {', '.join(dropped[:5])}")
        # A queue put that can block briefly under backpressure - off the loop. The SDK
        # stamps `_timestamp` on every row, so step=None rows keep a time anchor.
        await asyncio.to_thread(self.run.log_metrics, metrics, step=step)

    async def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind) -> None:
        """Every episode ships whole as it arrives, keyed by the SDK to its work type and
        dispatch step off ``run.work``."""
        if self.run is not None:
            # A queue put that can block briefly under backpressure - off the loop.
            await asyncio.to_thread(self.run.log_episodes, episodes)
            return
        by_env: dict[str, list[vf.Episode]] = {}
        for episode in episodes:
            if episode.env.name is not None:
                by_env.setdefault(episode.env.name, []).append(episode)
        for env_name, batch in by_env.items():
            evaluation = await self._evaluation(env_name, step)
            if evaluation is not None:
                await asyncio.to_thread(evaluation.log_episodes, batch)

    def _open_evaluation(self, env_name: str, step: int, expected: int | None) -> pr.Run:
        """Open the Prime evaluation of one epoch. Blocking: runs in a worker thread."""
        common = dict(kind="eval", mode=self.mode, base_url=_base_url(), finish_timeout=FINISH_TIMEOUT)
        if self.evaluation_id:
            # A hosted launch pre-created the evaluation and injected its id.
            if self.evaluations:
                raise RuntimeError(f"${EVAL_ID_VAR} holds one epoch, and it already took one")
            return pr.init(id=self.evaluation_id, **common)
        source = self.sources[env_name]
        return pr.init(
            name=self.config.name if len(self.sources) == 1 else f"{self.config.name}--{env_name}",
            environments=[source.env.taskset.id],
            model=self.model,
            framework="prime-rl",
            config={
                "model": self.model,
                "step": step,
                "run_id": os.getenv("PRL_RUN_ID"),
                "num_examples": expected // source.group_size if expected else None,
                "rollouts_per_example": source.group_size,
            },
            **common,
        )

    async def _evaluation(self, env_name: str, step: int, expected: int | None = None) -> pr.Run | None:
        """The epoch's evaluation, opened on first use."""
        key = (env_name, step)
        if key in self.evaluations:
            return self.evaluations[key]
        try:
            evaluation = await asyncio.to_thread(self._open_evaluation, env_name, step, expected)
        except Exception as e:
            self.logger.warning(f"Failed to open the {env_name} (Step {step}) evaluation: {type(e).__name__}: {e}")
            evaluation = None
        if key in self.evaluations:  # another arrival opened it meanwhile
            return self.evaluations[key]
        self.evaluations[key] = evaluation
        if evaluation is not None and evaluation.url:
            attached = f" (attached via ${EVAL_ID_VAR})" if evaluation.attached else ""
            self.logger.info(f"Streaming {env_name} (Step {step}) evaluation - {evaluation.url}{attached}")
        await asyncio.to_thread(self._record_evaluation, env_name, step, evaluation)
        return evaluation

    def _record_evaluation(self, env_name: str, step: int, evaluation: pr.Run | None) -> None:
        url = evaluation.url if evaluation is not None else None
        if url:

            def update(record: dict[str, Any]) -> dict[str, Any]:
                record.setdefault("evaluations", {})[env_name] = {"step": step, "id": evaluation.id, "url": url}
                return record

            _merge_platform_record(self.context.output_dir, update)
        publish_link(self.context.output_dir, PRIME_LINKS, env_name, url)

    async def log_eval_plan(self, env_name: str, step: int, expected: int) -> None:
        if self.run is None:
            await self._evaluation(env_name, step, expected)

    async def log_eval_epoch(self, env_name: str, step: int, episodes: list[vf.Episode]) -> None:
        if self.run is not None:
            return
        evaluation = await self._evaluation(env_name, step)
        if evaluation is None:
            return
        try:
            await asyncio.to_thread(evaluation.finish, pr.metrics.from_episodes(episodes))
        except Exception as e:
            self.logger.warning(f"Failed to finish the {env_name} (Step {step}) evaluation: {type(e).__name__}: {e}")
            return
        self.logger.info(f"Uploaded {env_name} (Step {step}) evaluation - {evaluation.url}")

    async def finalize(self, paused: bool = False) -> None:
        # An epoch the run did not finish leaves its evaluation open: close it as cancelled.
        for (env_name, step), evaluation in self.evaluations.items():
            if evaluation is None or evaluation.finished:
                continue
            try:
                await asyncio.to_thread(evaluation.finish, status=pr.RunStatus.CANCELLED, error="interrupted")
            except Exception as e:
                self.logger.warning(f"Failed to close the {env_name} (Step {step}) evaluation: {type(e).__name__}: {e}")
        if self.run is None:
            return
        producer = self.context.producer
        if paused or os.getenv(RUN_FINISHER_VAR, producer) != producer:
            # Drain only: a paused run continues on resume, and a shared run is finished
            # by the process that outlives this one. A clean exit is not a crash.
            await asyncio.to_thread(self.run.flush, FINISH_TIMEOUT)
            atexit.unregister(self.run._atexit_hook)
            return
        await asyncio.to_thread(self.run.finish)
