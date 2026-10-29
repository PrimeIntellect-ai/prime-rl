from __future__ import annotations

import asyncio
import atexit
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import prime_runs as pr
from prime_runs.config import Config

from prime_rl.configs.eval import EvalConfig, SFTOnlineEvalConfig
from prime_rl.configs.monitors import PrimeMonitorConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.links import publish_link, register_links
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_platform_run_path
from prime_rl.utils.record import update_record
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    import verifiers.v1 as vf

BASE_URL_VAR = "PRIME_API_BASE"
EVAL_ID_VAR = "PRIME_RUNS_EVAL_ID"
# How long finish() and the SDK's atexit crash hook let queued uploads drain. The SDK
# default (300 s) is sized for eval sample batches; a crashed training process should
# not linger that long, and a clean finish rarely has more than the last step queued.
FINISH_TIMEOUT = 60.0


OFF_HINT = "turn the monitor off with --no-monitors.prime"
NO_KEY = (
    "No Prime API key found - the run is not logged to Prime. Authenticate with "
    f"`prime login` (or set PRIME_API_KEY), or {OFF_HINT}"
)


def _base_url() -> str | None:
    """$PRIME_API_BASE historically points at the RFT API root (``.../api/v1/rft``);
    the SDK takes the platform base URL. Unset means the SDK resolves it."""
    base = os.getenv(BASE_URL_VAR)
    return base.rstrip("/").removesuffix("/rft") if base else None


def register(
    config: RLConfig | OrchestratorConfig | SFTConfig | None, name: str | None, mode: str, id: str | None = None
) -> pr.Run:
    """Attach to the platform run ``id``, or register ``config``'s training run with the
    whole config: an ``rl`` launch's covers the trainer and inference too."""
    init_kwargs: dict[str, Any]
    if id is not None:
        init_kwargs = {"id": id}
    elif config is None:
        # The RFT API requires a base model.
        init_kwargs = {"name": name, "model": "unknown"}
    else:
        if isinstance(config, SFTConfig):
            model = config.model.name
            # SFT trains on a dataset, not on environments, and has no rollouts.
            environments = []
            training = pr.TrainingSpec(
                max_steps=config.max_steps or 0,
                batch_size=config.data.batch_size,
                seq_len=config.data.seq_len,
                wandb_project=config.monitors.wandb.project if config.monitors.wandb else None,
            )
        else:
            orchestrator = config.orchestrator if isinstance(config, RLConfig) else config
            model = orchestrator.model.name
            environments = [env.env_id for env in orchestrator.train.source]
            training = pr.TrainingSpec(
                max_steps=orchestrator.max_steps or 0,
                batch_size=orchestrator.batch_size,
                rollouts_per_example=max(
                    (source.group_size for source in orchestrator.train.source), default=orchestrator.train.group_size
                ),
                seq_len=orchestrator.seq_len,
                wandb_project=orchestrator.monitors.wandb.project if orchestrator.monitors.wandb else None,
            )
        init_kwargs = dict(
            name=name,
            model=model,
            environments=environments,
            training=training,
            config=config.model_dump(exclude_none=True, mode="json"),
        )
    try:
        return pr.init(kind="train", mode=mode, base_url=_base_url(), finish_timeout=FINISH_TIMEOUT, **init_kwargs)
    except (pr.ConfigurationError, pr.APIError) as e:
        # An attached run's error is the launcher's to explain.
        if id is not None:
            raise
        if "needs an API key" in str(e):
            get_logger().warning(NO_KEY)
            return pr.init(kind="train", mode="disabled", finish_timeout=FINISH_TIMEOUT, **init_kwargs)
        # Registering fails without a team, or for a team outside the external-runs allowlist.
        raise RuntimeError(
            f"Failed to register the Prime run ({type(e).__name__}: {e}). Select a team with "
            f"`prime switch` (or set PRIME_TEAM_ID), or {OFF_HINT}"
        ) from e


@contextmanager
def launch_run(config: RLConfig | SFTConfig, prime: PrimeMonitorConfig | None) -> Iterator[None]:
    """Register the launch's Prime run around starting its components, which attach to
    it: its id becomes ``$PRL_RUN_ID`` and ``$RUN_ID``. A launch that fails marks the
    run failed, and one that ends cleanly finishes it once every component exited - the
    components only flush their attached run. So the process that registers the run
    supervises the launch: the local ``rl`` or ``sft`` launcher, or a multi-node SLURM
    job's ``run_owner``. A run a managed launch pre-created (``$RUN_ID``) is the platform's.
    Without a Prime API key, the components run the monitor disabled."""
    if prime is None or os.getenv("RUN_ID") or (os.getenv(pr.MODE_ENV) or "online") != "online":
        yield
        return
    run = register(config, prime.name, "online")
    if run.mode == "disabled":
        # register already warned; the components need not warn again each.
        os.environ[pr.MODE_ENV] = "disabled"
        atexit.unregister(run._atexit_hook)
        yield
        return
    os.environ["PRL_RUN_ID"] = os.environ["RUN_ID"] = run.id
    try:
        yield
    except BaseException as e:
        # A failed component exits the launcher with a SystemExit chained to its error.
        run.fail(e.__cause__ or e, timeout=FINISH_TIMEOUT)
        raise
    run.finish(timeout=FINISH_TIMEOUT)


class PrimeMonitor(Monitor):
    """Logs metrics and episodes to the Prime platform through ``prime_runs``.

    A training run's processes log to one Prime training run: the ``rl`` or ``sft``
    launcher registers it (``launch_run``) and its components attach through ``$RUN_ID``.
    The process that registered a run finishes it; an attached process only flushes, and
    a component started without a launcher registers and finishes its own. A standalone eval instead opens one evaluation per env and epoch when
    the epoch starts (``log_eval_plan``), streams its episodes into it, and finishes
    it with the epoch's aggregates (``log_eval_epoch``).

    The run handle owns the RFT lifecycle (register or attach, finalize), the
    per-step metrics POSTs, the episode uploads to Prime Traces, and the
    terminal status. Delivery is BEST-EFFORT: the SDK's atexit hook
    reports a process that exits without finalizing, but an abrupt kill
    (SIGKILL, OOM, node loss) skips the hook and a lost finalize is not
    retried here — the platform owns the terminal state of attached
    ($RUN_ID) and managed runs. `FINISH_TIMEOUT` bounds the queued-upload
    drain, not the SDK's finalize HTTP retries.

    ``init``/``finish`` do network I/O and run in worker threads; the log
    calls are queue puts onto the SDK's uploader thread, which owns retries
    and backpressure, so they never stall the loop.
    """

    config: PrimeMonitorConfig

    async def init(self, config: BaseConfig | None, output_dir: Path, producer: str | None = None) -> None:
        self.output_dir = output_dir
        self.producer = producer
        self.mode = mode = os.getenv(pr.MODE_ENV) or "online"
        self.run: pr.Run | None = None
        # (env, step) -> its open evaluation; None once opening it failed, so the epoch's
        # episodes do not retry the platform on every arrival
        self.runs: dict[tuple[str, int], pr.Run | None] = {}
        if isinstance(config, EvalConfig):
            await self._init_evaluations(config)
            return
        # The SFT online-eval process and the RL trainer log into the run their launcher registers.
        attaches = isinstance(config, (SFTOnlineEvalConfig, TrainerConfig))
        run_id = os.getenv("RUN_ID")
        if attaches and run_id is None and mode == "online":
            raise RuntimeError(f"This process logs to its launcher's run: launch it with `rl` or `sft`, or {OFF_HINT}")
        # The launcher registered the run (or a managed launch pre-created it) - attach
        # instead of registering a duplicate. The registering process owns the attached
        # run's failure marking; a clean finish() completes it.
        train_config = None if attaches else config
        self.run = await asyncio.to_thread(register, train_config, self.config.name, mode, run_id)
        if self.run.url:
            attached = " (attached via $RUN_ID)" if self.run.attached else ""
            self.logger.info(f"Logging metrics and episodes to Prime run {self.run.id} ({self.run.url}){attached}")
        else:
            self.logger.info("Prime run disabled")
        if not attaches:
            # Off the event loop: the flock retry must not stall it.
            await asyncio.to_thread(self._record_run)

    def _record_run(self) -> None:
        register_links(self.output_dir, "prime", 1 if self.run.url else 0)
        if self.run.url:
            record = {"kind": "train", "id": self.run.id, "url": self.run.url}
            update_record(get_platform_run_path(self.output_dir), lambda _: record)
            publish_link(self.output_dir, "prime", "run", self.run.url)

    async def log_metrics(self, metrics: dict[str, Any], step: int | None) -> None:
        # A standalone eval's evaluations carry their own aggregates (``log_eval_epoch``).
        if self.run is None:
            return
        # The SDK also drops non-finite values, but silently; sanitize first so
        # the dropped paths are named in the log.
        metrics, dropped = sanitize(metrics)
        if dropped:
            self.logger.warning(f"Dropping {len(dropped)} non-finite metric value(s): {', '.join(dropped[:5])}")
        # A queue put that can block briefly under backpressure - off the loop. The SDK
        # stamps `_timestamp` on every row, so step=None rows keep a time anchor.
        await asyncio.to_thread(self.run.log_metrics, metrics, step=step)

    async def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None:
        """Every episode ships whole as it arrives (the ``all`` subset), errored and
        filtered ones included, online evals too. The SDK keys each to its dispatch
        step and work type off ``run.work`` - the ``TrainRunInfo`` the dispatcher
        stamps at emit time."""
        if subset != "all" or not episodes:
            return
        if self.run is not None:
            # A queue put that can block briefly under backpressure - off the loop.
            await asyncio.to_thread(self.run.log_episodes, episodes)
            return
        by_env: dict[str, list[vf.Episode]] = {}
        for episode in episodes:
            if episode.env.name is not None:
                by_env.setdefault(episode.env.name, []).append(episode)
        for env_name, batch in by_env.items():
            run = await self.run_for(env_name, step)
            if run is not None:
                await asyncio.to_thread(run.log_episodes, batch)  # a queue put, off the loop

    async def finalize(self, paused: bool = False) -> None:
        # An epoch the run did not finish leaves its evaluation open: close it as cancelled.
        for (env_name, step), run in self.runs.items():
            if run is None or run.finished:
                continue
            try:
                await asyncio.to_thread(run.finish, status=pr.RunStatus.CANCELLED, error="interrupted")
            except Exception as e:
                self.logger.warning(f"Failed to close the {env_name} (Step {step}) evaluation: {type(e).__name__}: {e}")
        if self.run is None:
            return
        if paused or self.run.attached:
            # Drain only: a paused run continues on resume, and an attached run is
            # finished by the process that registered it, once every component exited.
            # A clean exit is not a crash.
            await asyncio.to_thread(self.run.flush, FINISH_TIMEOUT)
            atexit.unregister(self.run._atexit_hook)
            return
        # Drains queued uploads so the final step's metrics and episodes land, then finalizes.
        await asyncio.to_thread(self.run.finish)

    async def _init_evaluations(self, config: EvalConfig) -> None:
        self.model: str = config.model
        self.sources = {source.resolved_name: source for source in config.source}
        self.run_id = os.getenv("PRL_RUN_ID")
        self._lock = asyncio.Lock()
        self.evaluation_id = os.getenv(EVAL_ID_VAR)
        if self.evaluation_id and len(self.sources) != 1:
            raise ValueError(
                f"${EVAL_ID_VAR} names one platform evaluation, so the run needs "
                f"exactly one eval source (got {len(self.sources)})"
            )
        online = self.mode == "online"
        # Evaluations open lazily, when their epoch starts; decide at launch instead.
        if online and not Config().api_key:
            self.logger.warning(NO_KEY)
            self.mode, online = "disabled", False
        self.logger.info("Streaming eval epochs to the Prime platform" if online else "Prime evaluations disabled")

        def record() -> None:
            if online:
                record = {"kind": "eval", "run_id": self.run_id, "evaluations": {}}
                update_record(get_platform_run_path(self.output_dir), lambda _: record)
            register_links(self.output_dir, "prime", len(self.sources) if online else 0)

        # Off the event loop: the flock retry must not stall it.
        await asyncio.to_thread(record)

    def open(self, env_name: str, step: int, expected: int | None) -> pr.Run:
        """Open the platform evaluation of one epoch. Blocking: runs in a worker thread."""
        if self.evaluation_id:
            # A hosted launch pre-created the platform evaluation and injected its id -
            # attach instead of registering a duplicate. The backend owns its failure
            # marking then; a clean finish still completes it.
            if self.runs:
                raise RuntimeError(f"${EVAL_ID_VAR} holds one epoch, and it already took one")
            return pr.init(
                kind="eval",
                mode=self.mode,
                base_url=_base_url(),
                id=self.evaluation_id,
                finish_timeout=FINISH_TIMEOUT,
            )
        source = self.sources[env_name]
        name = self.config.name if len(self.sources) == 1 else f"{self.config.name}--{env_name}"
        return pr.init(
            kind="eval",
            mode=self.mode,
            base_url=_base_url(),
            name=name,
            environments=[source.env.taskset.id],
            model=self.model,
            framework="prime-rl",
            finish_timeout=FINISH_TIMEOUT,
            config={
                "model": self.model,
                "step": step,
                "run_id": self.run_id,
                "num_examples": expected // source.group_size if expected else None,
                "rollouts_per_example": source.group_size,
            },
        )

    async def run_for(self, env_name: str, step: int, expected: int | None = None) -> pr.Run | None:
        """The epoch's evaluation, opened on first use."""
        key = (env_name, step)
        async with self._lock:
            if key in self.runs:
                return self.runs[key]
            try:
                run = await asyncio.to_thread(self.open, env_name, step, expected)
            except Exception as e:
                self.logger.warning(f"Failed to open the {env_name} (Step {step}) evaluation: {type(e).__name__}: {e}")
                self.runs[key] = None
                await asyncio.to_thread(publish_link, self.output_dir, "prime", env_name, None)
                return None
            self.runs[key] = run
        if run.url:
            attached = f" (attached via ${EVAL_ID_VAR})" if run.attached else ""
            self.logger.info(f"Streaming {env_name} (Step {step}) evaluation - {run.url}{attached}")
        # Off the event loop: the flock retry must not stall it.
        await asyncio.to_thread(self._record_evaluation, env_name, step, run)
        return run

    def _record_evaluation(self, env_name: str, step: int, run: pr.Run) -> None:
        if run.url:

            def update(record: dict[str, Any]) -> dict[str, Any]:
                record.setdefault("kind", "eval")
                record.setdefault("run_id", self.run_id)
                record.setdefault("evaluations", {})[env_name] = {"step": step, "id": run.id, "url": run.url}
                return record

            update_record(get_platform_run_path(self.output_dir), update)
        publish_link(self.output_dir, "prime", env_name, run.url)

    async def log_eval_plan(self, env_name: str, step: int, expected: int) -> None:
        if self.run is None:
            await self.run_for(env_name, step, expected)

    async def log_eval_epoch(self, env_name: str, step: int, episodes: list[vf.Episode]) -> None:
        if self.run is not None:
            return
        run = await self.run_for(env_name, step)
        if run is None:
            return
        try:
            await asyncio.to_thread(run.finish, pr.metrics.from_episodes(episodes))
        except Exception as e:
            self.logger.warning(f"Failed to finish the {env_name} (Step {step}) evaluation: {type(e).__name__}: {e}")
            return
        self.logger.info(f"Uploaded {env_name} (Step {step}) evaluation - {run.url}")
