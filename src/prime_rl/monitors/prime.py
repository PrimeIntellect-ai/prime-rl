from __future__ import annotations

import asyncio
import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import orjson
import prime_runs as pr
from prime_sandboxes import Config as PrimeConfig

from prime_rl.configs.monitors import PrimeEvalMonitorConfig, PrimeTrainMonitorConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.monitors.base import Kind, Monitor, Subset
from prime_rl.utils.config import BaseConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_platform_run_path
from prime_rl.utils.utils import sanitize

if TYPE_CHECKING:
    import verifiers.v1 as vf

BASE_URL_VAR = "PRIME_API_BASE"
EVAL_ID_VAR = "PRIME_RUNS_EVAL_ID"
# How long finish() and the SDK's atexit crash hook let queued uploads drain. The SDK
# default (300 s) is sized for eval sample batches; a crashed training process should
# not linger that long, and a clean finish rarely has more than the last step queued.
FINISH_TIMEOUT = 60.0


def write_platform_record(output_dir: Path, record: dict[str, Any]) -> None:
    """Leave the run's platform identity in the run directory for the dashboard's
    "view on platform" link (atomic replace: a reader never sees a torn file).
    The tmp file is pid-suffixed: the trainer and the online-eval process
    share the run directory, and two writers sharing one tmp path can tear
    each other's staged file mid-write."""
    path = get_platform_run_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_platform_record(output_dir: Path) -> dict[str, Any] | None:
    path = get_platform_run_path(output_dir)
    return orjson.loads(path.read_bytes()) if path.is_file() else None


# Bounded wait for the record lock: a blocking flock(LOCK_EX) would stall the
# async event-loop thread for as long as the other writer holds the lock.
RECORD_LOCK_TIMEOUT = 10.0
RECORD_LOCK_POLL = 0.05


@contextmanager
def _platform_record_lock(output_dir: Path) -> Iterator[None]:
    """Cross-process read-modify-write lock for the platform record: the
    trainer and the online-eval process merge into the SAME run.json, and an
    unlocked read-modify-write lets one clobber the other's snapshot. POSIX
    flock on a sibling .lock file. The lock file is NEVER unlinked: a waiting
    writer can still hold the unlinked inode's lock while a new writer
    creates and locks a fresh file — two simultaneous "exclusive" locks and a
    lost merge. A stable lock inode is the whole guarantee; releasing is
    enough.

    Acquisition is BOUNDED (`RECORD_LOCK_TIMEOUT`): contention is retried
    with non-blocking flock instead of blocking the async event-loop thread
    indefinitely; `TimeoutError` past the bound. Storage errors that make
    locking impossible here (e.g. ENOLCK on a filesystem without flock
    support) raise immediately instead of retrying.

    Storage footprint: exactly one empty .lock file next to the run.json it
    guards, for the run directory's whole lifetime — the run-dir lifecycle
    already bounds it (deleting the run dir deletes the lock); no per-write
    files are ever created."""
    path = get_platform_run_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+b") as lock_file:
        deadline = time.monotonic() + RECORD_LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"platform record lock at {lock_path} still held after {RECORD_LOCK_TIMEOUT}s")
                time.sleep(RECORD_LOCK_POLL)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _merge_platform_record(
    output_dir: Path,
    merge: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Best-effort locked read-modify-write of the platform record: run
    `merge` on the current record under the record lock and atomically
    replace the file. The record only backs the dashboard's "view on
    platform" link, so a lock or storage failure (TimeoutError, ENOLCK, ...)
    is logged and SKIPPED, never fatal to training; the dashboard just misses
    the link for that run."""
    try:
        with _platform_record_lock(output_dir):
            write_platform_record(output_dir, merge(read_platform_record(output_dir) or {}))
    except OSError:
        get_logger().warning(
            f"Failed to persist the platform record at {get_platform_run_path(output_dir)} "
            f"(lock or storage error); the dashboard's platform link may be missing. "
            f"Training continues."
        )


def update_platform_record(output_dir: Path, update: dict[str, Any]) -> None:
    """Best-effort locked read-modify-write of the platform record: `update`
    overlays the current record (preserving keys it doesn't set). Use for
    every writer that shares the file — a naive overwrite drops concurrent
    writers' keys."""
    _merge_platform_record(output_dir, lambda record: {**record, **update})


def _base_url() -> str | None:
    """$PRIME_API_BASE historically points at the RFT API root (``.../api/v1/rft``);
    the SDK takes the platform base URL. Unset means the SDK resolves it."""
    base = os.getenv(BASE_URL_VAR)
    return base.rstrip("/").removesuffix("/rft") if base else None


class PrimeTrainMonitor(Monitor):
    """Logs metrics and episodes to the Prime platform through ``prime_runs``.

    The run handle owns what ``TrainRun`` used to do by hand: the RFT
    lifecycle (register or attach, finalize), the per-step metrics POSTs, the
    every-10th-step Parquet sample uploads (presign -> PUT -> confirm), and
    the terminal status. Delivery is BEST-EFFORT: the SDK's atexit hook
    reports a process that exits without finalizing, but an abrupt kill
    (SIGKILL, OOM, node loss) skips the hook and a lost finalize is not
    retried here — the platform owns the terminal state of attached
    ($RUN_ID) and managed runs. `FINISH_TIMEOUT` bounds the queued-upload
    drain, not the SDK's finalize HTTP retries.

    ``init``/``finish`` do network I/O and run in worker threads; the log
    calls are queue puts onto the SDK's uploader thread, which owns retries
    and backpressure, so they never stall the loop.
    """

    config: PrimeTrainMonitorConfig
    run: pr.Run

    async def init(self, config: BaseConfig | None = None, output_dir: Path | None = None) -> None:
        init_kwargs: dict[str, Any]
        if run_id := os.getenv("RUN_ID"):
            # A managed launch pre-created the platform run and injected its id -
            # attach instead of registering a duplicate. The platform owns the
            # attached run's failure/completion marking; a clean finish() is
            # delivered best-effort and does not by itself mark it completed.
            init_kwargs = {"id": run_id}
        elif isinstance(config, SFTConfig):
            # SFT trains on a dataset, not on environments, and has no rollouts - the
            # platform run is registered without both.
            init_kwargs = dict(
                name=self.config.name,
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
            if output_dir is not None:
                # Merge, not overwrite: a concurrent eval process (SFT
                # online evals share the trainer's run dir) may already
                # have written its evaluations record. Locked RMW: the
                # eval process merges into the same file.
                update_platform_record(output_dir, {"kind": "train", "id": self.run.id, "url": self.run.url})
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


class PrimeEvalMonitor(Monitor):
    """Streams each eval epoch to the Prime platform through ``prime_runs``: one
    evaluation per env and step, opened when the epoch starts (``log_eval_plan``), fed
    every episode as it lands (``log_episodes``) and finished with the epoch's aggregates
    (``log_eval_epoch``), so the platform page follows the run from its first episode.
    Metrics are not forwarded; the evaluation is the epoch."""

    config: PrimeEvalMonitorConfig

    async def init(self, config: BaseConfig | None = None, output_dir: Path | None = None) -> None:
        self.mode = os.getenv(pr.MODE_ENV) or "online"
        # A configured monitor must work: the SDK looks the key up when an epoch opens,
        # which is too late to find out there is none.
        if self.mode == "online" and not (os.getenv("PRIME_API_KEY") or PrimeConfig().api_key):
            raise RuntimeError("API key not found - set PRIME_API_KEY or run `prime login`")
        self.model: str = config.model if config is not None else "unknown"
        self.sources = {source.resolved_name: source for source in config.source} if config is not None else {}
        self.run_id = os.getenv("PRL_RUN_ID")
        self.output_dir = output_dir
        # (env, step) -> its open evaluation; None once opening it failed, so the epoch's
        # episodes do not retry the platform on every arrival
        self.runs: dict[tuple[str, int], pr.Run | None] = {}
        self._lock = asyncio.Lock()
        self.evaluation_id = os.getenv(EVAL_ID_VAR)
        if self.evaluation_id and len(self.sources) != 1:
            raise ValueError(
                f"${EVAL_ID_VAR} names one platform evaluation, so the run needs "
                f"exactly one eval source (got {len(self.sources)})"
            )
        if self.mode == "online":
            self.logger.info("Streaming eval epochs to the Prime platform")
            if output_dir is not None:
                # Merge, not overwrite: when the eval process shares the
                # trainer's run dir (SFT online evals), the train record's
                # platform link (kind/id/url) must survive; a standalone
                # eval keeps the plain eval-shaped record. Locked RMW.
                def _eval_init_update(record: dict[str, Any]) -> dict[str, Any]:
                    if record.get("kind") == "train":
                        record.setdefault("evaluations", {})
                        record["run_id"] = self.run_id
                        return record
                    return {"kind": "eval", "run_id": self.run_id, "evaluations": {}}

                _merge_platform_record(output_dir, _eval_init_update)
        else:
            self.logger.info(f"Platform evaluations disabled ({pr.MODE_ENV}=disabled)")

    async def log_metrics(self, metrics: dict[str, Any], step: int | None) -> None:
        pass

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
                return None
            self.runs[key] = run
        if run.url:
            attached = f" (attached via ${EVAL_ID_VAR})" if run.attached else ""
            self.logger.info(f"Streaming {env_name} (Step {step}) evaluation - {run.url}{attached}")
            if self.output_dir is not None:

                def _eval_epoch_update(record: dict[str, Any]) -> dict[str, Any]:
                    record.setdefault("kind", "eval")
                    record.setdefault("run_id", self.run_id)
                    record.setdefault("evaluations", {})[env_name] = {
                        "step": step,
                        "id": run.id,
                        "url": run.url,
                    }
                    return record

                _merge_platform_record(self.output_dir, _eval_epoch_update)
        return run

    async def log_eval_plan(self, env_name: str, step: int, expected: int) -> None:
        await self.run_for(env_name, step, expected)

    async def log_episodes(self, episodes: list[vf.Episode], step: int, kind: Kind, subset: Subset) -> None:
        """Every eval episode as it lands (the ``all`` subset is the arrival stream)."""
        if kind != "eval" or subset != "all":
            return
        by_env: dict[str, list[vf.Episode]] = {}
        for episode in episodes:
            if episode.env.name is not None:
                by_env.setdefault(episode.env.name, []).append(episode)
        for env_name, batch in by_env.items():
            run = await self.run_for(env_name, step)
            if run is not None:
                await asyncio.to_thread(run.log_episodes, batch)  # a queue put, off the loop

    async def log_eval_epoch(self, env_name: str, step: int, episodes: list[vf.Episode]) -> None:
        run = await self.run_for(env_name, step)
        if run is None:
            return
        try:
            await asyncio.to_thread(run.finish, pr.metrics.from_episodes(episodes))
        except Exception as e:
            self.logger.warning(f"Failed to finish the {env_name} (Step {step}) evaluation: {type(e).__name__}: {e}")
            return
        self.logger.info(f"Uploaded {env_name} (Step {step}) evaluation - {run.url}")

    async def finalize(self) -> None:
        # An epoch the run did not finish leaves its evaluation open: close it as cancelled.
        for (env_name, step), run in self.runs.items():
            if run is None or run.finished:
                continue
            try:
                await asyncio.to_thread(run.finish, status=pr.RunStatus.CANCELLED, error="interrupted")
            except Exception as e:
                self.logger.warning(f"Failed to close the {env_name} (Step {step}) evaluation: {type(e).__name__}: {e}")
