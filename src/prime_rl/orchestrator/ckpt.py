"""Checkpoint manager for orchestrator progress and train-source state, and with
``ckpt.rollouts`` for the pipeline as well: the finished but unshipped train episodes
and the train rollouts in flight. Layout under ``<output_dir>/checkpoints``::

    step_N/orchestrator/progress.pt                 progress, train source, configs; written last
    step_N/orchestrator/rollouts/open_work.pkl      the unshipped episodes and the open groups
    step_N/orchestrator/rollouts/<trace id>.restore an in-flight rollout's restore point (env workers write it)
    step_N/orchestrator/rollouts/arrays/<blob id>   the training arrays of the unshipped episodes
    rollout_snapshots.json                          the box snapshots each step's restore points hold
    resumed_arrays/<blob id>                        the arrays of the checkpoint this run resumed from
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import gc
import json
import os
import pickle
import shutil
import signal
import tempfile
import time
import traceback
import warnings
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import msgspec
import torch
import verifiers.v1 as vf
from verifiers.v1.delta import dump

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.metrics import TrainEpisodes
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import DispatchFailure, DispatchResult, GroupCancellation, Progress, TrainBatch
from prime_rl.transports.batch import TrainingSample
from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY
from prime_rl.transports.payload import link_or_copy, persist_payloads
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path

ROLLOUTS_DIR = "rollouts"
ARRAYS_DIR = "arrays"
OPEN_WORK_FILE = "open_work.pkl"
RESUMED_ARRAYS_DIR = "resumed_arrays"
SNAPSHOTS_FILE = "rollout_snapshots.json"
PERSISTED_URL = "checkpoint"
"""The URL of a training array in a written checkpoint: the checkpoint's own arrays,
wherever a resumed run reads it from (``read_open_work``)."""
WRITER_TIMEOUT_SECONDS = 1800.0


class CheckpointManager:
    def __init__(self, output_dir: Path, config: CheckpointConfig) -> None:
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.snapshots = SnapshotLedger(self.ckpt_dir / SNAPSHOTS_FILE)
        self.rollout_stats: dict[str, float] = {}
        """Counts, seconds and bytes of the last rollout checkpoint (``gauges``)."""

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def save(self, progress: Progress, train_source: TrainSource, config_state: dict, step: int) -> None:
        self.save_state(build_state(progress, train_source, config_state), step)

    def save_state(self, state: dict, step: int) -> None:
        """Save a ``build_state`` snapshot as the checkpoint of ``step``."""
        ckpt_path = self.get_ckpt_path(step)
        start = time.perf_counter()
        write_state(state, ckpt_path)
        get_logger().debug(
            f"Orchestrator checkpoint saved to {ckpt_path} in {format_time(time.perf_counter() - start)}"
        )

    def clear(self, step: int) -> None:
        """Remove what an earlier attempt wrote for ``step``, so the step counts as a
        checkpoint again only once ``fork_writer``'s child finished."""
        ckpt_path = self.get_ckpt_path(step)
        (ckpt_path / "progress.pt").unlink(missing_ok=True)
        shutil.rmtree(ckpt_path / ROLLOUTS_DIR, ignore_errors=True)

    def fork_writer(self, state: dict, work: dict, step: int, freeze: bool) -> int:
        """Write the rollout checkpoint of ``step`` from a forked child and return its
        pid (``wait_for_writer``). The child holds the process's memory as of the fork,
        so ``state`` and ``work`` need no copy and the caller goes on changing what
        they reference.

        ``freeze`` keeps the collector off every object alive now until the caller
        unfreezes: a collection in this process would otherwise write to the pages it
        shares with the child, and so copy them."""
        if freeze:
            gc.freeze()
        with warnings.catch_warnings():
            # The child runs no other thread, logs nothing, and leaves with os._exit.
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:
            code = 1
            try:
                gc.disable()
                ckpt_path = self.get_ckpt_path(step)
                write_open_work(work, ckpt_path / ROLLOUTS_DIR)
                write_state({**state, "rollouts": True}, ckpt_path)
                code = 0
            except BaseException:
                traceback.print_exc()
            finally:
                os._exit(code)
        return pid

    def read_state(self, step: int, path: Path | None = None) -> dict:
        """``path`` overrides where the checkpoint is read from (an external run's
        ``step_<N>/orchestrator``)."""
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        state_file = ckpt_path / "progress.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"Orchestrator checkpoint not found at {state_file}")
        get_logger().debug(f"Loading checkpoint from {state_file}")
        with open(state_file, "rb") as f:
            return torch.load(f, weights_only=False)

    def read_config_state(self, step: int, path: Path | None = None) -> dict | None:
        """The checkpoint's ``{base, applied, revision}`` configs, None for checkpoints
        written before configs were checkpointed."""
        return self.read_state(step, path).get("config")

    def load(self, progress: Progress, train_source: TrainSource, step: int, path: Path | None = None) -> dict | None:
        """Load progress and the train source. Returns the checkpoint's open work
        (``Orchestrator.open_work``) when it has one and this run checkpoints rollouts;
        the train source then re-runs no open group, since the open work continues
        them."""
        start = time.perf_counter()
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        state = self.read_state(step, path)
        saved: Progress = state["progress"]
        for key, value in asdict(saved).items():
            if hasattr(progress, key):
                setattr(progress, key, value)
        work = None
        if state.get("rollouts") and self.config.rollouts:
            work = self.load_open_work(ckpt_path / ROLLOUTS_DIR, owned=path is None)
        elif state.get("rollouts"):
            get_logger().info("The checkpoint holds rollouts, but ckpt.rollouts is off: its open groups re-run whole")
        train_source.load_state_dict(state["train_source"], reopen=work is None)
        get_logger().debug(f"Orchestrator checkpoint loaded in {format_time(time.perf_counter() - start)}")
        return work

    def load_open_work(self, rollouts: Path, owned: bool) -> dict:
        """Read the open work in ``rollouts``. Its training arrays are linked into this
        run's ``resumed_arrays``: the run reads them until the episodes ship, which can
        be after the trainer's cleanup removed the checkpoint. Without ``owned`` the
        checkpoint is another run's, which keeps its restore points' snapshots."""
        arrays = self.ckpt_dir / RESUMED_ARRAYS_DIR
        shutil.rmtree(arrays, ignore_errors=True)
        arrays.mkdir(parents=True)
        if (rollouts / ARRAYS_DIR).exists():
            for file in (rollouts / ARRAYS_DIR).iterdir():
                link_or_copy(file, arrays / file.name)
        # Reading makes many objects and frees none: a collection would only scan them.
        collecting = gc.isenabled()
        gc.disable()
        try:
            work = read_open_work(rollouts, arrays.resolve().as_uri())
        finally:
            if collecting:
                gc.enable()
        for episode in work["episodes"]:
            episode.owned = episode.owned and owned
            if episode.restore is not None:
                # The checkpoint may have moved since the env workers wrote the path.
                episode.restore.path = str(rollouts.resolve() / Path(episode.restore.path).name)
        return work

    def gauges(self) -> dict[str, float]:
        return {f"ckpt/rollouts/{name}": value for name, value in self.rollout_stats.items()}


def write_state(state: dict, ckpt_path: Path) -> None:
    """Write ``state`` to ``<ckpt_path>/progress.pt``."""
    ckpt_path.mkdir(parents=True, exist_ok=True)
    # Save to a temporary file and do an atomic rename, to avoid corrupting the last
    # file if the process gets killed while writing
    fd, tmp_name = tempfile.mkstemp(dir=ckpt_path, prefix="progress.pt.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
        os.replace(tmp_name, ckpt_path / "progress.pt")
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def build_state(progress: Progress, train_source: TrainSource, config_state: dict) -> dict:
    """A snapshot of the orchestrator's checkpointed state, independent of later changes.
    ``config_state`` holds the launch config (``base``), the config that applies from the
    resume step on (``applied``) and its ``revision``, all dumped to JSON-ready dicts."""
    return copy.deepcopy({"progress": progress, "train_source": train_source.state_dict(), "config": config_state})


async def wait_for_writer(pid: int, timeout: float = WRITER_TIMEOUT_SECONDS) -> float:
    """Wait for ``fork_writer``'s child without blocking the loop, and return the CPU
    seconds it used. Raises if the child fails or takes longer than ``timeout``."""
    deadline = time.monotonic() + timeout
    while True:
        done, status, usage = os.wait4(pid, os.WNOHANG)
        if done:
            break
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise RuntimeError(f"Orchestrator checkpoint writer did not finish within {timeout:.0f}s")
        await asyncio.sleep(0.05)
    if (code := os.waitstatus_to_exitcode(status)) != 0:
        raise RuntimeError(f"Orchestrator checkpoint writer exited with {code}")
    return usage.ru_utime + usage.ru_stime


def is_episode(result: DispatchResult) -> bool:
    return not isinstance(result, DispatchFailure | GroupCancellation)


def is_train_result(result: DispatchResult) -> bool:
    if is_episode(result):
        return result.run is not None and result.run.work.type == "train"
    return result.kind == "train"


def result_env_name(result: DispatchResult) -> str:
    return result.env.name if is_episode(result) else result.env_name


def count_open_work(work: dict) -> dict[str, int]:
    """How many episodes ``work`` holds: finished ones waiting to ship (``in_queue``),
    and in-flight ones with a restore point (``restorable``) and without (``restart``)."""
    sink = work["sink"]
    in_queue = sum(len(group) for group in sink["pending_groups"].values()) + len(sink["episode_by_trace"])
    in_queue += sum(is_episode(result) for result in work["results"])
    if work["batch"] is not None:
        in_queue += len(work["batch"].cohort.episodes)
    restorable = sum(episode.restore is not None for episode in work["episodes"])
    return {"in_queue": in_queue, "restorable": restorable, "restart": len(work["episodes"]) - restorable}


def owned_snapshots(work: dict) -> list[dict]:
    """The box snapshots the checkpoint of ``work`` restores from and this run deletes."""
    return [e.restore.snapshot for e in work["episodes"] if e.restore is not None and e.owned]


def payload_refs(work: dict) -> dict[str, set[str]]:
    """The training arrays ``work`` still needs, as blob ids by the URL that serves
    them: every array of an episode no group processed yet, and the arrays of the
    samples queued for a batch (the sink already deleted the others of their traces)."""
    refs: dict[str, set[str]] = defaultdict(set)
    sink = work["sink"]
    episodes = [episode for group in sink["pending_groups"].values() for episode in group]
    episodes += [result for result in work["results"] if is_episode(result)]
    for episode in episodes:
        for trace in episode.traces:
            if payload := trace.info.get(PAYLOAD_INFO_KEY):
                refs[payload["url"]].update(reference["id"] for reference in payload["branches"].values())
    samples = [sample for samples in sink["pending_batch"].values() for sample in samples]
    if work["batch"] is not None:
        samples += work["batch"].samples
    for sample in samples:
        if sample.payload is not None:
            refs[sample.payload.url].add(sample.payload.id)
    return refs


async def persist_open_work(work: dict, rollouts: Path) -> tuple[int, int]:
    """Bring what ``work`` references into ``rollouts``, the directory its checkpoint
    is written to: the env workers write the training arrays they keep in memory, the
    arrays of an earlier checkpoint are linked, and so is the restore file of an
    episode this run resumed and has not dispatched yet (if that file is gone, the
    episode starts over). Returns how many arrays ``work`` needs and how many were
    written."""
    rollouts.mkdir(parents=True, exist_ok=True)
    refs = payload_refs(work)
    urls = sorted(refs)
    written = await asyncio.gather(
        *(asyncio.to_thread(persist_payloads, url, rollouts / ARRAYS_DIR, sorted(refs[url])) for url in urls)
    )
    for episode in work["episodes"]:
        if episode.restore is None:
            continue
        source = Path(episode.restore.path)
        target = rollouts / source.name
        if await asyncio.to_thread(link_or_copy, source, target):
            episode.restore.path = str(target)
        else:
            episode.restore = None
    return sum(map(len, refs.values())), sum(map(len, written))


def _dump_episode(episode: vf.Episode) -> dict:
    """An episode as the checkpoint file holds it: its nodes' token streams as raw bytes,
    which pickle without a Python object per token. A served episode's type is a
    parametrized generic, which pickle cannot find by name, so the file holds it dumped."""
    dumped = dump(episode, context={"token_arrays": True})
    for trace in dumped["traces"]:
        if payload := trace["info"].get(PAYLOAD_INFO_KEY):
            trace["info"][PAYLOAD_INFO_KEY] = {**payload, "url": PERSISTED_URL}
    return dumped


def _dump_sample(sample: TrainingSample) -> TrainingSample:
    if sample.payload is None:
        return sample
    return msgspec.structs.replace(sample, payload=msgspec.structs.replace(sample.payload, url=PERSISTED_URL))


def _load_episode(dumped: dict, arrays_url: str) -> vf.WireEpisode:
    for trace in dumped["traces"]:
        payload = trace["info"].get(PAYLOAD_INFO_KEY)
        if payload and payload["url"] == PERSISTED_URL:
            payload["url"] = arrays_url
    return vf.WireEpisode.model_validate(dumped)


def _load_sample(sample: TrainingSample, arrays_url: str) -> TrainingSample:
    if sample.payload is not None and sample.payload.url == PERSISTED_URL:
        sample.payload.url = arrays_url
    return sample


def _map_open_work(work: dict, episode: Callable[[Any], Any], sample: Callable[[Any], Any], batch: Callable) -> dict:
    """``work`` with ``episode`` applied to each finished episode, ``sample`` to each
    queued sample and ``batch`` to the batch that was cut but not shipped."""
    sink = work["sink"]
    return {
        **work,
        "sink": {
            **sink,
            "pending_groups": {gid: [episode(e) for e in group] for gid, group in sink["pending_groups"].items()},
            "episode_by_trace": {trace_id: episode(e) for trace_id, e in sink["episode_by_trace"].items()},
            "pending_batch": {
                trace_id: [sample(s) for s in samples] for trace_id, samples in sink["pending_batch"].items()
            },
        },
        "results": [episode(r) if is_episode(r) else r for r in work["results"]],
        "batch": batch(work["batch"]) if work["batch"] is not None else None,
    }


def write_open_work(work: dict, directory: Path) -> None:
    """Write ``work`` (``Orchestrator.open_work``) to ``<directory>/open_work.pkl``.
    Every training array in it points at ``PERSISTED_URL``, also one that
    ``persist_open_work`` could not write: the env worker that serves it now is gone
    when a run resumes. ``work`` itself is left as it is."""

    def dump_batch(batch: TrainBatch) -> dict:
        return {
            "episodes": [_dump_episode(episode) for episode in batch.cohort.episodes],
            "sampled_trace_ids": sorted(batch.cohort.sampled_trace_ids),
            "samples": [_dump_sample(sample) for sample in batch.samples],
        }

    dumped = _map_open_work(work, _dump_episode, _dump_sample, dump_batch)
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f"{OPEN_WORK_FILE}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(dumped, f, protocol=5)
        os.replace(tmp_name, directory / OPEN_WORK_FILE)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def read_open_work(directory: Path, arrays_url: str) -> dict:
    """The open work ``write_open_work`` wrote to ``directory``, with its training
    arrays read from ``arrays_url``."""
    with open(directory / OPEN_WORK_FILE, "rb") as f:
        dumped = pickle.load(f)

    def load_batch(batch: dict) -> TrainBatch:
        episodes = [_load_episode(episode, arrays_url) for episode in batch["episodes"]]
        return TrainBatch(
            cohort=TrainEpisodes(episodes, sampled_trace_ids=set(batch["sampled_trace_ids"])),
            samples=[_load_sample(sample, arrays_url) for sample in batch["samples"]],
        )

    return _map_open_work(
        dumped,
        lambda episode: _load_episode(episode, arrays_url),
        lambda sample: _load_sample(sample, arrays_url),
        load_batch,
    )


def directory_size(directory: Path) -> int:
    return sum(file.stat().st_size for file in directory.rglob("*") if file.is_file())


class SnapshotLedger:
    """The box snapshots this run owns: those its rollout checkpoints restore from, by
    checkpoint step, and ``orphans`` that nothing restores from. It lives in a file
    under ``checkpoints/``, so a resumed run goes on deleting them. A run only ever
    records snapshots its own env workers took, so it never deletes those of a run it
    was forked from."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.steps: dict[int, list[dict]] = {}
        self.orphans: list[dict] = []
        if path.exists():
            saved = json.loads(path.read_text())
            self.steps = {int(step): snapshots for step, snapshots in saved["steps"].items()}
            self.orphans = saved["orphans"]

    def record(self, step: int, snapshots: list[dict]) -> None:
        """The checkpoint of ``step`` restores from ``snapshots``. What an earlier
        checkpoint of the same step held is orphaned."""
        held = {snapshot["id"] for snapshot in snapshots}
        self.orphans += [snapshot for snapshot in self.steps.get(step, []) if snapshot["id"] not in held]
        self.steps[step] = snapshots
        self.write()

    def orphan(self, snapshots: list[dict]) -> None:
        if snapshots:
            self.orphans += snapshots
            self.write()

    async def sweep(self, ckpt_dir: Path, discard: Callable[[list[dict]], Awaitable[list[str]]]) -> int:
        """Delete, through ``discard``, the snapshots nothing can restore from anymore:
        the orphans, and those of steps whose checkpoint directory is gone (the
        trainer's ``keep_last`` cleanup removes whole step directories), except the ones
        a remaining step also holds. ``discard`` returns the ids it deleted; the rest
        stay for the next sweep. Returns how many were deleted."""
        gone = [step for step in self.steps if not get_step_path(ckpt_dir, step).exists()]
        held = {snapshot["id"] for step, snapshots in self.steps.items() if step not in gone for snapshot in snapshots}
        unheld = [
            snapshot
            for snapshot in [*self.orphans, *(snapshot for step in gone for snapshot in self.steps[step])]
            if snapshot["id"] not in held
        ]
        candidates = list({snapshot["id"]: snapshot for snapshot in unheld}.values())
        deleted = set(await discard(candidates)) if candidates else set()
        left = [snapshot for snapshot in candidates if snapshot["id"] not in deleted]
        if left != self.orphans or gone:
            for step in gone:
                del self.steps[step]
            self.orphans = left
            self.write()
        return len(deleted)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp")
        tmp.write_text(json.dumps({"steps": self.steps, "orphans": self.orphans}))
        tmp.replace(self.path)


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager:
    """The checkpoint manager always exists: ``resume`` decides whether it loads,
    ``ckpt`` whether it saves (a resume without ``ckpt`` loads but saves nothing)."""
    return CheckpointManager(output_dir, config or CheckpointConfig())
