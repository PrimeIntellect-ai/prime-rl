"""Checkpoint manager for orchestrator progress, train-source state and the in-flight and
in-queue rollouts. Layout: ``<output_dir>/checkpoints/step_N/orchestrator/progress.pt``,
and next to it ``rollouts/``, a file per in-queue rollout and in-flight restore point
(``write_rollouts``), and ``arrays/``, the training arrays they need (``persist_arrays``)."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import gc
import os
import pickle
import shutil
import tempfile
import time
import traceback
import warnings
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import verifiers.v1 as vf
from verifiers.v1.serve import RestoreRequest
from verifiers.v1.serve.delta import dump

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, Progress
from prime_rl.orchestrator.utils import result_kind
from prime_rl.transports.batch.types import PAYLOAD_ARRAYS_KEY, PAYLOAD_INFO_KEY, PAYLOAD_WORKER_KEY
from prime_rl.transports.payload import persist_payloads
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


class CheckpointManager:
    def __init__(self, output_dir: Path, config: CheckpointConfig) -> None:
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.last_rollouts: Path | None = None
        """The ``rollouts/`` of the last checkpoint written or loaded: a new checkpoint
        links the in-queue rollouts it still holds from there."""

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def save(
        self,
        progress: Progress,
        train_source: TrainSource,
        config_state: dict,
        step: int,
        open_work: dict | None = None,
    ) -> None:
        self.save_state(build_state(progress, train_source, config_state, open_work), step)

    def save_state(self, state: dict, step: int) -> None:
        """Save a ``build_state`` snapshot as the checkpoint of ``step``. It changes the
        snapshot's rollouts, so it runs on a copy (``save_forked``)."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        if state.get("open_work") is not None:
            persist_arrays(state["open_work"], (ckpt_path / ARRAYS_DIR).resolve())
            rollouts = write_rollouts(state["open_work"], ckpt_path / ROLLOUTS_DIR, self.last_rollouts)
            state = {**state, "open_work": rollouts}
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
        get_logger().debug(
            f"Orchestrator checkpoint saved to {ckpt_path} in {format_time(time.perf_counter() - start)}"
        )

    async def save_forked(self, state: dict, step: int) -> None:
        """Save ``state`` from a forked child, which holds the process's memory as of the
        fork: the caller goes on changing it meanwhile, with no copy to take. Raises if
        the child fails."""
        # Frozen objects are not traversed by the collector, so neither process writes
        # to the shared pages just to collect.
        gc.freeze()
        try:
            with warnings.catch_warnings():
                # The child runs no thread, takes no lock another thread may hold, and
                # leaves with os._exit.
                warnings.simplefilter("ignore", DeprecationWarning)
                pid = os.fork()
            if pid == 0:
                code = 1
                try:
                    gc.disable()
                    self.save_state(state, step)
                    code = 0
                except BaseException:
                    traceback.print_exc()
                finally:
                    os._exit(code)
            _, status = await asyncio.to_thread(os.waitpid, pid, 0)
        finally:
            gc.unfreeze()
        if (code := os.waitstatus_to_exitcode(status)) != 0:
            raise RuntimeError(f"Orchestrator checkpoint writer for step {step} exited with {code}")
        if state.get("open_work") is not None:
            self.last_rollouts = self.get_ckpt_path(step) / ROLLOUTS_DIR

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
        """Load progress and the train source, and return the checkpoint's in-flight
        and in-queue rollouts (``build_state``), if it has them."""
        start = time.perf_counter()
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        state = self.read_state(step, path)
        saved: Progress = state["progress"]
        for key, value in asdict(saved).items():
            if hasattr(progress, key):
                setattr(progress, key, value)
        train_source.load_state_dict(state["train_source"])
        for name in state["train_source"]["envs"]:
            if name in train_source.curricula:
                get_logger().info(f"Resumed curriculum state for env {name}")
        get_logger().debug(f"Orchestrator checkpoint loaded in {format_time(time.perf_counter() - start)}")
        work = state.get("open_work")
        if work is not None:
            # The rollouts make many objects at once and free none: a collection while
            # reading them back only scans them.
            gc.disable()
            try:
                work = read_rollouts(work, ckpt_path / ROLLOUTS_DIR)
            finally:
                gc.enable()
            resolve_arrays(work, (ckpt_path / ARRAYS_DIR).resolve())
            self.last_rollouts = ckpt_path / ROLLOUTS_DIR
        return work


def build_state(
    progress: Progress, train_source: TrainSource, config_state: dict, open_work: dict | None = None
) -> dict:
    """The orchestrator's checkpointed state. It holds the live rollouts, not copies:
    write it at once or from a fork (``CheckpointManager.save_forked``). ``config_state`` holds the launch config (``base``), the config that applies from the
    resume step on (``applied``) and its ``revision``, all dumped to JSON-ready dicts.
    ``open_work`` holds the in-flight and in-queue rollouts a resumed run picks up where
    this one left them: the open groups with their in-flight rollouts' restore points,
    and the in-queue ones in the sink's buffers and the results not yet consumed."""
    state = {"progress": progress, "train_source": train_source.state_dict(), "config": config_state}
    if open_work is not None:
        state["open_work"] = open_work
    return state


ARRAYS_DIR = "arrays"
ROLLOUTS_DIR = "rollouts"
PERSISTED_URL = "checkpoint"
"""Where a persisted blob's URL points in the saved state: the checkpoint's own
``arrays/``, wherever a resumed run finds it (``resolve_arrays``)."""


def dump_episode(episode: Any) -> dict:
    """An episode as the checkpoint file holds it: its nodes' token streams as raw bytes,
    which pickle without a Python object per token."""
    return dump(episode, context={"token_arrays": True})


EVAL_PLACES = ("eval-queued", "eval")


def _queued_place(result: Any) -> str:
    """Where a result in ``out_q`` waits; a written one's file name already says it."""
    if isinstance(result, str):
        return result.split(".")[1]
    return "queued" if result_kind(result) == "train" else "eval-queued"


def _map_episodes(work: dict, fn: Callable[[Any, str], Any]) -> dict:
    """``work`` with ``fn(episode, place)`` applied to each of its in-queue episodes, where
    ``place`` is where the episode waits: ``queued`` (``out_q``), ``group`` (a train sink
    group still collecting), ``batch`` (the unshipped batch), ``eval-queued`` (``out_q``)
    or ``eval`` (an eval epoch still collecting)."""
    sink = work["sink"]
    mapped = {
        **work,
        "sink": {
            **sink,
            "pending_groups": {gid: [fn(e, "group") for e in group] for gid, group in sink["pending_groups"].items()},
            "episode_by_trace": {tid: fn(e, "batch") for tid, e in sink["episode_by_trace"].items()},
        },
        "results": [
            r if isinstance(r, DispatchFailure | GroupCancellation) else fn(r, _queued_place(r))
            for r in work["results"]
        ],
    }
    if (eval_work := work.get("eval")) is not None and eval_work["sink"] is not None:
        eval_sink = eval_work["sink"]
        batches = {key: [fn(e, "eval") for e in batch] for key, batch in eval_sink["pending_batches"].items()}
        mapped["eval"] = {**eval_work, "sink": {**eval_sink, "pending_batches": batches}}
    return mapped


def write_rollouts(work: dict, directory: Path, previous: Path | None) -> dict:
    """Write each in-queue episode and in-flight restore point of ``work`` to its own file
    in ``directory``, one at a time, and return ``work`` with the file names in their
    place. An episode does not change while it waits in one place, so a file of that
    name in the ``previous`` checkpoint's directory is linked, not written again. Its
    payload URLs depend on whether this checkpoint persisted its arrays
    (``persist_arrays``), so the name says that too. A served episode's type is a
    parametrized generic, which pickle cannot find by name, so the files hold the
    episodes dumped."""
    directory.mkdir(parents=True, exist_ok=True)

    def write_episode(episode: Any, place: str) -> str:
        evaluated = place in EVAL_PLACES
        persisted = evaluated or all(
            (trace.info.get(PAYLOAD_INFO_KEY) or {}).get("url", PERSISTED_URL) == PERSISTED_URL
            for trace in episode.traces
        )
        name = f"{episode.id}.{place}" + ("" if persisted else ".unpersisted")
        if previous is None or not _link(previous / name, directory / name):
            dumped = dump_episode(episode)
            if evaluated:
                # Eval never trains: a resumed run has no arrays to point at.
                for trace in dumped["traces"]:
                    trace["info"].pop(PAYLOAD_INFO_KEY, None)
            _write(directory / name, dumped)
        return name

    groups = []
    for group in work["groups"]:
        names = []
        for restore in group.restores:
            name = f"{restore.trace['id']}.restore"
            _write(directory / name, restore.model_dump())
            names.append(name)
        groups.append(dataclasses.replace(group, restores=names))
    return {**_map_episodes(work, write_episode), "groups": groups}


def read_rollouts(work: dict, directory: Path) -> dict:
    """``work`` with the files ``write_rollouts`` named read back."""

    def read(name: str) -> Any:
        with open(directory / name, "rb") as f:
            return pickle.load(f)

    groups = [
        dataclasses.replace(group, restores=[RestoreRequest.model_validate(read(name)) for name in group.restores])
        for group in work["groups"]
    ]
    work = _map_episodes(work, lambda name, _: vf.WireEpisode.model_validate(read(name)))
    return {**work, "groups": groups}


def _write(path: Path, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _link(source: Path, target: Path) -> bool:
    """Link ``source`` to ``target``; False when there is no ``source``."""
    try:
        os.link(source, target)
    except FileExistsError:
        pass
    except FileNotFoundError:
        return False
    except OSError:
        shutil.copyfile(source, target)
    return True


def _work_episodes(work: dict) -> list[Any]:
    """The in-queue train episodes: the ones whose arrays a resumed run trains on."""
    sink = work["sink"]
    episodes = [episode for group in sink["pending_groups"].values() for episode in group]
    episodes += sink["episode_by_trace"].values()
    episodes += [
        r
        for r in work["results"]
        if not isinstance(r, DispatchFailure | GroupCancellation) and result_kind(r) == "train"
    ]
    return episodes


def _work_samples(work: dict) -> list[Any]:
    return [s for samples in work["sink"]["pending_batch"].values() for s in samples if s.payload is not None]


def persist_arrays(work: dict, directory: Path) -> None:
    """Have the env servers write the training arrays ``work`` needs to ``directory``:
    they keep them in memory, and a resumed run has new env servers. Those are the blobs
    of its finished but unshipped episodes and samples, and the arrays of each in-flight
    rollout's nodes up to its restore point. Points ``work`` at what was written; a
    rollout whose arrays were not written starts over."""
    blobs: dict[str, set[str]] = defaultdict(set)
    for episode in _work_episodes(work):
        for trace in episode.traces:
            if payload := trace.info.get(PAYLOAD_INFO_KEY):
                blobs[payload["url"]].update(reference["id"] for reference in payload["branches"].values())
    for sample in _work_samples(work):
        blobs[sample.payload.url].add(sample.payload.id)
    traces: dict[str, dict[str, tuple[str, int]]] = defaultdict(dict)
    for group in work["groups"]:
        for restore in group.restores:
            info = restore.trace.get("info", {})
            if path := info.get(PAYLOAD_ARRAYS_KEY):
                # Not dispatched since a resume: its arrays are in the last checkpoint.
                blobs[Path(path).parent.as_uri()].add(Path(path).name)
            elif url := info.get(PAYLOAD_WORKER_KEY):
                checkpoint = restore.point["checkpoint"]
                num_nodes = 0 if checkpoint.get("node") is None else checkpoint["node"] + 1
                traces[url][restore.trace["id"]] = (checkpoint["id"], num_nodes)

    urls = sorted(set(blobs) | set(traces))
    with ThreadPoolExecutor(max(len(urls), 1)) as pool:
        results = pool.map(lambda url: persist_payloads(url, directory, sorted(blobs[url]), traces[url]), urls)
        written = dict(zip(urls, results))
    persisted = {(url, key) for url, (keys, _) in written.items() for key in keys}
    restorable = {trace_id for _, trace_ids in written.values() for trace_id in trace_ids}
    for group in work["groups"]:
        for restore in group.restores:
            if path := restore.trace.get("info", {}).get(PAYLOAD_ARRAYS_KEY):
                if (Path(path).parent.as_uri(), Path(path).name) in persisted:
                    restorable.add(restore.trace["id"])

    for episode in _work_episodes(work):
        for trace in episode.traces:
            payload = trace.info.get(PAYLOAD_INFO_KEY)
            if payload and all((payload["url"], r["id"]) in persisted for r in payload["branches"].values()):
                payload["url"] = PERSISTED_URL
    for sample in _work_samples(work):
        if (sample.payload.url, sample.payload.id) in persisted:
            sample.payload.url = PERSISTED_URL
    dropped = 0
    for group in work["groups"]:
        kept = [restore for restore in group.restores if restore.trace["id"] in restorable]
        dropped += len(group.restores) - len(kept)
        group.restores = kept
        for restore in kept:
            restore.trace["info"][PAYLOAD_ARRAYS_KEY] = restore.trace["id"]
    missing = sum(len(keys) for keys in blobs.values()) + sum(map(len, traces.values())) - len(persisted)
    get_logger().info(
        f"Persisted {len(persisted) + len(restorable)} array files, {len(restorable)} of in-flight rollouts"
        + (f" ({missing - dropped} missing)" if missing > dropped else "")
        + (f"; {dropped} in-flight rollouts start over" if dropped else "")
    )


def resolve_arrays(work: dict, directory: Path) -> None:
    """Point a loaded checkpoint's work at the arrays ``persist_arrays`` wrote."""
    url = directory.as_uri()
    for episode in _work_episodes(work):
        for trace in episode.traces:
            payload = trace.info.get(PAYLOAD_INFO_KEY)
            if payload and payload["url"] == PERSISTED_URL:
                payload["url"] = url
    for sample in _work_samples(work):
        if sample.payload.url == PERSISTED_URL:
            sample.payload.url = url
    for group in work["groups"]:
        for restore in group.restores:
            info = restore.trace["info"]
            if PAYLOAD_ARRAYS_KEY in info:
                info[PAYLOAD_ARRAYS_KEY] = str(directory / info[PAYLOAD_ARRAYS_KEY])


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager:
    """The checkpoint manager always exists: ``resume`` decides whether it loads,
    ``ckpt`` whether it saves (a resume without ``ckpt`` loads but saves nothing)."""
    return CheckpointManager(output_dir, config or CheckpointConfig())
