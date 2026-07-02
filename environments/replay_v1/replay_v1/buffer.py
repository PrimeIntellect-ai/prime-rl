"""The replay buffer: an index over saved rollout files, sampled into candidates.

The index holds only ``(path, offset, length)`` handles plus the few fields selection
needs — saved lines average ~1MB, so records are read back lazily, one line at a time,
when a task is materialized. Steps are scanned newest-first under the recency window
and candidate cap, which doubles as eviction for online buffers.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

from replay_v1.surgery import build_children, compaction_forks, main_tree, usable

logger = logging.getLogger(__name__)

ROLLOUT_FILE = "train_rollouts.jsonl"
BARRIER_FILE = "train_rollouts.bin"
_STEP_RE = re.compile(r"^step_(\d+)$")


@dataclass(frozen=True)
class Candidate:
    """One replayable task: a lazy handle onto a saved rollout line (plus, for continue
    mode, which compaction fork of it)."""

    path: str
    offset: int
    length: int
    step: int
    original_reward: float
    source_id: str
    fork_node: int | None = None


def resolve_rollout_dir(buffer_dir: str) -> Path:
    """Accept either a rollouts dir (containing step_*) or a run dir containing one."""
    path = Path(buffer_dir)
    if (path / "rollouts").is_dir() and not any(_STEP_RE.match(p.name) for p in path.glob("step_*")):
        return path / "rollouts"
    return path


def complete_steps(rollout_dir: Path, require_barrier: bool) -> list[tuple[int, Path]]:
    """(step, jsonl path) for every step whose rollout file is safe to read, newest first.

    The jsonl itself is written non-atomically, but the orchestrator writes and closes it
    strictly before the atomic rename that creates the sibling ``train_rollouts.bin`` —
    so for online buffers the barrier file marks the jsonl complete. Offline buffers
    (finished runs) skip the barrier: their files are complete by definition, and a run
    with a non-filesystem rollout transport never writes the .bin at all."""
    steps = []
    for step_dir in rollout_dir.glob("step_*"):
        match = _STEP_RE.match(step_dir.name)
        if match is None or not (step_dir / ROLLOUT_FILE).is_file():
            continue
        if require_barrier and not (step_dir / BARRIER_FILE).is_file():
            continue
        steps.append((int(match.group(1)), step_dir / ROLLOUT_FILE))
    return sorted(steps, reverse=True)


class ReplayBuffer:
    """Scans rollout files into candidates and picks one per resolved task.

    Offline: the index is built once and ``pick(idx)`` is deterministic, so GRPO group
    members dispatched as independent rollouts still bind the same source. Online: the
    index is rescanned (throttled) as the run writes new steps, ``sample()`` draws
    freshly — which is why online replay envs force group dispatch.

    Concurrency: rescans run in a thread while the event loop keeps sampling, so the
    index is swapped atomically (fresh lists, single attribute assignment), never
    mutated in place.
    """

    def __init__(
        self,
        buffer_dir: str,
        mode: str,
        online: bool,
        stop_conditions: list[str] | None,
        env_name: str | None,
        allow_container: bool,
        success_threshold: float,
        balance_labels: bool,
        max_candidates: int,
        max_steps_back: int | None,
        seed: int,
        rescan_seconds: float = 10.0,
        empty_wait_seconds: float = 60.0,
    ) -> None:
        self.rollout_dir = resolve_rollout_dir(buffer_dir)
        self.mode = mode
        self.online = online
        self.stop_conditions = set(stop_conditions) if stop_conditions else None
        self.env_name = env_name
        self.allow_container = allow_container
        self.success_threshold = success_threshold
        self.balance_labels = balance_labels
        self.max_candidates = max_candidates
        self.max_steps_back = max_steps_back
        self.rescan_seconds = rescan_seconds
        self.empty_wait_seconds = empty_wait_seconds
        # Pool workers each hold their own buffer instance; mix the pid in so online
        # sampling isn't replicated across worker processes.
        self._rng = random.Random(seed ^ os.getpid())
        self._all: list[Candidate] = []  # retained candidates, newest step first
        self._view: list[Candidate] = []  # what pick/sample draw from (label-balanced for judge)
        self._scanned_steps: set[int] = set()
        self._last_scan = 0.0
        self._rescan_lock: asyncio.Lock | None = None
        self._warned_empty = False

    # ------------------------------------------------------------------ scanning

    def scan(self) -> list[Candidate]:
        """Index all unscanned complete steps, newest first, under the caps. Synchronous
        file IO — called directly at server startup, via a thread during rollouts. The
        retained index keeps every candidate under the caps; label balancing is a derived
        view, so candidates dropped from one balanced view can pair up in a later one."""
        steps = complete_steps(self.rollout_dir, require_barrier=self.online)
        if steps and self.max_steps_back is not None:
            newest = steps[0][0]
            steps = [(step, path) for step, path in steps if step > newest - self.max_steps_back]
        retained = list(self._all)
        for step, path in steps:
            if step in self._scanned_steps:
                continue
            if len(retained) >= self.max_candidates and not self.online:
                break
            retained.extend(self._scan_file(step, path))
            self._scanned_steps.add(step)
        # Newest steps win the cap; within the window this evicts the oldest candidates.
        retained.sort(key=lambda c: c.step, reverse=True)
        del retained[self.max_candidates :]
        if self.max_steps_back is not None and self._scanned_steps:
            newest = max(self._scanned_steps)
            retained = [c for c in retained if c.step > newest - self.max_steps_back]
        self._all = retained
        self._view = self._balanced(retained) if self.balance_labels and self.mode == "judge" else retained
        self._last_scan = time.monotonic()
        return self._view

    def _scan_file(self, step: int, path: Path) -> list[Candidate]:
        candidates: list[Candidate] = []
        offset = 0
        with open(path, "rb") as f:
            for raw in f:
                length = len(raw)
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed line at %s:%d", path, offset)
                    offset += length
                    continue
                candidates.extend(self._candidates_from(record, str(path), offset, length, step))
                offset += length
        return candidates

    def _candidates_from(self, record: dict, path: str, offset: int, length: int, step: int) -> list[Candidate]:
        if not usable(record):
            return []
        task = record.get("task") or {}
        # Never replay a replay: online "self" buffers see the replay env's own saved
        # rollouts (judge-of-judge, recheck-of-recheck) — a compounding feedback loop.
        if "kind" in task and "source_task" in task:
            return []
        if self.stop_conditions is not None and record.get("stop_condition") not in self.stop_conditions:
            return []
        if self.env_name is not None:
            stamped = (record.get("info") or {}).get("prime_rl", {}).get("env_name")
            if stamped != self.env_name:
                return []
        if self.mode != "judge" and not self.allow_container and task.get("image"):
            return []
        reward = sum((record.get("rewards") or {}).values())
        base = dict(
            path=path,
            offset=offset,
            length=length,
            step=step,
            original_reward=reward,
            source_id=record.get("id", ""),
        )
        if self.mode == "continue":
            children, _ = build_children(record["nodes"])
            tree = main_tree(children)
            return [Candidate(**base, fork_node=fork) for fork in compaction_forks(record["nodes"], children, tree)]
        return [Candidate(**base)]

    def _balanced(self, candidates: list[Candidate]) -> list[Candidate]:
        """Interleave successes and failures 1:1 so judge rewards aren't gameable by a
        constant verdict (observed buffers skew 98:1). Truncates to the smaller label."""
        positive = [c for c in candidates if c.original_reward > self.success_threshold]
        negative = [c for c in candidates if c.original_reward <= self.success_threshold]
        if not positive or not negative:
            logger.warning(
                "judge buffer has one-sided labels (%d correct / %d incorrect); serving unbalanced",
                len(positive),
                len(negative),
            )
            return candidates
        interleaved = []
        for pair in zip(positive, negative):
            interleaved.extend(pair)
        return interleaved

    # ------------------------------------------------------------------ picking

    def __len__(self) -> int:
        return len(self._view)

    def pick(self, idx: int) -> Candidate:
        """Deterministic candidate for a task index (offline buffers)."""
        view = self._view
        return view[idx % len(view)]

    def discard(self, candidate: Candidate) -> None:
        """Drop a candidate whose source line is gone (its run was resumed or cleaned)."""
        retained = [c for c in self._all if c != candidate]
        self._all = retained
        self._view = self._balanced(retained) if self.balance_labels and self.mode == "judge" else retained

    async def sample(self) -> Candidate:
        """A fresh draw for online buffers: rescan when stale, then sample uniformly
        (label-balance for judge comes from the interleaved view). While the run has not
        produced any replayable rollouts yet (early steps), wait briefly, then fail the
        request — the errored group releases its dispatch permits and is retried later,
        instead of hoarding capacity the run needs to produce the first rollouts."""
        if self._rescan_lock is None:
            self._rescan_lock = asyncio.Lock()
        waited = 0.0
        while True:
            if time.monotonic() - self._last_scan > self.rescan_seconds or not self._view:
                async with self._rescan_lock:
                    if time.monotonic() - self._last_scan > self.rescan_seconds or not self._view:
                        await asyncio.to_thread(self.scan)
            view = self._view
            if view:
                return self._rng.choice(view)
            if waited >= self.empty_wait_seconds:
                raise RuntimeError(
                    f"replay buffer at {self.rollout_dir} has no replayable {self.mode!r} "
                    f"candidates after {waited:.0f}s; failing this request so its permits free up"
                )
            if not self._warned_empty:
                logger.warning(
                    "replay buffer at %s is empty (no replayable %s candidates yet); replay requests "
                    "will fail and retry until the run writes its first rollouts",
                    self.rollout_dir,
                    self.mode,
                )
                self._warned_empty = True
            await asyncio.sleep(5.0)
            waited += 5.0

    async def read_record(self, candidate: Candidate) -> dict:
        """Load the candidate's saved rollout line (~1MB) off the event loop."""

        def _read() -> dict:
            with open(candidate.path, "rb") as f:
                f.seek(candidate.offset)
                return json.loads(f.read(candidate.length))

        return await asyncio.to_thread(_read)
