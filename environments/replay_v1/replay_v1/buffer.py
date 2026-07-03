"""The replay buffer: an index over saved rollout files, sampled into candidates.

The index holds only ``(path, offset, length)`` handles plus the few fields selection
needs — saved lines average ~1MB, so records are read back lazily, one line at a time,
when a task is materialized. Steps are scanned newest-first under the recency window
and candidate cap, which doubles as eviction for online buffers.
"""

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

from replay_v1.surgery import build_children, compaction_forks, is_replay_derived, main_tree, unwrap_source_task, usable

logger = logging.getLogger(__name__)

ROLLOUT_FILE = "train_rollouts.jsonl"
BARRIER_FILE = "train_rollouts.bin"
MAX_CANDIDATES = 4096
RESCAN_SECONDS = 10.0
EMPTY_POLL_SECONDS = 5.0
EMPTY_WAIT_SECONDS = 60.0
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


def complete_steps(
    rollout_dir: Path,
    require_barrier: bool,
    skip: set[int] = frozenset(),
) -> list[tuple[int, Path]]:
    """(step, jsonl path) for every step whose rollout file is safe to read, newest first.
    Steps in `skip` (already scanned) are pruned by dirname alone, before any file stat —
    rescans run every few seconds over runs with thousands of steps, all but a handful
    already scanned.

    The jsonl itself is written non-atomically, but the orchestrator writes and closes it
    strictly before the atomic rename that creates the sibling ``train_rollouts.bin`` —
    so for online buffers the barrier file marks the jsonl complete. Offline buffers
    (finished runs) skip the barrier: their files are complete by definition, and a run
    with a non-filesystem rollout transport never writes the .bin at all."""
    steps = []
    for step_dir in rollout_dir.glob("step_*"):
        match = _STEP_RE.match(step_dir.name)
        if match is None or int(match.group(1)) in skip:
            continue
        if not (step_dir / ROLLOUT_FILE).is_file():
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
        source_envs: list[str] | None,
        allow_container: bool,
        success_threshold: float,
    ) -> None:
        self.rollout_dir = resolve_rollout_dir(buffer_dir)
        self.mode = mode
        self.online = online
        self.stop_conditions = set(stop_conditions) if stop_conditions else None
        self.source_envs = set(source_envs) if source_envs else None
        self.allow_container = allow_container
        self.success_threshold = success_threshold
        # Pool workers each hold their own buffer instance; a fresh OS-entropy rng per
        # instance keeps online sampling uncorrelated across worker processes.
        self._rng = random.Random()
        self._all: list[Candidate] = []  # retained candidates, newest step first
        self._view: list[Candidate] = []  # what pick/sample draw from (label-balanced for judge)
        self._scanned_steps: set[int] = set()
        self._last_scan = 0.0
        self._rescan_lock = asyncio.Lock()
        self._warned_empty = False

    # ------------------------------------------------------------------ scanning

    def scan(self) -> list[Candidate]:
        """Index all unscanned complete steps, newest first, under the capacity cap
        (which doubles as recency eviction: newest candidates win). Synchronous file
        IO — called directly at server startup, via a thread during rollouts. The
        retained index keeps every candidate under the cap; label balancing is a derived
        view, so candidates dropped from one balanced view can pair up in a later one."""
        steps = complete_steps(self.rollout_dir, require_barrier=self.online, skip=self._scanned_steps)
        retained = list(self._all)
        fresh = 0
        for i, (step, path) in enumerate(steps):
            candidates = self._scan_file(step, path)
            retained.extend(candidates)
            fresh += len(candidates)
            self._scanned_steps.add(step)
            if fresh >= MAX_CANDIDATES:
                # Steps iterate newest-first, so everything older is evictable now and
                # forever (candidates only get newer) — never worth scanning.
                self._scanned_steps.update(step for step, _ in steps[i + 1 :])
                break
        if fresh:
            # Newest steps win the cap; this evicts the oldest candidates.
            retained.sort(key=lambda c: c.step, reverse=True)
            del retained[MAX_CANDIDATES:]
            self._set_index(retained)
        self._last_scan = time.monotonic()
        return self._view

    def _set_index(self, retained: list[Candidate]) -> None:
        """Atomically swap in the retained index and the view pick/sample draw from."""
        view = self._balanced(retained) if self.mode == "judge" else retained
        self._all = retained
        self._view = view

    def _scan_file(self, step: int, path: Path) -> list[Candidate]:
        candidates: list[Candidate] = []
        offset = 0
        with open(path, "rb") as f:
            for raw in f:
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed line at %s:%d", path, offset)
                else:
                    candidates.extend(self._candidates_from(record, str(path), offset, len(raw), step))
                offset += len(raw)
        return candidates

    def _candidates_from(self, record: dict, path: str, offset: int, length: int, step: int) -> list[Candidate]:
        task = record.get("task") or {}
        if self.source_envs is None:
            # Default: replay any env EXCEPT replay envs — online "self" buffers see the
            # replay envs' own saved rollouts (judge-of-judge), a compounding feedback
            # loop unless chosen deliberately by listing the replay env in source_envs.
            if is_replay_derived(task):
                return []
        else:
            stamped = (record.get("info") or {}).get("prime_rl", {}).get("env_name")
            if stamped not in self.source_envs:
                return []
        if self.stop_conditions is not None and record.get("stop_condition") not in self.stop_conditions:
            return []
        if not usable(record):
            return []
        # Container provisioning is keyed on the innermost original task, however deep
        # the derivation chain.
        if self.mode != "judge" and not self.allow_container and unwrap_source_task(task).get("image"):
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
        self._set_index([c for c in self._all if c != candidate])

    async def sample(self) -> Candidate:
        """A fresh draw for online buffers: rescan when stale, then sample uniformly
        (label-balance for judge comes from the interleaved view). While the run has not
        produced any replayable rollouts yet (early steps), wait briefly, then fail the
        request — the errored group releases its dispatch permits and is retried later,
        instead of hoarding capacity the run needs to produce the first rollouts."""
        waited = 0.0
        while True:
            if time.monotonic() - self._last_scan > RESCAN_SECONDS or not self._view:
                async with self._rescan_lock:
                    if time.monotonic() - self._last_scan > RESCAN_SECONDS or not self._view:
                        await asyncio.to_thread(self.scan)
            view = self._view
            if view:
                return self._rng.choice(view)
            if waited >= EMPTY_WAIT_SECONDS:
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
            await asyncio.sleep(EMPTY_POLL_SECONDS)
            waited += EMPTY_POLL_SECONDS

    def read_record(self, candidate: Candidate) -> dict:
        """Load the candidate's saved rollout line (~1MB). Synchronous — callers on the
        event loop run it (with materialization) in a thread."""
        with open(candidate.path, "rb") as f:
            f.seek(candidate.offset)
            return json.loads(f.read(candidate.length))
