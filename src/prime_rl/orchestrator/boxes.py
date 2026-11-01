"""The live boxes of in-flight rollouts across a restart (``ckpt.rollouts``).

A checkpointed in-flight rollout's restore point names the box it runs in
(``point["live"]``). On shutdown the env servers pause those boxes instead of deleting them
and list them under ``get_suspend_dir``; a crash that skips the shutdown leaves them
running, their harnesses retrying the model until a server answers again. A resumed run
pauses the boxes its checkpoint resumes from as soon as it has read the checkpoint, so they
wait frozen while the run comes up, and deletes the listed ones nothing resumes from: the
boxes of rollouts started after the checkpoint, and those of groups the resume drops. The
orchestrator lists every live box too, as its rollout streams it (`LiveBoxJournal`): a
crash that suspends nothing leaves those boxes running, and only that list names them.
Restoring an in-flight rollout then adopts its box where it is (verifiers); a rollout
whose box is gone starts over."""

import json
from collections.abc import Iterable
from pathlib import Path

from verifiers.v1.serve import RestoreRequest

from prime_rl.orchestrator.types import GroupState
from prime_rl.utils.logger import get_logger


def live_tunnel(restore: RestoreRequest | None) -> str | None:
    """The tunnel a restore's live box reaches its rollout's server through, if it has one:
    the restores sharing it must go to one env-server replica (the one that adopts it)."""
    if restore is None or not (live := restore.point.get("live")):
        return None
    return live.get("tunnel_id")


def live_boxes(groups: Iterable[GroupState]) -> set[str]:
    """The boxes the groups' in-flight restore points resume in."""
    return {live["sandbox_id"] for group in groups for restore in group.restores if (live := restore.point.get("live"))}


def forget_live_boxes(groups: Iterable[GroupState]) -> int:
    """Drop the groups' restore points, whose rollouts then start over: a forked run must
    not adopt the boxes of the run it forks, which may still be running. Returns how many
    it dropped."""
    dropped = 0
    for group in groups:
        dropped += len(group.restores)
        group.restores = []
    return dropped


def read_suspended(directory: Path) -> tuple[set[str], list[Path]]:
    """The boxes the env servers listed as they shut down and the orchestrator as they
    streamed (`LiveBoxJournal`), and the files listing them."""
    files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    boxes: set[str] = set()
    for path in files:
        for line in path.read_text().splitlines():
            try:
                boxes.add(json.loads(line)["sandbox_id"])
            except (ValueError, KeyError, TypeError):
                get_logger().warning(f"Skipping a malformed line of {path}: {line[:200]!r}")
    return boxes, files


JOURNAL = "orchestrator.jsonl"
"""The orchestrator's `LiveBoxJournal`, next to the env servers' lists."""


class LiveBoxJournal:
    """The live boxes of this run's rollouts, listed next to the suspended ones as their
    restore points stream in, and rewritten at each checkpoint to the ones still in
    flight: the boxes a resume deletes when its checkpoint does not resume them."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.listed: set[str] = set()

    def record(self, box: str) -> None:
        if box in self.listed:
            return
        self.listed.add(box)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as journal:
            journal.write(json.dumps({"sandbox_id": box}) + "\n")

    def rewrite(self, boxes: set[str]) -> None:
        self.listed = set(boxes)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps({"sandbox_id": box}) + "\n" for box in sorted(boxes)))
        tmp.replace(self.path)


async def pause_boxes(boxes: set[str]) -> None:
    if not boxes:
        return
    from verifiers.v1.runtimes.backends.prime import pause_boxes

    paused = await pause_boxes(sorted(boxes))
    get_logger().info(f"Paused {len(paused)}/{len(boxes)} live boxes until their rollouts restore")


async def delete_boxes(boxes: set[str], reason: str) -> set[str]:
    """Delete `boxes`, best-effort; returns the ones deleted."""
    if not boxes:
        return set()
    from verifiers.v1.runtimes.backends.prime import delete_boxes

    deleted = await delete_boxes(sorted(boxes))
    get_logger().info(f"Deleted {len(deleted)}/{len(boxes)} live boxes of {reason}")
    return set(deleted)


async def settle_live_boxes(groups: list[GroupState], directory: Path, *, adopt: bool) -> None:
    """Get the boxes ready for a resume, before any rollout restores: pause the ones the
    groups resume in (``adopt``), so they stop while the run comes up, and delete the
    listed ones nothing resumes from. Without ``adopt`` the restore points lose their boxes
    first (``forget_live_boxes``), and every listed box goes."""
    if not adopt and (dropped := forget_live_boxes(groups)):
        get_logger().info(f"Starting {dropped} in-flight rollouts over, not adopting their live boxes")
    resumed = live_boxes(groups)
    suspended, files = read_suspended(directory)
    await pause_boxes(resumed)
    deleted = await delete_boxes(suspended - resumed, "rollouts the checkpoint does not resume")
    # What is left goes to this run's journal, so a resume that crashes before its first
    # checkpoint still leaves every box listed for the next one.
    LiveBoxJournal(directory / JOURNAL).rewrite(resumed | (suspended - deleted))
    for path in files:
        if path.name != JOURNAL:
            path.unlink(missing_ok=True)
