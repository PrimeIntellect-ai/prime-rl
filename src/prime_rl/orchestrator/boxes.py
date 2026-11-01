"""The live boxes of in-flight rollouts across a restart (``ckpt.live_rollouts``).

A checkpointed in-flight rollout's restore point names the box it runs in
(``point["live"]``). On shutdown the env servers pause those boxes instead of deleting them
and list them under ``get_suspend_dir``; a crash that skips the shutdown leaves them
running, their harnesses retrying the model until a server answers again. A resumed run
pauses the boxes its checkpoint resumes from as soon as it has read the checkpoint, so they
wait frozen while the run comes up, and deletes the listed ones nothing resumes from: the
boxes of rollouts started after the checkpoint, and those of groups the resume drops.
Restoring an in-flight rollout then adopts its box where it is (verifiers); a box that is
gone falls back to the snapshot restore."""

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
    """Drop the live boxes from the groups' restore points, which then restore from their
    snapshots: a forked run must not adopt the boxes of the run it forks, which may still
    be running. Returns how many it dropped."""
    dropped = 0
    for group in groups:
        for restore in group.restores:
            if restore.point.get("live"):
                restore.point["live"] = None
                dropped += 1
    return dropped


def read_suspended(directory: Path) -> tuple[set[str], list[Path]]:
    """The boxes the env servers listed as they shut down, and the files listing them."""
    files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    boxes: set[str] = set()
    for path in files:
        for line in path.read_text().splitlines():
            try:
                boxes.add(json.loads(line)["sandbox_id"])
            except (ValueError, KeyError, TypeError):
                get_logger().warning(f"Skipping a malformed line of {path}: {line[:200]!r}")
    return boxes, files


async def pause_boxes(boxes: set[str]) -> None:
    if not boxes:
        return
    from verifiers.v1.runtimes.backends.prime import pause_boxes

    paused = await pause_boxes(sorted(boxes))
    get_logger().info(f"Paused {len(paused)}/{len(boxes)} live boxes until their rollouts restore")


async def delete_boxes(boxes: set[str], reason: str) -> None:
    if not boxes:
        return
    from verifiers.v1.runtimes.backends.prime import delete_boxes

    deleted = await delete_boxes(sorted(boxes))
    get_logger().info(f"Deleted {len(deleted)}/{len(boxes)} live boxes of {reason}")


async def settle_live_boxes(groups: list[GroupState], directory: Path, *, adopt: bool) -> None:
    """Get the boxes ready for a resume, before any rollout restores: pause the ones the
    groups resume in (``adopt``), so they stop while the run comes up, and delete the
    listed ones nothing resumes from. Without ``adopt`` the restore points lose their boxes
    first (``forget_live_boxes``), and every listed box goes."""
    if not adopt and (dropped := forget_live_boxes(groups)):
        get_logger().info(f"Restoring {dropped} in-flight rollouts from snapshots, not their live boxes")
    resumed = live_boxes(groups)
    suspended, files = read_suspended(directory)
    await pause_boxes(resumed)
    await delete_boxes(suspended - resumed, "rollouts the checkpoint does not resume")
    # Listed once: the next shutdown lists the boxes it pauses then.
    for path in files:
        path.unlink(missing_ok=True)
