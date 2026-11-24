"""Ownership of Prime sandboxes provisioned by launcher-managed env pools."""

import fcntl
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from prime_rl.utils.logger import get_logger


def _label(marker: Path) -> str:
    # Copying a run directory must not give the copy ownership of the source's VMs.
    return "prl-pool-" + uuid.uuid5(uuid.NAMESPACE_URL, marker.as_uri()).hex


@contextmanager
def register_sandbox_pool(config_dir: Path) -> Iterator[str]:
    """Record the pool before it can provision a VM. Every replica gets its own label."""
    directory = config_dir.resolve() / "sandbox-owners"
    directory.mkdir(exist_ok=True)
    marker = directory / uuid.uuid4().hex
    with marker.open("x") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not marker.exists():
            raise RuntimeError(f"Sandbox pool ownership was already reclaimed: {marker}")
        yield _label(marker)


async def reap_previous_sandbox_pools(config_dir: Path) -> None:
    """Terminate VMs from earlier attempts in this run. Snapshots have a separate ledger."""
    attempt = config_dir.resolve().parent
    number = attempt.name.removeprefix("attempt_")
    if not attempt.name.startswith("attempt_") or not number.isdecimal():
        return
    markers = [
        marker
        for directory in attempt.parent.glob("attempt_*")
        if directory.name.removeprefix("attempt_").isdecimal()
        and int(directory.name.removeprefix("attempt_")) < int(number)
        for marker in (directory / "resolved" / "sandbox-owners").glob("*")
    ]
    if not markers:
        return

    from prime_sandboxes import AsyncSandboxClient

    async with AsyncSandboxClient() as client:
        for marker in markers:
            # Even an older attempt can still be serving. Only a dead pool releases its lock.
            with marker.open() as owner:
                try:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(f"Previous sandbox pool is still active: {marker}") from error
                result = await client.bulk_delete(labels=[_label(marker)])
                if result.failed:
                    raise RuntimeError(f"Failed to terminate sandboxes from {marker}: {result.failed}")
                if result.succeeded:
                    get_logger().info(
                        f"Requested termination of {len(result.succeeded)} sandboxes from {marker.parent.parent.parent.name}"
                    )
                marker.unlink(missing_ok=True)
