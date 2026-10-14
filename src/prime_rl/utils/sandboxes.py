"""A run's sandbox label, and the resume sweep of its leftover Prime sandboxes and tunnels.

The launchers stamp ``$PRL_RUN_LABEL`` (``run_label``) on every sandbox and tunnel the
run's env servers and orchestrator open (verifiers' base labels). The label is the run
name under a ``prl-`` namespace with a digest of the run's output dir, so every attempt
of one run carries it and no other run does, not even one of the same name in another
output dir. An attempt that dies without its teardown leaves its sandboxes and tunnels
running: sandboxes bill until their idle timeout, tunnels hold the team's tunnel cap. A
resume names the run it continues, so whatever still carries its label belongs to a dead
attempt; the launcher deletes it before it starts the new one.
"""

import asyncio
import hashlib
import re
from pathlib import Path

from prime_rl.utils.logger import get_logger

RUN_LABEL_VAR = "PRL_RUN_LABEL"
"""The env var carrying the run's sandbox label to every process the launcher starts."""


def run_label(run_name: str, output_dir: Path) -> str:
    """The sandbox label of run ``run_name`` writing to ``output_dir``:
    ``prl-<name>-<digest>``, the digest over the name and the resolved output dir."""
    digest = hashlib.sha256(f"{Path(output_dir).resolve()}:{run_name}".encode()).hexdigest()[:10]
    stem = re.sub(r"[^a-z0-9]+", "-", run_name.lower()).strip("-") or "run"
    return f"prl-{stem}-{digest}"


def sweep_run_leftovers(run_name: str, output_dir: Path, *, run_dir: Path | None = None) -> None:
    """Delete the sandboxes and tunnels labelled ``run_label(run_name, output_dir)``. A
    no-op without Prime credentials (a run on local runtimes), and when ``run_dir``'s run
    control names an eval of the run still alive on this host (its sandboxes are in use);
    failures are logged, never raised."""
    from verifiers.v1.runtimes.sweep import sweep
    from verifiers.v1.utils.prime import prime_authenticated

    logger = get_logger()
    if run_dir is not None:
        from prime_rl.eval.control import live_pid

        if (pid := live_pid(run_dir)) is not None:
            logger.warning(f"An attempt of {run_name} still runs (pid {pid}), not sweeping its sandboxes")
            return
    if not prime_authenticated():
        logger.debug("No Prime credentials, skipping the leftover sandbox sweep")
        return
    label = run_label(run_name, output_dir)
    try:
        report = asyncio.run(sweep([label]))
    except Exception as e:
        logger.warning(f"Leftover sandbox sweep for {run_name} ({label}) failed: {e}")
        return
    # verifiers' sweep reports -1 for a kind whose sweep failed (and logs the manual cleanup)
    counts = {"sandboxes": report.sandboxes.get(label, 0), "tunnels": report.tunnels.get(label, 0)}
    if failed := [kind for kind, n in counts.items() if n < 0]:
        logger.warning(
            f"Leftover {' and '.join(failed)} sweep for {run_name} ({label}) failed; "
            "see the sweep's log for the cleanup"
        )
    if deleted := [f"{n} {kind}" for kind, n in counts.items() if n > 0]:
        logger.info(f"Deleted {' and '.join(deleted)} left by an earlier attempt of {run_name} ({label})")
