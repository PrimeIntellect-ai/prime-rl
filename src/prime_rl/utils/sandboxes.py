"""Sweep a resumed run's leftover Prime sandboxes and tunnels.

The launchers stamp ``$PRL_RUN_NAME`` on every sandbox and tunnel the run's env servers
and orchestrator open (verifiers' base labels). An attempt that dies without its
teardown leaves them running: sandboxes bill until their idle timeout, tunnels hold the
team's tunnel cap. A resume names the run it continues, so whatever still carries that
name belongs to a dead attempt; the launcher deletes it before it starts the new one.
"""

import asyncio

from prime_rl.utils.logger import get_logger


def sweep_run_leftovers(run_name: str) -> None:
    """Delete the sandboxes and tunnels labelled ``run_name``. A no-op without Prime
    credentials (a run on local runtimes); failures are logged, never raised."""
    from verifiers.v1.runtimes.sweep import prime_authenticated, sweep

    logger = get_logger()
    if not prime_authenticated():
        logger.debug("No Prime credentials, skipping the leftover sandbox sweep")
        return
    try:
        report = asyncio.run(sweep([run_name]))
    except Exception as e:
        logger.warning(f"Leftover sandbox sweep for {run_name} failed: {e}")
        return
    # verifiers' sweep reports -1 for a kind whose sweep failed (and logs the manual cleanup)
    counts = {"sandboxes": report.sandboxes.get(run_name, 0), "tunnels": report.tunnels.get(run_name, 0)}
    if failed := [kind for kind, n in counts.items() if n < 0]:
        logger.warning(
            f"Leftover {' and '.join(failed)} sweep for {run_name} failed; see the sweep's log for the cleanup"
        )
    if deleted := [f"{n} {kind}" for kind, n in counts.items() if n > 0]:
        logger.info(f"Deleted {' and '.join(deleted)} left by an earlier attempt of {run_name}")
