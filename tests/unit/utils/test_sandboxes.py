"""The resume sweep of a run's leftover sandboxes and tunnels."""

from types import SimpleNamespace

import verifiers.v1.runtimes.sweep as vf_sweep
import verifiers.v1.utils.prime as vf_prime

from prime_rl.utils import sandboxes
from prime_rl.utils.sandboxes import sweep_run_leftovers


def test_resume_sweeps_the_run_name(monkeypatch):
    swept: list[list[str]] = []

    async def sweep(labels):
        swept.append(labels)
        return vf_sweep.SweepReport(sandboxes={labels[0]: 3}, tunnels={labels[0]: 1})

    monkeypatch.setattr(vf_prime, "prime_authenticated", lambda: True)
    monkeypatch.setattr(vf_sweep, "sweep", sweep)
    sweep_run_leftovers("my-run")
    assert swept == [["my-run"]]


def test_no_credentials_skip_the_sweep(monkeypatch):
    def sweep(labels):
        raise AssertionError("swept without credentials")

    monkeypatch.setattr(vf_prime, "prime_authenticated", lambda: False)
    monkeypatch.setattr(vf_sweep, "sweep", sweep)
    sweep_run_leftovers("my-run")


def test_a_failed_sweep_does_not_stop_the_launch(monkeypatch):
    async def sweep(labels):
        raise RuntimeError("api down")

    monkeypatch.setattr(vf_prime, "prime_authenticated", lambda: True)
    monkeypatch.setattr(vf_sweep, "sweep", sweep)
    sweep_run_leftovers("my-run")


def test_a_failed_label_is_logged_as_a_failure_not_a_count(monkeypatch):
    async def sweep(labels):
        return vf_sweep.SweepReport(sandboxes={labels[0]: -1}, tunnels={labels[0]: 2})

    messages: list[tuple[str, str]] = []
    logger = SimpleNamespace(
        info=lambda m: messages.append(("info", m)),
        warning=lambda m: messages.append(("warning", m)),
        debug=lambda m: None,
    )
    monkeypatch.setattr(vf_prime, "prime_authenticated", lambda: True)
    monkeypatch.setattr(vf_sweep, "sweep", sweep)
    monkeypatch.setattr(sandboxes, "get_logger", lambda: logger)
    sweep_run_leftovers("my-run")
    assert messages == [
        ("warning", "Leftover sandboxes sweep for my-run failed; see the sweep's log for the cleanup"),
        ("info", "Deleted 2 tunnels left by an earlier attempt of my-run"),
    ]
    assert "-1" not in str(messages)
