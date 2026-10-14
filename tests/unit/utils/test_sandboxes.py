"""A run's sandbox label and the resume sweep of its leftover sandboxes and tunnels."""

import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import verifiers.v1.runtimes.sweep as vf_sweep
import verifiers.v1.utils.prime as vf_prime

from prime_rl.eval.control import control_path, live_pid
from prime_rl.utils import sandboxes
from prime_rl.utils.sandboxes import run_label, sweep_run_leftovers

OUT = "/tmp/outputs"


def test_the_label_is_namespaced_and_unique_to_the_run_name_and_output_dir(tmp_path):
    label = run_label("My Run", tmp_path / "a")
    assert label.startswith("prl-my-run-") and label != "My Run"
    assert run_label("My Run", tmp_path / "a") == label  # every attempt of the run
    assert run_label("My Run", tmp_path / "a" / ".." / "a") == label
    assert run_label("My Run", tmp_path / "b") != label  # the same name elsewhere
    assert run_label("my-run", tmp_path / "a") != label  # names that sanitize alike


def test_resume_sweeps_the_run_label(monkeypatch, tmp_path):
    swept: list[list[str]] = []

    async def sweep(labels):
        swept.append(labels)
        return vf_sweep.SweepReport(sandboxes={labels[0]: 3}, tunnels={labels[0]: 1})

    monkeypatch.setattr(vf_prime, "prime_authenticated", lambda: True)
    monkeypatch.setattr(vf_sweep, "sweep", sweep)
    sweep_run_leftovers("my-run", tmp_path)
    assert swept == [[run_label("my-run", tmp_path)]]


def test_a_live_attempt_is_found_through_its_run_control(tmp_path):
    def control(**info) -> None:
        control_path(tmp_path).write_text(json.dumps(info))

    assert live_pid(tmp_path) is None  # no control file
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        control(host=socket.gethostname(), pid=other.pid, address="tcp://127.0.0.1:1")
        assert live_pid(tmp_path) == other.pid
        control(host="elsewhere", pid=other.pid)
        assert live_pid(tmp_path) is None  # not this host's process
        control(host=socket.gethostname(), pid=os.getpid())
        assert live_pid(tmp_path) is None  # this very launch
    finally:
        other.kill()
        other.wait()
    control(host=socket.gethostname(), pid=other.pid)
    assert live_pid(tmp_path) is None  # it died
    control_path(tmp_path).write_text("[")
    assert live_pid(tmp_path) is None


def test_an_eval_resume_leaves_a_live_attempts_sandboxes_alone(monkeypatch, tmp_path):
    swept: list[list[str]] = []

    async def sweep(labels):
        swept.append(labels)
        return vf_sweep.SweepReport(sandboxes={}, tunnels={})

    monkeypatch.setattr(vf_prime, "prime_authenticated", lambda: True)
    monkeypatch.setattr(vf_sweep, "sweep", sweep)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        control_path(tmp_path).write_text(json.dumps({"host": socket.gethostname(), "pid": other.pid}))
        sweep_run_leftovers("my-run", OUT, run_dir=tmp_path)
        assert swept == []
    finally:
        other.kill()
        other.wait()
    sweep_run_leftovers("my-run", OUT, run_dir=tmp_path)
    assert swept == [[run_label("my-run", OUT)]]


def test_no_credentials_skip_the_sweep(monkeypatch):
    def sweep(labels):
        raise AssertionError("swept without credentials")

    monkeypatch.setattr(vf_prime, "prime_authenticated", lambda: False)
    monkeypatch.setattr(vf_sweep, "sweep", sweep)
    sweep_run_leftovers("my-run", OUT)


def test_a_failed_sweep_does_not_stop_the_launch(monkeypatch):
    async def sweep(labels):
        raise RuntimeError("api down")

    monkeypatch.setattr(vf_prime, "prime_authenticated", lambda: True)
    monkeypatch.setattr(vf_sweep, "sweep", sweep)
    sweep_run_leftovers("my-run", OUT)


def test_a_failed_label_is_logged_as_a_failure_not_a_count(monkeypatch):
    async def sweep(labels):
        return vf_sweep.SweepReport(sandboxes={labels[0]: -1}, tunnels={labels[0]: 2})

    label = run_label("my-run", OUT)

    messages: list[tuple[str, str]] = []
    logger = SimpleNamespace(
        info=lambda m: messages.append(("info", m)),
        warning=lambda m: messages.append(("warning", m)),
        debug=lambda m: None,
    )
    monkeypatch.setattr(vf_prime, "prime_authenticated", lambda: True)
    monkeypatch.setattr(vf_sweep, "sweep", sweep)
    monkeypatch.setattr(sandboxes, "get_logger", lambda: logger)
    sweep_run_leftovers("my-run", OUT)
    assert messages == [
        ("warning", f"Leftover sandboxes sweep for my-run ({label}) failed; see the sweep's log for the cleanup"),
        ("info", f"Deleted 2 tunnels left by an earlier attempt of my-run ({label})"),
    ]
    assert "-1" not in str(messages)
