"""The env server ships its env config to workers as plain data: a JSON-mode
`model_dump` and `resolve_env_config` rebuilds the env's own config class, role-typed seats
included."""

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from verifiers.v1.utils.loaders import resolve_env_config


@pytest.mark.parametrize(
    "data",
    [
        {"taskset": {"id": "kuhn-poker"}, "player0": {"max_turns": 3}},
        {
            "id": "best-of-n",
            "n": 2,
            "taskset": {"id": "reverse-text"},
            "agent": {"harness": {"id": "null"}, "max_turns": 2},
        },
    ],
    ids=["role-typed-seats", "bundled-env"],
)
def test_env_config_survives_the_wire(data: dict) -> None:
    config = resolve_env_config(data)
    rebuilt = resolve_env_config(config.model_dump(mode="json"))
    assert type(rebuilt) is type(config)
    assert rebuilt == config


def test_sigterm_exits_143_after_teardown(tmp_path: Path) -> None:
    address_file = tmp_path / "address"
    cmd = [sys.executable, "-m", "prime_rl.entrypoints.env_server", "--env.taskset.id", "reverse-text"]
    cmd += ["--address-file", str(address_file), "--output-dir", str(tmp_path)]
    proc = subprocess.Popen(cmd)
    deadline = time.monotonic() + 60
    while not address_file.exists():
        assert proc.poll() is None and time.monotonic() < deadline
        time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=60) == 128 + signal.SIGTERM
