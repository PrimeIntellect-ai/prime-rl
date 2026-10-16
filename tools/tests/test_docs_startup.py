import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "nightly_fft_startup.py"
spec = importlib.util.spec_from_file_location("nightly_fft_startup", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    ("events", "code", "outcome"),
    [
        ([("RUNNING", 1)], 0, "started"),
        ([("COMPLETED", 2)], 0, "started"),
        ([("FAILED", 1)], 1, "training_failed"),
        ([("STOPPED", 1)], 1, "training_failed"),
        ([("COMPLETED", None)], 1, "startup_unverified"),
        ([("QUEUED", 1)] * 3, 1, "startup_unverified"),
        ([("RUNNING", None)] * 3, 1, "startup_unverified"),
        ([KeyboardInterrupt()], 1, "observation_detached"),
        ([ValueError("private-secret")], 1, "startup_unverified"),
    ],
)
def test_startup_gate(tmp_path, capsys, events, code, outcome):
    path = tmp_path / "result.json"
    path.write_text('{"run_id":"r1","name":"smoke"}')
    now = [0]
    events = iter(events)

    def lookup(*args):
        event = next(events)
        if isinstance(event, BaseException):
            raise event
        return event

    def wait(seconds):
        now[0] += seconds

    assert module.monitor(path, 3, 1, lookup, lambda: now[0], wait) == code
    assert json.loads(path.read_text())["outcome"] == outcome
    assert "private-secret" not in capsys.readouterr().out + path.read_text()


@pytest.mark.parametrize(
    "run",
    [
        None,
        {},
        {"id": "other", "name": "smoke", "status": "RUNNING"},
        {"id": "r1", "name": "other", "status": "RUNNING"},
        {"id": "r1", "name": "smoke", "status": "private-secret"},
    ],
)
def test_inspect_rejects_bad_identity_or_status(run):
    with pytest.raises((ValueError, KeyError, TypeError)):
        module.inspect_run({"run_id": "r1", "name": "smoke"}, 3, lambda *args: {"run": run})


@pytest.mark.parametrize("step", ["1", True, -1])
def test_inspect_rejects_invalid_progress(step):
    def read(args, remaining):
        return (
            {"run": {"id": "r1", "name": "smoke", "status": "RUNNING"}} if args[0] == "get" else {"latest_step": step}
        )

    with pytest.raises(ValueError):
        module.inspect_run({"run_id": "r1", "name": "smoke"}, 3, read)


def test_late_progress_not_green(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"run_id":"r1","name":"smoke"}')
    now = [0]

    def lookup(*args):
        now[0] = 4
        return "RUNNING", 1

    assert module.monitor(path, 3, 1, lookup, lambda: now[0], lambda _: None) == 1


def test_prepare_finite_cap_preserves_model_topology(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_docs_smoke", PATH.parent / "prepare_docs_smoke.py")
    prepare_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_module)
    import tomllib

    for source in sorted((PATH.parents[1] / "configs/ci/docs-checkpoints").glob("*.toml")):
        destination = tmp_path / source.name
        prepare_module.prepare(source, destination, 2)
        original = tomllib.loads(source.read_text())
        actual = tomllib.loads(destination.read_text())
        assert actual["max_steps"] == 2
        assert "name" not in actual
        for field in ("model", "deployment"):
            assert actual[field] == original[field]
    with pytest.raises(ValueError):
        prepare_module.prepare(source, destination, 0)
