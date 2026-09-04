"""Phase timing for the mx_refit refit cycle."""

import json
import sys
from pathlib import Path
from runpy import run_path

import pytest

_MODULE = run_path(Path(__file__).parents[3] / "src" / "prime_rl" / "transports" / "weights" / "mx_phases.py")
timed_refit = _MODULE["timed_refit"]
PhaseTimer = _MODULE["PhaseTimer"]
RECORD = _MODULE["RECORD"]


def test_a_cycle_records_every_phase_it_was_given():
    with timed_refit("trainer", 7, "abc123:7") as timer:
        with timer.phase("publish"):
            pass
        with timer.phase("rendezvous"):
            pass

    record = timer.payload()
    assert record["record"] == RECORD
    assert record["role"] == "trainer"
    assert record["step"] == 7
    assert record["version_uid"] == "abc123:7"
    assert set(record["phases_s"]) == {"publish", "rendezvous"}


def test_accounted_time_is_the_sum_of_the_phases():
    """The whole point is closing an attribution gap, so the arithmetic that gap
    is computed from has to hold within the record itself."""
    timer = PhaseTimer("generator", 3, "abc123:3")
    for name in ("wire", "install", "release"):
        with timer.phase(name):
            pass

    record = timer.payload()
    assert record["accounted_s"] == pytest.approx(sum(record["phases_s"].values()), abs=1e-6)


def test_a_failing_phase_is_still_recorded_and_the_error_propagates():
    """A refit that died after 30s on the wire is the case the split most needs
    to explain; dropping the span would make the failure look instantaneous."""
    timer = PhaseTimer("generator", 1, "abc123:1")
    with pytest.raises(RuntimeError, match="staging blew up"):
        with timer.phase("wire"):
            raise RuntimeError("staging blew up")

    assert "wire" in timer.payload()["phases_s"]


def test_a_repeated_phase_accumulates():
    """Retries inside one cycle must add up rather than overwrite, or a phase
    that ran three times would be reported as if it ran once."""
    timer = PhaseTimer("orchestrator", 2, "abc123:2")
    with timer.phase("discovery"):
        pass
    with timer.phase("discovery"):
        pass

    record = timer.payload()
    assert set(record["phases_s"]) == {"discovery"}
    assert record["accounted_s"] == pytest.approx(record["phases_s"]["discovery"], abs=1e-6)


def test_the_record_survives_a_cycle_that_raises():
    """timed_refit emits from a finally, so a broadcast that blows up mid-cycle
    still reports where its time went."""
    emitted: list[dict] = []
    with pytest.raises(RuntimeError):
        with timed_refit("trainer", 4, "abc123:4") as timer:
            timer.emit = lambda: emitted.append(timer.payload())  # type: ignore[method-assign]
            with timer.phase("publish"):
                pass
            raise RuntimeError("cycle blew up")

    assert [record["step"] for record in emitted] == [4]


def test_emission_falls_back_to_stdout_when_the_logger_is_unavailable(capsys, monkeypatch):
    """The generator role runs in a vLLM worker process prime-RL does not own,
    where the logger may be unconfigured or unimportable. Measurement must not
    be what takes a refit down."""
    monkeypatch.setitem(sys.modules, "prime_rl.utils.logger", None)

    timer = PhaseTimer("generator", 5, "abc123:5")
    with timer.phase("wire"):
        pass
    timer.emit()

    lines = [line for line in capsys.readouterr().out.splitlines() if RECORD in line]
    assert len(lines) == 1
    assert json.loads(lines[0])["step"] == 5
