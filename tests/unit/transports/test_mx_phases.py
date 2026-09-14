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


def test_records_phases():
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
    assert record["status"] == "complete"
    assert record["elapsed_s"] >= record["accounted_s"] - 1e-6
    assert timer.payload()["elapsed_s"] == record["elapsed_s"]


def test_accounted_time_is_the_sum_of_the_phases():
    timer = PhaseTimer("generator", 3, "abc123:3")
    for name in ("wire", "install", "release"):
        with timer.phase(name):
            pass

    record = timer.payload()
    assert record["accounted_s"] == pytest.approx(sum(record["phases_s"].values()), abs=1e-6)


def test_records_failed_phase():
    timer = PhaseTimer("generator", 1, "abc123:1")
    with pytest.raises(RuntimeError, match="staging blew up"):
        with timer.phase("wire"):
            with timer.child("read_s"):
                raise RuntimeError("staging blew up")

    assert "wire" in timer.payload()["phases_s"]
    assert "read_s" in timer.payload()["marks"]
    assert "read_s" not in timer.payload()["phases_s"]


def test_repeated_phase_accumulates():
    timer = PhaseTimer("orchestrator", 2, "abc123:2")
    with timer.phase("discovery"):
        pass
    with timer.phase("discovery"):
        pass

    record = timer.payload()
    assert set(record["phases_s"]) == {"discovery"}
    assert record["accounted_s"] == pytest.approx(record["phases_s"]["discovery"], abs=1e-6)


def test_failed_cycle_emits_record():
    emitted: list[dict] = []
    with pytest.raises(RuntimeError):
        with timed_refit("trainer", 4, "abc123:4") as timer:
            timer.emit = lambda: emitted.append(timer.payload())  # type: ignore[method-assign]
            with timer.phase("publish"):
                pass
            raise RuntimeError("cycle blew up")

    assert [record["step"] for record in emitted] == [4]
    assert emitted[0]["status"] == "failed"
    assert emitted[0]["elapsed_s"] >= emitted[0]["accounted_s"] - 1e-6


def test_emission_falls_back_to_stdout_when_the_logger_is_unavailable(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "prime_rl.utils.logger", None)

    timer = PhaseTimer("generator", 5, "abc123:5")
    with timer.phase("wire"):
        pass
    timer.emit()

    lines = [line for line in capsys.readouterr().out.splitlines() if RECORD in line]
    assert len(lines) == 1
    assert json.loads(lines[0])["step"] == 5


def test_marks_are_not_counted_as_time():
    with timed_refit("orchestrator", 3, "abc123:3") as timer:
        with timer.phase("discovery"):
            with timer.child("poll_s"):
                pass
        timer.mark("offer_lag_s", 29.7)

    record = timer.payload()
    assert record["marks"]["offer_lag_s"] == 29.7
    assert record["marks"]["poll_s"] >= 0
    assert "poll_s" not in record["phases_s"]
    assert "offer_lag_s" not in record["phases_s"]
    assert record["accounted_s"] == pytest.approx(sum(record["phases_s"].values()))


def test_payload_omits_empty_marks():
    with timed_refit("trainer", 1, "abc123:1") as timer:
        with timer.phase("publish"):
            pass

    assert "marks" not in timer.payload()


@pytest.mark.parametrize("failed", [False, True])
def test_timeline_preserves_child_intervals_and_uncovered_gaps(failed, monkeypatch):
    ticks = iter([10.0, 11.0, 12.0, 13.0, 15.0, 17.0, 20.0])
    with monkeypatch.context() as clock:
        clock.setattr(_MODULE["time"], "perf_counter", lambda: next(ticks))
        timer = PhaseTimer("orchestrator", 3, "test:3")
        try:
            with timer.phase("update_rpc", timeline=True):
                with timer.span("admin_pause"):
                    pass
                with timer.span("admin_update"):
                    if failed:
                        raise RuntimeError("update failed")
        except RuntimeError:
            assert failed
        timer.elapsed = 10.0
    record = timer.payload()
    assert record["hostname"]
    assert [span["name"] for span in record["spans"]] == ["admin_pause", "admin_update", "update_rpc"]
    assert record["spans"][1] == {
        "name": "admin_update",
        "start_offset_s": 5.0,
        "end_offset_s": 7.0,
        "duration_s": 2.0,
        "status": "failed" if failed else "complete",
    }
    assert record["spans"][-1]["status"] == ("failed" if failed else "complete")
    assert [(gap["start_offset_s"], gap["end_offset_s"]) for gap in record["span_gaps"]] == [
        (1.0, 2.0),
        (3.0, 5.0),
        (7.0, 10.0),
    ]
    assert record["marks"]["update_rpc_uncovered_s"] == 6.0
    assert record["accounted_s"] == 9.0


def test_timeline_gaps_use_interval_union_for_nested_children(monkeypatch):
    ticks = iter([0.0, 1.0, 2.0, 3.0, 6.0, 8.0, 10.0])
    with monkeypatch.context() as clock:
        clock.setattr(_MODULE["time"], "perf_counter", lambda: next(ticks))
        timer = PhaseTimer("orchestrator", 2, "test:2")
        with timer.phase("update_rpc", timeline=True):
            with timer.span("outer"):
                with timer.span("inner"):
                    pass
    assert timer.phases["update_rpc"] == 9.0
    assert timer.marks["outer_s"] == 6.0
    assert timer.marks["inner_s"] == 3.0
    assert timer.marks["update_rpc_uncovered_s"] == 3.0
