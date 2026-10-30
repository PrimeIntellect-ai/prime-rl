import threading
from types import SimpleNamespace

import pytest

from prime_rl.trainer.rl import annotations
from prime_rl.trainer.rl.annotations import AnnotationWriter


@pytest.fixture
def gate():
    return threading.Event()


@pytest.fixture
def written(monkeypatch, gate):
    calls: list[list[dict]] = []

    async def fake_log_annotations(records):
        assert gate.wait(timeout=5)
        calls.append(records)

    monkeypatch.setattr(annotations.monitors, "log_annotations", fake_log_annotations)
    return calls


@pytest.fixture
def writer(gate):
    parallel_dims = SimpleNamespace(cp_enabled=False, world_mesh=None)
    world = SimpleNamespace(world_size=1, rank=0)
    writer = AnnotationWriter(parallel_dims, world, float_decimals=None)
    yield writer
    gate.set()
    writer.close()


def test_flush_returns_before_write_finishes(writer, written, gate):
    writer._pending = [{"trace_id": "a"}]
    writer.flush()
    assert written == []
    gate.set()
    writer.close()
    assert written == [[{"trace_id": "a"}]]


def test_flushes_write_in_step_order_exactly_once(writer, written, gate):
    gate.set()
    writer._pending = [{"trace_id": "a"}]
    writer.flush()
    writer._pending = [{"trace_id": "b"}, {"trace_id": "c"}]
    writer.flush()
    writer.close()
    assert written == [[{"trace_id": "a"}], [{"trace_id": "b"}, {"trace_id": "c"}]]


def test_close_waits_for_pending_write(writer, written, gate):
    writer._pending = [{"trace_id": "a"}]
    writer.flush()
    closer = threading.Thread(target=writer.close)
    closer.start()
    closer.join(timeout=0.2)
    assert closer.is_alive()
    gate.set()
    closer.join()
    assert written == [[{"trace_id": "a"}]]


def test_flush_without_records_writes_nothing(writer, written, gate):
    gate.set()
    writer.flush()
    writer.close()
    assert written == []


def test_write_error_surfaces_at_next_flush(writer, monkeypatch):
    async def failing_log_annotations(records):
        raise RuntimeError("disk full")

    monkeypatch.setattr(annotations.monitors, "log_annotations", failing_log_annotations)
    writer._pending = [{"trace_id": "a"}]
    writer.flush()
    with pytest.raises(RuntimeError, match="disk full"):
        writer.flush()
