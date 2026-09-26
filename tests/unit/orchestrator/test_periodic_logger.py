import pytest

from prime_rl.orchestrator.periodic_logger import PeriodicLogger
from tests.unit.orchestrator.fakes import RecordingMonitors


@pytest.mark.asyncio
async def test_collect_joins_statuses_and_merges_gauges():
    logger = PeriodicLogger(name="test", interval=60)
    monitors = RecordingMonitors()
    logger.bind(monitors=monitors)
    logger.register(status=lambda: "a", gauges=lambda: {"x": 1.0})
    logger.register(status=lambda: None)
    logger.register(status=lambda: "b", gauges=lambda: {"y": 2.0})
    body, payload = logger.collect()
    assert body == "a; b"
    assert payload == {"x": 1.0, "y": 2.0}
    await logger.emit()
    assert monitors.metrics == [({"x": 1.0, "y": 2.0}, None)]
