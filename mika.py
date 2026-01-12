import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Protocol


class MetricTransport(Protocol):
    def send(self, metrics: dict[str, float]) -> None: ...
    def receive_all(self) -> list[dict[str, float]]: ...


class FileTransport:
    def __init__(self, source_id: str | None = None):
        self.directory = Path("/tmp/metrics-file-transport")
        self.source_id = source_id or f"source-{uuid.uuid4().hex}"
        self.filepath = self.directory / f"{self.source_id}.jsonl"

    def send(self, metrics: dict[str, float]) -> None:
        if not self.filepath.parent.exists():
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "a") as f:
            json.dump(metrics, f)

    def receive_all(self) -> dict[str, float]:
        metrics = {}
        filepaths = list(self.directory.glob("*.jsonl"))
        for filepath in filepaths:
            with open(filepath, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                    except:
                        print(f"Error loading metrics from {filepath}: {line}")
                        continue
                    metrics.update(data)
        return metrics

    def reset(self) -> None:
        for filepath in self.directory.glob("*.jsonl"):
            filepath.unlink()


class MetricsSource:
    """Mixin that adds automatic metric collection."""

    def __init__(self):
        self.transport = FileTransport(self.__class__.__name__)

    def push_metric(self, name: str, value: float):
        self.transport.send({name: value})

    def push_metrics(self, metrics: dict[str, float]):
        self.transport.send(metrics)


class MetricsCollector:
    def __init__(self):
        self.transport = FileTransport("collector")

    def collect(self) -> dict[str, float]:
        return self.transport.receive_all()

    def reset(self) -> None:
        self.transport.reset()


class MeanEventLoopLag:
    def __init__(self, name: str):
        self.name = name
        self.total_lag = 0.0
        self.count = 0

    def update(self, lag: float) -> None:
        self.total_lag += lag
        self.count += 1

    def compute(self) -> float:
        return self.total_lag / self.count

    def reset(self) -> None:
        self.total_lag = 0.0
        self.count = 0


class EventLoopLagMonitor(MetricsSource):
    """A class to monitor how busy the main event loop is."""

    def __init__(self, interval: float = 0.1):
        super().__init__()
        assert interval > 0
        self.interval = interval
        self.mean_lag = MeanEventLoopLag("event_loop_lag")
        self.task: asyncio.Task | None = None

    async def measure_lag(self) -> float:
        """Measures event loop lag by asynchronously sleeping for interval seconds"""
        next_time = time.perf_counter() + self.interval
        await asyncio.sleep(self.interval)
        now = time.perf_counter()
        lag = now - next_time
        return lag

    async def measure_lag_loop(self) -> None:
        while True:
            lag = await self.measure_lag()
            self.mean_lag.update(lag)
            self.push_metric("event_loop_lag/mean", self.mean_lag.compute())

    def start(self) -> "EventLoopLagMonitor":
        """Start the event loop lag monitor as a background task."""

        self.task = asyncio.create_task(self.measure_lag_loop())

        return self


class Trainer(MetricsSource):
    def __init__(self):
        super().__init__()
        self.metrics_collector = MetricsCollector()
        self.lag_monitor = EventLoopLagMonitor(interval=0.1)

    async def start(self) -> None:
        self.lag_monitor.start()
        for step in range(10):
            await asyncio.sleep(2)

            # Update progress
            loss = 1 / (step + 1)
            reward = 1 - 1 / (step + 1)
            self.push_metrics({"step": step, "loss": loss, "reward": reward})

            step_metrics = self.metrics_collector.collect()
            print(f"Got metrics: {step_metrics}")
            self.metrics_collector.reset()


if __name__ == "__main__":
    asyncio.run(Trainer().start())
