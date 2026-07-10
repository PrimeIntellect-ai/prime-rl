from pathlib import Path
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.train_finalization import finalize_train_batch
from prime_rl.orchestrator.types import Progress


class _Stat:
    def mean(self) -> float:
        return 1.0


class _Metrics:
    reward = _Stat()
    num_turns = _Stat()
    num_branches = _Stat()
    is_truncated = _Stat()
    has_error = _Stat()

    def to_wandb(self, *, prefix: str, subset: str) -> dict[str, float]:
        return {f"{prefix}/{subset}/metric": 1.0}


class _Rollout:
    is_trainable = True
    off_policy_steps = 0
    num_total_tokens = 3
    num_input_tokens = 1
    num_output_tokens = 2
    group_id = "group"
    reward = 1.0

    def to_record(self) -> dict:
        return {"group": self.group_id}

    def scalar_advantage(self) -> float:
        return 0.5


class _Rollouts(list):
    metrics = _Metrics()

    @property
    def effective(self):
        return self

    @property
    def rollouts(self):
        return list(self)

    def by_env(self) -> dict:
        return {}


class _Monitor:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.logged_metrics: dict[str, float] = {}

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self.events.append("monitor:log")
        self.logged_metrics = metrics

    def log_samples(self, _rollouts, *, step: int) -> None:
        self.events.append("monitor:samples")

    def log_distributions(self, *, distributions, step: int) -> None:
        self.events.append("monitor:distributions")


class _Host:
    def __init__(self, tmp_path: Path, events: list[str]) -> None:
        self.config = SimpleNamespace(output_dir=tmp_path, max_steps=None)
        self.progress = Progress()
        self.last_batch_at = 1.0
        self.draining = False
        self.consecutive_empty_batches = 0
        self.wait_for_policy_time = 0.25
        self.monitor = _Monitor(events)
        self.usage_reporter = None
        self.heart = None
        self.train_envs = []
        self.train_sink = SimpleNamespace(
            pre_filter_seen=0,
            pre_filter_dropped=0,
            pre_filter_dropped_by_name={},
            reset_pre_filter_stats=lambda: events.append("reset-filters"),
        )
        self.events = events

    async def _send_to_trainer(self, _batch) -> None:
        self.events.append("send")

    def update_dispatch_gate(self) -> None:
        assert self.progress.step == 2
        self.events.append("dispatch-gate")

    async def maybe_save_ckpt(self, step: int) -> float:
        assert step == 1
        self.events.append("checkpoint")
        return 0.5

    def maybe_trigger_eval(self, step: int) -> None:
        assert step == 2
        self.events.append("eval")


@pytest.mark.asyncio
async def test_finalize_train_batch_preserves_ship_and_reporting_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    host = _Host(tmp_path, events)
    batch = SimpleNamespace(samples=[object()], rollouts=_Rollouts([_Rollout()]))

    monkeypatch.setattr(
        "prime_rl.orchestrator.train_finalization.save_rollouts",
        lambda *_args: events.append("save-rollouts"),
    )
    monkeypatch.setattr(
        "prime_rl.orchestrator.train_finalization.trim_process_memory",
        lambda: events.append("trim"),
    )
    monkeypatch.setattr(
        "prime_rl.orchestrator.train_finalization.get_logger",
        lambda: SimpleNamespace(success=lambda _message: events.append("success")),
    )

    await finalize_train_batch(host, batch)

    assert events == [
        "save-rollouts",
        "send",
        "dispatch-gate",
        "checkpoint",
        "trim",
        "monitor:log",
        "monitor:samples",
        "monitor:distributions",
        "success",
        "reset-filters",
        "eval",
        "trim",
    ]
    assert host.progress.step == 2
    assert host.progress.total_tokens == 3
    assert host.progress.total_samples == 1
    assert host.progress.total_problems == 1
    assert host.monitor.logged_metrics["progress/total_tokens"] == 0
    assert host.wait_for_policy_time == 0.0
