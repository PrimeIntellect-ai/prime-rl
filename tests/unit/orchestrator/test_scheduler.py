from types import SimpleNamespace

from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.scheduler import Scheduler


def _make_scheduler(config: OrchestratorConfig) -> Scheduler:
    return Scheduler(
        env=SimpleNamespace(),
        inference_pool=SimpleNamespace(clients=[]),
        buffer=SimpleNamespace(),
        config=config,
    )


def test_max_inflight_groups_matches_legacy_rollout_mode_behavior() -> None:
    config = OrchestratorConfig(batch_size=128, rollouts_per_example=16)
    scheduler = _make_scheduler(config)
    assert scheduler.max_inflight_groups == 8


def test_max_inflight_groups_is_at_least_one() -> None:
    config = OrchestratorConfig(token_batch_size=2048, max_inflight_rollouts=1, max_concurrent=8)
    scheduler = _make_scheduler(config)
    assert scheduler.max_inflight_groups == 1
