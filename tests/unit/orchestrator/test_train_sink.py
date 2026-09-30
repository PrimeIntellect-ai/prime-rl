from collections import defaultdict

from prime_rl.orchestrator.train_sink import TrainSink


def test_missing_routed_group_budget_is_cumulative_per_env():
    sink = TrainSink.__new__(TrainSink)
    sink.missing_routed_groups_by_env = defaultdict(int)

    assert sink._record_missing_routed_group("automationbench") == 1
    assert sink._record_missing_routed_group("other") == 1
    assert sink._record_missing_routed_group("automationbench") == 2
    assert sink._record_missing_routed_group("automationbench") == 3
