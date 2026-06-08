from prime_rl.configs.orchestrator import AdvantageFilterConfig


def should_filter_by_advantage(advantage: float | None, config: AdvantageFilterConfig | None) -> bool:
    return config is not None and advantage is not None and advantage <= config.threshold
