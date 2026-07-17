from types import SimpleNamespace

import wandb_workspaces.reports.v2 as wr

from prime_rl.utils.monitor.wandb import build_sections, has_effective_train_reward_panels, section


def test_overview_requires_effective_train_reward_panels():
    current_sections = build_sections(train_envs=["env-a", "env-b"])
    deserialized_sections = [
        SimpleNamespace(
            name="train/agg",
            panels=[SimpleNamespace(y=[wr.Metric("train/agg/effective/reward/mean")])],
        )
    ]
    legacy_sections = [section("train/agg", metrics=["train/agg/all/reward/mean"])]

    assert has_effective_train_reward_panels(current_sections)
    assert has_effective_train_reward_panels(deserialized_sections)
    assert not has_effective_train_reward_panels(legacy_sections)
