from __future__ import annotations

import json
from pathlib import Path

import pytest

from glyphbench.pro_harness import ProConfig, ProHarness


def _saved_harness(tmp_path: Path) -> ProHarness:
    cfg = ProConfig(
        model="test-model", num_episodes=2, wandb_run_id="test-run",
        api_key="private-api-key", azure_api_key="private-azure-key",
        base_url="https://" + "user:private-password@example.test/v1?token=private-token",
        azure_endpoint="https://example.test/responses?api-version=preview&api-key=private-query-key",
    )
    harness = ProHarness(cfg, client=object())
    harness._run_dir = str(tmp_path)
    harness._save(
        [harness._failed_episode(0, cfg.seed, "test failure")],
        [{"episode": 0, "seed": cfg.seed, "interactions": []}],
    )
    return harness


def test_saved_and_logged_config_excludes_credentials(tmp_path: Path) -> None:
    harness = _saved_harness(tmp_path)
    text = (tmp_path / "config.json").read_text()
    assert "private-" not in text
    assert "private-" not in harness.cfg.describe()
    config = json.loads(text)
    assert "api_key" not in config
    assert "azure_api_key" not in config
    assert "api-version=preview" in config["azure_endpoint"]


def test_resume_rejects_changed_experiment(tmp_path: Path) -> None:
    harness = _saved_harness(tmp_path)
    assert len(harness._load_checkpointed_results()[0]) == 1
    harness.cfg.seed += 1
    with pytest.raises(ValueError, match="checkpoint configuration differs: seed"):
        harness._load_checkpointed_results()


def test_failed_save_preserves_previous_results(tmp_path: Path) -> None:
    harness = _saved_harness(tmp_path)
    previous = (tmp_path / "metrics.json").read_text()
    with pytest.raises(TypeError):
        harness._save([], [{"unserializable": object()}])
    assert (tmp_path / "metrics.json").read_text() == previous


def test_interrupted_metrics_commit_still_resumes_completed_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _saved_harness(tmp_path)
    results, interactions = harness._load_checkpointed_results()
    previous = (tmp_path / "metrics.json").read_text()
    replace = Path.replace

    def fail_metrics_replace(path: Path, target: Path) -> Path:
        if target.name == "metrics.json":
            raise OSError("simulated failure committing metrics")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_metrics_replace)
    with pytest.raises(OSError, match="simulated failure"):
        harness._save(
            [*results, harness._failed_episode(1, harness.cfg.seed + 1, "test failure")],
            [*interactions, {"episode": 1, "seed": harness.cfg.seed + 1, "interactions": []}],
        )

    assert (tmp_path / "metrics.json").read_text() == previous
    assert len(json.loads((tmp_path / "interactions.json").read_text())) == 2
    resumed_results, resumed_interactions = harness._load_checkpointed_results()
    assert [result.episode for result in resumed_results] == [0]
    assert resumed_interactions == interactions


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "wrong_seed"])
def test_resume_requires_matching_committed_interactions(tmp_path: Path, invalid: str) -> None:
    harness = _saved_harness(tmp_path)
    path = tmp_path / "interactions.json"
    interactions = json.loads(path.read_text())
    if invalid == "missing":
        interactions.clear()
    elif invalid == "duplicate":
        interactions.append(interactions[0])
    else:
        interactions[0]["seed"] += 1
    path.write_text(json.dumps(interactions))
    with pytest.raises(ValueError, match="checkpoint interactions do not match"):
        harness._load_checkpointed_results()
