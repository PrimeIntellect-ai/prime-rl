import json

import pytest

from prime_rl.baselines.benchmark import ModelSpec, artifact_complete, filter_blocked_specs, select_specs, slug


def test_slug_normalizes_model_ids():
    assert slug("Qwen/Qwen3.5-35B_A3B") == "qwen35-35b-a3b"


def test_artifact_complete_requires_records_and_clean_summary(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text(json.dumps({"num_rollouts": 2, "error_rate": 0.0}))

    assert not artifact_complete(output_dir, 2)

    (output_dir / "records.jsonl").write_text("{}\n{}\n")
    assert artifact_complete(output_dir, 2)

    (output_dir / "records.jsonl").write_text("{}\n")
    assert not artifact_complete(output_dir, 2)

    (output_dir / "records.jsonl").write_text("{}\n{}\n")
    (output_dir / "summary.json").write_text(json.dumps({"num_rollouts": 2}))
    assert not artifact_complete(output_dir, 2)

    (output_dir / "summary.json").write_text(json.dumps({"num_rollouts": 2, "error_rate": 0.5}))
    assert not artifact_complete(output_dir, 2)


def test_select_specs_rejects_unmatched_requested_models():
    ok = ModelSpec("org/ok-model", "ok", "test", tp=1, dp=1, max_concurrency=1)
    other = ModelSpec("org/other-model", "other", "test", tp=1, dp=1, max_concurrency=1)

    assert select_specs([ok, other], {"ok", "other-model"}) == [ok, other]

    with pytest.raises(ValueError, match="typo"):
        select_specs([ok, other], {"ok", "typo"})


def test_filter_blocked_specs_skips_defaults_and_errors_on_explicit_selection():
    ok = ModelSpec("org/ok", "ok", "test", tp=1, dp=1, max_concurrency=1)
    blocked = ModelSpec(
        "org/bad",
        "bad",
        "test",
        tp=1,
        dp=1,
        max_concurrency=1,
        blocked_reason="known bad",
    )

    specs, summaries = filter_blocked_specs([ok, blocked], explicitly_requested=False, include_blocked=False)
    assert specs == [ok]
    assert summaries == {"bad": {"skipped": True, "blocked": True, "blocked_reason": "known bad"}}

    with pytest.raises(ValueError, match="Explicitly selected blocked"):
        filter_blocked_specs([blocked], explicitly_requested=True, include_blocked=False)
