import argparse
import json
from pathlib import Path

from scripts.audit_sampling_kernel_goal_pt2 import (
    DEFAULT_PRODUCTION_CONFIG,
    DEFAULT_TRAINER_TEMPLATE,
    audit_production_config,
    audit_production_readiness,
    audit_tail_specialization,
    audit_token_export_run,
    audit_trainer_nsys_hook,
    expand_host_list,
    production_preflight,
)


def _token_export_record(*, mismatch: bool = False) -> dict:
    token_ids = [101, 102, 103]
    same_length_values = [-1.0, -2.0, -3.0]
    trainer_logprobs = [-1.0, -2.0] if mismatch else same_length_values
    return {
        "token_ids": token_ids,
        "loss_mask": [True, False, True],
        "trainer_logprobs": trainer_logprobs,
        "inference_logprobs": same_length_values,
        "log_importance_ratio": [0.0, 0.0, 0.0],
        "importance_ratio": [1.0, 1.0, 1.0],
        "mismatch_kl": [0.0, 0.0, 0.0],
        "entropy": [0.1, 0.2, 0.3],
    }


def _write_token_export_run(root: Path, *, mismatch: bool = False) -> None:
    (root / "logs" / "trainer").mkdir(parents=True)
    (root / "logs" / "trainer" / "node_0.log").write_text("RL trainer finished\n")
    token_exports = root / "run_default" / "token_exports"
    for step in range(2):
        step_dir = token_exports / f"step_{step}"
        step_dir.mkdir(parents=True)
        (step_dir / "STABLE").write_text("")
        for rank in range(2):
            record = _token_export_record(mismatch=mismatch and step == 0 and rank == 0)
            (step_dir / f"rank_{rank}.jsonl").write_text(json.dumps(record) + "\n")


def test_expand_host_list_handles_ranges_and_comma_groups() -> None:
    assert expand_host_list("nid[011153,011166,011175-011176],nid011195") == [
        "nid011153",
        "nid011166",
        "nid011175",
        "nid011176",
        "nid011195",
    ]


def test_production_readiness_requires_full_topology_on_allowed_lane() -> None:
    args = argparse.Namespace(
        allocation_hosts="nid[011153,011166,011175,011195]",
        allowed_hosts="nid011175,nid011195",
        required_train_nodes=4,
        required_inference_replicas=12,
    )

    gate = audit_production_readiness(args)

    assert gate.required_total_nodes == 16
    assert gate.allocation_hosts == ["nid011153", "nid011166", "nid011175", "nid011195"]
    assert gate.allowed_hosts == ["nid011175", "nid011195"]
    assert gate.allocation_has_required_nodes is False
    assert gate.allowed_lane_has_required_nodes is False
    assert gate.pass_gate is False


def test_production_readiness_passes_only_when_all_required_nodes_are_allowed() -> None:
    args = argparse.Namespace(
        allocation_hosts="nid[011100-011115]",
        allowed_hosts="nid[011100-011115]",
        required_train_nodes=4,
        required_inference_replicas=12,
    )

    gate = audit_production_readiness(args)

    assert len(gate.allocation_hosts) == 16
    assert len(gate.allowed_hosts) == 16
    assert gate.allocation_has_required_nodes is True
    assert gate.allowed_lane_has_required_nodes is True
    assert gate.pass_gate is True


def test_default_production_config_shape_matches_gate() -> None:
    gate = audit_production_config(DEFAULT_PRODUCTION_CONFIG)

    assert gate.pass_gate is True
    assert gate.checked_fields == 24
    assert gate.mismatches == []


def test_production_config_shape_rejects_sampling_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "prod.toml"
    config_text = DEFAULT_PRODUCTION_CONFIG.read_text().replace(
        "extra_body = { top_k = 20, min_p = 0.0, presence_penalty = 1.5 }",
        "extra_body = { top_k = 64, min_p = 0.0, presence_penalty = 1.5 }",
        1,
    )
    config_path.write_text(config_text)

    gate = audit_production_config(config_path)

    assert gate.pass_gate is False
    assert gate.mismatches == ["orchestrator.train.sampling.extra_body.top_k: expected 20, got 64"]


def test_trainer_nsys_hook_exists_in_multi_node_template() -> None:
    gate = audit_trainer_nsys_hook(DEFAULT_TRAINER_TEMPLATE)

    assert gate.pass_gate is True
    assert gate.checked_snippets == 8
    assert gate.missing_snippets == []


def test_trainer_nsys_hook_rejects_missing_env_gate(tmp_path: Path) -> None:
    template = tmp_path / "multi_node_rl.sbatch.j2"
    template.write_text(DEFAULT_TRAINER_TEMPLATE.read_text().replace("PRIME_RL_NSYS_TRAINER", "REMOVED"))

    gate = audit_trainer_nsys_hook(template)

    assert gate.pass_gate is False
    assert gate.missing_snippets == ["PRIME_RL_NSYS_TRAINER"]


def test_tail_specialization_gate_accepts_runtime_top_p() -> None:
    gate = audit_tail_specialization()

    assert gate.pass_gate is True
    assert gate.kernel_constexprs == [7]
    assert gate.precompile_top_p_values == [0.95]
    assert gate.error is None


def test_tail_specialization_gate_rejects_duplicate_top_p_values(monkeypatch) -> None:
    from prime_rl.inference.vllm import flashinfer_sampler

    monkeypatch.setattr(
        flashinfer_sampler,
        "_precompile_tail_top_p_values",
        lambda: [0.95, 0.949999988079071],
    )

    gate = audit_tail_specialization()

    assert gate.pass_gate is False
    assert gate.precompile_top_p_values == [0.95, 0.949999988079071]


def test_production_preflight_passes_without_run_logs_on_full_lane() -> None:
    args = argparse.Namespace(
        allocation_hosts="nid[011100-011115]",
        allowed_hosts="nid[011100-011115]",
        required_train_nodes=4,
        required_inference_replicas=12,
        production_config=DEFAULT_PRODUCTION_CONFIG,
        trainer_template=DEFAULT_TRAINER_TEMPLATE,
    )

    preflight = production_preflight(args)

    assert preflight.production_config.pass_gate is True
    assert preflight.trainer_nsys_hook.pass_gate is True
    assert preflight.tail_specialization.pass_gate is True
    assert preflight.production_readiness.pass_gate is True
    assert preflight.full_production_ready is True


def test_production_preflight_fails_on_two_node_lane() -> None:
    args = argparse.Namespace(
        allocation_hosts="nid[011153,011166,011175,011195]",
        allowed_hosts="nid011175,nid011195",
        required_train_nodes=4,
        required_inference_replicas=12,
        production_config=DEFAULT_PRODUCTION_CONFIG,
        trainer_template=DEFAULT_TRAINER_TEMPLATE,
    )

    preflight = production_preflight(args)

    assert preflight.production_config.pass_gate is True
    assert preflight.trainer_nsys_hook.pass_gate is True
    assert preflight.tail_specialization.pass_gate is True
    assert preflight.production_readiness.pass_gate is False
    assert preflight.full_production_ready is False


def test_token_export_canary_accepts_finished_stable_finite_matching_shapes(tmp_path: Path) -> None:
    _write_token_export_run(tmp_path)

    run = audit_token_export_run("patched", tmp_path)

    assert run.trainer_finished is True
    assert run.jsonl_files == 4
    assert run.stable_files == 2
    assert run.rows == 4
    assert run.tokens == 12
    assert run.loss_tokens == 8
    assert run.shape_mismatches == 0
    assert run.bad_numeric_values == 0
    assert run.pass_gate is True


def test_token_export_canary_rejects_shape_mismatch(tmp_path: Path) -> None:
    _write_token_export_run(tmp_path, mismatch=True)

    run = audit_token_export_run("patched", tmp_path)

    assert run.shape_mismatches == 1
    assert run.pass_gate is False
