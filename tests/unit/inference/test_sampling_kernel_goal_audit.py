import argparse
import json
from pathlib import Path

from scripts.audit_sampling_kernel_goal_pt2 import (
    DEFAULT_INFERENCE_TEMPLATES,
    DEFAULT_PATCHES_PATH,
    DEFAULT_PRODUCTION_CONFIG,
    DEFAULT_PYPROJECT,
    DEFAULT_SAMPLER_CONTRACT_SWEEP,
    DEFAULT_SAMPLER_SOURCE,
    DEFAULT_TRAINER_TEMPLATE,
    audit_inference_sampler_runtime_hooks,
    audit_production_config,
    audit_production_readiness,
    audit_sampler_contract_sweep_result,
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


def _sampler_contract_rows(rows: int) -> list[dict]:
    specs = [
        (16, 20, 0, False),
        (16, 20, 0, True),
        (64, 20, 0, False),
        (64, 20, 0, True),
        (64, 64, 0, False),
        (64, 64, 0, True),
        (128, 20, 0, True),
        (128, 64, 0, True),
        (16, 20, 1, False),
        (16, 20, 1, True),
        (64, 64, 1, False),
        (64, 64, 1, True),
    ][:rows]
    return [
        {
            "batch_size": batch_size,
            "dense_presence_enabled": dense_presence,
            "device": "cuda",
            "env_enabled": True,
            "expected_patched_logprob_token_id_mismatches": 0,
            "expected_patched_sampled_token_mismatches": 0,
            "expected_patched_selected_rank_mismatches": 0,
            "fastpath_allowed": True,
            "max_expected_patched_logprob_diff": 4.76837158203125e-07,
            "max_forced_logprob_diff": 9.5367431640625e-07,
            "max_num_logprobs": max_num_logprobs,
            "max_patched_logprob_diff": 9.5367431640625e-07,
            "native_cols": max_num_logprobs + 1,
            "patch_marker": True,
            "patched_cols": max_num_logprobs + 1,
            "patched_sampled_ids_shape": [batch_size, 1],
            "presence_penalty": 1.5 if dense_presence else 0.0,
            "prompt_len": 16 if dense_presence else 0,
            "top_k": top_k,
            "top_p": 0.95,
            "unique_output_tokens": 8 if dense_presence else 0,
            "vocab_size": 248320,
        }
        for batch_size, top_k, max_num_logprobs, dense_presence in specs
    ]


def _write_sampler_contract_sweep_result(
    path: Path,
    *,
    failures: int = 0,
    rows: int = 12,
    results: list[dict] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "results": _sampler_contract_rows(rows) if results is None else results,
                "summary": {
                    "rows": rows,
                    "failures": failures,
                    "atol": 1e-6,
                    "max_expected_patched_logprob_diff": 4.76837158203125e-07,
                    "max_forced_logprob_diff": 9.5367431640625e-07,
                    "native_cols": [1, 2],
                    "patched_cols": [1, 2],
                },
            }
        )
    )


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
    assert gate.checked_fields == 36
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


def test_production_config_shape_requires_config_backed_sampler_env(tmp_path: Path) -> None:
    config_path = tmp_path / "prod.toml"
    config_path.write_text(
        DEFAULT_PRODUCTION_CONFIG.read_text().replace(
            """
[inference.finite_topk_sampled_logprob]
enabled = true
tail = "triton"
dense_presence = true
stats_interval = 1000
hit_log_limit = 6
log_fallback = true
precompile_tail = true
precompile_top_k = 20
precompile_top_p = 0.95
precompile_vocab = 248320
precompile_batches = [1, 128, 256]
boundary_tie_guard = true
""",
            "",
        )
    )

    gate = audit_production_config(config_path)

    assert gate.pass_gate is False
    assert "inference.finite_topk_sampled_logprob.enabled: expected True, got <missing>" in gate.mismatches


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


def test_inference_sampler_runtime_hooks_cover_plugin_and_stats_templates() -> None:
    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
    )

    assert gate.pass_gate is True
    assert gate.checked_snippets == 24
    assert gate.missing_snippets == []


def test_inference_sampler_runtime_hooks_reject_missing_plugin_entrypoint(tmp_path: Path) -> None:
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(DEFAULT_PYPROJECT.read_text().replace("vllm.general_plugins", "removed.plugins"))

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=pyproject_path,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == [
        'pyproject.toml: [project.entry-points."vllm.general_plugins"]',
    ]


def test_inference_sampler_runtime_hooks_reject_missing_stats_log_env(tmp_path: Path) -> None:
    template = tmp_path / "inference.sbatch.j2"
    template.write_text(
        DEFAULT_INFERENCE_TEMPLATES[0]
        .read_text()
        .replace("PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_LOG_DIR", "REMOVED_SAMPLER_STATS_LOG_DIR")
    )

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=(template, *DEFAULT_INFERENCE_TEMPLATES[1:]),
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == [
        "inference.sbatch.j2: PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_LOG_DIR",
    ]


def test_inference_sampler_runtime_hooks_rejects_renderer_logprobs_one(tmp_path: Path) -> None:
    renderer_client_path = tmp_path / "client.py"
    renderer_client_path.write_text(
        Path("deps/renderers/renderers/client.py").read_text().replace('sp["logprobs"] = 0', 'sp["logprobs"] = 1')
    )

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
        renderer_client_path=renderer_client_path,
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == ['client.py: sp["logprobs"] = 0']


def test_inference_sampler_runtime_hooks_rejects_missing_top_p_boundary_tie_guard(tmp_path: Path) -> None:
    sampler_source_path = tmp_path / "flashinfer_sampler.py"
    sampler_source_path.write_text(DEFAULT_SAMPLER_SOURCE.read_text().replace("top_p_boundary_tie", "removed"))

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
        sampler_source_path=sampler_source_path,
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == [
        "flashinfer_sampler.py: _has_top_p_boundary_tie",
        'flashinfer_sampler.py: "top_p_boundary_tie"',
    ]


def test_inference_sampler_runtime_hooks_rejects_missing_top_p_tolerance(tmp_path: Path) -> None:
    sampler_source_path = tmp_path / "flashinfer_sampler.py"
    sampler_source_path.write_text(
        DEFAULT_SAMPLER_SOURCE.read_text()
        .replace("_TOP_P_BOUNDARY_TIE_ATOL = 5e-7", "_TOP_P_BOUNDARY_TIE_ATOL = 0.0")
        .replace("atol=_TOP_P_BOUNDARY_TIE_ATOL", "atol=0.0")
    )

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
        sampler_source_path=sampler_source_path,
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == [
        "flashinfer_sampler.py: _TOP_P_BOUNDARY_TIE_ATOL = 5e-7",
        "flashinfer_sampler.py: atol=_TOP_P_BOUNDARY_TIE_ATOL",
    ]


def test_inference_sampler_runtime_hooks_rejects_contract_sweep_without_production_vocab(tmp_path: Path) -> None:
    contract_sweep_path = tmp_path / "probe_prime_flashinfer_sampler_contract_sweep.py"
    contract_sweep_path.write_text(DEFAULT_SAMPLER_CONTRACT_SWEEP.read_text().replace("248320", "4096"))

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
        contract_sweep_path=contract_sweep_path,
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == [
        "probe_prime_flashinfer_sampler_contract_sweep.py: 248320",
    ]


def test_inference_sampler_runtime_hooks_rejects_missing_renderer_token_id_check(tmp_path: Path) -> None:
    renderer_client_path = tmp_path / "client.py"
    renderer_client_path.write_text(
        Path("deps/renderers/renderers/client.py")
        .read_text()
        .replace("isinstance(raw_completion_ids, list)", "removed_token_id_check")
    )

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
        renderer_client_path=renderer_client_path,
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == [
        "client.py: isinstance(raw_completion_ids, list)",
    ]


def test_inference_sampler_runtime_hooks_rejects_missing_renderer_token_id_value_check(tmp_path: Path) -> None:
    renderer_client_path = tmp_path / "client.py"
    renderer_client_path.write_text(
        Path("deps/renderers/renderers/client.py").read_text().replace("token_id < 0", "removed_token_id_value_check")
    )

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
        renderer_client_path=renderer_client_path,
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == [
        "client.py: token_id < 0",
    ]


def test_inference_sampler_runtime_hooks_rejects_missing_renderer_logprob_length_check(tmp_path: Path) -> None:
    renderer_client_path = tmp_path / "client.py"
    renderer_client_path.write_text(
        Path("deps/renderers/renderers/client.py")
        .read_text()
        .replace("len(completion_logprobs) != len(completion_ids)", "removed_logprob_length_check")
    )

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
        renderer_client_path=renderer_client_path,
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == [
        "client.py: len(completion_logprobs) != len(completion_ids)",
    ]


def test_inference_sampler_runtime_hooks_rejects_missing_renderer_logprob_finite_check(tmp_path: Path) -> None:
    renderer_client_path = tmp_path / "client.py"
    renderer_client_path.write_text(
        Path("deps/renderers/renderers/client.py")
        .read_text()
        .replace("math.isfinite(logprob)", "removed_logprob_finite_check")
    )

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
        renderer_client_path=renderer_client_path,
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == [
        "client.py: math.isfinite(logprob)",
    ]


def test_inference_sampler_runtime_hooks_rejects_missing_logprobs_zero_serving_contract(tmp_path: Path) -> None:
    serving_tokens_test_path = tmp_path / "test_serving_tokens.py"
    serving_tokens_test_path.write_text(
        Path("tests/unit/inference/test_serving_tokens.py")
        .read_text()
        .replace("test_generate_logprobs_zero_serializes_sampled_completion_logprob", "removed_contract")
    )

    gate = audit_inference_sampler_runtime_hooks(
        pyproject_path=DEFAULT_PYPROJECT,
        patches_path=DEFAULT_PATCHES_PATH,
        template_paths=DEFAULT_INFERENCE_TEMPLATES,
        serving_tokens_test_path=serving_tokens_test_path,
    )

    assert gate.pass_gate is False
    assert gate.missing_snippets == [
        "test_serving_tokens.py: test_generate_logprobs_zero_serializes_sampled_completion_logprob",
    ]


def test_sampler_contract_sweep_result_accepts_production_vocab_json(tmp_path: Path) -> None:
    result_path = tmp_path / "sampler_contract.json"
    _write_sampler_contract_sweep_result(result_path)

    gate = audit_sampler_contract_sweep_result(result_path)

    assert gate.pass_gate is True
    assert gate.rows == 12
    assert gate.failures == 0
    assert gate.mismatches == []


def test_sampler_contract_sweep_result_rejects_failures(tmp_path: Path) -> None:
    result_path = tmp_path / "sampler_contract.json"
    _write_sampler_contract_sweep_result(result_path, failures=1)

    gate = audit_sampler_contract_sweep_result(result_path)

    assert gate.pass_gate is False
    assert gate.mismatches == ["failures: expected 0, got 1"]


def test_sampler_contract_sweep_result_rejects_missing_width1(tmp_path: Path) -> None:
    result_path = tmp_path / "sampler_contract.json"
    _write_sampler_contract_sweep_result(result_path, rows=8)

    gate = audit_sampler_contract_sweep_result(result_path)

    assert gate.pass_gate is False
    assert gate.mismatches == [
        "rows: expected at least 12, got 8",
        "results: expected at least 12 rows, got 8",
    ]


def test_sampler_contract_sweep_result_rejects_empty_rows_even_with_passing_summary(tmp_path: Path) -> None:
    result_path = tmp_path / "sampler_contract.json"
    _write_sampler_contract_sweep_result(result_path, results=[])

    gate = audit_sampler_contract_sweep_result(result_path)

    assert gate.pass_gate is False
    assert gate.mismatches == [
        "summary rows: expected 0, got 12",
        "results: expected at least 12 rows, got 0",
    ]


def test_sampler_contract_sweep_result_rejects_row_fastpath_miss(tmp_path: Path) -> None:
    result_path = tmp_path / "sampler_contract.json"
    results = _sampler_contract_rows(12)
    results[0]["fastpath_allowed"] = False
    _write_sampler_contract_sweep_result(result_path, results=results)

    gate = audit_sampler_contract_sweep_result(result_path)

    assert gate.pass_gate is False
    assert gate.mismatches == ["results[0].fastpath_allowed: expected True, got False"]


def test_tail_specialization_gate_accepts_runtime_top_p() -> None:
    gate = audit_tail_specialization()

    assert gate.pass_gate is True
    assert gate.kernel_constexprs == [gate.kernel_parameters.index("K_BLOCK")]
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
    assert preflight.inference_sampler_runtime_hooks.pass_gate is True
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
    assert preflight.inference_sampler_runtime_hooks.pass_gate is True
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
