from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints import launch
from prime_rl.entrypoints import rl as rl_entrypoint


def make_gpu_layout_config(tmp_path: Path, *, hosts: list[str] | None = None, slurm: bool = False) -> RLConfig:
    config = RLConfig(
        max_steps=1,
        output_dir=tmp_path / "run",
        dry_run=True,
        model={"name": "Qwen/Qwen3-0.6B"},
        weight_broadcast={"type": "nccl"},
        trainer={},
        orchestrator={"client": {"dp_rank_count": 6}},
        deployment={
            "type": "gpu_layout",
            "gpus_per_node": 4,
            "hosts": hosts,
            "nodes": [
                {"inference": [0, 1, 2, 3]},
                {"inference": [0, 1], "trainer": [2, 3]},
            ],
        },
        inference={"parallel": {"tp": 1, "dp": 1}},
        slurm={"project_dir": tmp_path, "partition": "debug"} if slurm else None,
    )
    config.output_dir.mkdir(parents=True)
    return config


def test_rlvr_command_uses_existing_rl_entrypoint() -> None:
    args = argparse.Namespace(
        config=[Path("configs/omni_math2/run.toml")],
        output_dir=Path("/tmp/run"),
        dry_run=True,
        extra=[],
    )

    assert launch._build_rlvr_command(args) == [
        "uv",
        "run",
        "--no-sync",
        "rl",
        "@",
        "configs/omni_math2/run.toml",
        "--output-dir",
        "/tmp/run",
        "--dry-run",
    ]


def test_shell_join_keeps_slurm_job_id_expansion_live() -> None:
    rendered = launch._shell_join(["cmd", "--job", "${SLURM_JOB_ID}", "--literal", "a b"])

    assert '--job "$SLURM_JOB_ID"' in rendered
    assert "--literal 'a b'" in rendered


def test_offline_eval_args_default_to_routed_multinode() -> None:
    args = argparse.Namespace(
        arm="arm",
        run_root=Path("outputs/run/run_default"),
        weights_root=None,
        output_dir=Path("/tmp/eval"),
        base_model="model",
        num_examples=600,
        rollouts_per_example=8,
        max_concurrency=64,
        score_max_concurrency=1024,
        max_retries=3,
        ks=[1, 2, 8],
        steps=[25],
        step_interval=25,
        min_step=25,
        max_step=100,
        base_url=None,
        admin_url=[],
        nodes=8,
        gpus_per_node=4,
        dp_per_node=4,
        tp=1,
        api_server_count=4,
        dp_local=4,
        port=9800,
        router_port=None,
        backend_port=9900,
        gpu_memory_utilization=0.95,
        max_model_len=16384,
        max_num_seqs=192,
        max_num_batched_tokens=65536,
    )

    rendered = launch._offline_eval_args(args)

    assert "--launch-mode" in rendered
    assert "srun_multinode" in rendered
    assert "--launch-nodes" in rendered
    assert "8" in rendered
    assert "--launch-srun-job-id" in rendered
    assert "${SLURM_JOB_ID}" in rendered


def test_offline_eval_parser_defaults_to_high_concurrency(monkeypatch) -> None:
    monkeypatch.delenv("OFFLINE_EVAL_MAX_CONCURRENCY", raising=False)

    args = launch.build_parser().parse_args(
        [
            "offline-eval",
            "--arm",
            "arm",
            "--run-root",
            "outputs/run/run_default",
        ]
    )

    assert args.max_concurrency == 256


def test_offline_eval_env_script_preflights_requested_weights() -> None:
    args = argparse.Namespace(
        root=Path("/repo"),
        run_root=Path("outputs/run/run_default"),
        weights_root=Path("outputs/run/run_default/broadcasts"),
        output_dir=Path("/tmp/eval"),
        wait_step=None,
        base_url=None,
        driver_node_count=0,
        nodes=8,
        patched_verifiers=Path("/tmp/verifiers"),
        omni_env_path=Path("/repo/environments/omni_math2_singleturn"),
        disable_router=False,
        router_policy="round_robin",
        compare_output=Path("/tmp/compare.md"),
        steps=[75, 25],
    )

    rendered = launch._offline_eval_env_script(args, command=["uv", "run", "ok"])

    assert "offline eval weight preflight" in rendered
    assert "outputs/run/run_default/broadcasts/step_25" in rendered
    assert "outputs/run/run_default/broadcasts/step_75" in rendered
    assert "missing requested checkpoint directory" in rendered
    assert "compgen -G" in rendered


def test_gpu_layout_step_resources_use_shared_margins(monkeypatch) -> None:
    def fake_run(cmd, capture_output, text):
        assert cmd == ["scontrol", "show", "node", "node-a", "-o"]
        return SimpleNamespace(returncode=0, stdout="NodeName=node-a CPUTot=128 RealMemory=250000", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert rl_entrypoint._gpu_layout_step_resources("node-a") == (112, 233616)


def test_gpu_layout_host_selection_rejects_ambiguous_extra_allocation_nodes(monkeypatch, tmp_path) -> None:
    config = make_gpu_layout_config(tmp_path)

    def fake_run(cmd, capture_output, text):
        assert cmd == ["scontrol", "show", "hostnames", "node-[1-3]"]
        return SimpleNamespace(returncode=0, stdout="node-1\nnode-2\nnode-3\n", stderr="")

    monkeypatch.setenv("SLURM_JOB_NODELIST", "node-[1-3]")
    monkeypatch.setattr(subprocess, "run", fake_run)

    try:
        rl_entrypoint._select_gpu_layout_hosts(config.deployment)
    except RuntimeError as exc:
        assert "deployment.hosts" in str(exc)
    else:
        raise AssertionError("expected ambiguous gpu_layout allocation to fail")


def test_gpu_layout_slurm_dry_run_writes_referenced_launcher_script(tmp_path) -> None:
    config = make_gpu_layout_config(tmp_path, hosts=["node-a", "node-b"], slurm=True)

    rl_entrypoint.rl_slurm(config)

    launcher = config.output_dir / rl_entrypoint.GPU_LAYOUT_SCRIPT
    sbatch = config.output_dir / rl_entrypoint.RL_SBATCH
    assert launcher.exists()
    assert launcher.stat().st_mode & 0o111
    assert f"bash {launcher}" in sbatch.read_text()
    assert str(rl_entrypoint._GPU_LAYOUT_CPU_MARGIN) in sbatch.read_text()
    assert str(rl_entrypoint._GPU_LAYOUT_MEM_MARGIN_MB) in sbatch.read_text()
