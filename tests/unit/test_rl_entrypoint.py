import subprocess
from pathlib import Path
from types import SimpleNamespace

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints import rl as rl_entrypoint


def make_gpu_layout_config(tmp_path: Path, *, hosts: list[str] | None = None, slurm: bool = False) -> RLConfig:
    config = RLConfig(
        max_steps=1,
        output_dir=tmp_path / "run",
        dry_run=True,
        model={"name": "Qwen/Qwen3-0.6B"},
        weight_broadcast={"type": "nccl"},
        trainer={},
        orchestrator={"student": {"client": {"dp_rank_count": 6}}},
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


def test_gpu_layout_step_resources_use_shared_margins(monkeypatch):
    def fake_run(cmd, capture_output, text):
        assert cmd == ["scontrol", "show", "node", "node-a", "-o"]
        return SimpleNamespace(returncode=0, stdout="NodeName=node-a CPUTot=128 RealMemory=250000", stderr="")

    monkeypatch.setattr(rl_entrypoint.subprocess, "run", fake_run)

    assert rl_entrypoint._gpu_layout_step_resources("node-a") == (112, 233616)


def test_gpu_layout_host_selection_rejects_ambiguous_extra_allocation_nodes(monkeypatch, tmp_path):
    config = make_gpu_layout_config(tmp_path)

    def fake_run(cmd, capture_output, text):
        assert cmd == ["scontrol", "show", "hostnames", "node-[1-3]"]
        return SimpleNamespace(returncode=0, stdout="node-1\nnode-2\nnode-3\n", stderr="")

    monkeypatch.setenv("SLURM_JOB_NODELIST", "node-[1-3]")
    monkeypatch.setattr(rl_entrypoint.subprocess, "run", fake_run)

    try:
        rl_entrypoint._select_gpu_layout_hosts(config.deployment)
    except RuntimeError as exc:
        assert "deployment.hosts" in str(exc)
    else:
        raise AssertionError("expected ambiguous gpu_layout allocation to fail")


def test_gpu_layout_slurm_dry_run_writes_current_dialect_launcher(tmp_path):
    config = make_gpu_layout_config(tmp_path, hosts=["node-a", "node-b"], slurm=True)

    rl_entrypoint.rl_slurm(config)

    launcher = config.output_dir / rl_entrypoint.GPU_LAYOUT_SCRIPT
    sbatch = config.output_dir / rl_entrypoint.RL_SBATCH
    assert launcher.exists()
    assert launcher.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", launcher], check=True)

    sbatch_text = sbatch.read_text()
    assert f"bash {launcher}" in sbatch_text
    assert str(rl_entrypoint._GPU_LAYOUT_CPU_MARGIN) in sbatch_text
    assert str(rl_entrypoint._GPU_LAYOUT_MEM_MARGIN_MB) in sbatch_text

    launcher_text = launcher.read_text()
    assert "--student.client.base-url" in launcher_text
    assert "--student.client.admin-base-url" in launcher_text
    assert "--client.base-url" not in launcher_text
