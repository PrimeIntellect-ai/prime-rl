"""Render the SLURM templates and check the output is valid bash.

The multi-node launch blocks live inside a single-quoted ``srun bash -c '...'``
body, so a stray single quote anywhere in the template (or in an included
helper) — an apostrophe in a comment is the easy one to write — ends the quoted
body and the rest of the script is parsed by the outer shell. Rendering alone
does not catch it; the job only dies at submit time, on the cluster.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.inference import write_slurm_script as write_inference_slurm_script
from prime_rl.entrypoints.rl import write_slurm_script as write_rl_slurm_script

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to syntax-check the scripts")

MOONCAKE = {"type": "mooncake", "cpu": {"num_bytes": 1_000_000_000}}
SLURM = {"partition": "cluster", "job_name": "test"}

# Deployment shapes that exercise every launch block of `inference.sbatch.j2`.
INFERENCE_CONFIGS: dict[str, dict[str, Any]] = {
    "single_node": {
        "vllm": {"tensor_parallel_size": 8},
        "deployment": {"type": "single_node"},
    },
    # tp=8 EP: one engine per node, the dp=1 branch of the multi-node block.
    "multi_node_ep_dp1": {
        "vllm": {"tensor_parallel_size": 8, "enable_expert_parallel": True},
        "deployment": {"type": "multi_node", "num_nodes": 2},
        "kv_cache_offload": MOONCAKE,
    },
    # tp=4 EP: two engines per node, the external-LB branch with per-rank overrides.
    "multi_node_ep_dp2": {
        "vllm": {"tensor_parallel_size": 4, "enable_expert_parallel": True},
        "deployment": {"type": "multi_node", "num_nodes": 2},
        "kv_cache_offload": MOONCAKE,
    },
    "multi_node_llmd_router": {
        "vllm": {"tensor_parallel_size": 8},
        "deployment": {"type": "multi_node", "num_nodes": 2},
        "router": {"type": "llm-d"},
    },
    "disaggregated": {
        "vllm": {"tensor_parallel_size": 4},
        "deployment": {"type": "disaggregated"},
        "kv_cache_offload": MOONCAKE,
    },
    "disaggregated_llmd_router": {
        "vllm": {"tensor_parallel_size": 4},
        "deployment": {"type": "disaggregated"},
        "router": {"type": "llm-d"},
    },
}

RL_CONFIGS: dict[str, dict[str, Any]] = {
    "single_node": {
        "deployment": {"type": "single_node"},
    },
    "multi_node": {
        "inference": {
            "vllm": {"tensor_parallel_size": 8, "enable_expert_parallel": True},
            "deployment": {"type": "multi_node", "num_nodes": 1},
            "kv_cache_offload": MOONCAKE,
        },
        "deployment": {"type": "multi_node", "num_train_nodes": 1, "num_infer_nodes": 1},
    },
    "multi_node_disaggregated": {
        "inference": {
            "vllm": {"tensor_parallel_size": 4},
            "deployment": {"type": "disaggregated"},
            "kv_cache_offload": MOONCAKE,
        },
        "deployment": {"type": "multi_node", "num_train_nodes": 1, "num_infer_nodes": 2},
    },
}


def check_bash_syntax(script_path: Path) -> None:
    result = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"rendered {script_path.name} is not valid bash:\n{result.stderr}"


@pytest.mark.parametrize("name", INFERENCE_CONFIGS)
def test_rendered_inference_slurm_scripts_are_valid_bash(name: str, tmp_path: Path):
    config = InferenceConfig.model_validate({**INFERENCE_CONFIGS[name], "slurm": SLURM, "output_dir": str(tmp_path)})
    script_path = tmp_path / "inference.sbatch"
    write_inference_slurm_script(config, tmp_path / "inference.toml", script_path)
    check_bash_syntax(script_path)


@pytest.mark.parametrize("name", RL_CONFIGS)
def test_rendered_rl_slurm_scripts_are_valid_bash(name: str, tmp_path: Path):
    config = RLConfig.model_validate(
        {
            **RL_CONFIGS[name],
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "orchestrator": {"renderer": {"name": "default"}},
            "trainer": {},
            "slurm": SLURM,
            "output_dir": str(tmp_path),
        }
    )
    script_path = tmp_path / "rl.sbatch"
    write_rl_slurm_script(config, tmp_path, script_path)
    check_bash_syntax(script_path)
