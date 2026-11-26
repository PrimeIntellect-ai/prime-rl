from pathlib import Path

from prime_rl.configs.inference import InferenceConfig
from prime_rl.entrypoints.inference import write_slurm_script

BREAKABLE_OFF = "export VLLM_USE_BREAKABLE_CUDAGRAPH=0"


def _script(tmp_path: Path, prefill_vllm_overrides: dict) -> str:
    deployment = {"type": "disaggregated", "prefill_vllm_overrides": prefill_vllm_overrides}
    config = InferenceConfig(deployment=deployment, slurm={}, output_dir=tmp_path)
    script_path = tmp_path / "inference.sbatch"
    write_slurm_script(config, tmp_path / "inference.json", tmp_path / "logs", script_path)
    return script_path.read_text()


def test_prefill_keeps_breakable_cudagraphs_unless_it_runs_deepep_high_throughput(tmp_path):
    """vLLM cannot capture the DeepEP high-throughput dispatch in a breakable CUDA graph, so prefill turns them off
    for that backend (the default). With another backend it keeps them, and with them its prefill CUDA graphs."""
    assert BREAKABLE_OFF in _script(tmp_path, {})
    assert BREAKABLE_OFF in _script(tmp_path, {"all2all_backend": "deepep_high_throughput"})
    assert BREAKABLE_OFF not in _script(tmp_path, {"all2all_backend": "flashinfer_nvlink_one_sided"})
