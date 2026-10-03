"""Rendered SLURM scripts for representative configs, checked against golden files.

Regenerate the golden files after an intended template change with
`uv run python tests/unit/test_slurm_templates.py` and review the diff.
"""

import tempfile
from pathlib import Path

import pytest

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.entrypoints import inference, rl, sft
from prime_rl.utils.config import cli

GOLDEN_DIR = Path(__file__).parent / "slurm_golden"


def one_prefill_one_decode(prefix: str) -> list[str]:
    """Shrink a P/D topology to one prefill and one decode node to keep the llm-d goldens small."""
    fields = ["prefill-nodes-per-replica", "num-prefill-replicas", "decode-nodes-per-replica", "num-decode-replicas"]
    return [arg for field in fields for arg in (f"{prefix}.{field}", "1")]


CASES = {
    "rl_glm53_pd": (RLConfig, ["examples/advanced/glm-5.3/swe.toml"], []),
    "rl_glm53_pd_llmd": (
        RLConfig,
        ["examples/advanced/glm-5.3/swe.toml", "examples/advanced/glm-5.3/swe-llmd.toml"],
        one_prefill_one_decode("--inference.deployment"),
    ),
    "rl_minimax_m25": (RLConfig, ["configs/advanced/minimax-m2.5/swe.toml"], []),
    "infer_glm53_pd": (InferenceConfig, ["examples/advanced/glm-5.3/infer/pd.toml"], []),
    "infer_glm53_pd_llmd": (
        InferenceConfig,
        ["examples/advanced/glm-5.3/infer/pd-llmd.toml"],
        one_prefill_one_decode("--deployment"),
    ),
    "infer_multi_node": (
        InferenceConfig,
        ["configs/advanced/deepseek-v4-flash/inference.toml"],
        ["--deployment.type", "multi_node", "--vllm.tensor-parallel-size", "4", "--slurm.partition", "cluster"],
    ),
    "sft_glm53_multi_node": (SFTConfig, ["examples/advanced/glm-5.3/sft/h200/base.toml"], []),
    "sft_online_eval": (
        SFTConfig,
        ["configs/basic/reverse-text/sft.toml"],
        ["--deployment.type", "multi_node", "--deployment.num-infer-nodes", "2", "--slurm.partition", "cluster"],
    ),
}


def render(name: str, tmp_path: Path) -> str:
    config_cls, tomls, overrides = CASES[name]
    args = [arg for toml in tomls for arg in ("@", toml)]
    args += ["--output-dir", str(tmp_path), "--slurm.project-dir", "/project", *overrides]
    if config_cls is not InferenceConfig:
        args += ["--run.name", "golden"]
    config = cli(config_cls, args=args)
    script_path = tmp_path / "script.sbatch"
    config_dir, log_dir = Path("/run/configs"), Path("/run/logs")
    if config_cls is RLConfig:
        rl.write_slurm_script(config, config_dir, log_dir, script_path)
    elif config_cls is InferenceConfig:
        inference.write_slurm_script(config, config_dir / "inference.json", log_dir, script_path)
    else:
        sft.write_slurm_script(config, config_dir / "sft.json", log_dir, script_path, "run-id", wandb_shared=False)
    return script_path.read_text().replace(str(tmp_path), "/output")


@pytest.mark.parametrize("name", CASES)
def test_rendered_slurm_script_matches_golden(name: str, tmp_path: Path):
    assert render(name, tmp_path) == (GOLDEN_DIR / f"{name}.sbatch").read_text()


if __name__ == "__main__":
    GOLDEN_DIR.mkdir(exist_ok=True)
    for name in CASES:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (GOLDEN_DIR / f"{name}.sbatch").write_text(render(name, Path(tmp_dir)))
