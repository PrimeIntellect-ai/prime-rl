import json
from pathlib import Path
from typing import Callable

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from tests.conftest import ProcessResult

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

BASE_MODEL = "PrimeIntellect/Qwen3-0.6B"
TIMEOUT = 5 * 60
NUM_STEPS = 4
# name -> (ranks, extra trainer args). All ranks share GPU 0.
LAYOUTS = {
    "base": (1, []),
    "dp2": (2, []),
    "cp2": (2, ["--model.cp", "2"]),
}
# CP splits attention across ranks, which moves bf16 gradients by ~1e-4 (relative)
STEP_1_REL_TOL = {"loss/mean": 1e-5, "optim/grad_norm": 2e-3}
LATER_LOSS_REL_TOL = 2e-4


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = AutoConfig.from_pretrained(
        BASE_MODEL,
        num_hidden_layers=2,
        layer_types=["full_attention"] * 2,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
    )
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-qwen3")
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
    return path


def read_metrics(run_dir: Path) -> dict[int, dict]:
    metrics = {}
    for line in next(run_dir.rglob("metrics.jsonl")).read_text().splitlines():
        row = json.loads(line)
        if "step" in row:
            metrics.setdefault(row["step"], {}).update(row)
    return metrics


@pytest.fixture(scope="module")
def layout_metrics(
    run_process: Callable[..., ProcessResult], output_dir: Path, tiny_model: Path
) -> dict[str, dict[int, dict]]:
    run_output_dir = output_dir / "one-gpu-parallel-invariance"
    metrics = {}
    for name, (num_ranks, extra) in LAYOUTS.items():
        cmd = [
            "uv",
            "run",
            "torchrun",
            f"--nproc-per-node={num_ranks}",
            "src/prime_rl/trainer/sft/train.py",
            "--model.name",
            tiny_model.as_posix(),
            "--tokenizer.name",
            BASE_MODEL,
            "--data.name",
            "PrimeIntellect/Reverse-Text-SFT",
            # Shorter than two samples, so every row holds exactly one sample and all layouts train the same samples
            "--data.seq-len",
            "96",
            "--data.batch-size",
            "4",
            "--optim.lr",
            "1e-3",
            "--max-steps",
            str(NUM_STEPS),
            "--output-dir",
            run_output_dir.as_posix(),
            "--run.name",
            name,
            "--clean",
            "--monitors.file.path",
            "metrics.jsonl",
            "--no-monitors.prime",
            *extra,
        ]
        env = {"CUDA_VISIBLE_DEVICES": "0", "NCCL_MULTI_RANK_GPU_ENABLE": "1"}
        assert run_process(cmd, env=env, timeout=TIMEOUT).returncode == 0, f"{name} failed"
        metrics[name] = read_metrics(run_output_dir / name)
    return metrics


@pytest.mark.parametrize("layout", [name for name in LAYOUTS if name != "base"])
def test_layout_matches_single_rank(layout_metrics: dict[str, dict[int, dict]], layout: str):
    base, other = layout_metrics["base"], layout_metrics[layout]
    assert sorted(other) == sorted(base) == list(range(1, NUM_STEPS + 1))
    for metric, rel_tol in STEP_1_REL_TOL.items():
        assert other[1][metric] == pytest.approx(base[1][metric], rel=rel_tol), metric
    for step in range(2, NUM_STEPS + 1):
        assert other[step]["loss/mean"] == pytest.approx(base[step]["loss/mean"], rel=LATER_LOSS_REL_TOL), step
