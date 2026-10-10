import json
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MODEL = "samsja/mini-glm-moe"
TIMEOUT = 10 * 60
# name -> (GPUs, extra trainer args). Fake data is seeded by the global micro batch, so every
# layout trains the same global batch from the same weights.
LAYOUTS = {
    "base": (1, []),
    "dp2": (2, []),
    "cp2": (2, ["--model.cp", "2"]),
    "ep2": (2, ["--model.ep", "2"]),
    "cp2_ep2": (4, ["--model.cp", "2", "--model.ep", "2"]),
}
# Step-1 metrics that must not depend on the layout
METRICS = {"optim/grad_norm": 0.02, "loss/mean": 0.02, "entropy/all/mean": 1e-3, "mismatch_kl/all/mean": 1e-3}


def step_metrics(output_dir: Path, step: int) -> dict:
    metrics = {}
    for path in output_dir.rglob("metrics.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("step") == step:
                metrics.update(row)
    return metrics


@pytest.fixture(scope="module")
def layout_metrics(run_process: Callable[..., ProcessResult], output_dir: Path) -> dict[str, dict]:
    metrics = {}
    for name, (num_gpus, extra) in LAYOUTS.items():
        run_dir = output_dir / "parallel-invariance" / name
        cmd = [
            "uv",
            "run",
            "torchrun",
            f"--nproc-per-node={num_gpus}",
            "src/prime_rl/trainer/rl/train.py",
            "--model.name",
            MODEL,
            "--model.seq-len",
            "2048",
            "--data.fake.batch-size",
            "4",
            "--max-steps",
            "1",
            "--output-dir",
            run_dir.as_posix(),
            "--monitors.file.path",
            "metrics.jsonl",
            *extra,
        ]
        env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, range(num_gpus)))}
        assert run_process(cmd, env=env, timeout=TIMEOUT).returncode == 0, f"{name} failed"
        metrics[name] = step_metrics(run_dir, step=1)
    return metrics


@pytest.mark.parametrize("layout", [name for name in LAYOUTS if name != "base"])
def test_layout_matches_single_gpu(layout_metrics: dict[str, dict], layout: str):
    base, other = layout_metrics["base"], layout_metrics[layout]
    for metric, rel_tol in METRICS.items():
        assert other[metric] == pytest.approx(base[metric], rel=rel_tol), metric
