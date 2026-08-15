import importlib.util
import json
from pathlib import Path
from typing import Callable

import pytest
import torch

from tests.conftest import ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

RUN_NAME = "parallelism"
CONFIG = "configs/ci/integration/parallelism/sft.toml"
TIMEOUT = 600  # 10 minutes

# Every layout trains the same global batch of fake data (a pure function of the sample
# index), so the per-step loss must match the single-GPU baseline. Parallelism only
# reorders bf16 reductions, which drifts the loss far less than this; a bad shard, a
# wrong reduction group or a wrong gradient scale moves it by orders of magnitude more.
LOSS_RTOL = 5e-3

# (name, number of GPUs, config overrides)
PARALLELISMS: list[tuple[str, int, list[str]]] = [
    ("fsdp", 2, []),
    ("hsdp", 2, ["--model.dp-replicate", "2"]),
    ("cp-ring", 2, ["--model.cp", "2", "--model.cp-style", "ring"]),
    ("cp-ulysses", 2, ["--model.cp", "2", "--model.cp-style", "ulysses"]),
    ("ep", 2, ["--model.ep", "2"]),
]

# DeepEP forces gradient clipping off, so it is compared against its own baseline.
NO_GRAD_CLIPPING = ["--optim.max-norm", "None"]


def has_deepep() -> bool:
    """Whether the DeepEP backend can run here. It is an optional extra, pre-built for Hopper onwards."""
    if importlib.util.find_spec("deep_ep") is None:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def read_losses(run_dir: Path) -> list[float]:
    """Read the per-step training loss from a run's JSONL metric sink."""
    metrics_path = run_dir / "metrics.jsonl"
    with open(metrics_path, "r") as f:
        metrics = [json.loads(line) for line in f]
    losses = {metric["step"]: metric["loss/mean"] for metric in metrics if "loss/mean" in metric}
    assert losses, f"No loss metrics found in {metrics_path}"
    return [losses[step] for step in sorted(losses)]


def check_losses_match(name: str, losses: list[float], baseline_losses: list[float]) -> None:
    """Helper to assert that a run reproduces the baseline loss at every step"""
    assert len(losses) == len(baseline_losses), (
        f"{name} logged {len(losses)} steps, baseline logged {len(baseline_losses)}"
    )
    for step, (loss, baseline_loss) in enumerate(zip(losses, baseline_losses), start=1):
        assert loss == pytest.approx(baseline_loss, rel=LOSS_RTOL), (
            f"{name} loss at step {step} ({loss}) differs from baseline ({baseline_loss}) "
            f"by more than {LOSS_RTOL:.1%} ({name}: {losses}, baseline: {baseline_losses})"
        )


@pytest.fixture(scope="module")
def run_sft(
    run_process: Callable[..., ProcessResult], output_dir: Path
) -> Callable[[str, int, list[str]], list[float]]:
    """Factory fixture running SFT for one parallelism layout, returning its per-step losses."""

    def _run_sft(name: str, num_gpus: int, args: list[str]) -> list[float]:
        run_name = f"{RUN_NAME}-{name}"
        cmd = [
            "uv",
            "run",
            "sft",
            "@",
            CONFIG,
            "--deployment.num-gpus",
            str(num_gpus),
            "--clean",
            "--output-dir",
            output_dir.as_posix(),
            "--run.name",
            run_name,
            *args,
        ]
        process = run_process(cmd, timeout=TIMEOUT)
        assert process.returncode == 0, f"SFT process for {name} has non-zero return code ({process})"
        return read_losses(output_dir / run_name)

    return _run_sft


@pytest.fixture(scope="module")
def baseline_losses(run_sft: Callable[[str, int, list[str]], list[float]]) -> list[float]:
    """Fixture for the losses of a single GPU run without any parallelism."""
    return run_sft("baseline", 1, [])


@pytest.fixture(scope="module")
def unclipped_baseline_losses(run_sft: Callable[[str, int, list[str]], list[float]]) -> list[float]:
    """Fixture for the single GPU losses without gradient clipping, to compare DeepEP against."""
    return run_sft("baseline-unclipped", 1, NO_GRAD_CLIPPING)


def test_baseline_loss_changes(baseline_losses: list[float]):
    """Tests that the baseline trains, else comparing losses would not test the backward pass."""
    assert len(set(baseline_losses)) > 1, f"Baseline loss is constant across steps ({baseline_losses})"


@pytest.mark.parametrize(("name", "num_gpus", "args"), PARALLELISMS, ids=[name for name, _, _ in PARALLELISMS])
def test_loss_matches_baseline(
    name: str,
    num_gpus: int,
    args: list[str],
    run_sft: Callable[[str, int, list[str]], list[float]],
    baseline_losses: list[float],
):
    """Tests that a parallelism technique yields the baseline loss on the same fixed data."""
    check_losses_match(name, run_sft(name, num_gpus, args), baseline_losses)


@pytest.mark.skipif(not has_deepep(), reason="DeepEP is not installed or unsupported on this GPU")
def test_deepep_loss_matches_baseline(
    run_sft: Callable[[str, int, list[str]], list[float]], unclipped_baseline_losses: list[float]
):
    """Tests that expert parallelism with the DeepEP all-to-all kernels yields the baseline loss."""
    args = ["--model.ep", "2", "--model.ep-comm-backend", "deepep", *NO_GRAD_CLIPPING]
    check_losses_match("ep-deepep", run_sft("ep-deepep", 2, args), unclipped_baseline_losses)
