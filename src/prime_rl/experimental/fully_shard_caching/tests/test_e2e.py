import json
import os
import subprocess
import sys

import pytest
import torch

LAYERS = 2
GRAD_ACCUM = 2
WRAPPED_PARAMETERS_PER_MOE_LAYER = 2
MOE_LAYERS = LAYERS - 1
WRAPPED_PARAMETERS = WRAPPED_PARAMETERS_PER_MOE_LAYER * MOE_LAYERS

BASE_ARGS = [
    "--layers",
    str(LAYERS),
    "--seq-len",
    "512",
    "--steps",
    "2",
    "--warmup-steps",
    "0",
    "--grad-accum",
    str(GRAD_ACCUM),
    "--deterministic",
]

HELD_ARGS = [
    "--wrap",
    "fp8",
    "--expert-reshard-after-forward",
    "false",
    "--expert-reshard-after-backward",
    "false",
]

RUNS = {
    "none": ["--wrap", "none"],
    "fp8": ["--wrap", "fp8"],
    "fp8_ac": ["--wrap", "fp8", "--ac", "full"],
    "fp8_held": HELD_ARGS,
    "fp8_held_ac": [*HELD_ARGS, "--ac", "full"],
    "toy": ["--wrap", "toy"],
    "toy_uninstalled": ["--wrap", "toy", "--install-prepared", "false"],
}


@pytest.fixture(scope="session")
def metrics(tmp_path_factory) -> dict[str, dict]:
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    output_dir = tmp_path_factory.mktemp("fully_shard_caching")
    environment = dict(os.environ, DG_JIT_CACHE_DIR=os.environ.get("DG_JIT_CACHE_DIR", ""))
    results = {}
    for name, extra in RUNS.items():
        target = output_dir / f"{name}.json"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node",
            "8",
            "-m",
            "prime_rl.experimental.fully_shard_caching.train_mini",
            *BASE_ARGS,
            *extra,
            "--metrics-out",
            str(target),
        ]
        completed = subprocess.run(command, env=environment, capture_output=True, text=True)
        if completed.returncode != 0:
            pytest.fail(f"{name} failed:\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")
        results[name] = json.loads(target.read_text())
    return results


def test_wrapped_fp8_matches_the_unwrapped_kernels(metrics):
    assert metrics["fp8"]["first_step_microbatch_losses"] == metrics["none"]["first_step_microbatch_losses"]


def test_wrapped_toy_matches_its_own_unwrapped_branch(metrics):
    assert metrics["toy"]["first_step_microbatch_losses"] == metrics["toy_uninstalled"]["first_step_microbatch_losses"]


def test_unwrapped_run_never_prepares(metrics):
    assert metrics["none"]["prepare_calls_total"] == 0
    assert metrics["toy_uninstalled"]["prepare_calls_total"] == 0


def test_reshard_after_forward_prepares_twice_per_microbatch(metrics):
    expected = 2 * WRAPPED_PARAMETERS * GRAD_ACCUM
    assert metrics["fp8"]["prepare_calls_per_measured_step"] == [expected, expected]


def test_held_experts_prepare_once_per_optimizer_step(metrics):
    assert metrics["fp8_held"]["prepare_calls_per_measured_step"] == [WRAPPED_PARAMETERS, WRAPPED_PARAMETERS]


def test_activation_checkpointing_adds_no_prepare_calls(metrics):
    assert metrics["fp8_ac"]["prepare_calls_per_measured_step"] == metrics["fp8"]["prepare_calls_per_measured_step"]
    assert (
        metrics["fp8_held_ac"]["prepare_calls_per_measured_step"]
        == metrics["fp8_held"]["prepare_calls_per_measured_step"]
    )


def test_activation_checkpointing_leaves_the_loss_unchanged(metrics):
    reference = metrics["fp8"]["first_step_microbatch_losses"]
    assert metrics["fp8_ac"]["first_step_microbatch_losses"] == reference
    assert metrics["fp8_held_ac"]["first_step_microbatch_losses"] == reference


def test_held_experts_match_the_resharding_run(metrics):
    assert metrics["fp8_held"]["first_step_microbatch_losses"] == metrics["fp8"]["first_step_microbatch_losses"]
