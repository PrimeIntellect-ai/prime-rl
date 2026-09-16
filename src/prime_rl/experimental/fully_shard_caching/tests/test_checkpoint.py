import json
import os
import subprocess
import sys

import pytest
import torch

DRIVER = "prime_rl.experimental.fully_shard_caching.checkpoint_driver"
RANKS = 8
BASE_ARGS = ["--layers", "2", "--seq-len", "512", "--grad-accum", "2", "--eval-microbatches", "2"]
SAVE_SEED = "0"
LOAD_SEED = "1"


def tail(completed: subprocess.CompletedProcess) -> str:
    return f"stdout:\n{completed.stdout[-8000:]}\nstderr:\n{completed.stderr[-8000:]}"


def run_phase(workspace, name: str, extra: list[str]) -> tuple[subprocess.CompletedProcess, dict | None]:
    report_path = workspace / f"{name}.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(RANKS),
        "-m",
        DRIVER,
        *BASE_ARGS,
        *extra,
        "--out",
        str(report_path),
    ]
    completed = subprocess.run(command, env=dict(os.environ), capture_output=True, text=True)
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return completed, report


def require_phase(workspace, name: str, extra: list[str]) -> dict:
    completed, report = run_phase(workspace, name, extra)
    if completed.returncode != 0 or report is None:
        pytest.fail(f"{name} exited {completed.returncode}\n{tail(completed)}")
    return report


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    if torch.cuda.device_count() < RANKS:
        pytest.skip(f"needs {RANKS} GPUs")
    return tmp_path_factory.mktemp("fully_shard_caching_checkpoint")


@pytest.fixture(scope="session")
def fp8_save(workspace):
    return require_phase(
        workspace,
        "fp8_save",
        ["--phase", "save", "--wrap", "fp8", "--ep", "4", "--seed", SAVE_SEED, "--ckpt", str(workspace / "fp8")],
    )


@pytest.fixture(scope="session")
def none_save(workspace):
    return require_phase(
        workspace,
        "none_save",
        ["--phase", "save", "--wrap", "none", "--ep", "4", "--seed", SAVE_SEED, "--ckpt", str(workspace / "none")],
    )


@pytest.fixture(scope="session")
def fp8_loaded_from_none(workspace, none_save):
    return require_phase(
        workspace,
        "fp8_loaded_from_none",
        ["--phase", "load", "--wrap", "fp8", "--ep", "4", "--seed", LOAD_SEED, "--ckpt", str(workspace / "none")],
    )


def test_load_keeps_the_wrapper_installed(fp8_loaded_from_none):
    wrapped_parameters = fp8_loaded_from_none["local_types_after_load"]
    assert set(wrapped_parameters.values()) == {"ShardedPreparedTensor"}
    assert fp8_loaded_from_none["shard_values_changed"]
    assert fp8_loaded_from_none["wrapper_objects_preserved"]
    assert fp8_loaded_from_none["shard_storage_preserved"]
    assert fp8_loaded_from_none["prepare_calls_after_load"] >= len(wrapped_parameters)


def test_checkpoints_interoperate_across_wrap_modes(workspace, fp8_save, none_save, fp8_loaded_from_none):
    assert fp8_loaded_from_none["eval_losses_before_load"] != none_save["eval_losses"]
    assert fp8_loaded_from_none["eval_losses"] == none_save["eval_losses"]
    completed, none_loaded_from_fp8 = run_phase(
        workspace,
        "none_loaded_from_fp8",
        ["--phase", "load", "--wrap", "none", "--ep", "4", "--seed", LOAD_SEED, "--ckpt", str(workspace / "fp8")],
    )
    assert completed.returncode == 0, tail(completed)
    assert none_loaded_from_fp8["eval_losses_before_load"] != fp8_save["eval_losses"]
    assert none_loaded_from_fp8["eval_losses"] == fp8_save["eval_losses"]


def test_round_trip_restores_master_weights_and_prepared_tensors(workspace):
    completed, report = run_phase(
        workspace,
        "round_trip",
        [
            "--phase",
            "roundtrip",
            "--wrap",
            "fp8",
            "--ep",
            "4",
            "--seed",
            SAVE_SEED,
            "--ckpt",
            str(workspace / "round_trip"),
        ],
    )
    assert completed.returncode == 0, tail(completed)
    assert report["perturbed_before_load"]
    assert report["master_weights_restored"]
    assert report["prepared_restored"]


def test_reshard_across_expert_parallel_degree(workspace, fp8_save):
    completed, report = run_phase(
        workspace,
        "fp8_loaded_at_ep8",
        ["--phase", "load", "--wrap", "fp8", "--ep", "8", "--seed", LOAD_SEED, "--ckpt", str(workspace / "fp8")],
    )
    assert completed.returncode == 0, tail(completed)
    assert set(report["local_types_after_load"].values()) == {"ShardedPreparedTensor"}
    assert report["shard_values_changed"]
    assert report["eval_losses"] == fp8_save["eval_losses"]


def test_optimizer_state_is_never_wrapped(fp8_save, fp8_loaded_from_none):
    for report in (fp8_save, fp8_loaded_from_none):
        state_types = report["optimizer_state_types"]
        moment_names = set().union(*(set(entry) for entry in state_types.values()))
        assert {"exp_avg", "exp_avg_sq"} <= moment_names
        assert {name for entry in state_types.values() for name in entry.values()} == {"Tensor"}
