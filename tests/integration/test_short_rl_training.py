import os
import subprocess

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="session")
def username():
    return os.environ.get("USERNAME_CI", os.getlogin())


@pytest.fixture(scope="session")
def branch_name():
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    return branch


@pytest.fixture(scope="session")
def commit_hash():
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    return commit


def test_short_rl_training(commit_hash: str, branch_name: str, username: str):
    wandb_run_name = f"{branch_name}-{commit_hash}"

    if username == "CI_RUNNER":
        project = "ci_run_prime_rl"
    else:
        project = "ci_run_prime_rl_local"

    training_cmd = f"uv run torchrun --nproc_per_node=1 src/zeroband/train.py @ configs/training/simple_reverse_two_gpu.toml --optim.total_steps 70 --wandb_run_name {wandb_run_name} --project {project}".split()
    inference_cmd = "uv run python src/zeroband/infer.py @ configs/inference/simple_reverse_two_gpus.toml --total_step 35".split()

    training_process = subprocess.Popen(training_cmd)
    envs = os.environ.copy()

    envs["CUDA_VISIBLE_DEVICES"] = "1"
    inference_process = subprocess.Popen(inference_cmd, env=envs)

    training_process.wait()
    inference_process.wait()

    assert training_process.returncode == 0, f"Training process failed with return code {training_process.returncode}"
    assert inference_process.returncode == 0, f"Inference process failed with return code {inference_process.returncode}"
