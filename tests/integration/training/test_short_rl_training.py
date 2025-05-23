import os
import subprocess

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_short_rl_training():
    training_cmd = "uv run torchrun --nproc_per_node=1 src/zeroband/train.py @ configs/training/simple_reverse_two_gpu.toml --optim.total_steps 40".split()
    inference_cmd = "uv run python src/zeroband/infer.py @ configs/inference/simple_reverse_two_gpus.toml --total_step 20".split()

    training_process = subprocess.Popen(training_cmd)
    envs = os.environ.copy()

    envs["CUDA_VISIBLE_DEVICES"] = "1"
    inference_process = subprocess.Popen(inference_cmd, env=envs)

    training_process.wait()
    inference_process.wait()

    assert training_process.returncode == 0, f"Training process failed with return code {training_process.returncode}"
    assert inference_process.returncode == 0, f"Inference process failed with return code {inference_process.returncode}"
