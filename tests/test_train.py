import subprocess
import pytest


def _test_torchrun(num_gpus, config, extra_args=[]):
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "src/zeroband/train.py",
        f"@configs/training/{config}",
        *extra_args,
    ]

    process = subprocess.Popen(cmd)
    result = process.wait()
    if result != 0:
        pytest.fail(f"Process {result} failed {result}")


@pytest.mark.parametrize("num_gpus", [1, 2])
def test_train(num_gpus):
    _test_torchrun(num_gpus=num_gpus, config="debug.toml")


def test_train_with_rollout_file(fake_rollout_files_dir):
    """
    this test will create a fake rollout file and then train with it
    """
    path = fake_rollout_files_dir(0, 10)
    _test_torchrun(num_gpus=1, config="debug.toml", extra_args=["--data.path", str(path), "--no-data.fake"])
