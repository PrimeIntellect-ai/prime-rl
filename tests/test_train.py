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
        pytest.fail(f"Process  failed {result}")


@pytest.mark.parametrize("num_gpus", [1, 2])
@pytest.mark.parametrize("collate_mode", ["packing", "padding", "balancing"])
@pytest.mark.parametrize("kl_coef", [None, 0.02])
def test_train(num_gpus, collate_mode, kl_coef):
    extra_args = ["--collate_mode", collate_mode]
    if kl_coef is not None:
        extra_args.extend(["--kl_coef", str(kl_coef)])

    _test_torchrun(num_gpus=num_gpus, config="debug.toml", extra_args=extra_args)


def test_train_with_rollout_file(fake_rollout_files_dir):
    """
    this test will create a fake rollout file and then train with it
    """
    path = fake_rollout_files_dir(steps=list(range(5)), num_files=8, batch_size=8, seq_len=16)  # there is more file than batch here
    _test_torchrun(num_gpus=1, config="debug.toml", extra_args=["--data.path", str(path), "--no-data.fake"])
