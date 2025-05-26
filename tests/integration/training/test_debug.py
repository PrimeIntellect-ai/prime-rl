import pytest

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = ["uv", "run", "torchrun", "src/zeroband/train.py", "@configs/training/debug.toml"]


@pytest.fixture(scope="module")
def process(run_process):
    return run_process(CMD, {})


def test_no_error(process):
    assert process.returncode == 0, f"Process failed with return code {process.returncode}"
