import os
import subprocess
import pytest
import torch
import pyarrow.parquet as pq

# ruff: noqa
TIMEOUT = 300
IROH_NODE_ID_MAP = {
    0: "ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03",
    1: "ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337",
    2: "191fc38f134aaf1b7fdb1f86330b9d03e94bd4ba884f490389de964448e89b3f",
    3: "c5bbbb60e412879bbec7bb769804fa8e36e68af10d5477280b63deeaca931bed",
    4: "4f44e6c7bdfed3d9f48d86149ee3d29382cae8c83ca253e06a70be54a301828b",
    5: "e2e8aa145e1ec5cb01ebfaa40e10e12f0230c832fd8135470c001cb86d77de00",
    6: "17888c2ca502371245e5e35d5bcf35246c3bc36878e859938c9ead3c54db174f",
    7: "478243aed376da313d7cf3a60637c264cb36acc936efb341ff8d3d712092d244",
}


def get_command(dp, tp, pp_rank, pp_world_size):
    command = "python src/zeroband/infer.py"
    command += " @configs/inference/debug.toml"
    command += " --total-step 1"
    command += " --batch-size 8"
    command += " --max-samples 8"
    command += " --sampling.n 1"
    command += " --seed 69"
    command += f" --dp {dp}"
    command += f" --tp {tp}"
    command += f" --pp.rank {pp_rank}"
    command += f" --pp.world-size {pp_world_size}"
    if pp_world_size > 1:
        peer_id = IROH_NODE_ID_MAP[(pp_rank + 1) % pp_world_size]
        command += f" --pp.iroh-seed {pp_rank}"
        command += f" --pp.iroh-peer-id {peer_id}"
    return command.split()


@pytest.fixture(
    params=[
        {"name": "single-node", "dp": 1, "tp": 1, "pp": 1},
        {"name": "single-node-multi-gpu-dp", "dp": 2, "tp": 1, "pp": 1},
        {"name": "single-node-multi-gpu-tp", "dp": 1, "tp": 2, "pp": 1},
        {"name": "single-node-multi-gpu-pp", "dp": 1, "tp": 1, "pp": 2},
        {"name": "single-node-multi-gpu-tp-dp", "dp": 2, "tp": 2, "pp": 1},
        {"name": "single-node-multi-gpu-tp-pp", "dp": 1, "tp": 2, "pp": 2},
    ],
    ids=lambda x: x["name"],
)
def parallel_config(request):
    dp, tp, pp = request.param["dp"], request.param["tp"], request.param["pp"]
    if dp * tp * pp > torch.cuda.device_count():
        pytest.skip(
            f"Skipping {request.param['name']} because it requires {dp * tp * pp} GPUs, but only {torch.cuda.device_count()} are available"
        )
    return request.param


@pytest.fixture
def process(parallel_config):
    """Start memorize training process."""
    # Start process
    dp, tp, pp_world_size = parallel_config["dp"], parallel_config["tp"], parallel_config["pp"]
    processes = []
    for pp_rank in range(pp_world_size):
        command = get_command(dp=dp, tp=tp, pp_rank=pp_rank, pp_world_size=pp_world_size)
        cuda_visible_devices = ",".join(str(i) for i in range(pp_rank * tp * dp, (pp_rank + 1) * tp * dp))
        env = {"CUDA_VISIBLE_DEVICES": str(cuda_visible_devices), **os.environ}
        process = subprocess.Popen(command, env=env)
        processes.append(process)

    try:
        for process in processes:
            process.wait(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Process did not complete within {TIMEOUT} seconds")
    finally:
        for process in processes:
            process.terminate()
    yield process


def test_no_error(process):
    assert process.returncode == 0, f"Process failed with return code {process.returncode}"
