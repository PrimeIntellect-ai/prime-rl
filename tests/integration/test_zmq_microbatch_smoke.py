from __future__ import annotations

import asyncio
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import tomli_w

from prime_rl.transport.filesystem import FileSystemTrainingBatchSender
from prime_rl.transport.types import TrainingBatch, TrainingSample


def _flatten(values: list[list[float]]) -> list[float]:
    return [item for row in values for item in row]


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _free_zmq_base_port() -> int:
    for _ in range(100):
        base = random.randint(30_000, 60_000)
        sockets: list[socket.socket] = []
        try:
            for port in (base + 1, base + 2):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("0.0.0.0", port))
                sockets.append(sock)
        except OSError:
            pass
        else:
            return base
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("Could not find free ZMQ base port")


def _create_run(output_dir: Path) -> Path:
    run_dir = output_dir / "run_zmq_smoke"
    control_dir = run_dir / "control"
    control_dir.mkdir(parents=True)
    config = {
        "model": {"name": "test-model"},
        "batch_size": 2,
        "group_size": 1,
        "env": [{"id": "test-env"}],
        "sampling": {"temperature": 1.0},
        # test-model is synthetic; bypass model->renderer validation.
        "renderer": "None",
    }
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config, f)
    return run_dir


def _training_sample(prompt_id: int, completion_id: int, advantage: float) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[prompt_id],
        prompt_mask=[False],
        completion_ids=[completion_id],
        completion_mask=[True],
        completion_logprobs=[-0.1],
        completion_temperatures=[0.7],
        env_name="test-env",
        advantage=advantage,
        reward=1.0,
    )


def test_dataloader_splits_filesystem_rollouts_and_zmq_micro_batches(tmp_path: Path) -> None:
    run_dir = _create_run(tmp_path)
    asyncio.run(
        FileSystemTrainingBatchSender(run_dir).send(
            TrainingBatch(
                examples=[
                    _training_sample(10, 11, 1.0),
                    _training_sample(20, 21, 2.0),
                ],
                step=0,
            )
        )
    )

    dist_port = _free_tcp_port()
    zmq_base_port = _free_zmq_base_port()
    script = Path(__file__).with_name("zmq_microbatch_smoke.py")
    env = {
        **os.environ,
        "MASTER_ADDR": "127.0.0.1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            "--master_port",
            str(dist_port),
            script.as_posix(),
            tmp_path.as_posix(),
            str(zmq_base_port),
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    outputs = [json.loads((tmp_path / "rank_outputs" / f"rank_{rank}.json").read_text()) for rank in range(2)]
    expected = [
        {
            "input_ids": [[10, 11]],
            "position_ids": [[0, 1]],
            "loss_mask": [[False, True]],
            "advantages": [[1.0, 1.0]],
            "rewards": [[1.0, 1.0]],
            "inference_logprobs": [[0.0, -0.1]],
            "temperatures": [[0.7, 0.7]],
            "env_names": ["test-env", "test-env"],
            "lora_num_tokens": [2],
            "training_mode": "rl",
            "mm_kwargs": None,
            "mm_token_type_ids": None,
        },
        {
            "input_ids": [[20, 21]],
            "position_ids": [[0, 1]],
            "loss_mask": [[False, True]],
            "advantages": [[2.0, 2.0]],
            "rewards": [[1.0, 1.0]],
            "inference_logprobs": [[0.0, -0.1]],
            "temperatures": [[0.7, 0.7]],
            "env_names": ["test-env", "test-env"],
            "lora_num_tokens": [2],
            "training_mode": "rl",
            "mm_kwargs": None,
            "mm_token_type_ids": None,
        },
    ]
    for actual, expected_rank in zip(outputs, expected, strict=True):
        for key in (
            "input_ids",
            "position_ids",
            "loss_mask",
            "env_names",
            "lora_num_tokens",
            "training_mode",
            "mm_kwargs",
            "mm_token_type_ids",
        ):
            assert actual[key] == expected_rank[key]
        for key in ("advantages", "rewards", "inference_logprobs", "temperatures"):
            assert _flatten(actual[key]) == pytest.approx(_flatten(expected_rank[key]))
