"""Trainer logprobs on a Qwen3.8 video prompt against vLLM's. Needs GPUs and a Qwen3.8 checkpoint:

    PRIME_RL_QWEN38_CHECKPOINT=/path/to/checkpoint PRIME_RL_TEST_VIDEO=clip.mp4 PRIME_RL_QWEN38_GPUS=8 \\
        uv run pytest tests/unit/train/models/qwen3_8_flash_next/test_video_parity.py -m gpu -s
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.unit.train.models.qwen3_8_flash_next.test_video import write_synthetic_video

CHECKPOINT_ENV = "PRIME_RL_QWEN38_CHECKPOINT"
SCRIPT = Path(__file__).with_name("video_parity.py")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not os.environ.get(CHECKPOINT_ENV), reason=f"{CHECKPOINT_ENV} is not set"),
]


def test_video_logprobs_match_vllm(tmp_path):
    from transformers import AutoConfig

    from tests.unit.train.models.qwen3_8_flash_next.video_parity import compare

    checkpoint = os.environ[CHECKPOINT_ENV]
    gpus = int(os.environ.get("PRIME_RL_QWEN38_GPUS", "1"))
    video = os.environ.get("PRIME_RL_TEST_VIDEO")
    if video is None:
        video = str(tmp_path / "synthetic.mp4")
        write_synthetic_video(Path(video))
    common = ["--model", checkpoint, "--video", video]
    vllm_out, trainer_out = tmp_path / "vllm.pt", tmp_path / "trainer.pt"
    subprocess.run(
        [sys.executable, SCRIPT, "vllm", *common, "--out", vllm_out, "--tensor-parallel-size", str(min(gpus, 4))],
        check=True,
    )
    subprocess.run(
        ["torchrun", f"--nproc-per-node={gpus}", SCRIPT, "trainer", *common, "--vllm", vllm_out, "--out", trainer_out],
        check=True,
    )

    video_token_id = AutoConfig.from_pretrained(checkpoint).video_token_id
    result = compare(torch.load(vllm_out), torch.load(trainer_out), video_token_id)
    print(json.dumps(result, indent=2))
    # The video tokens may differ no more than the sample's text does: wrong video embeddings or positions show up
    # there first (a mirrored clip puts 10x the text difference on the pads of a 4-layer Qwen3.8).
    text = result["text"]["mean"]
    assert text < 0.1, result
    for name in ("video pads", "video timestamps and markers"):
        assert result[name]["mean"] < 2 * text + 0.01, result
