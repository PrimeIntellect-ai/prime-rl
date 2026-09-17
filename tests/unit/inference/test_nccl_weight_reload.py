"""GPU regression test for the quantized NCCL weight-reload path.

At tiny scale this exercises the full production contract end-to-end:

* a vLLM engine initialized from a **serialized blockwise-FP8 checkpoint** of
  the GLM-MoE-DSA architecture (TP-sharded, expert parallelism on and off),
* receiving **two successive** quantized NCCL weight updates streamed through
  the real worker extension and vLLM's checkpoint load path,
* whose post-update state is compared against a **fresh load** of the very
  same quantized tensors written to disk — greedy generations must match
  exactly, and must differ from the pre-update baseline (the updates actually
  mutated the policy).

A second test pins the receiver-side contract check: an engine that quantizes
a bf16 checkpoint on the fly (``quantization="fp8"`` with no serialized fp8
checkpoint) must reject ``quantize_in_weight_transfer`` at
``init_broadcaster``, loudly and before any collective can strand a worker.

Each engine runs in its own subprocess (see ``weight_reload_driver.py``) so
every invocation gets a clean CUDA/distributed lifetime. Requires SM90+
(blockwise fp8) and enough GPUs to leave one dedicated to the NCCL sender
rank next to the engine's TP ranks: TP is the largest of 2/4/8 that fits, so
a 9-GPU runner exercises the production TP8 topology and an 8-GPU node runs
TP4.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="needs CUDA (vLLM TP engines + NCCL)",
    ),
    pytest.mark.skipif(
        torch.cuda.device_count() < 3,
        reason="needs at least 3 GPUs: 2+ engine TP ranks plus a dedicated sender GPU",
    ),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9,
        reason="blockwise-fp8 weights need SM90+ (Hopper)",
    ),
]

DRIVER = Path(__file__).parent / "weight_reload_driver.py"
DRIVER_TIMEOUT_S = 1800


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_driver(*args: str, timeout: int = DRIVER_TIMEOUT_S) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return subprocess.run(
        [sys.executable, str(DRIVER), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _tp_size(gpu_count: int) -> int:
    """The largest TP (of the head-count divisors 2/4/8) that still leaves a
    dedicated GPU for the NCCL sender rank."""
    for tp in (8, 4, 2):
        if gpu_count >= tp + 1:
            return tp
    return 0


def _reference_tokens(model_dir: Path, tp: int, ep: str, out: Path) -> list[list[int]]:
    result = _run_driver(
        "reference",
        "--model-dir",
        str(model_dir),
        "--out",
        str(out),
        "--tp",
        str(tp),
        "--ep",
        ep,
    )
    assert result.returncode == 0, f"reference ({model_dir.name}, ep={ep}) failed:\n{result.stdout}\n{result.stderr}"
    return json.loads(out.read_text())["tokens"]


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The tiny serialized-FP8 and plain-bf16 GLM-MoE-DSA checkpoints."""
    base = tmp_path_factory.mktemp("tiny-glm")
    fp8_dir = base / "glm-fp8"
    bf16_dir = base / "glm-bf16"
    result = _run_driver("prepare", "--fp8-dir", str(fp8_dir), "--bf16-dir", str(bf16_dir))
    assert result.returncode == 0, f"checkpoint prepare failed:\n{result.stdout}\n{result.stderr}"
    return {"fp8": fp8_dir, "bf16": bf16_dir}


@pytest.fixture(scope="module")
def reload_rounds(checkpoints: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """Two successive quantized NCCL updates per EP setting, each round's wire
    stream also saved as an on-disk FP8 checkpoint for the fresh-load reference."""
    import torch

    tp = _tp_size(torch.cuda.device_count())
    base = tmp_path_factory.mktemp("reload")
    rounds: dict[str, dict] = {}
    for ep in ("off", "on"):
        out = base / f"update-{ep}.json"
        ref1_dir = base / f"ref1-{ep}"
        ref2_dir = base / f"ref2-{ep}"
        result = _run_driver(
            "update",
            "--model-dir",
            str(checkpoints["fp8"]),
            "--ref1-dir",
            str(ref1_dir),
            "--ref2-dir",
            str(ref2_dir),
            "--out",
            str(out),
            "--tp",
            str(tp),
            "--ep",
            ep,
            "--port",
            str(_free_port()),
        )
        assert result.returncode == 0, f"update (ep={ep}) failed:\n{result.stdout}\n{result.stderr}"
        rounds[ep] = {
            "tp": tp,
            "payload": json.loads(out.read_text()),
            "ref1": ref1_dir,
            "ref2": ref2_dir,
            "refs_out": base,
        }
    return rounds


@pytest.mark.parametrize("ep", ["off", "on"])
def test_reload_reproduces_fresh_load(reload_rounds: dict[str, dict], ep: str):
    """The two successive NCCL updates must leave the engine in exactly the
    state a fresh engine loads from the same quantized tensors on disk."""
    rounds = reload_rounds[ep]
    tp = rounds["tp"]
    payload = rounds["payload"]

    ref1 = _reference_tokens(rounds["ref1"], tp, ep, rounds["refs_out"] / f"ref1-tokens-{ep}.json")
    assert payload["after_first"] == ref1, (
        f"generations after NCCL update 1 (ep={ep}) differ from the fresh-load reference"
    )

    ref2 = _reference_tokens(rounds["ref2"], tp, ep, rounds["refs_out"] / f"ref2-tokens-{ep}.json")
    assert payload["after_second"] == ref2, (
        f"generations after NCCL update 2 (ep={ep}) differ from the fresh-load reference"
    )

    # The updates must have actually changed the policy (not a silent no-op).
    assert payload["baseline"] != payload["after_first"], f"NCCL update 1 (ep={ep}) did not change the engine outputs"
    assert payload["after_first"] != payload["after_second"], (
        f"NCCL update 2 (ep={ep}) did not change the engine outputs"
    )


def test_online_fp8_engine_rejects_quantized_wire(checkpoints: dict[str, Path], tmp_path: Path):
    """Engines without a serialized blockwise-fp8 checkpoint must fail
    init_broadcaster loudly, before entering any NCCL collective."""
    out = tmp_path / "reject.json"
    result = _run_driver(
        "reject",
        "--model-dir",
        str(checkpoints["bf16"]),
        "--out",
        str(out),
        "--port",
        str(_free_port()),
    )
    assert result.returncode == 0, f"reject driver crashed:\n{result.stdout}\n{result.stderr}"
    payload = json.loads(out.read_text())
    assert payload["rejected"], f"online-fp8 engine did not reject the quantized wire format: {payload['message']}"
