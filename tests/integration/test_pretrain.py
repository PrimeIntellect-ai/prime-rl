import json
import re
import tomllib
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from tests.conftest import ProcessResult
from tests.utils import check_loss_goes_down, strip_escape_codes

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

RUN_NAME = "pretrain"
MODEL = "PrimeIntellect/Qwen3-0.6B"
TIMEOUT = 300  # 5 minutes


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-pretrain:{branch_name}"


@pytest.fixture(scope="module")
def token_dir(output_dir: Path) -> Path:
    """Reverse-Text-SFT as one datatrove token shard: every conversation one document, ending with Qwen's document
    separator <|endoftext|> (its EOS, <|im_end|>, ends a chat turn)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    separator = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    documents = [
        tokenizer.encode("\n".join(message["content"] for message in row["prompt"] + row["completion"])) + [separator]
        for row in load_dataset("PrimeIntellect/Reverse-Text-SFT", split="train")
    ]
    path = output_dir / "pretrain-tokens"
    path.mkdir(parents=True, exist_ok=True)
    np.array([token for document in documents for token in document], dtype="<u4").tofile(path / "00.ds")
    np.cumsum([len(document) for document in documents], dtype="<u8").tofile(path / "00.ds.index")
    (path / "00.ds.metadata").write_text(f"{MODEL}|4\n")
    return path


def held_out_sets(token_dir: Path) -> list[dict]:
    """The config's [[val.data]] sets, scoring the shards the test wrote."""
    config = tomllib.loads(Path("configs/ci/integration/pretrain/start.toml").read_text())
    return [{**data, "paths": [token_dir.as_posix()]} for data in config["val"]["data"]]


def pretrain_command(
    configs: list[str], token_dir: Path, wandb_project: str, wandb_name: str, output_dir: Path
) -> list[str]:
    return [
        "uv",
        "run",
        "pretrain",
        *[arg for config in configs for arg in ("@", f"configs/ci/integration/pretrain/{config}")],
        "--data.paths",
        f'["{token_dir.as_posix()}"]',
        "--val.data",
        json.dumps(held_out_sets(token_dir)),
        "--deployment.num-train-gpus",
        "2",
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        wandb_name,
        "--no-monitors.prime",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]


@pytest.fixture(scope="module")
def pretrain_process(
    run_process: Callable[..., ProcessResult], token_dir: Path, wandb_project: str, wandb_name: str, output_dir: Path
) -> ProcessResult:
    cmd = pretrain_command(["start.toml"], token_dir, wandb_project, wandb_name, output_dir)
    return run_process([*cmd, "--clean"], timeout=TIMEOUT)


@pytest.fixture(scope="module")
def pretrain_resume_process(
    pretrain_process: ProcessResult,
    run_process: Callable[..., ProcessResult],
    token_dir: Path,
    wandb_project: str,
    wandb_name: str,
    output_dir: Path,
) -> ProcessResult:
    cmd = pretrain_command(["start.toml", "resume.toml"], token_dir, wandb_project, f"{wandb_name}-resume", output_dir)
    return run_process(cmd, timeout=TIMEOUT)


def trainer_log(run_dir: Path, attempt: str = "latest") -> list[str]:
    with open(run_dir / "logs" / attempt / "trainer.log") as f:
        return strip_escape_codes(f.read()).splitlines()


def step_losses(lines: list[str]) -> dict[int, float]:
    matches = (re.search(r"SUCCESS Step (\d+) \|.*? Loss (\d+\.\d+)", line) for line in lines)
    return {int(match[1]): float(match[2]) for match in matches if match}


def test_no_error(pretrain_process: ProcessResult):
    assert pretrain_process.returncode == 0, f"Process has non-zero return code ({pretrain_process})"


def test_loss_goes_down(pretrain_process: ProcessResult, run_dir: Path):
    check_loss_goes_down(trainer_log(run_dir))


def test_validation(pretrain_process: ProcessResult, run_dir: Path):
    lines = trainer_log(run_dir)
    for length in ("512", "1k"):
        assert any(f"Validation | Step 10 | reverse-text@{length} | Loss" in line for line in lines)


def test_no_error_resume(pretrain_resume_process: ProcessResult):
    assert pretrain_resume_process.returncode == 0, f"Process has non-zero return code ({pretrain_resume_process})"


def test_loss_goes_down_resume(pretrain_resume_process: ProcessResult, run_dir: Path):
    check_loss_goes_down(trainer_log(run_dir))


def test_resume_continues_from_the_checkpoint(pretrain_resume_process: ProcessResult, run_dir: Path):
    """The first resumed step's loss is close to the start run's last one, not back at the from-scratch loss."""
    last = step_losses(trainer_log(run_dir, "attempt_1"))[20]
    first = step_losses(trainer_log(run_dir))[21]
    assert abs(first - last) < 0.1 * last, f"Loss jumped from {last} at step 20 to {first} at step 21"
