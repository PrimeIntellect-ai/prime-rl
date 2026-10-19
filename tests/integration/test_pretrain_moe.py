from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from tests.conftest import ProcessResult
from tests.utils import check_loss_goes_down, strip_escape_codes

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

RUN_NAME = "pretrain-moe"
TIMEOUT = 300  # 5 minutes


@pytest.fixture(scope="module")
def run_dir(output_dir: Path) -> Path:
    return output_dir / RUN_NAME


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    return f"test-pretrain-moe:{branch_name}"


@pytest.fixture(scope="module")
def token_dir(output_dir: Path) -> Path:
    """Reverse-Text-SFT as one datatrove token shard: every conversation one document, ending with EOS."""
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    documents = [
        tokenizer.encode("\n".join(message["content"] for message in row["prompt"] + row["completion"]))
        + [tokenizer.eos_token_id]
        for row in load_dataset("PrimeIntellect/Reverse-Text-SFT", split="train")
    ]
    path = output_dir / "pretrain-tokens"
    path.mkdir(parents=True, exist_ok=True)
    np.array([token for document in documents for token in document], dtype="<u4").tofile(path / "00.ds")
    np.cumsum([len(document) for document in documents], dtype="<u8").tofile(path / "00.ds.index")
    (path / "00.ds.metadata").write_text("PrimeIntellect/Qwen3-0.6B|4\n0\n0")
    return path


def pretrain_command(config: str, token_dir: Path, wandb_project: str, wandb_name: str, output_dir: Path) -> list[str]:
    return [
        "uv",
        "run",
        "pretrain",
        "@",
        f"configs/ci/integration/pretrain-moe/{config}",
        "--data.sources.reverse.path",
        token_dir.as_posix(),
        "--deployment.num-train-gpus",
        "2",
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        RUN_NAME,
    ]


@pytest.fixture(scope="module")
def pretrain_process(
    run_process: Callable[..., ProcessResult], token_dir: Path, wandb_project: str, wandb_name: str, output_dir: Path
) -> ProcessResult:
    cmd = pretrain_command("start.toml", token_dir, wandb_project, wandb_name, output_dir)
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
    cmd = pretrain_command("resume.toml", token_dir, wandb_project, f"{wandb_name}-resume", output_dir)
    return run_process(cmd, timeout=TIMEOUT)


def trainer_log(run_dir: Path) -> list[str]:
    with open(run_dir / "logs" / "latest" / "trainer.log") as f:
        return strip_escape_codes(f.read()).splitlines()


def test_no_error(pretrain_process: ProcessResult):
    assert pretrain_process.returncode == 0, f"Process has non-zero return code ({pretrain_process})"


def test_loss_goes_down(pretrain_process: ProcessResult, run_dir: Path):
    check_loss_goes_down(trainer_log(run_dir))


def test_no_error_resume(pretrain_resume_process: ProcessResult):
    assert pretrain_resume_process.returncode == 0, f"Process has non-zero return code ({pretrain_resume_process})"


def test_loss_goes_down_resume(pretrain_resume_process: ProcessResult, run_dir: Path):
    check_loss_goes_down(trainer_log(run_dir))
