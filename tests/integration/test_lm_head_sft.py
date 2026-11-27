"""SFT on System One data of the models that answer through their LM head: a stock Qwen3.5 (NextTokenStrategy) and a slot
decision model (SlotLMStrategy, two-block attention)."""

import json
import re
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.unit.train.models.qwen3_5.lm_head_release import MODELING_OVERRIDES, write_lm_head_release
from tests.utils import convert_checkpoint, strip_escape_codes

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

TIMEOUT = 600  # 10 minutes
STRATEGIES = {"stock": "NextTokenStrategy", "slot": "SlotLMStrategy"}
QUESTIONS = {
    "outage": {"type": "noul", "instructions": "Is a service down?", "criteria": {"true": "A service is down."}},
    "team": {"type": "choice", "instructions": "Who handles it?", "criteria": {"technical": "Bugs", "billing": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "This week", "Today"]},
}


@pytest.fixture(scope="module", params=list(STRATEGIES))
def kind(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def run_name(kind: str) -> str:
    return f"lm-head-sft-{kind}"


@pytest.fixture(scope="module")
def run_dir(output_dir: Path, run_name: str) -> Path:
    return output_dir / run_name


@pytest.fixture(scope="module")
def release(output_dir: Path, kind: str) -> Path:
    path = output_dir / f"lm-head-{kind}-release"
    write_lm_head_release(path, MODELING_OVERRIDES if kind == "slot" else None)
    return path


@pytest.fixture(scope="module")
def records(output_dir: Path) -> Path:
    """System One records, some questions labelled hard, some soft, some not at all."""
    path = output_dir / "lm-head-records"
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"r{i}",
            "state": json.dumps({"ticket": f"Checkout fails for order {i}.", "plan": ["free", "pro"][i % 2]}),
            "questions": json.dumps(QUESTIONS),
            "label": json.dumps({"outage": i % 2 == 0, "urgency": i % 3}),
            "soft_label": json.dumps({"team": {"technical": 0.75, "billing": 0.25}} if i % 4 else {}),
        }
        for i in range(64)
    ]
    (path / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture(scope="module")
def sft_process(
    run_process: Callable[..., ProcessResult],
    release: Path,
    records: Path,
    run_name: str,
    wandb_project: str,
    branch_name: str,
    output_dir: Path,
) -> ProcessResult:
    source = json.dumps([{"name": "records", "dataset": records.as_posix()}])
    cmd = [
        "uv",
        "run",
        "sft",
        "@",
        "configs/ci/integration/lm-head-sft/start.toml",
        "--model.name",
        release.as_posix(),
        "--data.source",
        source,
        "--val.data.source",
        source,
        "--deployment.num-train-gpus",
        "2",
        "--clean",
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        f"test-{run_name}:{branch_name}",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        run_name,
    ]
    return run_process(cmd, timeout=TIMEOUT)


def trainer_log(run_dir: Path) -> list[str]:
    with open(run_dir / "logs" / "latest" / "trainer.log") as f:
        return strip_escape_codes(f.read()).splitlines()


def test_no_error(sft_process: ProcessResult):
    assert sft_process.returncode == 0, f"Process has non-zero return code ({sft_process})"


def test_trains_with_the_checkpoints_strategy(sft_process: ProcessResult, run_dir: Path, kind: str):
    lines = trainer_log(run_dir)
    assert any(f"Training on System One data with {STRATEGIES[kind]}" in line for line in lines)
    losses = [float(match[1]) for line in lines if (match := re.search(r"SUCCESS Step \d+ \|.*? Loss (\S+)", line))]
    validations = [
        float(match[1]) for line in lines if (match := re.search(r"Validation \| Step \d+ \| Loss (\S+)", line))
    ]
    assert len(losses) == 4 and all(0 < loss < 100 for loss in losses), losses
    assert len(validations) == 2 and all(0 < loss < 100 for loss in validations), validations


def test_the_export_has_the_releases_layout(sft_process: ProcessResult, run_dir: Path, kind: str):
    assert sft_process.returncode == 0
    step_dir = run_dir / "checkpoints" / "step_4"
    convert_checkpoint("dcp_to_bf16", step_dir)
    config = json.loads((step_dir / "weights" / "config.json").read_text())
    assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert config.get("modeling_overrides") == (MODELING_OVERRIDES if kind == "slot" else None)
