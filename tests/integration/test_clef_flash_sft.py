import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image
from safetensors.torch import load_file

from tests.conftest import ProcessResult
from tests.unit.train.models.qwen3_5.clef_flash_release import write_processor, write_release
from tests.utils import convert_checkpoint, strip_escape_codes

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

TIMEOUT = 600  # 10 minutes
QUESTIONS = {
    "outage": {"type": "noul", "instructions": "Is a service down?", "criteria": {"true": "A service is down."}},
    "team": {"type": "choice", "instructions": "Who handles it?", "criteria": {"technical": "Bugs", "billing": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Can wait", "This week", "Today"]},
}


def image(i: int) -> str:
    """A small image as a data URL."""
    buffer = BytesIO()
    Image.new("RGB", (64, 64 + 32 * (i % 3)), (i % 256, 128, 0)).save(buffer, "PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


@pytest.fixture(scope="module", params=["text", "images"])
def images(request: pytest.FixtureRequest) -> bool:
    return request.param == "images"


@pytest.fixture(scope="module")
def run_name(images: bool) -> str:
    return "clef-flash-sft-images" if images else "clef-flash-sft"


@pytest.fixture(scope="module")
def run_dir(output_dir: Path, run_name: str) -> Path:
    return output_dir / run_name


@pytest.fixture(scope="module")
def wandb_name(branch_name: str, run_name: str) -> str:
    return f"test-{run_name}:{branch_name}"


@pytest.fixture(scope="module")
def release(output_dir: Path) -> Path:
    """A tiny Clef-Flash release, laid out as Cloudflare/clef-flash is."""
    path = output_dir / "clef-flash-release"
    write_release(path)
    write_processor(path)
    return path


@pytest.fixture(scope="module")
def records(output_dir: Path, images: bool) -> Path:
    """System One records, some questions labelled hard, some soft, some not at all. With ``images``, half of them have
    one or two images."""
    path = output_dir / f"clef-flash-records-{'images' if images else 'text'}"
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"r{i}",
            "state": json.dumps({"ticket": f"Checkout fails for order {i}.", "plan": ["free", "pro"][i % 2]}),
            "questions": json.dumps(QUESTIONS),
            "label": json.dumps({"outage": i % 2 == 0, "urgency": i % 3}),
            "soft_label": json.dumps({"team": {"technical": 0.75, "billing": 0.25}} if i % 4 else {}),
            **({"images": json.dumps([image(i)] * (i % 4 % 3))} if images else {}),
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
    images: bool,
    run_name: str,
    wandb_project: str,
    wandb_name: str,
    output_dir: Path,
) -> ProcessResult:
    source = json.dumps([{"name": "records", "dataset": records.as_posix()}])
    cmd = [
        "uv",
        "run",
        "sft",
        "@",
        "configs/ci/integration/clef-flash-sft/start.toml",
        *(["@", "configs/ci/integration/clef-flash-sft/images.toml"] if images else []),
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
        wandb_name,
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


def test_steps_and_validation_have_finite_losses(sft_process: ProcessResult, run_dir: Path):
    lines = trainer_log(run_dir)
    losses = [float(match[1]) for line in lines if (match := re.search(r"SUCCESS Step \d+ \|.*? Loss (\S+)", line))]
    validations = [
        float(match[1]) for line in lines if (match := re.search(r"Validation \| Step \d+ \| Loss (\S+)", line))
    ]
    assert len(losses) == 4 and all(0 < loss < 100 for loss in losses), losses
    assert len(validations) == 2 and all(0 < loss < 100 for loss in validations), validations


def test_the_export_is_a_clef_flash_release(sft_process: ProcessResult, run_dir: Path, release: Path):
    assert sft_process.returncode == 0
    step_dir = run_dir / "checkpoints" / "step_4"
    convert_checkpoint("dcp_to_bf16", step_dir)
    weights = step_dir / "weights"
    assert json.loads((weights / "config.json").read_text())["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    head_config = json.loads((weights / "joint_head_config.json").read_text())
    assert head_config == json.loads((release / "joint_head_config.json").read_text())
    assert load_file(weights / "joint_head.safetensors").keys() == load_file(release / "joint_head.safetensors").keys()
