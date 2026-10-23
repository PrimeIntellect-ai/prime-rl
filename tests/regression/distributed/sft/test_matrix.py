from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable

import pytest
from huggingface_hub import snapshot_download

from tests.regression.distributed.sft.cases import (
    CASES,
    Case,
    assert_training_rows,
    completed_step,
    graph_break_reports,
    load_metrics,
    ranks_without_dynamo_logs,
    sft_args,
)
from tests.regression.distributed.tiny_models import TOKENIZERS

pytestmark = [pytest.mark.distributed, pytest.mark.gpu]

TRAIN_STEPS, RESUME_STEPS, TIMEOUT_SEC = 5, 3, 600


@dataclass
class Launch:
    run_dir: Path
    returncodes: list[int]


@pytest.fixture(scope="module")
def launch(tiny_checkpoint, run_process, tmp_path_factory) -> Callable[[Case], Launch]:
    """Run each case's `uv run sft` launches once, shared by every test that checks that case."""

    @cache
    def _launch(case: Case) -> Launch:
        model_name = tiny_checkpoint(case.arch.name).as_posix()
        snapshot_download(TOKENIZERS[case.arch.name], allow_patterns=["*.json", "*.txt", "tokenizer*", "*.jinja"])
        output_dir = tmp_path_factory.mktemp(case.id)
        env = {"TORCH_LOGS": "graph_breaks,dynamo"} if case.check_graph_breaks else {}

        def sft(*extra: str) -> int:
            cmd = ["uv", "run", "sft", *sft_args(case, model_name), "--output-dir", output_dir.as_posix()]
            cmd += ["--run.name", case.id, *extra]
            return run_process(cmd, env=env, timeout=TIMEOUT_SEC).returncode

        returncodes = [sft("--max-steps", str(TRAIN_STEPS), *(["--ckpt"] if case.resume else []))]
        if case.resume and returncodes[0] == 0:
            returncodes.append(sft("--max-steps", str(TRAIN_STEPS + RESUME_STEPS), "--resume.step", str(TRAIN_STEPS)))
        return Launch(output_dir / case.id, returncodes)

    return _launch


@pytest.mark.parametrize("case", [case.param() for case in CASES])
def test_sft(case, launch):
    result = launch(case)
    assert result.returncodes == [0] * (2 if case.resume else 1)
    last_step = TRAIN_STEPS + RESUME_STEPS if case.resume else TRAIN_STEPS
    assert_training_rows(load_metrics(result.run_dir), range(1, last_step + 1))


@pytest.mark.parametrize("case", [case.param() for case in CASES if case.check_graph_breaks])
def test_no_graph_breaks(case, launch):
    result = launch(case)
    metrics = result.run_dir / "monitors" / "file" / "metrics.jsonl"
    if not metrics.exists() or not completed_step(load_metrics(result.run_dir), 1):
        pytest.skip("launch ended before step 1; graph breaks not checked")
    missing = ranks_without_dynamo_logs(result.run_dir)
    assert not missing, f"dynamo logging did not reach {missing}, so graph breaks cannot be checked"
    breaks = graph_break_reports(result.run_dir)
    assert not breaks, f"{len(breaks)} graph break(s):\n" + "\n".join(breaks)
