from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable

import pytest

from tests.regression.distributed.sft.archs import CASES
from tests.regression.distributed.sft.cases import (
    Case,
    assert_training_rows,
    completed_step,
    failure_report,
    graph_break_reports,
    load_metrics,
    ranks_without_dynamo_logs,
    sft_args,
)

pytestmark = [pytest.mark.distributed, pytest.mark.gpu]

TRAIN_STEPS, RESUME_STEPS, TIMEOUT_SEC = 5, 3, 600


@dataclass
class SFTRun:
    """One `uv run sft` process: the train run, or the resume run that follows it in a resume case."""

    label: str
    returncode: int
    timed_out: bool
    log_dir: Path | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def describe(self) -> str:
        outcome = f"timed out after {TIMEOUT_SEC} s" if self.timed_out else f"exited with code {self.returncode}"
        report = failure_report(self.log_dir) if self.log_dir else "no logs: the run ended before creating them"
        return f"{self.label} run {outcome}\n{report}"


@dataclass
class CaseResult:
    run_dir: Path
    runs: list[SFTRun]


@pytest.fixture(scope="module")
def run_case(tiny_checkpoint, tokenizer_files, run_process, output_dir) -> Callable[[Case], CaseResult]:
    """Run each case once, shared by every test that checks that case."""

    @cache
    def _run_case(case: Case) -> CaseResult:
        model_name = tiny_checkpoint(case.arch).as_posix()
        tokenizer_files(case.arch)
        run_dir = output_dir / case.id
        env = {"HF_HUB_OFFLINE": "1"}
        if case.check_graph_breaks:
            env["TORCH_LOGS"] = "graph_breaks,dynamo"

        def sft(label: str, *extra: str) -> SFTRun:
            cmd = ["uv", "run", "--no-sync", "sft", *sft_args(case, model_name), "--output-dir", output_dir.as_posix()]
            cmd += ["--run.name", case.id, *extra]
            result = run_process(cmd, env=env, timeout=TIMEOUT_SEC)
            latest_log_dir = run_dir / "logs" / "latest"
            log_dir = latest_log_dir.resolve() if latest_log_dir.exists() else None
            return SFTRun(label, result.returncode, result.timed_out, log_dir)

        train = sft("train", "--max-steps", str(TRAIN_STEPS), "--clean", *(["--ckpt"] if case.resume else []))
        runs = [train]
        if case.resume and train.ok:
            runs.append(
                sft("resume", "--max-steps", str(TRAIN_STEPS + RESUME_STEPS), "--resume.step", str(TRAIN_STEPS))
            )
        return CaseResult(run_dir, runs)

    return _run_case


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_sft(case, run_case):
    result = run_case(case)
    for run in result.runs:
        if not run.ok:
            pytest.fail(f"run dir: {result.run_dir}\n{run.describe()}", pytrace=False)
    last_step = TRAIN_STEPS + RESUME_STEPS if case.resume else TRAIN_STEPS
    assert_training_rows(load_metrics(result.run_dir), range(1, last_step + 1))


@pytest.mark.parametrize("case", [case for case in CASES if case.check_graph_breaks], ids=lambda case: case.id)
def test_no_graph_breaks(case, run_case):
    result = run_case(case)
    metrics = result.run_dir / "monitors" / "file" / "metrics.jsonl"
    if not metrics.exists() or not completed_step(load_metrics(result.run_dir), 1):
        pytest.skip("run ended before step 1; graph breaks not checked")
    missing = ranks_without_dynamo_logs(result.run_dir)
    assert not missing, f"dynamo logging did not reach {missing}, so graph breaks cannot be checked"
    breaks = graph_break_reports(result.run_dir)
    assert not breaks, f"{len(breaks)} graph break(s):\n" + "\n".join(breaks)
