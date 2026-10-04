from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest

from tests.utils import (
    check_avg_reward_in_range,
    check_mismatch_kl_in_range,
    check_no_error,
    check_reward_goes_up,
    check_reward_in_range,
    strip_escape_codes,
)

if TYPE_CHECKING:
    from tests.conftest import ProcessResult

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@dataclass
class Example:
    name: str
    config: str
    extra_args: list[str] = field(default_factory=list)
    min_final_reward: float | None = None
    min_avg_reward: tuple[int, float] | None = None  # (last_n_steps, min_threshold)
    mismatch_kl_band: tuple[float, float] | None = None


EXAMPLES = [
    Example("alphabet-sort", "examples/basic/alphabet-sort/rl.toml"),
    Example(
        "hendrycks-sanity",
        "examples/basic/hendrycks-sanity/rl.toml",
        extra_args=["--max-steps", "1000"],  # do less steps to finish in time
        min_final_reward=0.75,
        mismatch_kl_band=(0.0, 0.0005),
    ),
    Example(
        "multimodal-color-codeword",
        "configs/ci/nightly/multimodal_color_codeword.toml",
        min_avg_reward=(7, 0.85),
    ),
    Example("reverse-text", "examples/basic/reverse-text/rl.toml", min_final_reward=0.65),
    Example("wiki-search", "examples/basic/wiki-search/rl.toml"),
    Example("wordle", "examples/basic/wordle/rl.toml"),
]


def read_log(run_dir: Path, name: str) -> list[str]:
    with open(run_dir / "logs" / "latest" / f"{name}.log", "r") as f:
        return strip_escape_codes(f.read()).splitlines()


@pytest.mark.parametrize("example", [pytest.param(e, id=e.name) for e in EXAMPLES])
def test_example(
    example: Example,
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    branch_name: str,
):
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        example.config,
        "--monitors.wandb.project",
        wandb_project,
        "--monitors.wandb.name",
        f"{example.name}-{branch_name}",
        "--output-dir",
        output_dir.as_posix(),
        "--run.name",
        example.name,
        *example.extra_args,
    ]
    run_dir = output_dir / example.name
    check_no_error(run_process(cmd), run_dir)

    orchestrator_log = read_log(run_dir, "orchestrator")
    check_reward_goes_up(orchestrator_log)
    if example.min_final_reward is not None:
        check_reward_in_range(orchestrator_log, min_threshold=example.min_final_reward)
    if example.min_avg_reward is not None:
        last_n_steps, min_threshold = example.min_avg_reward
        check_avg_reward_in_range(orchestrator_log, last_n_steps=last_n_steps, min_threshold=min_threshold)
    if example.mismatch_kl_band is not None:
        min_kl, max_kl = example.mismatch_kl_band
        check_mismatch_kl_in_range(read_log(run_dir, "trainer"), min_threshold=min_kl, max_threshold=max_kl)
