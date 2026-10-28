"""The SFT functionality matrix, rendered as `uv run sft` arguments.

Every case runs on `NUM_GPUS` GPUs of one node from `base.toml`, which keeps the `ModelConfig` defaults. Each `Arch`
in `ARCHS` (in `archs.py`, with its tiny config and tokenizer) contributes two kinds of case:

- A base case on every layout in its `layouts`. It trains `TRAIN_STEPS` steps (see `test_matrix.py`). The base case
  on the arch's `resume_layout`, if set, also saves a checkpoint and resumes in a fresh process.
- One case per entry in its `features`. Each feature changes one setting and runs on its single pinned layout.

To save launches, only qwen3_moe sets `features` and `resume_layout`. It is the plainest MoE arch, with standard
attention and routing, so its features and resume exercise only shared code and stand in for every arch. Every other
arch runs just its base cases.

Base cases on the CP4 layouts (CP and, for MoE archs, EP, with default `fullgraph=False`) also launch with
`TORCH_LOGS=graph_breaks,dynamo`. A second test, `test_no_graph_breaks`, reads the same launch's rank logs and fails on
any graph break, so graph breaks and training failures are reported as separate tests. It also fails if any rank's
logs lack dynamo's tracing messages, since then an empty graph-break log would prove nothing. Recompile-limit hits,
which also fall back to eager, are not graph breaks and are not checked.

`Arch.layouts` picks the layout family: `MOE_LAYOUTS` (expert parallel) or `DENSE_LAYOUTS` (ep=1).
`Arch.overrides` holds arch-wide config, such as an explicit renderer. Case ids read `{arch}-{layout}-{feature or
base}`. No name is a substring of another, so `-k` selects cases by arch, layout or
feature name. For example, one case:

    uv run pytest tests/regression/distributed -k "qwen3_moe and cp4_ep4 and base"

or every case on one layout, including the feature cases pinned to it:

    uv run pytest tests/regression/distributed -k cp4_ep4

`--shard I/N` runs shard I of N. Each arch's base cases form one shard unit and its feature cases another, so a shard
builds each checkpoint it needs once. `assign_shards` packs the units longest first by `Arch.launch_s` times the
unit's launch count, and tests outside the matrix run in shard 1.

`scripts/ci/run_distributed_nightly.sh` runs the same pytest command and passes extra args through to it.
"""

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

from transformers import PretrainedConfig

CONFIG_DIR = Path("tests/regression/configs/distributed/sft")
NUM_GPUS = 4
Overrides = dict[str, object]


@dataclass(frozen=True, eq=False)
class Layout:
    name: str
    dp_replicate: int
    cp: int
    ep: int

    def overrides(self) -> Overrides:
        return {
            "model.dp_replicate": self.dp_replicate,
            "model.cp": self.cp,
            "model.ep": self.ep,
        }


FSDP4_EP4 = Layout("fsdp4_ep4", dp_replicate=1, cp=1, ep=4)
CP4_EP4 = Layout("cp4_ep4", dp_replicate=1, cp=4, ep=4)
MOE_LAYOUTS = (FSDP4_EP4, CP4_EP4)

FSDP4_EP1 = Layout("fsdp4_ep1", dp_replicate=1, cp=1, ep=1)
CP4_EP1 = Layout("cp4_ep1", dp_replicate=1, cp=4, ep=1)
DENSE_LAYOUTS = (FSDP4_EP1, CP4_EP1)


@dataclass(frozen=True, eq=False)
class Feature:
    name: str
    overrides: Overrides
    layout: Layout = FSDP4_EP4


FEATURES = (
    Feature("ac_selective", {"model.ac.mode": "selective"}),
    Feature("act_offload_off", {"model.ac_offloading": None}),
    Feature("optim_offload_off", {"model.offload": "none"}),
    Feature("full_offload", {"model.offload": "full"}),
    Feature("fullgraph", {"model.compile.fullgraph": True}, layout=CP4_EP4),
)


@dataclass(frozen=True, eq=False)
class Arch:
    """A trainer arch: its tiny config, the tokenizer matching its vocab, and how the suite runs it."""

    name: str
    config: PretrainedConfig
    tokenizer: str
    layouts: tuple[Layout, ...] = MOE_LAYOUTS
    features: tuple[Feature, ...] = ()
    overrides: Overrides = field(default_factory=dict)
    resume_layout: Layout | None = None
    launch_s: int = 160

    def cases(self) -> list["Case"]:
        for feature in self.features:
            if feature.layout not in self.layouts:
                raise ValueError(
                    f"{self.name}: feature {feature.name} runs on {feature.layout.name}, outside its layouts"
                )
        return [Case(self, layout) for layout in self.layouts] + [
            Case(self, feature.layout, feature) for feature in self.features
        ]


@dataclass(frozen=True, eq=False)
class Case:
    arch: Arch
    layout: Layout
    feature: Feature | None = None

    @property
    def id_parts(self) -> tuple[str, str, str]:
        return self.arch.name, self.layout.name, self.feature.name if self.feature else "base"

    @property
    def id(self) -> str:
        return "-".join(self.id_parts)

    @property
    def resume(self) -> bool:
        return self.feature is None and self.layout is self.arch.resume_layout

    @property
    def check_graph_breaks(self) -> bool:
        return self.feature is None and self.layout in (CP4_EP4, CP4_EP1)

    @property
    def shard_unit(self) -> tuple[str, bool]:
        return self.arch.name, self.feature is not None

    def overrides(self) -> Overrides:
        return self.arch.overrides | self.layout.overrides() | (self.feature.overrides if self.feature else {})


def assign_shards(cases: list[Case], num_shards: int) -> dict[tuple[str, bool], int]:
    """Map each shard unit (an arch's base cases, or its feature cases) to a shard, balancing estimated launch time."""
    costs: defaultdict[tuple[str, bool], int] = defaultdict(int)
    for case in cases:
        launches = 2 if case.resume else 1
        costs[case.shard_unit] += launches * case.arch.launch_s
    loads = [0] * num_shards
    shards = {}
    for unit, cost in sorted(costs.items(), key=lambda item: (-item[1], item[0])):
        shards[unit] = loads.index(min(loads))
        loads[shards[unit]] += cost
    return shards


def check_k_selectable(names: set[str]) -> None:
    for name, other in permutations(sorted(names), 2):
        if name in other:
            raise ValueError(f"-k {name} would also select {other}")


def sft_args(case: Case, model_name: str) -> list[str]:
    return [
        "@",
        (CONFIG_DIR / "base.toml").as_posix(),
        "--model.name",
        model_name,
        "--tokenizer.name",
        case.arch.tokenizer,
        "--deployment.num-train-gpus",
        str(NUM_GPUS),
        *to_cli(case.overrides()),
    ]


def to_cli(overrides: Overrides) -> list[str]:
    args = []
    for key, value in overrides.items():
        rendered = "None" if value is None else json.dumps(value) if isinstance(value, list | dict) else str(value)
        args += ["--" + key.replace("_", "-"), rendered]
    return args


def load_metrics(run_dir: Path) -> list[dict]:
    with open(run_dir / "monitors" / "file" / "metrics.jsonl") as f:
        return [json.loads(line) for line in f]


def assert_training_rows(rows: list[dict], expected_steps: range) -> None:
    train_rows = [row for row in rows if "loss/nan_count" in row]
    assert [row["step"] for row in train_rows] == list(expected_steps)
    for row in train_rows:
        assert row["loss/nan_count"] == 0, row
        assert math.isfinite(row.get("loss/mean", math.nan)), row


GRAPH_BREAK_TAG = "[__graph_breaks]"
GRAPH_BREAK_STARTS = ("Graph break", "WON'T CONVERT")
GRAPH_BREAK_LOCATION = re.compile(r"in user code at (\S+:\d+)")
DYNAMO_TRACE_MARKER = "torchdynamo start tracing"


def ranks_without_dynamo_logs(run_dir: Path) -> list[str]:
    """Launch ranks whose logs show no dynamo tracing, which would make a clean graph-break log meaningless."""
    missing = []
    for launch in sorted(run_dir.glob("logs/attempt_*")):
        rank_dirs = {path.name: path for path in launch.glob("trainer/torchrun/*/attempt_0/*") if path.is_dir()}
        for rank in map(str, range(NUM_GPUS)):
            logs = rank_dirs[rank].glob("*.log") if rank in rank_dirs else []
            if not any(DYNAMO_TRACE_MARKER in log.read_text(errors="replace") for log in logs):
                missing.append(f"{launch.name}/rank{rank}")
    return missing


def graph_break_reports(run_dir: Path) -> list[str]:
    """Distinct graph breaks that dynamo logged on any rank, as `location: detail`.

    The `graph_breaks` log artifact tags every line of every graph-break record, so a record is found by its first line.
    """
    details: dict[str, str] = {}
    for log in run_dir.glob("logs/attempt_*/trainer/torchrun/**/*.log"):
        messages = [
            line.split(GRAPH_BREAK_TAG, 1)[1].strip()
            for line in log.read_text(errors="replace").splitlines()
            if GRAPH_BREAK_TAG in line
        ]
        for i, message in enumerate(messages):
            if not message.startswith(GRAPH_BREAK_STARTS):
                continue
            location = GRAPH_BREAK_LOCATION.search(message)
            key = location.group(1) if location else message
            details.setdefault(key, "")
            following = messages[i + 1 : i + 6]
            reason_at = next((j for j, m in enumerate(following) if m.startswith("Graph Break Reason:")), None)
            if reason_at is None:
                details[key] = details[key] or next((m for m in following if "fall back to eager" in m), "")
                continue
            title = next((m for m in following[reason_at + 1 :] if m), None)
            if title is not None:
                details[key] = title
            elif not details[key]:
                details[key] = following[reason_at].removeprefix("Graph Break Reason:").strip()
    return sorted(f"{key}: {detail}" for key, detail in details.items())


def completed_step(rows: list[dict], step: int) -> bool:
    return any(row.get("step") == step and "loss/nan_count" in row for row in rows)


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
ROOT_CAUSE_LOCAL_RANK = re.compile(r"Root Cause \(first observed failure\):.*?local_rank:\s*(\d+)", re.DOTALL)
REPORT_TAIL_LINES = 40


def log_tail(path: Path) -> str:
    lines = ANSI_ESCAPE.sub("", path.read_text(errors="replace")).splitlines()
    return f"--- last {REPORT_TAIL_LINES} lines of {path} ---\n" + "\n".join(lines[-REPORT_TAIL_LINES:])


def failure_report(log_dir: Path) -> str:
    """The tails of a launch attempt's `trainer.log` and of the rank torchrun names as the root cause (else rank 0)."""
    trainer_log = log_dir / "trainer.log"
    if not trainer_log.exists():
        return f"no trainer log at {trainer_log}"
    root_cause = ROOT_CAUSE_LOCAL_RANK.search(ANSI_ESCAPE.sub("", trainer_log.read_text(errors="replace")))
    rank = root_cause.group(1) if root_cause else "0"
    rank_dirs = log_dir.glob(f"trainer/torchrun/*/attempt_0/{rank}")
    rank_logs = [rank_dir / name for rank_dir in rank_dirs for name in ("stdout.log", "stderr.log")]
    return "\n".join(log_tail(path) for path in [trainer_log, *rank_logs] if path.exists())
