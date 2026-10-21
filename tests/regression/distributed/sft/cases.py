"""The SFT functionality matrix, rendered as `uv run sft` arguments.

Every case runs on one 8-GPU node from `base.toml`, which is the `ModelConfig` defaults plus an fp32 MoE router.
Each arch contributes two kinds of case:

- A base case on every layout in its `layouts`. It trains, saves a checkpoint, then resumes in a fresh process.
- One case per entry in its `features`. Each feature changes one setting and runs on its single pinned layout.

`Arch.layouts` picks the layout family: `MOE_LAYOUTS` (expert parallel) or `DENSE_LAYOUTS` (ep=1). `Arch.skip`
removes a layout (along with the feature cases pinned to it) or a single feature, and the reason shows in the
pytest report. Case ids read `{arch}-{layout}-{feature or base}`. No name is a substring of another, so `-k`
selects cases by arch, layout or feature name. For example, one case:

    uv run pytest tests/regression/distributed -k "qwen3_moe and cp2_ep4 and base"

or every case on one layout, including the feature cases pinned to it:

    uv run pytest tests/regression/distributed -k cp2_ep4

`scripts/ci/run_distributed_nightly.sh` runs the same pytest command and passes extra args through to it.
"""

import json
import math
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import pytest

from tests.regression.distributed.tiny_models import TOKENIZERS

CONFIG_DIR = Path("tests/regression/configs/distributed/sft")
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


FSDP8_EP8 = Layout("fsdp8_ep8", dp_replicate=1, cp=1, ep=8)
HSDP2X4_EP4 = Layout("hsdp2x4_ep4", dp_replicate=2, cp=1, ep=4)
CP2_EP4 = Layout("cp2_ep4", dp_replicate=1, cp=2, ep=4)
CP4_EP8 = Layout("cp4_ep8", dp_replicate=1, cp=4, ep=8)
MOE_LAYOUTS = (FSDP8_EP8, HSDP2X4_EP4, CP2_EP4, CP4_EP8)

FSDP8_EP1 = Layout("fsdp8_ep1", dp_replicate=1, cp=1, ep=1)
HSDP2X4_EP1 = Layout("hsdp2x4_ep1", dp_replicate=2, cp=1, ep=1)
CP2_EP1 = Layout("cp2_ep1", dp_replicate=1, cp=2, ep=1)
CP4_EP1 = Layout("cp4_ep1", dp_replicate=1, cp=4, ep=1)
DENSE_LAYOUTS = (FSDP8_EP1, HSDP2X4_EP1, CP2_EP1, CP4_EP1)


@dataclass(frozen=True, eq=False)
class Feature:
    name: str
    overrides: Overrides
    layout: Layout = FSDP8_EP8


AC_OFF = Feature("ac_off", {"model.ac": None})
AC_SELECTIVE = Feature("ac_selective", {"model.ac.mode": "selective"})
ACT_OFFLOAD_OFF = Feature("act_offload_off", {"model.ac_offloading": None})
OPTIM_OFFLOAD_OFF = Feature("optim_offload_off", {"model.offload": "none"})
FULL_OFFLOAD = Feature("full_offload", {"model.offload": "full"})
EAGER = Feature("eager", {"model.compile": None})
UNFUSED = Feature("unfused", {"model.fusions.enabled": []})
BF16_ROUTER = Feature("bf16_router", {"model.moe_router_dtype": "bfloat16"}, layout=CP2_EP4)
FEATURES = (AC_OFF, AC_SELECTIVE, ACT_OFFLOAD_OFF, OPTIM_OFFLOAD_OFF, FULL_OFFLOAD, EAGER, UNFUSED, BF16_ROUTER)


@dataclass(frozen=True, eq=False)
class Arch:
    name: str
    layouts: tuple[Layout, ...] = MOE_LAYOUTS
    features: tuple[Feature, ...] = ()
    skip: dict[Layout | Feature, str] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class Case:
    arch: Arch
    layout: Layout
    feature: Feature | None = None

    @property
    def id(self) -> str:
        return f"{self.arch.name}-{self.layout.name}-{self.feature.name if self.feature else 'base'}"

    @property
    def resume(self) -> bool:
        return self.feature is None

    def overrides(self) -> Overrides:
        return self.layout.overrides() | (self.feature.overrides if self.feature else {})

    def param(self):
        reason = self.arch.skip.get(self.layout) or self.arch.skip.get(self.feature)
        return pytest.param(self, id=self.id, marks=[pytest.mark.skip(reason=reason)] if reason else [])


def cases_for(arch: Arch) -> list[Case]:
    for feature in arch.features:
        if feature.layout not in arch.layouts:
            raise ValueError(f"{arch.name}: feature {feature.name} runs on {feature.layout.name}, outside its layouts")
    return [Case(arch, layout) for layout in arch.layouts] + [
        Case(arch, feature.layout, feature) for feature in arch.features
    ]


def check_k_selectable(names: list[str]) -> None:
    for name, other in permutations(names, 2):
        if name in other:
            raise ValueError(f"-k {name} would also select {other}")


QWEN3_MOE = Arch("qwen3_moe", features=FEATURES)
GLM_MOE_DSA = Arch("glm_moe_dsa")
DEEPSEEK_V4 = Arch("deepseek_v4")
LLAMA = Arch(
    "llama",
    layouts=DENSE_LAYOUTS,
    skip={layout: "SFT's llama renderer key is the gated meta-llama repo, not accessible" for layout in DENSE_LAYOUTS},
)
QWEN3_DENSE = Arch("qwen3_dense", layouts=DENSE_LAYOUTS)
QWEN3_5_DENSE = Arch("qwen3_5_dense", layouts=DENSE_LAYOUTS)
GLM4_MOE = Arch("glm4_moe")
GPT_OSS = Arch("gpt_oss")
LAGUNA = Arch("laguna")
MINIMAX_M2 = Arch("minimax_m2")
NEMOTRON_H = Arch("nemotron_h")
QWEN3_5_MOE = Arch("qwen3_5_moe")
QWEN4_EXP = Arch("qwen4_exp")
AFMOE = Arch(
    "afmoe", skip={layout: "SFT needs a typed renderer and none is registered for afmoe" for layout in MOE_LAYOUTS}
)
ARCHS = (
    QWEN3_MOE,
    GLM_MOE_DSA,
    DEEPSEEK_V4,
    LLAMA,
    QWEN3_DENSE,
    QWEN3_5_DENSE,
    GLM4_MOE,
    GPT_OSS,
    LAGUNA,
    MINIMAX_M2,
    NEMOTRON_H,
    QWEN3_5_MOE,
    QWEN4_EXP,
    AFMOE,
)
CASES = [case for arch in ARCHS for case in cases_for(arch)]
check_k_selectable(
    [
        "base",
        *(a.name for a in ARCHS),
        *(l.name for l in (*MOE_LAYOUTS, *DENSE_LAYOUTS)),
        *(f.name for f in FEATURES),
    ]
)


def sft_args(case: Case, model_name: str) -> list[str]:
    return [
        "@",
        (CONFIG_DIR / "base.toml").as_posix(),
        "--model.name",
        model_name,
        "--tokenizer.name",
        TOKENIZERS[case.arch.name],
        "--deployment.num-train-gpus",
        "8",
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
