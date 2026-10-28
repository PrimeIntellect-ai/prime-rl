"""The matrix the SFT suite runs: parallel layouts, features and archs, expanded into `CASES`.

Each `Arch` contributes two kinds of case:

- A base case on every layout in its `layouts`. It trains `TRAIN_STEPS` steps (see `test_matrix.py`). The base case
  on the arch's `resume_layout`, if set, also saves a checkpoint and resumes in a fresh process.
- One case per entry in its `features`. Each feature changes one setting and runs on its single pinned layout.

README.md explains why only qwen3_moe sets `features` and `resume_layout`. `Arch.layouts` picks the layout family,
`MOE_LAYOUTS` (expert parallel) or `DENSE_LAYOUTS` (ep=1). Each arch's tiny config, tokenizer and renderer live on its
`TinyModel` in `tiny_models.py`. Case ids read `{arch}-{layout}-{feature or base}`. No name is a substring of
another, so `-k` selects cases by arch, layout or feature name.

TODO: afmoe is the one registered trainer arch left out: SFT has no typed renderer for it.
"""

from dataclasses import dataclass
from itertools import permutations

from tests.regression.distributed import tiny_models

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
    Feature("deepep_full_ac", {"model.moe.dispatch.type": "deepep"}),
)


@dataclass(frozen=True, eq=False)
class Arch:
    """A tiny model and how the SFT suite runs it."""

    model: tiny_models.TinyModel
    layouts: tuple[Layout, ...] = MOE_LAYOUTS
    features: tuple[Feature, ...] = ()
    resume_layout: Layout | None = None
    launch_s: int = 160

    @property
    def name(self) -> str:
        return self.model.name

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
        return self.layout.overrides() | (self.feature.overrides if self.feature else {})


def check_k_selectable(names: set[str]) -> None:
    for name, other in permutations(sorted(names), 2):
        if name in other:
            raise ValueError(f"-k {name} would also select {other}")


ARCHS = (
    # The only arch with feature rows and resume, as README.md explains.
    Arch(tiny_models.qwen3_moe, features=FEATURES, resume_layout=FSDP4_EP4),
    Arch(tiny_models.glm_moe_dsa),
    Arch(tiny_models.deepseek_v4, launch_s=350),
    Arch(tiny_models.llama, layouts=DENSE_LAYOUTS),
    Arch(tiny_models.qwen3_dense, layouts=DENSE_LAYOUTS),
    Arch(tiny_models.qwen3_5_dense, layouts=DENSE_LAYOUTS),
    Arch(tiny_models.glm4_moe),
    Arch(tiny_models.gpt_oss),
    Arch(tiny_models.laguna),
    Arch(tiny_models.minimax_m2),
    Arch(tiny_models.nemotron_h),
    Arch(tiny_models.qwen3_5_moe),
    Arch(tiny_models.qwen4_exp),
)
CASES = [case for arch in ARCHS for case in arch.cases()]
check_k_selectable({part for case in CASES for part in case.id_parts})
