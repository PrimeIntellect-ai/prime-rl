"""Algorithm abstraction: bundles sampling, scoring, and loss routing.

An algorithm is a preset of three components:

1. **Sampling** — who generates train rollouts (student or teacher pool).
   Group sizing stays on the env config (``group_size``).
2. **Scoring** — how rollouts become per-token training signal:
   - ``advantage``: group-level advantage assignment (e.g. GRPO group-norm).
   - ``token_scorer``: optional async per-sample scorer that attaches
     per-token data by querying auxiliary models (e.g. teacher logprobs).
3. **Loss routing** — which loss core applies to which tokens:
   - ``action``: core for model-generated tokens (``rl`` / ``ce`` / ``teacher_kl``).
   - ``observation``: core for env-provided tokens in multi-turn rollouts
     (``none`` drops them from the loss — the default; ``ce`` trains on them).

Presets are vetted bundles; every component can be overridden individually
for research. The trainer is algorithm-blind: routing ships per token on the
wire and the trainer just executes loss cores.
"""

import warnings
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.utils.config import BaseConfig

AlgorithmName: TypeAlias = Literal["grpo", "opd", "sft_distill", "self_distill", "echo"]

# Deprecated `orchestrator.training_mode` values map onto algorithm presets.
TrainingMode: TypeAlias = Literal["rl", "opd", "sft"]
TRAINING_MODE_TO_ALGORITHM: dict[str, str] = {"rl": "grpo", "opd": "opd", "sft": "sft_distill"}


# ---------------------------------------------------------------------------
# Component 1: sampling
# ---------------------------------------------------------------------------


class SamplingConfig(BaseConfig):
    source: Literal["student", "teacher"] = "student"
    """Which inference pool generates train rollouts. ``student`` samples the live
    policy (prefix caches are salted per policy version); ``teacher`` samples the
    frozen teacher pool (no version salt, sampling logprobs are not requested)."""


# ---------------------------------------------------------------------------
# Component 2a: group-level advantage assignment
# ---------------------------------------------------------------------------


class TokensLengthPenaltyConfig(BaseConfig):
    type: Literal["tokens"] = "tokens"

    completion_weight: float = Field(1.0, ge=0, allow_inf_nan=False)
    """Weight on model completion tokens. Finite and non-negative."""

    tool_response_weight: float = Field(1.0, ge=0, allow_inf_nan=False)
    """Weight on tool-response tokens (read from the rollout's ``*_total_tool_response_tokens`` harness metric; 0 if absent). Finite and non-negative."""


class TurnsLengthPenaltyConfig(BaseConfig):
    type: Literal["turns"] = "turns"


LengthPenaltyConfig: TypeAlias = Annotated[
    TokensLengthPenaltyConfig | TurnsLengthPenaltyConfig,
    Field(discriminator="type"),
]


class DefaultAdvantageConfig(BaseConfig):
    type: Literal["default"] = "default"

    length_penalty: LengthPenaltyConfig | None = None
    """Correctness-gated length penalty. ``tokens`` shapes by weighted token cost; ``turns`` shapes by trajectory turn count; None disables shaping. In mixed groups, lower-cost correct rollouts get amplified advantage (up to 2x), higher-cost correct rollouts are unchanged, incorrect untouched. In all-correct groups, below-average-cost rollouts get advantage in [0, 1], others get 0."""


class RewardAdvantageConfig(BaseConfig):
    type: Literal["reward"] = "reward"
    """Advantage = raw reward, no group baseline."""


class NoAdvantageConfig(BaseConfig):
    type: Literal["none"] = "none"
    """No group-level advantage: rollouts keep ``advantage=None`` (so the
    zero-advantage filter never fires) and samples ship a neutral 0.0. Use for
    algorithms whose training signal does not come from rewards (e.g.
    self-distillation)."""


class CustomAdvantageConfig(BaseConfig):
    type: Literal["custom"] = "custom"

    import_path: str
    """Import path to the advantage function (e.g. ``my_module.my_advantage``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the advantage function."""


AdvantageConfig: TypeAlias = Annotated[
    DefaultAdvantageConfig | RewardAdvantageConfig | NoAdvantageConfig | CustomAdvantageConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Component 2b: per-sample token scorers (async, may query auxiliary models)
# ---------------------------------------------------------------------------


class TeacherLogprobsConfig(BaseConfig):
    type: Literal["teacher_logprobs"] = "teacher_logprobs"
    """Score each sample's tokens under the teacher via prefill on the sample's
    own context. Fills ``TrainingSample.teacher_logprobs`` — consumed by the
    ``teacher_kl`` loss core (on-policy distillation)."""

    max_concurrent: int = Field(32, ge=1)
    """Maximum concurrent teacher prefill requests per batch."""


class DemoTeacherLogprobsConfig(BaseConfig):
    type: Literal["demo_teacher_logprobs"] = "demo_teacher_logprobs"
    """Score each sample's completion under the teacher conditioned on an expert
    demonstration (SDFT, https://arxiv.org/abs/2601.19897). The teacher prefix is
    rebuilt from the rollout's first-turn messages with the demonstration woven
    into the user message via ``template``; completion logprobs are aligned back
    onto the sample. Requires single-step trajectories."""

    demo_key: str = "demonstration"
    """Key under the example's ``info`` dict holding the expert demonstration text."""

    template: str = (
        "{question}\n\n"
        "Here is an example of an expert response:\n"
        "<demonstration>\n{demonstration}\n</demonstration>\n\n"
        "Answer with a response of your own."
    )
    """Template for the demo-conditioned user message. Receives ``{question}``
    (the original user message text) and ``{demonstration}``."""

    max_concurrent: int = Field(32, ge=1)
    """Maximum concurrent teacher prefill requests per batch."""


TokenScorerConfig: TypeAlias = Annotated[
    TeacherLogprobsConfig | DemoTeacherLogprobsConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Component 3: loss routing
# ---------------------------------------------------------------------------

ActionLossCore: TypeAlias = Literal["rl", "ce", "teacher_kl"]
ObservationLossCore: TypeAlias = Literal["none", "ce"]


class LossRoutingConfig(BaseConfig):
    action: ActionLossCore = "rl"
    """Loss core for model-generated tokens. ``rl`` is the configured RL loss
    (``trainer.loss``); ``ce`` is masked NLL (SFT); ``teacher_kl`` uses the
    per-token teacher KL as the policy-gradient signal (requires a
    ``teacher_logprobs``-family token scorer)."""

    observation: ObservationLossCore = "none"
    """Loss core for env-provided tokens of later turns (tool output, terminal
    responses). ``none`` masks them out (standard RL); ``ce`` trains on them with
    weight ``observation_weight`` (ECHO)."""

    observation_weight: float = Field(0.1, gt=0)
    """Per-token loss weight for observation tokens (ECHO's lambda). Only used
    when ``observation != 'none'``."""


# ---------------------------------------------------------------------------
# The algorithm bundle
# ---------------------------------------------------------------------------

# Preset component tables. ``None`` entries mean "component absent" (e.g. no
# token scorer); fields the user sets explicitly always win.
_PRESETS: dict[str, dict[str, Any]] = {
    "grpo": dict(
        sampling=lambda: SamplingConfig(source="student"),
        advantage=lambda: DefaultAdvantageConfig(),
        token_scorer=lambda: None,
        loss=lambda: LossRoutingConfig(action="rl"),
    ),
    "opd": dict(
        sampling=lambda: SamplingConfig(source="student"),
        advantage=lambda: DefaultAdvantageConfig(),
        token_scorer=lambda: TeacherLogprobsConfig(),
        loss=lambda: LossRoutingConfig(action="teacher_kl"),
    ),
    "sft_distill": dict(
        sampling=lambda: SamplingConfig(source="teacher"),
        advantage=lambda: DefaultAdvantageConfig(),
        token_scorer=lambda: None,
        loss=lambda: LossRoutingConfig(action="ce"),
    ),
    "self_distill": dict(
        sampling=lambda: SamplingConfig(source="student"),
        advantage=lambda: NoAdvantageConfig(),
        token_scorer=lambda: DemoTeacherLogprobsConfig(),
        loss=lambda: LossRoutingConfig(action="teacher_kl"),
    ),
    "echo": dict(
        sampling=lambda: SamplingConfig(source="student"),
        advantage=lambda: DefaultAdvantageConfig(),
        token_scorer=lambda: None,
        loss=lambda: LossRoutingConfig(action="rl", observation="ce"),
    ),
}


class AlgorithmConfig(BaseConfig):
    name: AlgorithmName = "grpo"
    """Algorithm preset. Resolves any component left unset:

    - ``grpo`` — student group sampling, group-relative advantage, RL loss.
    - ``opd`` — on-policy distillation: student samples, teacher logprobs, per-token teacher-KL signal.
    - ``sft_distill`` — teacher samples, student trains with CE on teacher tokens.
    - ``self_distill`` — SDFT: student samples, demo-conditioned self-teacher logprobs, teacher-KL signal.
    - ``echo`` — GRPO on action tokens + weighted CE on env-observation tokens.
    """

    sampling: SamplingConfig | None = None
    """Sampling component override. Unset inherits from the preset."""

    advantage: AdvantageConfig | None = None
    """Group-level advantage component override. Unset inherits from the preset."""

    token_scorer: TokenScorerConfig | None = None
    """Per-sample token scorer override. Unset inherits from the preset."""

    loss: LossRoutingConfig | None = None
    """Loss routing override. Unset inherits from the preset."""

    @property
    def requires_teacher(self) -> bool:
        return self.token_scorer is not None or (self.sampling is not None and self.sampling.source == "teacher")

    @property
    def requires_group_advantage(self) -> bool:
        """True when the advantage component is group-relative, i.e. degenerate
        with ``group_size=1``."""
        return isinstance(self.advantage, DefaultAdvantageConfig)

    @model_validator(mode="after")
    def resolve_preset(self):
        """Fill unset components from the preset table.

        ``sampling`` / ``loss`` merge at the field level: a partial override
        (e.g. just ``loss.observation_weight`` on ``echo``) keeps the preset's
        other fields. ``advantage`` / ``token_scorer`` are discriminated unions
        — setting one replaces the preset's choice wholesale, and
        ``token_scorer`` may be explicitly set to None to disable the preset's
        scorer."""
        preset = _PRESETS[self.name]
        for component in ("sampling", "loss"):
            preset_value = preset[component]()
            current = getattr(self, component)
            if current is None:
                setattr(self, component, preset_value)
            else:
                for field_name in type(current).model_fields:
                    if field_name not in current.model_fields_set:
                        setattr(current, field_name, getattr(preset_value, field_name))
        if self.advantage is None:
            self.advantage = preset["advantage"]()
        if "token_scorer" not in self.model_fields_set:
            self.token_scorer = preset["token_scorer"]()
        return self

    @model_validator(mode="after")
    def validate_component_compatibility(self):
        assert self.loss is not None and self.sampling is not None  # resolved above
        if self.loss.action == "teacher_kl" and self.token_scorer is None:
            raise ValueError(
                f"algorithm '{self.name}': loss.action='teacher_kl' requires a token_scorer that fills "
                "teacher logprobs (teacher_logprobs or demo_teacher_logprobs)."
            )
        if self.loss.action == "rl" and self.sampling.source == "teacher":
            raise ValueError(
                f"algorithm '{self.name}': loss.action='rl' with sampling.source='teacher' is invalid — "
                "importance ratios need the student's own sampling logprobs. Use loss.action='ce' "
                "to distill teacher tokens."
            )
        return self

    @classmethod
    def from_training_mode(cls, training_mode: str) -> "AlgorithmConfig":
        return cls(name=TRAINING_MODE_TO_ALGORITHM[training_mode])

    def warn_group_size(self, group_size: int, env_name: str) -> None:
        """Group-relative scoring with a single rollout per example collapses
        every advantage to zero. Warn loudly — this is the classic footgun the
        algorithm presets exist to prevent."""
        if self.requires_group_advantage and group_size == 1:
            warnings.warn(
                f"Env '{env_name}' uses group-relative advantage (algorithm '{self.name}') with "
                "group_size=1 — every advantage is 0 and (with the default zero-advantage filter) "
                "no rollout will train. Set group_size >= 2 or advantage.type='none'.",
                stacklevel=2,
            )
