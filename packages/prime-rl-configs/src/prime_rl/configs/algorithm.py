"""Algorithm abstraction: sampling, advantage, and loss routing.

An algorithm is a preset of three pieces:

1. **Sampling** — which model generates train rollouts. ``source`` is a model
   reference: ``"policy"`` (the live policy) or a ``[orchestrator.models]``
   registry key (a frozen hosted model).
2. **Advantage** — the per-token training signal, one concept at different
   granularities and evaluation sites: group-relative strategies compute
   scalars on the orchestrator and ship numbers; reference-KL strategies ship
   reference prefill logprobs and the trainer evaluates the per-token signal
   against the live policy. The strategy determines which loss core consumes
   the action tokens (``rl`` / ``ce`` / ``ref_kl``).
3. **Loss routing** — what happens to env-provided observation tokens in
   multi-turn rollouts (``none`` drops them from the loss — the default;
   ``ce`` trains on them with a per-token weight, ECHO).

There are no model roles ("teacher", "judge") in the schema — advantage
strategies hold *references* to named hosted models, and the same registry
entry can serve different algorithms in the same run. Presets are vetted
bundles; every piece can be overridden individually for research. The trainer
is algorithm-blind: routing ships per token on the wire and the trainer just
executes loss cores.
"""

import warnings
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.utils.config import BaseConfig

AlgorithmName: TypeAlias = Literal["grpo", "opd", "sft_distill", "self_distill", "echo"]

# Reserved model-registry key for the live policy (weight-updated, prefix
# caches salted per version). Every other key names a frozen hosted model
# from ``[orchestrator.models]``.
POLICY_MODEL: str = "policy"

ActionLossCore: TypeAlias = Literal["rl", "ce", "ref_kl"]
ObservationLossCore: TypeAlias = Literal["none", "ce"]


# ---------------------------------------------------------------------------
# Component 1: sampling
# ---------------------------------------------------------------------------


class SamplingConfig(BaseConfig):
    source: str | None = POLICY_MODEL
    """Model reference for train rollout generation: ``"policy"`` (the live
    policy — prefix caches salted per version, sampling logprobs requested,
    rollouts age off-policy) or a ``[orchestrator.models]`` key (frozen hosted
    model — stable prefix cache, no sampling logprobs, rollouts never go
    stale). ``None`` is only set by presets that require a frozen source and
    must be resolved via ``algorithm.model`` or an explicit value."""


# ---------------------------------------------------------------------------
# Component 2: advantage strategies
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


class GroupNormAdvantageConfig(BaseConfig):
    type: Literal["group_norm"] = "group_norm"
    """GRPO: scalar advantage = reward minus the per-group mean baseline,
    consumed by the ``rl`` loss core on the rollout's action tokens."""

    action_core: ClassVar[ActionLossCore] = "rl"
    group_relative: ClassVar[bool] = True

    length_penalty: LengthPenaltyConfig | None = None
    """Correctness-gated length penalty. ``tokens`` shapes by weighted token cost; ``turns`` shapes by trajectory turn count; None disables shaping. In mixed groups, lower-cost correct rollouts get amplified advantage (up to 2x), higher-cost correct rollouts are unchanged, incorrect untouched. In all-correct groups, below-average-cost rollouts get advantage in [0, 1], others get 0."""


class RewardAdvantageConfig(BaseConfig):
    type: Literal["reward"] = "reward"
    """Scalar advantage = raw reward, no group baseline. Consumed by the
    ``rl`` loss core."""

    action_core: ClassVar[ActionLossCore] = "rl"
    group_relative: ClassVar[bool] = False


class RefKLAdvantageConfig(BaseConfig):
    type: Literal["ref_kl"] = "ref_kl"
    """On-policy distillation (OPD): the per-token signal is the reverse KL to
    a reference model, evaluated in the trainer from reference prefill
    logprobs scored over each sample's own context (``ref_logprobs`` on the
    wire, ``ref_kl`` loss core). Group-relative scalars are still assigned:
    their sign steers the DPPO masking direction in the loss, and the
    zero-advantage filter reads them."""

    action_core: ClassVar[ActionLossCore] = "ref_kl"
    group_relative: ClassVar[bool] = True

    model: str | None = None
    """Registry key of the reference model (a ``[orchestrator.models]``
    entry). Required — set it here or fold via ``algorithm.model``.
    ``"policy"`` is rejected: scoring the policy under itself yields zero KL
    signal (use ``demo_ref_kl`` for demo-conditioned self-teaching)."""

    max_concurrent: int = Field(32, ge=1)
    """Maximum concurrent prefill requests per batch."""

    length_penalty: LengthPenaltyConfig | None = None
    """Length penalty applied to the group-relative scalars (see
    ``group_norm``)."""


class DemoRefKLAdvantageConfig(BaseConfig):
    type: Literal["demo_ref_kl"] = "demo_ref_kl"
    """Self-distillation (SDFT, https://arxiv.org/abs/2601.19897): the
    per-token signal is the reverse KL to a reference model conditioned on an
    expert demonstration. The scoring prefix is rebuilt from the rollout's
    first-turn messages with the demonstration woven into the user message via
    ``template``; completion logprobs are aligned back onto the sample.
    Requires single-step trajectories. No scalar advantage is assigned —
    rollouts keep ``advantage=None`` (advantage-based filters never fire) and
    samples ship a neutral 0.0."""

    action_core: ClassVar[ActionLossCore] = "ref_kl"
    group_relative: ClassVar[bool] = False

    model: str | None = None
    """Registry key of the reference model. ``"policy"`` is the SDFT paper's
    setting — the current model conditioned on the demo *is* the reference —
    and needs no extra deployment. Point at a ``[orchestrator.models]`` entry
    to score under a frozen copy instead."""

    demo_key: str = "demonstration"
    """Key holding the expert demonstration text — looked up in the example's
    ``info`` dict first, then as a top-level rollout field (e.g. ``answer``)."""

    template: str = (
        "{question}\n\n"
        "Here is an example of an expert response:\n"
        "<demonstration>\n{demonstration}\n</demonstration>\n\n"
        "Answer with a response of your own."
    )
    """Template for the demo-conditioned user message. Receives ``{question}``
    (the original user message text) and ``{demonstration}``."""

    max_concurrent: int = Field(32, ge=1)
    """Maximum concurrent prefill requests per batch."""


class SupervisedAdvantageConfig(BaseConfig):
    type: Literal["supervised"] = "supervised"
    """Cross-entropy on the sampled tokens (SFT distillation). The ``ce``
    loss core ignores scalar advantages, but group-relative scalars are still
    assigned so reward-based filtering keeps working (the zero-advantage
    filter drops uniform-reward groups)."""

    action_core: ClassVar[ActionLossCore] = "ce"
    group_relative: ClassVar[bool] = True


class CustomAdvantageConfig(BaseConfig):
    type: Literal["custom"] = "custom"
    """Custom scalar advantage function, consumed by the ``rl`` loss core."""

    action_core: ClassVar[ActionLossCore] = "rl"
    group_relative: ClassVar[bool] = False

    import_path: str
    """Import path to the advantage function (e.g. ``my_module.my_advantage``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the advantage function."""


AdvantageConfig: TypeAlias = Annotated[
    GroupNormAdvantageConfig
    | RewardAdvantageConfig
    | RefKLAdvantageConfig
    | DemoRefKLAdvantageConfig
    | SupervisedAdvantageConfig
    | CustomAdvantageConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Component 3: loss routing
# ---------------------------------------------------------------------------


class LossRoutingConfig(BaseConfig):
    """Routing for tokens the advantage strategy doesn't already determine.
    The action-token core is derived from the advantage strategy
    (``advantage.action_core``); this config only routes env-provided
    observation tokens."""

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

# Preset component tables. Fields the user sets explicitly always win. Model
# references (``sampling.source`` of sft_distill, ``advantage.model`` of opd /
# self_distill) have no sensible default — presets leave them ``None`` and
# validation demands ``algorithm.model`` (or the component field).
_PRESETS: dict[str, dict[str, Any]] = {
    "grpo": dict(
        sampling=lambda: SamplingConfig(),
        advantage=lambda: GroupNormAdvantageConfig(),
        loss=lambda: LossRoutingConfig(),
    ),
    "opd": dict(
        sampling=lambda: SamplingConfig(),
        advantage=lambda: RefKLAdvantageConfig(),
        loss=lambda: LossRoutingConfig(),
    ),
    "sft_distill": dict(
        sampling=lambda: SamplingConfig(source=None),
        advantage=lambda: SupervisedAdvantageConfig(),
        loss=lambda: LossRoutingConfig(),
    ),
    "self_distill": dict(
        sampling=lambda: SamplingConfig(),
        advantage=lambda: DemoRefKLAdvantageConfig(),
        loss=lambda: LossRoutingConfig(),
    ),
    "echo": dict(
        sampling=lambda: SamplingConfig(),
        advantage=lambda: GroupNormAdvantageConfig(),
        loss=lambda: LossRoutingConfig(observation="ce"),
    ),
}


class AlgorithmConfig(BaseConfig):
    name: AlgorithmName = "grpo"
    """Algorithm preset. Resolves any component left unset:

    - ``grpo`` — policy group sampling, group-relative advantage, RL loss.
    - ``opd`` — on-policy distillation: policy samples, ``ref_kl`` advantage against a reference model. Needs ``model``.
    - ``sft_distill`` — a frozen model samples, the policy trains with CE on its tokens (``supervised``). Needs ``model``.
    - ``self_distill`` — SDFT: policy samples, ``demo_ref_kl`` advantage. ``model = "policy"`` is the paper's setting.
    - ``echo`` — GRPO on action tokens + weighted CE on env-observation tokens.
    """

    model: str | None = Field(None, exclude=True)
    """Model reference for whichever component the preset leaves unresolved:
    ``advantage.model`` (opd / self_distill) or ``sampling.source``
    (sft_distill). ``"policy"`` or a ``[orchestrator.models]`` key. Set the
    component fields directly for multi-model setups. Write-only input sugar —
    folded by validation and excluded from dumps so resolved configs
    round-trip."""

    sampling: SamplingConfig | None = None
    """Sampling component override. Unset inherits from the preset."""

    advantage: AdvantageConfig | None = None
    """Advantage strategy override. Unset inherits from the preset; setting
    one replaces the preset's choice wholesale."""

    loss: LossRoutingConfig | None = None
    """Loss routing override. Unset inherits from the preset."""

    @property
    def model_refs(self) -> set[str]:
        """Every model the algorithm references (``"policy"`` included)."""
        refs: set[str] = set()
        if self.sampling is not None and self.sampling.source is not None:
            refs.add(self.sampling.source)
        advantage_model = getattr(self.advantage, "model", None)
        if advantage_model is not None:
            refs.add(advantage_model)
        return refs

    @property
    def requires_group_advantage(self) -> bool:
        """True when the advantage strategy assigns group-relative scalars,
        i.e. degenerate with ``group_size=1``."""
        return self.advantage is not None and self.advantage.group_relative

    @model_validator(mode="after")
    def resolve_preset(self):
        """Fill unset components from the preset table.

        ``sampling`` / ``loss`` merge at the field level: a partial override
        (e.g. just ``loss.observation_weight`` on ``echo``) keeps the preset's
        other fields. ``advantage`` is a discriminated union — setting one
        replaces the preset's choice wholesale. Preset-filled components are
        removed from ``model_fields_set`` again so downstream folding (the
        orchestrator-level ``advantage`` shorthand) can tell genuine user
        input from preset defaults."""
        preset = _PRESETS[self.name]
        for component in ("sampling", "loss"):
            preset_value = preset[component]()
            current = getattr(self, component)
            if current is None:
                setattr(self, component, preset_value)
                self.__pydantic_fields_set__.discard(component)
            else:
                for field_name in type(current).model_fields:
                    if field_name not in current.model_fields_set:
                        setattr(current, field_name, getattr(preset_value, field_name))
        if self.advantage is None:
            self.advantage = preset["advantage"]()
            self.__pydantic_fields_set__.discard("advantage")
        return self

    @model_validator(mode="after")
    def fold_model(self):
        """Fold the ``model`` shorthand into whichever component reference the
        preset left unresolved. Declared after ``resolve_preset`` so the preset
        components exist, before ``validate_component_compatibility`` so the
        unresolved-reference errors only fire when folding couldn't help."""
        if self.model is None:
            return self
        filled = False
        if getattr(self.advantage, "model", "<absent>") is None:
            self.advantage.model = self.model
            filled = True
        if self.sampling is not None and self.sampling.source is None:
            self.sampling.source = self.model
            filled = True
        if not filled:
            raise ValueError(
                f"algorithm '{self.name}': 'model' is set but no component needs it — every model "
                "reference is already set, or the algorithm references no model. Set the component "
                "field (advantage.model / sampling.source) directly instead."
            )
        return self

    @model_validator(mode="after")
    def validate_component_compatibility(self):
        assert self.sampling is not None and self.advantage is not None and self.loss is not None  # resolved above
        if self.sampling.source is None:
            raise ValueError(
                f"algorithm '{self.name}' samples rollouts from a frozen model — set model='<key>' "
                "on the algorithm (a [orchestrator.models] entry), or sampling.source explicitly."
            )
        if getattr(self.advantage, "model", "<absent>") is None:
            raise ValueError(
                f"algorithm '{self.name}': advantage '{self.advantage.type}' needs a reference model — "
                "set model='<key>' on the algorithm ('policy' or a [orchestrator.models] entry), or "
                "advantage.model explicitly."
            )
        if isinstance(self.advantage, RefKLAdvantageConfig) and self.advantage.model == POLICY_MODEL:
            raise ValueError(
                f"algorithm '{self.name}': advantage 'ref_kl' with model='policy' is degenerate — "
                "the reference distribution equals the policy, so the KL signal is zero. Point at a "
                "[orchestrator.models] entry, or use 'demo_ref_kl' for demo-conditioned self-teaching."
            )
        if self.advantage.action_core == "rl" and self.sampling.source != POLICY_MODEL:
            raise ValueError(
                f"algorithm '{self.name}': advantage '{self.advantage.type}' trains with the rl loss "
                f"core but sampling.source='{self.sampling.source}' — importance ratios need the live "
                "policy's own sampling logprobs. Use the 'supervised' advantage (sft_distill) to "
                "distill frozen-model tokens."
            )
        return self

    def warn_group_size(self, group_size: int, env_name: str) -> None:
        """Group-relative scoring with a single rollout per example collapses
        every advantage to zero. Warn loudly — this is the classic footgun the
        algorithm presets exist to prevent."""
        if self.requires_group_advantage and group_size == 1:
            warnings.warn(
                f"Env '{env_name}' uses group-relative advantage (algorithm '{self.name}') with "
                "group_size=1 — every advantage is 0 and (with the default zero-advantage filter) "
                "no rollout will train. Set group_size >= 2 or a non-group-relative advantage "
                "(e.g. advantage.type='reward').",
                stacklevel=2,
            )
