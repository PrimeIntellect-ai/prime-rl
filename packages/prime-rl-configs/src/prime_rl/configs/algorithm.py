"""Algorithm abstraction: sampling, advantage, and loss routing.

An algorithm is a preset of three pieces:

1. **Sampling** — which model generates train rollouts. ``source`` is a model
   reference: ``"policy"`` (the live policy) or an inline frozen hosted model.
2. **Advantage** — the per-token training signal, one concept at different
   granularities and evaluation sites: group-relative strategies compute
   scalars on the orchestrator and ship numbers; reference-KL strategies ship
   reference prefill logprobs and the trainer evaluates the per-token signal
   against the live policy. The strategy determines which loss type consumes
   the action tokens (``rl`` / ``ce`` / ``ref_kl``).
3. **Loss routing** — what happens to env-provided observation tokens in
   multi-turn rollouts (``none`` drops them from the loss — the default;
   ``ce`` trains on them with a per-token weight, ECHO).

prime-rl only ever hosts the trainable policy. Every other model an algorithm
uses is an external OpenAI-compatible endpoint, declared inline on the
component that uses it (a :class:`FrozenModelConfig`). Model roles like
"teacher" are algorithm-local vocabulary over these references; the pipeline
branches on liveness alone. Presets are vetted bundles; every piece can be
overridden individually for research. The trainer is algorithm-blind: routing
ships per token on the wire and the trainer just executes loss types.
"""

import warnings
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.config import BaseConfig

AlgorithmName: TypeAlias = Literal["grpo", "opd", "sft_distill", "self_distill", "echo"]


class FrozenModelConfig(ClientConfig):
    """An externally hosted model behind an OpenAI-compatible endpoint: the
    client config plus the served model's ``name``.

    prime-rl never launches or updates these — only the trainable policy is
    ever hosted by prime-rl itself. Frozen models are reachable-but-unmanaged:
    ``base_url`` is required, their weights never change, and rollouts or
    scores from them never go stale (stable prefix cache, no off-policy
    aging)."""

    name: str
    """Served model name, sent as the ``model`` field of every request."""

    @model_validator(mode="after")
    def require_explicit_endpoint(self):
        if "base_url" not in self.model_fields_set and not self.is_elastic:
            raise ValueError(
                "a frozen model reference needs base_url — frozen models are externally "
                "hosted; prime-rl only ever hosts the trainable policy."
            )
        return self


ModelReference: TypeAlias = Literal["policy"] | FrozenModelConfig
"""``"policy"`` (the live policy — weight-updated: prefix caches salted per
version, sampling logprobs carried, rollouts age off-policy) or an inline
externally-hosted frozen model."""

ActionLossType: TypeAlias = Literal["rl", "ce", "ref_kl"]
ObservationLossType: TypeAlias = Literal["none", "ce"]


# ---------------------------------------------------------------------------
# Component 1: sampling
# ---------------------------------------------------------------------------


class SamplingConfig(BaseConfig):
    source: ModelReference | None = "policy"
    """Model reference for train rollout generation: ``"policy"`` (the live
    policy — prefix caches salted per version, sampling logprobs requested,
    rollouts age off-policy) or an inline frozen hosted model (stable prefix
    cache, no sampling logprobs, rollouts never go stale). ``None`` is only
    set by presets that require a frozen source and must be resolved via
    ``algo.model`` or an explicit value."""


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
    consumed by the ``rl`` loss type on the rollout's action tokens."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    group_relative: ClassVar[bool] = True

    length_penalty: LengthPenaltyConfig | None = None
    """Correctness-gated length penalty. ``tokens`` shapes by weighted token cost; ``turns`` shapes by trajectory turn count; None disables shaping. In mixed groups, lower-cost correct rollouts get amplified advantage (up to 2x), higher-cost correct rollouts are unchanged, incorrect untouched. In all-correct groups, below-average-cost rollouts get advantage in [0, 1], others get 0."""


class RewardAdvantageConfig(BaseConfig):
    type: Literal["reward"] = "reward"
    """Scalar advantage = raw reward, no group baseline. Consumed by the
    ``rl`` loss type."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    group_relative: ClassVar[bool] = False


class RefKLAdvantageConfig(BaseConfig):
    type: Literal["ref_kl"] = "ref_kl"
    """On-policy distillation (OPD): the per-token signal is the reverse KL to
    a reference model, evaluated in the trainer from reference prefill
    logprobs scored over each sample's own context (``ref_logprobs`` on the
    wire, ``ref_kl`` loss type). Group-relative scalars are still assigned:
    their sign steers the DPPO masking direction in the loss, and the
    zero-advantage filter reads them."""

    action_loss_type: ClassVar[ActionLossType] = "ref_kl"
    group_relative: ClassVar[bool] = True

    model: ModelReference | None = None
    """The reference model — an inline frozen hosted model (``name`` +
    ``base_url``). Required — set it here or fold via ``algo.model``.
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

    action_loss_type: ClassVar[ActionLossType] = "ref_kl"
    group_relative: ClassVar[bool] = False

    model: ModelReference = "policy"
    """The reference model. ``"policy"`` (the default) is the SDFT paper's
    setting — the current model conditioned on the demo *is* the reference —
    and needs no extra deployment. Set an inline frozen hosted model to score
    under a frozen copy instead."""

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
    loss type ignores scalar advantages, but group-relative scalars are still
    assigned so reward-based filtering keeps working (the zero-advantage
    filter drops uniform-reward groups)."""

    action_loss_type: ClassVar[ActionLossType] = "ce"
    group_relative: ClassVar[bool] = True


class CustomAdvantageConfig(BaseConfig):
    type: Literal["custom"] = "custom"
    """Custom advantage function, consumed by the ``rl`` loss type. Returns
    one scalar per rollout, optionally with per-token advantages aligned to
    each rollout's completion tokens."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
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
    The action-token loss type is derived from the advantage strategy
    (``advantage.action_loss_type``); this config only routes env-provided
    observation tokens."""

    observation: ObservationLossType = "none"
    """Loss type for env-provided tokens of later turns (tool output, terminal
    responses). ``none`` masks them out (standard RL); ``ce`` trains on them with
    weight ``observation_weight`` (ECHO)."""

    observation_weight: float = Field(0.1, gt=0)
    """Per-token loss weight for observation tokens (ECHO's lambda). Only used
    when ``observation != 'none'``."""


# ---------------------------------------------------------------------------
# The algorithm bundle
# ---------------------------------------------------------------------------

# Preset component tables. Fields the user sets explicitly always win. Model
# references with no sensible default (``sampling.source`` of sft_distill,
# ``advantage.model`` of opd) are left ``None`` by the presets and validation
# demands ``algo.model`` (or the component field); ``demo_ref_kl`` defaults to
# the live policy (the SDFT setting).
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
    - ``self_distill`` — SDFT: policy samples, ``demo_ref_kl`` advantage against the live policy by default.
    - ``echo`` — GRPO on action tokens + weighted CE on env-observation tokens.
    """

    model: ModelReference | None = Field(None, exclude=True)
    """Model reference shorthand: ``"policy"`` or an inline frozen hosted
    model. Folds into the component reference the preset leaves unresolved
    (``advantage.model`` for opd, ``sampling.source`` for sft_distill) or a
    component default the user didn't set; an explicit component reference
    that already equals it is accepted, a disagreeing one is an error.
    Write-only input sugar — folded by validation and excluded from dumps so
    resolved configs round-trip."""

    sampling: SamplingConfig | None = None
    """Sampling component override. Unset inherits from the preset."""

    advantage: AdvantageConfig | None = None
    """Advantage strategy override. Unset inherits from the preset; setting
    one replaces the preset's choice wholesale."""

    loss: LossRoutingConfig | None = None
    """Loss routing override. Unset inherits from the preset."""

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

    def fold_model_shorthand(self) -> None:
        """Fold the ``model`` shorthand into the component references.

        Fill-or-agree: an unresolved reference (``None``, or a preset default
        the user didn't set) takes the shorthand; an explicit reference that
        already equals it is redundant-but-consistent; if no reference accepts
        it, that's an error. Re-run after the orchestrator-level ``advantage``
        shorthand replaces the advantage component wholesale."""
        if self.model is None:
            return
        matched = False
        advantage = self.advantage
        if advantage is not None and "model" in type(advantage).model_fields:
            if advantage.model is None or "model" not in advantage.model_fields_set:
                advantage.model = self.model
                matched = True
            elif advantage.model == self.model:
                matched = True
        if self.sampling is not None:
            if self.sampling.source is None:
                self.sampling.source = self.model
                matched = True
            elif self.sampling.source == self.model:
                matched = True
        if not matched:
            raise ValueError(
                f"algorithm '{self.name}': 'model' is set but no component reference accepts it — "
                "every reference is already explicitly set to a different value, or the algorithm "
                "references no model. Set advantage.model / sampling.source directly instead."
            )

    @model_validator(mode="after")
    def fold_model(self):
        """Declared after ``resolve_preset`` so the preset components exist,
        before ``validate_component_compatibility`` so unresolved-reference
        errors only fire when folding couldn't help."""
        self.fold_model_shorthand()
        return self

    @model_validator(mode="after")
    def validate_component_compatibility(self):
        assert self.sampling is not None and self.advantage is not None and self.loss is not None  # resolved above
        if self.sampling.source is None:
            raise ValueError(
                f"algorithm '{self.name}' samples rollouts from a frozen model — set 'model' on the "
                "algorithm (an inline hosted model: name + base_url), or sampling.source "
                "explicitly."
            )
        if getattr(self.advantage, "model", "<absent>") is None:
            raise ValueError(
                f"algorithm '{self.name}': advantage '{self.advantage.type}' needs a reference model — "
                "set 'model' on the algorithm (an inline hosted model: name + base_url), "
                "or advantage.model explicitly."
            )
        if isinstance(self.advantage, RefKLAdvantageConfig) and self.advantage.model == "policy":
            raise ValueError(
                f"algorithm '{self.name}': advantage 'ref_kl' with model='policy' is degenerate — "
                "the reference distribution equals the policy, so the KL signal is zero. Point at a "
                "frozen hosted model, or use 'demo_ref_kl' for demo-conditioned self-teaching."
            )
        if self.advantage.action_loss_type == "rl" and self.sampling.source != "policy":
            raise ValueError(
                f"algorithm '{self.name}': advantage '{self.advantage.type}' trains with the rl loss "
                "type but sampling.source is a frozen model — importance ratios need the live "
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
