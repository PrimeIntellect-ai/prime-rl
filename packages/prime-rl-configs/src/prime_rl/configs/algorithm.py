"""Algorithm abstraction: bundles sampling, scoring, and loss routing.

An algorithm is a preset of three components:

1. **Sampling** — which model generates train rollouts. ``source`` is a model
   reference: ``"policy"`` (the live policy) or a ``[orchestrator.models]``
   registry key (a frozen hosted model).
2. **Scoring** — how rollouts become per-token training signal:
   - ``advantage``: group-level advantage assignment (e.g. GRPO group-norm).
   - ``token_scorer``: optional async per-sample scorer that attaches
     per-token data by querying a reference model (e.g. prefill logprobs).
3. **Loss routing** — which loss core applies to which tokens:
   - ``action``: core for model-generated tokens (``rl`` / ``ce`` / ``ref_kl``).
   - ``observation``: core for env-provided tokens in multi-turn rollouts
     (``none`` drops them from the loss — the default; ``ce`` trains on them).

There are no model roles ("teacher", "judge") in the schema — components hold
*references* to named hosted models, and the same registry entry can serve
different algorithms in the same run. Presets are vetted bundles; every
component can be overridden individually for research. The trainer is
algorithm-blind: routing ships per token on the wire and the trainer just
executes loss cores.
"""

import warnings
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.utils.config import BaseConfig

AlgorithmName: TypeAlias = Literal["grpo", "opd", "sft_distill", "self_distill", "echo"]

# Reserved model-registry key for the live policy (weight-updated, prefix
# caches salted per version). Every other key names a frozen hosted model
# from ``[orchestrator.models]``.
POLICY_MODEL: str = "policy"


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
# Component 2b: per-sample token scorers (async, query a reference model)
# ---------------------------------------------------------------------------


class LogprobsScorerConfig(BaseConfig):
    type: Literal["logprobs"] = "logprobs"
    """Score each sample's own context under a reference model via prefill.
    Fills ``TrainingSample.ref_logprobs`` — consumed by the ``ref_kl`` loss
    core (on-policy distillation)."""

    model: str | None = None
    """Registry key of the scoring model (a ``[orchestrator.models]`` entry).
    Required — set it here or fold via ``algorithm.model``. ``"policy"`` is
    rejected: scoring the policy under itself yields zero KL signal (use
    ``demo_logprobs`` for demo-conditioned self-teaching)."""

    max_concurrent: int = Field(32, ge=1)
    """Maximum concurrent prefill requests per batch."""


class DemoLogprobsScorerConfig(BaseConfig):
    type: Literal["demo_logprobs"] = "demo_logprobs"
    """Score each sample's completion under a reference model conditioned on an
    expert demonstration (SDFT, https://arxiv.org/abs/2601.19897). The scoring
    prefix is rebuilt from the rollout's first-turn messages with the
    demonstration woven into the user message via ``template``; completion
    logprobs are aligned back onto the sample. Requires single-step
    trajectories."""

    model: str | None = None
    """Registry key of the scoring model. ``"policy"`` is the SDFT paper's
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


TokenScorerConfig: TypeAlias = Annotated[
    LogprobsScorerConfig | DemoLogprobsScorerConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Component 3: loss routing
# ---------------------------------------------------------------------------

ActionLossCore: TypeAlias = Literal["rl", "ce", "ref_kl"]
ObservationLossCore: TypeAlias = Literal["none", "ce"]


class LossRoutingConfig(BaseConfig):
    action: ActionLossCore = "rl"
    """Loss core for model-generated tokens. ``rl`` is the configured RL loss
    (``trainer.loss``); ``ce`` is masked NLL (SFT); ``ref_kl`` uses the
    per-token reverse KL to a reference model as the policy-gradient signal
    (requires a token scorer that fills ``ref_logprobs``)."""

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
# token scorer); fields the user sets explicitly always win. Model references
# (``sampling.source`` of sft_distill, ``token_scorer.model`` of opd /
# self_distill) have no sensible default — presets leave them ``None`` and
# validation demands ``algorithm.model`` (or the component field).
_PRESETS: dict[str, dict[str, Any]] = {
    "grpo": dict(
        sampling=lambda: SamplingConfig(),
        advantage=lambda: DefaultAdvantageConfig(),
        token_scorer=lambda: None,
        loss=lambda: LossRoutingConfig(action="rl"),
    ),
    "opd": dict(
        sampling=lambda: SamplingConfig(),
        advantage=lambda: DefaultAdvantageConfig(),
        token_scorer=lambda: LogprobsScorerConfig(),
        loss=lambda: LossRoutingConfig(action="ref_kl"),
    ),
    "sft_distill": dict(
        sampling=lambda: SamplingConfig(source=None),
        advantage=lambda: DefaultAdvantageConfig(),
        token_scorer=lambda: None,
        loss=lambda: LossRoutingConfig(action="ce"),
    ),
    "self_distill": dict(
        sampling=lambda: SamplingConfig(),
        advantage=lambda: NoAdvantageConfig(),
        token_scorer=lambda: DemoLogprobsScorerConfig(),
        loss=lambda: LossRoutingConfig(action="ref_kl"),
    ),
    "echo": dict(
        sampling=lambda: SamplingConfig(),
        advantage=lambda: DefaultAdvantageConfig(),
        token_scorer=lambda: None,
        loss=lambda: LossRoutingConfig(action="rl", observation="ce"),
    ),
}


class AlgorithmConfig(BaseConfig):
    name: AlgorithmName = "grpo"
    """Algorithm preset. Resolves any component left unset:

    - ``grpo`` — policy group sampling, group-relative advantage, RL loss.
    - ``opd`` — on-policy distillation: policy samples, reference-model logprobs, per-token reverse-KL signal. Needs ``model``.
    - ``sft_distill`` — a frozen model samples, the policy trains with CE on its tokens. Needs ``model``.
    - ``self_distill`` — SDFT: policy samples, demo-conditioned reference logprobs, reverse-KL signal. ``model = "policy"`` is the paper's setting.
    - ``echo`` — GRPO on action tokens + weighted CE on env-observation tokens.
    """

    model: str | None = Field(None, exclude=True)
    """Model reference for whichever component the preset leaves unresolved:
    ``token_scorer.model`` (opd / self_distill) or ``sampling.source``
    (sft_distill). ``"policy"`` or a ``[orchestrator.models]`` key. Set the
    component fields directly for multi-model setups. Write-only input sugar —
    folded by validation and excluded from dumps so resolved configs
    round-trip."""

    sampling: SamplingConfig | None = None
    """Sampling component override. Unset inherits from the preset."""

    advantage: AdvantageConfig | None = None
    """Group-level advantage component override. Unset inherits from the preset."""

    token_scorer: TokenScorerConfig | None = None
    """Per-sample token scorer override. Unset inherits from the preset."""

    loss: LossRoutingConfig | None = None
    """Loss routing override. Unset inherits from the preset."""

    @property
    def model_refs(self) -> set[str]:
        """Every model the algorithm references (``"policy"`` included)."""
        refs: set[str] = set()
        if self.sampling is not None and self.sampling.source is not None:
            refs.add(self.sampling.source)
        if self.token_scorer is not None and self.token_scorer.model is not None:
            refs.add(self.token_scorer.model)
        return refs

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
    def fold_model(self):
        """Fold the ``model`` shorthand into whichever component references the
        preset left unresolved. Declared after ``resolve_preset`` so the preset
        components exist, before ``validate_component_compatibility`` so the
        unresolved-reference errors only fire when folding couldn't help."""
        if self.model is None:
            return self
        filled = False
        if self.token_scorer is not None and self.token_scorer.model is None:
            self.token_scorer.model = self.model
            filled = True
        if self.sampling is not None and self.sampling.source is None:
            self.sampling.source = self.model
            filled = True
        if not filled:
            raise ValueError(
                f"algorithm '{self.name}': 'model' is set but no component needs it — every model "
                "reference is already set, or the algorithm references no model. Set the component "
                "field (token_scorer.model / sampling.source) directly instead."
            )
        return self

    @model_validator(mode="after")
    def validate_component_compatibility(self):
        assert self.loss is not None and self.sampling is not None  # resolved above
        if self.sampling.source is None:
            raise ValueError(
                f"algorithm '{self.name}' samples rollouts from a frozen model — set model='<key>' "
                "on the algorithm (a [orchestrator.models] entry), or sampling.source explicitly."
            )
        if self.token_scorer is not None and self.token_scorer.model is None:
            raise ValueError(
                f"algorithm '{self.name}': token_scorer '{self.token_scorer.type}' needs a scoring model — "
                "set model='<key>' on the algorithm ('policy' or a [orchestrator.models] entry), or "
                "token_scorer.model explicitly."
            )
        if isinstance(self.token_scorer, LogprobsScorerConfig) and self.token_scorer.model == POLICY_MODEL:
            raise ValueError(
                f"algorithm '{self.name}': token_scorer 'logprobs' with model='policy' is degenerate — "
                "the reference distribution equals the policy, so the KL signal is zero. Point at a "
                "[orchestrator.models] entry, or use 'demo_logprobs' for demo-conditioned self-teaching."
            )
        if self.loss.action == "ref_kl" and self.token_scorer is None:
            raise ValueError(
                f"algorithm '{self.name}': loss.action='ref_kl' requires a token_scorer that fills "
                "ref_logprobs (logprobs or demo_logprobs)."
            )
        if self.loss.action == "rl" and self.sampling.source != POLICY_MODEL:
            raise ValueError(
                f"algorithm '{self.name}': loss.action='rl' with sampling.source='{self.sampling.source}' "
                "is invalid — importance ratios need the live policy's own sampling logprobs. Use "
                "loss.action='ce' to distill frozen-model tokens."
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
                "no rollout will train. Set group_size >= 2 or advantage.type='none'.",
                stacklevel=2,
            )
