"""Algorithm abstraction: sampling and the per-token training signal.

An algorithm is a bundle of two pieces:

1. **Sampling** — which model generates train rollouts. ``source`` is a model
   reference: ``"policy"`` (the live policy) or an inline frozen hosted model.
2. **Advantage** — credit assignment and loss routing, fused: one mapping from
   a finalized rollout to per-token ``(loss component, weight)``.
   Group-relative strategies compute scalars on the orchestrator and ship
   numbers; reference-KL strategies ship reference prefill logprobs and the
   trainer evaluates the per-token signal against the live policy. The
   strategy determines which loss component consumes the action tokens
   (``rl`` / ``ce`` / ``ref_kl``) and what happens to env-provided observation
   tokens (masked out by default; ``echo`` trains on them with weighted CE).

prime-rl only ever hosts the trainable policy. Every other model an algorithm
uses is an external OpenAI-compatible endpoint, declared inline on the
component that uses it (a :class:`FrozenModelConfig`). Model roles like
"teacher" are algorithm-local vocabulary over these references; the pipeline
branches on liveness alone. The advantage ``type`` names the algorithm and its
class defaults are the vetted setting — ``type = "opd"`` with nothing else IS
on-policy distillation; any key you set is visibly your own assembly. The
trainer is algorithm-blind: the loss is a sum of three components (rl, ce,
ref_kl), each normalized by its own global token count; per-token component
weights ship on the wire and the trainer just executes them.
"""

import warnings
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import AliasChoices, Field, model_validator

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.config import BaseConfig


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


# ---------------------------------------------------------------------------
# Component 1: sampling
# ---------------------------------------------------------------------------


class SamplingConfig(BaseConfig):
    source: ModelReference = "policy"
    """Model reference for train rollout generation: ``"policy"`` (the live
    policy — prefix caches salted per version, sampling logprobs requested,
    rollouts age off-policy) or an inline frozen hosted model (stable prefix
    cache, no sampling logprobs, rollouts never go stale)."""


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


class GRPOAdvantageConfig(BaseConfig):
    type: Literal["grpo"] = "grpo"
    """GRPO: scalar advantage = reward minus the per-group mean baseline,
    consumed by the ``rl`` loss component on the rollout's action tokens."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    group_relative: ClassVar[bool] = True

    length_penalty: LengthPenaltyConfig | None = None
    """Correctness-gated length penalty. ``tokens`` shapes by weighted token cost; ``turns`` shapes by trajectory turn count; None disables shaping. In mixed groups, lower-cost correct rollouts get amplified advantage (up to 2x), higher-cost correct rollouts are unchanged, incorrect untouched. In all-correct groups, below-average-cost rollouts get advantage in [0, 1], others get 0."""


class EchoRoleConfig(BaseConfig):
    """Echo CE supervision for one message role."""

    alpha: float = Field(0.1, gt=0)
    """Per-token ce weight for this role's env-provided tokens (ECHO's lambda)."""


class EchoRolesConfig(BaseConfig):
    """Which env-provided message roles train, each at its own weight.
    Setting any role replaces the whole table — unset roles stay disabled."""

    system: EchoRoleConfig | None = None
    user: EchoRoleConfig | None = None
    assistant: EchoRoleConfig | None = None
    tool: EchoRoleConfig | None = None

    @model_validator(mode="after")
    def require_a_role(self):
        if self.system is None and self.user is None and self.assistant is None and self.tool is None:
            raise ValueError("echo needs at least one role enabled (system, user, assistant, or tool)")
        return self


class EchoFilterConfig(BaseConfig):
    """User-supplied per-token filter narrowing the role-selected echo tokens.

    The callable is imported at startup and invoked once per rollout as
    ``filter_fn(rollout, **kwargs) -> list[list[bool]]`` — one keep-mask per
    trajectory step, each spanning that step's ``prompt_ids`` +
    ``completion_ids``. Tokens with ``False`` never receive echo weight; the
    filter can only narrow the role selection, not widen it. The raw rollout
    exposes message text and sampling logprobs, so content filters (e.g.
    dropping tool-output warnings) and sampling-probability filters need no
    extra framework surface."""

    import_path: str
    """Import path to the filter callable (e.g. ``my_module.drop_warnings``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the filter."""


class EchoAdvantageConfig(GRPOAdvantageConfig):
    type: Literal["echo"] = "echo"  # type: ignore[assignment]
    """ECHO: group-relative advantage on action tokens (GRPO), plus weighted
    CE on env-provided tokens of later turns (tool output, user feedback),
    selected by message role via the renderer's per-token attribution
    (requires ``orchestrator.renderer``; MITO rollouts carry no attribution).
    Selected tokens feed the ``ce`` loss component at their role's ``alpha``
    and stay outside the rl mask and its denominator."""

    roles: EchoRolesConfig = EchoRolesConfig(tool=EchoRoleConfig())
    """The role table. The default — tool-response bodies at ``alpha = 0.1``
    — is the vetted ECHO setting."""

    filter: EchoFilterConfig | None = None
    """Optional user-supplied filter narrowing the role-selected tokens."""


class MaxRLAdvantageConfig(BaseConfig):
    type: Literal["max_rl"] = "max_rl"
    """MaxRL (arXiv:2602.02710): scalar advantage = (reward − group mean) /
    group mean, consumed by the ``rl`` loss component. Normalizing by the
    mean instead of GRPO's standard deviation makes the policy gradient
    unbiased for the order-``group_size`` truncation of the maximum-likelihood
    objective: low-pass-rate examples get ~1/p weight, and ``group_size`` is
    the truncation order interpolating REINFORCE (1) → exact maximum
    likelihood (∞). Designed for non-negative (canonically binary) rewards;
    a group with mean reward 0 carries zero advantages everywhere (the
    zero-advantage filter drops it, matching the paper's K=0 convention)."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    group_relative: ClassVar[bool] = True


class RewardAdvantageConfig(BaseConfig):
    type: Literal["reward"] = "reward"
    """Scalar advantage = raw reward, no group baseline. Consumed by the
    ``rl`` loss component."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    group_relative: ClassVar[bool] = False


class OPDAdvantageConfig(BaseConfig):
    type: Literal["opd"] = "opd"
    """On-policy distillation: the per-token signal is the reverse KL to
    a reference model, evaluated in the trainer from reference prefill
    logprobs scored over each sample's own context (``ref_logprobs`` on the
    wire, ``ref_kl`` loss component). No scalar advantage is assigned —
    rollouts keep ``advantage=None`` (advantage-based filters never fire) and
    samples ship a neutral 0.0; rewards still flow to metrics. ``group_size``
    only fans out sampling."""

    action_loss_type: ClassVar[ActionLossType] = "ref_kl"
    group_relative: ClassVar[bool] = False
    model_role: ClassVar[str] = "teacher"

    model: ModelReference | None = None
    """The teacher — an inline frozen hosted model (``name`` + ``base_url``).
    Required — set it here or fold via ``algo.model`` / ``algo.teacher``.
    ``"policy"`` is rejected: scoring the policy under itself yields zero KL
    signal (use ``opsd`` for demo-conditioned self-teaching)."""

    max_concurrent: int = Field(32, ge=1)
    """Maximum concurrent prefill requests per batch."""


class OPSDAdvantageConfig(BaseConfig):
    type: Literal["opsd"] = "opsd"
    """On-policy self-distillation (SDFT, https://arxiv.org/abs/2601.19897):
    the per-token signal is the reverse KL to a reference model conditioned on
    an expert demonstration. The scoring prefix is rebuilt from the rollout's
    first-turn messages with the demonstration woven into the user message via
    ``template``; completion logprobs are aligned back onto the sample.
    Requires single-step trajectories. No scalar advantage is assigned —
    rollouts keep ``advantage=None`` (advantage-based filters never fire) and
    samples ship a neutral 0.0."""

    action_loss_type: ClassVar[ActionLossType] = "ref_kl"
    group_relative: ClassVar[bool] = False
    model_role: ClassVar[str] = "teacher"

    model: ModelReference = "policy"
    """The teacher. ``"policy"`` (the default) is the SDFT paper's setting —
    the current model conditioned on the demo *is* the teacher — and needs no
    extra deployment. Set an inline frozen hosted model to score under a
    frozen copy instead."""

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


class RLCSDAdvantageConfig(BaseConfig):
    type: Literal["rlcsd"] = "rlcsd"
    """RLCSD (arXiv:2606.11709): GRPO with a contrastive self-distillation
    modulation. The teacher scores each rollout's tokens under a correct
    sibling rollout as a hint and under ``num_negative_hints`` incorrect
    sibling hints (identical template, so the privilege-induced style shift
    cancels in the subtraction); the squashed contrast ``λ·tanh(e/τ)``
    modulates the group-relative advantage at tokens where it exceeds
    ``delta``, with a sign-preserving clamp so the verifier keeps the update
    direction. Ships per-token advantages on the ``rl`` loss component.
    Groups without both correct and incorrect rollouts get no modulation
    (uniform groups already die in the zero-advantage filter, matching the
    paper's group-discard rule)."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    group_relative: ClassVar[bool] = True
    model_role: ClassVar[str] = "teacher"

    model: ModelReference = "policy"
    """The teacher the hinted distributions are computed under. ``"policy"``
    (the default) approximates the paper's setting — there the teacher is a
    snapshot of the student refreshed every 10 steps; the live policy
    refreshes every weight update. Set an inline frozen hosted model to
    contrast under a fixed teacher instead."""

    num_negative_hints: int = Field(4, ge=1)
    """K: incorrect sibling hints whose probabilities average into the
    negative branch — marginalizing over error types stabilizes the
    contrast."""

    tau: float = Field(0.02, gt=0)
    """Soft-threshold slope of the tanh squash on the raw contrast."""

    lam: float = Field(0.5, gt=0)
    """Scale of the modulation: ``r_t = lam · tanh(e_ctr / tau)`` ∈ (-lam, lam)."""

    delta: float = Field(0.02, ge=0)
    """Modulation mask threshold: only tokens with ``|r_t| > delta`` get
    their advantage modulated (~20-30% of tokens at the defaults)."""

    eta: float = Field(1.0, ge=0)
    """Weight of the modulated path relative to the unmodulated path; both
    paths are normalized independently per rollout so the modulated tokens
    never dilute."""

    correct_threshold: float = 1.0
    """Rollouts with ``reward >= correct_threshold`` form the correct hint
    pool, the rest the incorrect pool — the binary verifier generalized to
    continuous rewards."""

    template: str = (
        "{question}\n\n"
        "Here is a reference solution to this problem:\n"
        "=== Reference Solution Begin ===\n{hint}\n=== Reference Solution End ===\n\n"
        "After reading the reference solution above, make sure you understand the "
        "reasoning behind each step. Please reason step by step, and put your final "
        "answer within \\boxed{{}}."
    )
    """Template for the hinted teacher context. Receives ``{question}`` (the
    original user message text) and ``{hint}`` (a sibling rollout's full
    completion text). Byte-for-byte identical for correct and incorrect
    hints — that symmetry is what cancels the style component."""

    max_concurrent: int = Field(32, ge=1)
    """Maximum concurrent prefill requests per batch (each rollout costs
    ``1 + num_negative_hints`` prefills)."""


class SFTAdvantageConfig(BaseConfig):
    type: Literal["sft"] = "sft"
    """SFT distillation: cross-entropy on the sampled tokens. The ``ce``
    loss component ignores scalar advantages, but group-relative scalars are still
    assigned so reward-based filtering keeps working (the zero-advantage
    filter drops uniform-reward groups)."""

    action_loss_type: ClassVar[ActionLossType] = "ce"
    group_relative: ClassVar[bool] = True
    source_role: ClassVar[str] = "teacher"
    """The sampling source is this algorithm's teacher — the frozen model
    whose tokens the policy trains on. Required: CE on the policy's own
    tokens is rejected at validation."""


class CustomAdvantageConfig(BaseConfig):
    type: Literal["custom"] = "custom"
    """Custom advantage function, consumed by the ``rl`` loss component. Returns
    one scalar per rollout, optionally with per-token advantages aligned to
    each rollout's completion tokens."""

    action_loss_type: ClassVar[ActionLossType] = "rl"
    group_relative: ClassVar[bool] = False

    import_path: str
    """Import path to the advantage function (e.g. ``my_module.my_advantage``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the advantage function."""


AdvantageConfig: TypeAlias = Annotated[
    GRPOAdvantageConfig
    | EchoAdvantageConfig
    | MaxRLAdvantageConfig
    | RewardAdvantageConfig
    | OPDAdvantageConfig
    | OPSDAdvantageConfig
    | RLCSDAdvantageConfig
    | SFTAdvantageConfig
    | CustomAdvantageConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# The algorithm bundle
# ---------------------------------------------------------------------------


class AlgorithmConfig(BaseConfig):
    """The advantage ``type`` names the algorithm, and each type's class
    defaults are its vetted setting — ``advantage = { type = "opd" }`` with a
    teacher IS on-policy distillation; any other key you set is visibly your
    own assembly.

    The algorithms:

    - ``grpo`` — policy group sampling, group-relative advantage, RL loss (the default).
    - ``max_rl`` — GRPO with mean-normalized advantages (maximum-likelihood RL).
    - ``opd`` — on-policy distillation: policy samples, per-token reverse KL against a reference model. Needs ``teacher``.
    - ``opsd`` — SDFT: policy samples, demo-conditioned reverse KL against the live policy by default.
    - ``sft`` — a frozen model samples, the policy trains with CE on its tokens. Needs ``teacher``.
    - ``echo`` — GRPO on action tokens + weighted CE on tool-response observation tokens.
    - ``reward`` / ``custom`` — raw-reward and user-supplied advantage functions.
    """

    model: ModelReference | None = Field(None, exclude=True, validation_alias=AliasChoices("model", "teacher"))
    """Model reference shorthand: ``"policy"`` or an inline frozen hosted
    model. Folds into the slot the advantage type declares for it —
    ``advantage.model`` when the type has one (opd, opsd), ``sampling.source``
    when the type's teacher is its sampling source (sft). A slot the user
    didn't set takes the shorthand; an explicit reference that already equals
    it is accepted, a disagreeing one is an error. ``teacher`` is an accepted
    alias — the distillation algorithms declare their reference's role as
    "teacher", and this is the slot it fills. Write-only input sugar — folded
    by validation and excluded from dumps so resolved configs round-trip."""

    sampling: SamplingConfig = SamplingConfig()
    """Sampling component."""

    advantage: AdvantageConfig = GRPOAdvantageConfig()
    """The per-token training signal: credit assignment and loss routing,
    fused. The ``type`` selects the algorithm."""

    @property
    def requires_group_advantage(self) -> bool:
        """True when the advantage strategy assigns group-relative scalars,
        i.e. degenerate with ``group_size=1``."""
        return self.advantage.group_relative

    @model_validator(mode="after")
    def fold_model(self):
        """Fold the ``model`` shorthand into the component references.

        Fill-or-agree: the slot the advantage type declares (``model`` field,
        or ``sampling.source`` for source-role types) takes the shorthand
        when the user didn't set it; an explicit reference that already
        equals it is redundant-but-consistent; if no slot accepts it, that's
        an error."""
        if self.model is None:
            return self
        matched = False
        advantage = self.advantage
        if "model" in type(advantage).model_fields:
            if advantage.model is None or "model" not in advantage.model_fields_set:
                advantage.model = self.model
                matched = True
            elif advantage.model == self.model:
                matched = True
        if getattr(advantage, "source_role", None) is not None:
            if "source" not in self.sampling.model_fields_set:
                self.sampling.source = self.model
                matched = True
            elif self.sampling.source == self.model:
                matched = True
        if not matched:
            raise ValueError(
                f"advantage '{self.advantage.type}': 'model' is set but no component reference accepts it — "
                "every reference is already explicitly set to a different value, or the algorithm "
                "references no model. Set advantage.model / sampling.source directly instead."
            )
        return self

    @model_validator(mode="after")
    def validate_component_compatibility(self):
        source_role = getattr(self.advantage, "source_role", None)
        if source_role is not None and self.sampling.source == "policy":
            raise ValueError(
                f"advantage '{self.advantage.type}' needs a {source_role} to sample rollouts from — "
                f"CE on the policy's own tokens is not a distillation target. Set '{source_role}' on "
                "the algorithm (an inline hosted model: name + base_url), or sampling.source explicitly."
            )
        if getattr(self.advantage, "model", "<absent>") is None:
            role = getattr(self.advantage, "model_role", "reference model")
            raise ValueError(
                f"advantage '{self.advantage.type}' needs a {role} — "
                f"set '{role}' on the algorithm (an inline hosted model: name + base_url), "
                "or advantage.model explicitly."
            )
        if isinstance(self.advantage, OPDAdvantageConfig) and self.advantage.model == "policy":
            raise ValueError(
                "advantage 'opd' with model='policy' is degenerate — the reference distribution "
                "equals the policy, so the KL signal is zero. Point at a frozen hosted model, or "
                "use 'opsd' for demo-conditioned self-teaching."
            )
        if self.advantage.action_loss_type in ("rl", "ref_kl") and self.sampling.source != "policy":
            raise ValueError(
                f"advantage '{self.advantage.type}' trains with the "
                f"{self.advantage.action_loss_type} loss type but sampling.source is a frozen model — "
                "the importance ratio and trust region need the live policy's own sampling logprobs. "
                "Use the 'sft' advantage to distill frozen-model tokens."
            )
        return self

    def warn_group_size(self, group_size: int, env_name: str) -> None:
        """Group-relative scoring with a single rollout per example collapses
        every advantage to zero. Warn loudly — this is the classic footgun."""
        if self.requires_group_advantage and group_size == 1:
            warnings.warn(
                f"Env '{env_name}' uses group-relative advantage ('{self.advantage.type}') with "
                "group_size=1 — every advantage is 0 and (with the default zero-advantage filter) "
                "no rollout will train. Set group_size >= 2 or a non-group-relative advantage "
                "(e.g. advantage.type='reward').",
                stacklevel=2,
            )
