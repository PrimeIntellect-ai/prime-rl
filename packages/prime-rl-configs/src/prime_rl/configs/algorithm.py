"""Algorithm abstraction: sampling and the per-token training signal.

An algorithm is a named, self-contained config — a discriminated union keyed
on ``type`` (``grpo``, ``max_rl``, ``opd``, ``opsd``, ``sft``, ``echo``).
The bundle *is* the algorithm: each variant carries
its sampling component and its credit-assignment / loss-routing parameters,
and its class defaults are the vetted setting — ``type = "opd"`` with a
teacher IS on-policy distillation; any key you set is visibly your own
assembly. There is no separate ``advantage`` sub-component and no preset layer.

Each algorithm fixes two things:

1. **Sampling** — which model generates train rollouts. ``sampling.source`` is
   a model reference: ``"policy"`` (the live policy) or an inline frozen hosted
   model.
2. **The per-token training signal** — credit assignment and loss routing,
   fused: one mapping from a finalized rollout to per-token ``(loss component,
   weight)``. Group-relative algorithms compute scalars on the orchestrator and
   ship numbers; reference-KL algorithms ship reference prefill logprobs and the
   trainer evaluates the per-token signal against the live policy. The algorithm
   determines which loss component consumes the action tokens (``rl`` / ``ce`` /
   ``ref_kl``, via the ``action_loss_type`` class declaration) and what happens
   to env-provided observation tokens (masked out by default; ``echo`` trains on
   them with weighted CE).

prime-rl only ever hosts the trainable policy. Every other model an algorithm
uses is an external OpenAI-compatible endpoint, declared inline on the
algorithm that uses it (a :class:`FrozenModelConfig`). Model roles like
"teacher" are algorithm-local vocabulary over these references; the pipeline
branches on liveness alone. The trainer is algorithm-blind: the loss is a sum
of three components (rl, ce, ref_kl), each normalized by its own global token
count; per-token component weights ship on the wire and the trainer just
executes them.
"""

from typing import Annotated, Any, ClassVar, Literal, TypeAlias

from pydantic import Field, model_validator
from renderers import AutoRendererConfig, RendererConfig

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
# Sampling
# ---------------------------------------------------------------------------


class SamplingConfig(BaseConfig):
    source: ModelReference = "policy"
    """Model reference for train rollout generation: ``"policy"`` (the live
    policy — prefix caches salted per version, sampling logprobs requested,
    rollouts age off-policy) or an inline frozen hosted model (stable prefix
    cache, no sampling logprobs, rollouts never go stale)."""


# ---------------------------------------------------------------------------
# Shared sub-configs (length penalty, echo roles)
# ---------------------------------------------------------------------------


class LinearLengthPenaltyConfig(BaseConfig):
    """Linear ``pass_rate``-scaled penalty subtracted from each reward before the GRPO baseline — the sum of three terms (completion tokens, input tokens, turns), each normalized by the group's own max for that quantity and disabled by setting its coefficient to 0."""

    type: Literal["linear"] = "linear"

    num_output_tokens_weight: float = Field(0.25, ge=0, allow_inf_nan=False)
    """Scale on the output-token term. Each reward is reduced by ``num_output_tokens_weight * pass_rate * (rollout num_output_tokens / group's max num_output_tokens)`` — where ``pass_rate`` is the group's mean reward — before the GRPO baseline subtraction. Finite and non-negative; 0 disables the term."""

    num_input_tokens_weight: float = Field(0.1, ge=0, allow_inf_nan=False)
    """Scale on the input-token term — tokens the model conditioned on but did not generate (``num_total_tokens - num_output_tokens``: prompts, tool responses), as a fraction of the group's max input tokens. 0 disables the term."""

    num_turns_weight: float = Field(0.1, ge=0, allow_inf_nan=False)
    """Scale on the turns term (``pass_rate * (rollout num_turns / group's max num_turns)``). 0 disables the term."""


LengthPenaltyConfig: TypeAlias = LinearLengthPenaltyConfig


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
    trainable branch, each spanning that branch's ``token_ids``. Tokens with
    ``False`` never receive echo weight; the
    filter can only narrow the role selection, not widen it. The raw rollout
    exposes message text and sampling logprobs, so content filters (e.g.
    dropping tool-output warnings) and sampling-probability filters need no
    extra framework surface."""

    import_path: str
    """Import path to the filter callable (e.g. ``my_module.drop_warnings``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Kwargs forwarded to the filter."""


# ---------------------------------------------------------------------------
# The algorithms (a discriminated union keyed on ``type``)
# ---------------------------------------------------------------------------


class BaseAlgoConfig(BaseConfig):
    """Base for every algorithm: the shared sampling component and the
    cross-cutting source/loss compatibility check. Each subclass sets ``type``
    (the discriminator), declares its loss routing (``action_loss_type``), and
    adds its own parameters — including any reference model it needs, named
    where that model is actually used (opd scores against a frozen ``teacher``;
    sft samples from its ``sampling.source``; opsd self-distills against the
    live policy and names no model).

    The bundle IS the algorithm — there is no separate ``advantage``
    sub-component. ``algo.type`` names it, and the class defaults are the
    vetted setting."""

    action_loss_type: ClassVar[ActionLossType] = "rl"

    sampling: SamplingConfig = SamplingConfig()
    """Sampling component: which model generates train rollouts."""

    @model_validator(mode="after")
    def validate_sampling_source(self):
        """The on-policy loss types (rl, ref_kl) need the live policy's own
        sampling logprobs, so they must sample from ``"policy"``; only sft (ce)
        may sample from a frozen model."""
        if self.action_loss_type in ("rl", "ref_kl") and self.sampling.source != "policy":
            raise ValueError(
                f"algorithm '{self.type}' trains with the "
                f"{self.action_loss_type} loss type but sampling.source is a frozen model — "
                "the importance ratio and trust region need the live policy's own sampling logprobs. "
                "Use the 'sft' algorithm to distill frozen-model tokens."
            )
        return self


class GRPOAlgoConfig(BaseAlgoConfig):
    type: Literal["grpo"] = "grpo"
    """GRPO: scalar advantage = reward minus the per-group mean baseline,
    consumed by the ``rl`` loss component on the rollout's action tokens."""

    action_loss_type: ClassVar[ActionLossType] = "rl"

    length_penalty: LengthPenaltyConfig | None = None
    """Linear length penalty subtracted from each reward before the GRPO baseline (see ``LinearLengthPenaltyConfig``): a ``pass_rate``-scaled sum of output-token, input-token, and turns terms, each normalized by the group's own max for that quantity. None disables it."""


class SelfDistilledRLConfig(GRPOAlgoConfig):
    """Shared knobs for reward-grounded self-distillation variants.

    These algorithms keep the GRPO/RL loss and use self-teacher scores only to
    redistribute the scalar, verifier-derived advantage across sampled tokens.
    The sign of the update remains reward-grounded.
    """

    demo_key: str = "demonstration"
    """Key holding the privileged hint text for dataset-hint variants, looked
    up in rollout ``info`` first and then as a top-level task field."""

    template: str = "Here is a reference solution attempt:\n<demonstration>\n{demonstration}\n</demonstration>"
    """Content of the leading system message carrying a hint. Receives
    ``{demonstration}``; the sampled trajectory is scored verbatim after it."""

    max_score_tokens: int | None = None
    """Scoring context window. When set, over-budget samples are scored on the
    head and the unscored tail is masked out of the RL loss."""

    diag_top_k: int | None = 64
    """Collect the self-teacher's top-k token ids and logprobs per sampled
    position for measurement-only trainer exports; ``None`` disables it. The
    sampled-token teacher logprobs used by the algorithm remain scalar and the
    top-k distribution never participates in advantages, weights, or loss.
    Requires the inference server's ``max_logprobs`` to be at least this k."""

    renderer: RendererConfig = AutoRendererConfig()
    """Renderer family for the hint block. The tokenizer is the live policy's
    tokenizer, matching OPSD's self-teacher setup."""

    token_weight_clip_min: float = Field(0.25, gt=0, allow_inf_nan=False)
    """Lower bound for the per-token teacher/student reweighting factor."""

    token_weight_clip_max: float = Field(4.0, gt=0, allow_inf_nan=False)
    """Upper bound for the per-token teacher/student reweighting factor."""

    token_weight_temperature: float = Field(1.0, gt=0, allow_inf_nan=False)
    """Temperature applied to logprob gaps before exponentiating into token
    weights. Higher values make the modulation weaker."""

    normalize_token_weights: bool = True
    """Normalize trainable-token weights to mean 1 per rollout, preserving the
    scale of the original GRPO advantage while changing token allocation."""

    @model_validator(mode="after")
    def validate_weight_clip(self):
        if self.token_weight_clip_min > self.token_weight_clip_max:
            raise ValueError("token_weight_clip_min must be <= token_weight_clip_max")
        return self


class RLSDAlgoConfig(SelfDistilledRLConfig):
    type: Literal["rlsd"] = "rlsd"  # type: ignore[assignment]
    """RLSD-style reward-grounded self-distillation: compute a GRPO advantage,
    then modulate its per-token magnitude by the demo-conditioned self-teacher's
    sampled-token probability ratio. The verifier-derived advantage keeps the
    update direction."""


class RLRTAlgoConfig(SelfDistilledRLConfig):
    type: Literal["rlrt"] = "rlrt"  # type: ignore[assignment]
    """RLRT-style reversed-teacher exploration: on positive-advantage rollouts,
    upweight sampled tokens the student chose despite the demo-conditioned
    teacher assigning lower probability. Non-positive rollouts stay GRPO."""


class RLCSDAlgoConfig(SelfDistilledRLConfig):
    type: Literal["rlcsd"] = "rlcsd"  # type: ignore[assignment]
    """RLCSD-style contrastive self-distillation: build correct/incorrect
    hints from sibling rollouts in the same group, subtract the incorrect-hint
    score from the correct-hint score, and use the contrast to modulate the
    GRPO advantage. If a group lacks a usable contrast, it remains plain GRPO."""

    demo_key: str = "demonstration"
    """Unused by the default sibling-hint path; kept for schema compatibility
    with the shared self-distillation config."""

    max_hint_chars: int | None = Field(12000, gt=0)
    """Maximum characters from a sibling rollout to use as a hint. ``None``
    keeps the full assistant text."""

    num_negative_hints: int = Field(1, ge=1)
    """Number of incorrect sibling hints to marginalize over. Multiple
    negatives are averaged in probability space for the sampled token."""


class EchoAlgoConfig(GRPOAlgoConfig):
    type: Literal["echo"] = "echo"  # type: ignore[assignment]
    """ECHO: group-relative advantage on action tokens (GRPO), plus weighted
    CE on env-provided tokens of later turns (tool output, user feedback),
    selected by message role via the renderer's per-token ``is_content``
    attribution (renderers that don't attribute content fall back to weighting
    the whole non-sampled span). Selected tokens feed the ``ce`` loss component
    at their role's ``alpha`` and stay outside the rl mask and its denominator."""

    roles: EchoRolesConfig = EchoRolesConfig(tool=EchoRoleConfig())
    """The role table. The default — tool-response bodies at ``alpha = 0.1``
    — is the vetted ECHO setting."""

    filter: EchoFilterConfig | None = None
    """Optional user-supplied filter narrowing the role-selected tokens."""


class MaxRLAlgoConfig(BaseAlgoConfig):
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


class DistillationAlgoConfig(BaseAlgoConfig):
    """Shared base for the teacher-scored distillation algorithms (opd/opsd):
    both prefill-score each sample under a teacher and train with the
    ``ref_kl`` loss component; the scoring granularity is common vocabulary."""

    action_loss_type: ClassVar[ActionLossType] = "ref_kl"

    ref_logprob_granularity: Literal["single_token", "top_k"] = "single_token"
    """Granularity of teacher scoring: ``"single_token"`` ships only the
    teacher logprob of each sampled token; ``"top_k"`` additionally ships the
    teacher's top-k (token id, logprob) pairs per position, and the trainer
    minimizes a dense cross-entropy to the teacher over those k tokens."""

    ref_top_k: int = 64
    """k for the ``"top_k"`` granularity. Values above the inference server's
    ``max_logprobs`` cap (vLLM default: 20) are rejected at scoring time."""

    diag_top_k: int | None = 64
    """Collect the hinted-teacher's top-k (token id, logprob) pairs per position
    for diagnostics regardless of the objective; ``None`` disables. Only active
    with the ``"single_token"`` granularity (``"top_k"`` already ships the
    distribution as ``ref_topk_*``): samples additionally carry ``diag_topk_*``
    tensors, consumed only by token export — the training loss is unchanged. The
    trainer also computes, per position, the student's OWN top-k (ids+logprobs)
    and the teacher's truncated entropy for export (measurements 4 and 6). Lives
    on the shared distillation base so both opd and opsd inherit it. Requires the
    inference server's ``max_logprobs`` >= this k."""


class OPDAlgoConfig(DistillationAlgoConfig):
    type: Literal["opd"] = "opd"
    """On-policy distillation: the per-token signal is the reverse KL to
    a reference model, evaluated in the trainer from reference prefill
    logprobs scored over each sample's own context (``ref_logprobs`` on the
    wire, ``ref_kl`` loss component). No scalar advantage is assigned —
    rollouts keep ``advantages=None`` (advantage-based filters never fire) and
    samples ship no advantage stream. ``group_size`` only fans out sampling."""

    teacher: FrozenModelConfig
    """The teacher — an inline frozen hosted model (``name`` + ``base_url``)
    whose reverse KL the policy distills toward. Required, and necessarily a
    frozen endpoint: scoring the policy under itself yields zero KL signal, so
    ``"policy"`` is not even representable here (use ``opsd`` for
    demo-conditioned self-teaching)."""


class OPSDAlgoConfig(DistillationAlgoConfig):
    type: Literal["opsd"] = "opsd"
    """On-policy self-distillation (SDFT, https://arxiv.org/abs/2601.19897):
    the per-token signal is the reverse KL against the live policy conditioned
    on an expert demonstration. The teacher *is* the policy — self-distillation
    names no separate model — scoring each sample with the demonstration
    prepended as a leading system message. The sample is scored verbatim (no
    re-rendering), so it's robust to tool/multimodal prompts and works for any
    number of turns. No scalar advantage is assigned — rollouts keep
    ``advantages=None`` (advantage-based filters never fire) and samples ship no
    advantage stream."""

    action_loss_type: ClassVar[ActionLossType] = "ref_kl"

    demo_key: str = "demonstration"
    """Key holding the expert demonstration text — looked up in the example's
    ``info`` dict first, then as a top-level rollout field (e.g. ``answer``)."""

    demo_transform: Literal["identity", "tool_sequence_plan"] = "identity"
    """Optional deterministic transformation of the demonstration before it is
    inserted into ``template``. ``"identity"`` preserves the exact reference;
    ``"tool_sequence_plan"`` parses a General Agent ``gold.json`` chain and
    retains only the ordered tool names and argument names, deliberately
    withholding concrete argument values from the self-teacher."""

    template: str = "Here is an example of an expert response:\n<demonstration>\n{demonstration}\n</demonstration>"
    """Content of the leading system message carrying the demonstration.
    Receives ``{demonstration}``; the original question stays in the (verbatim)
    user turn, so it isn't templated here."""

    template_path: str | None = None
    """Optional UTF-8 file containing ``template``. This keeps long optimized
    prompts in one versioned artifact while flattened configs reference it."""

    max_score_tokens: int | None = None
    """Scoring context window — set to the inference server's ``max_model_len``.
    Generation reserves no headroom for the hint block, so a near-max-context
    trajectory + hint can exceed the server window. When set, over-budget
    samples are scored on their first ``max_score_tokens − len(hint)`` tokens
    and the unscored tail is masked out of the loss, keeping the (expensive)
    rollout instead of dropping it and paying a serial backfill. When unset,
    over-budget rollouts are dropped via the pre-filter path."""

    renderer: RendererConfig = AutoRendererConfig()
    """Renderer family for the hint block. The tokenizer is always the live
    policy's (self-distillation has no separate model — not configurable).
    Defaults to ``"auto"`` (resolved from the policy tokenizer); set explicitly
    to match a non-auto policy renderer."""


class SFTAlgoConfig(BaseAlgoConfig):
    type: Literal["sft"] = "sft"
    """SFT distillation: cross-entropy on the sampled tokens. The ``ce`` loss
    ignores advantages and SFT assigns none — it trains on every sampled token.
    Reward-based filtering, if wanted, is an explicit filter, not smuggled
    through an unused advantage stream."""

    action_loss_type: ClassVar[ActionLossType] = "ce"

    @model_validator(mode="after")
    def require_frozen_source(self):
        """sft's teacher is the model it samples from — ``sampling.source`` must
        be a frozen hosted model, not the policy (CE on the policy's own tokens
        is not a distillation target)."""
        if self.sampling.source == "policy":
            raise ValueError(
                f"algorithm '{self.type}' needs a teacher to sample rollouts from — "
                "CE on the policy's own tokens is not a distillation target. Set "
                "sampling.source to an inline hosted model (name + base_url)."
            )
        return self


AlgoConfig: TypeAlias = Annotated[
    GRPOAlgoConfig
    | EchoAlgoConfig
    | MaxRLAlgoConfig
    | OPDAlgoConfig
    | OPSDAlgoConfig
    | RLSDAlgoConfig
    | RLRTAlgoConfig
    | RLCSDAlgoConfig
    | SFTAlgoConfig,
    Field(discriminator="type"),
]
"""The training algorithm: sampling plus the per-token training signal (credit
assignment and loss routing, fused). The ``type`` selects the algorithm, and
its class defaults are the vetted setting.

- ``grpo`` — policy group sampling, group-relative advantage, RL loss (the default).
- ``max_rl`` — GRPO with mean-normalized advantages (maximum-likelihood RL).
- ``opd`` — on-policy distillation: policy samples, per-token reverse KL against a reference model. Needs ``teacher``.
- ``opsd`` — SDFT: policy samples, demo-conditioned reverse KL against the live policy (the teacher is the policy itself).
- ``rlsd`` — GRPO with demo-conditioned self-teacher token-magnitude reweighting.
- ``rlrt`` — GRPO with reversed-teacher reweighting on positive-advantage rollouts.
- ``rlcsd`` — GRPO with correct-vs-incorrect sibling-hint contrastive token reweighting.
- ``sft`` — a frozen model samples, the policy trains with CE on its tokens. Needs a frozen ``sampling.source``.
- ``echo`` — GRPO on action tokens + weighted CE on tool-response observation tokens.

A new credit-assignment scheme is a new named algorithm in code (subclass
``Algorithm``, register it), not a config that points at an import path.
"""
