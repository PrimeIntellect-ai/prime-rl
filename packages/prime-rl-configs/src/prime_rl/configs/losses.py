"""Composable loss-term config DSL (the unified ``losses`` surface).

A training run defines a list of named loss *terms*; per-env ``enabled_losses`` selects which apply.
Each term is a free composition of three independently-chosen axes, every one a discriminated-union
preset (``type`` + kwargs) with a ``custom`` import-path escape hatch:

- ``loss``    — the core (trainer-side): ``dppo_kl`` (RL) · ``ce`` (echo / NLL) · ``custom``.
- ``filters`` — token eligibility → mask (orchestrator-side): ``completion`` · ``role`` · ``custom``;
  the chain intersects (AND).
- ``weight``  — per-token weight (orchestrator-side): ``constant`` · ``advantage`` · ``custom``.

Common losses are one-line presets — ``{type = "rl", ...}`` / ``{type = "echo", ...}`` — that expand
into the canonical three-axis form (see ``LossTerm._expand_preset``). The **primary** (rl objective)
is the completion-filtered, advantage-weighted term, dispatched by ``training_mode``; every other term
is an additive **overlay** over context tokens. Terms are summed over one shared forward → one
backward. Default ``losses`` reproduces today's DPPO+KL. sft/opd remain separate ``training_mode``
paths with fixed cores. ``RLLossConfig`` / ``EchoLossConfig`` below are the resolved internal forms
the orchestrator/trainer consume.
"""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AfterValidator, Field, model_validator

from prime_rl.utils.config import BaseConfig

# --------------------------------------------------------------------------------------------------
# Axis 1 — cores (`loss`): trainer-side ``LossInputs -> LossOutputs``.
# --------------------------------------------------------------------------------------------------


class DPPOKLCoreConfig(BaseConfig):
    """DPPO+KL policy-gradient core (the RL objective). Trains its filtered tokens with the weight as
    the per-token advantage; masks trust-region violators and adds a squared-KL regularizer."""

    type: Literal["dppo_kl"] = "dppo_kl"

    dppo_mask_low: float = Field(0.2, ge=0)
    """Lower DPPO masking threshold."""

    dppo_mask_high: float = Field(0.2, ge=0)
    """Upper DPPO masking threshold."""

    kl_tau: float = Field(1e-3, ge=0)
    """Temperature for the KL term."""


class CECoreConfig(BaseConfig):
    """Cross-entropy core (echo / weighted NLL): ``-Σ weightₜ · logprobₜ`` over the filtered tokens."""

    type: Literal["ce"] = "ce"


class CustomCoreConfig(BaseConfig):
    """A user core resolved from a dotted import path (``def core(inputs, **kwargs) -> LossOutputs``)."""

    type: Literal["custom"] = "custom"

    import_path: str
    """Dotted import path to the loss core."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments forwarded to the core as ``**kwargs``."""


LossCoreConfig: TypeAlias = Annotated[
    DPPOKLCoreConfig | CECoreConfig | CustomCoreConfig,
    Field(discriminator="type"),
]

# --------------------------------------------------------------------------------------------------
# Axis 2 — filters: orchestrator-side token eligibility (a chain; filters intersect).
# --------------------------------------------------------------------------------------------------

Role: TypeAlias = Literal["system", "user", "assistant", "tool"]


class CompletionFilterConfig(BaseConfig):
    """The sampled completion tokens (the RL loss mask)."""

    type: Literal["completion"] = "completion"


class RoleFilterConfig(BaseConfig):
    """Context tokens attributed to the given roles (prompt-side roles need renderer
    ``prompt_attribution``; assistant completion tokens are always available)."""

    type: Literal["role"] = "role"

    roles: list[Role] = Field(min_length=1)
    """Roles whose content tokens are eligible."""

    tool_names: set[str] | None = Field(None, min_length=1)
    """When ``"tool"`` is among the roles, restrict to these tool function names; None = all tools."""

    @model_validator(mode="after")
    def validate_tool_names(self) -> "RoleFilterConfig":
        if self.tool_names is not None and "tool" not in self.roles:
            raise ValueError("role filter `tool_names` requires `roles` to include 'tool'.")
        return self


class CustomTokenFilterConfig(BaseConfig):
    """A user filter resolved from a dotted import path; narrows the tokens selected so far."""

    type: Literal["custom"] = "custom"

    import_path: str
    """Dotted import path to the filter callable."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments forwarded to the filter as ``**kwargs``."""


TokenFilterConfig: TypeAlias = Annotated[
    CompletionFilterConfig | RoleFilterConfig | CustomTokenFilterConfig,
    Field(discriminator="type"),
]

# --------------------------------------------------------------------------------------------------
# Axis 3 — weight: orchestrator-side per-token weight (resolved per-group, after advantages).
# --------------------------------------------------------------------------------------------------


class ConstantWeightConfig(BaseConfig):
    """A fixed per-token weight (echo's alpha). ``0`` disables the token; negative suppresses it
    (anti-echo)."""

    type: Literal["constant"] = "constant"

    alpha: float = Field(1.0, allow_inf_nan=False)
    """Per-token weight."""


class AdvantageWeightConfig(BaseConfig):
    """Use the GRPO advantage (× ``tau``) as the per-token weight (the RL signal)."""

    type: Literal["advantage"] = "advantage"

    tau: float = Field(1.0, ge=0)
    """Temperature on the advantage."""


class CustomWeightConfig(BaseConfig):
    """A user weight resolved per-rollout from a dotted import path. Signature:
    ``fn(inputs: WeightInputs, **kwargs) -> list[float]`` (length = prompt + completion of the
    sample). ``WeightInputs`` carries the sample and the full GRPO group's rollouts (each with its
    advantage/reward/raw trajectory), so the resolver can compute group-relative weights."""

    type: Literal["custom"] = "custom"

    import_path: str
    """Dotted import path to the weight callable."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments forwarded to the weight fn as ``**kwargs``."""


WeightConfig: TypeAlias = Annotated[
    ConstantWeightConfig | AdvantageWeightConfig | CustomWeightConfig,
    Field(discriminator="type"),
]

# --------------------------------------------------------------------------------------------------
# The advantage axis — the per-token advantage_fn (orchestrator-side). Produces one float per token
# (0 = masked); supersedes the filters + weight axes above. Resolved to a callable over a group of
# ``RenderHints`` by ``orchestrator.advantage.resolve_advantage_fn``.
# --------------------------------------------------------------------------------------------------


class GRPOAdvantageConfig(BaseConfig):
    """GRPO advantage (the RL objective / primary): the per-rollout reward-baseline advantage (× tau)
    broadcast over the sampled tokens, 0 elsewhere."""

    type: Literal["grpo"] = "grpo"

    tau: float = Field(1.0, ge=0)
    """Temperature on the advantage."""


class EchoAdvantageConfig(BaseConfig):
    """Echo advantage (overlay): ``alpha`` on role-matched context tokens, 0 elsewhere. ``by_advantage``
    multiplies it by the rollout's advantage (× ``tau``) for advantage-weighted echo."""

    type: Literal["echo"] = "echo"

    roles: list[Role] = Field(min_length=1)
    """Roles whose content tokens are echoed."""

    tool_names: set[str] | None = Field(None, min_length=1)
    """When ``"tool"`` is among the roles, restrict to these tool function names; None = all tools."""

    alpha: float = Field(1.0, allow_inf_nan=False)
    """Per-token weight (0 disables; negative is anti-echo)."""

    by_advantage: bool = False
    """Multiply ``alpha`` by the rollout's advantage × ``tau`` (advantage-weighted echo)."""

    tau: float = Field(1.0, ge=0)
    """Temperature on the advantage when ``by_advantage``."""


class SFTAdvantageConfig(BaseConfig):
    """SFT advantage: a constant ``alpha`` on the sampled tokens, 0 elsewhere (masked NLL)."""

    type: Literal["sft"] = "sft"

    alpha: float = Field(1.0, allow_inf_nan=False)
    """Per-token weight."""


class CustomAdvantageFnConfig(BaseConfig):
    """A user advantage_fn resolved from a dotted import path. Signature:
    ``fn(group: list[RenderHints], **kwargs) -> list[list[float]]`` (one list/unit, one float/token;
    ``0`` masks). The group's ``RenderHints`` carry each unit's reward/advantage/rollout + per-token
    attribution, so the fn can compute group-relative, advantage-weighted, or attribution-based signals."""

    type: Literal["custom"] = "custom"

    import_path: str
    """Dotted import path to the advantage_fn."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments forwarded to the fn as ``**kwargs``."""


AdvantageFnConfig: TypeAlias = Annotated[
    GRPOAdvantageConfig | EchoAdvantageConfig | SFTAdvantageConfig | CustomAdvantageFnConfig,
    Field(discriminator="type"),
]

# --------------------------------------------------------------------------------------------------
# The term: a free composition of the three axes.
# --------------------------------------------------------------------------------------------------


def _default_filters() -> list[TokenFilterConfig]:
    return [CompletionFilterConfig()]


class LossTerm(BaseConfig):
    """A single loss term — a core and an advantage_fn — summed into the total loss."""

    name: str
    """Unique term name (referenced by per-env ``enabled_losses`` / ``loss_overrides``)."""

    loss: LossCoreConfig
    """The core (trainer-side, required)."""

    advantage: AdvantageFnConfig
    """The advantage_fn (orchestrator-side, required): the per-token signal, ``0`` = masked."""

    @model_validator(mode="before")
    @classmethod
    def _expand_preset(cls, data: Any) -> Any:
        """Expand a one-line preset (``{type = "rl"|"echo", ...}``) into the full loss/advantage form.
        The preset seeds default axes; any explicit ``loss``/``advantage``/``name`` the user sets is
        deep-merged over them with the *same* semantics as a full term (so ``advantage = {alpha = 0.5}``
        keeps ``type = "echo"``). The full form (no top-level ``type``) and built terms pass through."""
        if not isinstance(data, dict) or "type" not in data:
            return data
        data = dict(data)
        preset = data.pop("type")
        if preset == "rl":
            defaults: dict[str, Any] = {"name": "rl", "loss": {"type": "dppo_kl"}, "advantage": {"type": "grpo"}}
        elif preset == "echo":
            defaults = {"name": "echo", "loss": {"type": "ce"}, "advantage": {"type": "echo", "roles": ["assistant"]}}
        else:
            raise ValueError(f"unknown loss preset type {preset!r}; use 'rl', 'echo', or the full loss/advantage form.")
        unknown = set(data) - {"name", "loss", "advantage"}
        if unknown:
            raise ValueError(
                f"loss preset {preset!r}: override the axes (loss/advantage) or name, not {sorted(unknown)}."
            )
        return deep_merge(defaults, data)

    @model_validator(mode="after")
    def validate_supported(self) -> "LossTerm":
        # The primary (rl objective) is the grpo-advantage term with a dppo_kl/custom core; every other
        # advantage (echo / sft / custom) is an additive overlay with a ce/custom core.
        core = self.loss.type
        if self.advantage.type == "grpo":
            if core not in ("dppo_kl", "custom"):
                raise ValueError(f"loss term {self.name!r}: a grpo (primary) advantage needs a dppo_kl or custom core.")
        elif core not in ("ce", "custom"):
            raise ValueError(
                f"loss term {self.name!r}: a {self.advantage.type} (overlay) advantage needs a ce or custom core."
            )
        return self


def is_primary(term: LossTerm) -> bool:
    """The rl objective is the grpo-advantage term; every other advantage is an additive overlay."""
    return term.advantage.type == "grpo"


def validate_loss_list(losses: list[LossTerm]) -> list[LossTerm]:
    """Reject duplicate term names and more than one primary (rl objective) term."""
    names = [term.name for term in losses]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate loss term names: {duplicates}. Each term in `losses` needs a unique name.")
    primaries = [term.name for term in losses if is_primary(term)]
    if len(primaries) > 1:
        raise ValueError(f"At most one primary (dppo_kl/custom core) loss term is allowed, got {primaries}.")
    # The trainer dispatches the per-run cores by these keys; an overlay sharing one would silently
    # overwrite the dispatch core (sft/opd) or the rl primary, so reserve them.
    for term in losses:
        if term.name in ("sft", "opd"):
            raise ValueError(f"loss term name {term.name!r} is reserved for the training_mode core; rename it.")
        if term.name == "rl" and not is_primary(term):
            raise ValueError("loss term name 'rl' is reserved for the rl primary; rename the overlay.")
    return losses


LossList: TypeAlias = Annotated[list[LossTerm], AfterValidator(validate_loss_list)]
"""``list[LossTerm]`` with the unique-name + single-primary check; the field type for ``losses``."""


def check_enabled_losses(loss_names: set[str], enabled: list[str], where: str) -> None:
    """Raise if ``enabled`` references unknown terms."""
    unknown = sorted(set(enabled) - loss_names)
    if unknown:
        raise ValueError(f"{where}: enabled_losses {unknown} not found in losses {sorted(loss_names)}.")


def check_loss_overrides(
    loss_names: set[str],
    overlay_names: set[str],
    enabled: list[str],
    overrides: dict[str, dict],
    where: str,
) -> None:
    """Raise if ``overrides`` references unknown/non-overlay/disabled terms, or overrides anything but
    the ``advantage`` axis. The advantage_fn is resolved per env (orchestrator-side); ``loss`` (the
    core) and ``name`` are global, so overriding them would validate but be silently ignored. An
    override on a term the env doesn't enable is a silent no-op too (``_resolve_overlays`` skips
    disabled terms before applying overrides), so reject that as well."""
    for name, override in overrides.items():
        if name not in loss_names:
            raise ValueError(f"{where}: loss_overrides key {name!r} not found in losses {sorted(loss_names)}.")
        if name not in overlay_names:
            raise ValueError(f"{where}: loss_overrides currently applies only to overlay terms, got {name!r}.")
        if name not in enabled:
            raise ValueError(
                f"{where}: loss_overrides[{name!r}] targets a term not in enabled_losses {sorted(enabled)}; "
                f"the override would be silently ignored. Add it to enabled_losses or drop the override."
            )
        unsupported = set(override) - {"advantage"}
        if unsupported:
            raise ValueError(
                f"{where}: loss_overrides[{name!r}] may only override the 'advantage' axis per env, "
                f"not {sorted(unsupported)}."
            )


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins on leaves)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_term_override(term: LossTerm, override: dict) -> LossTerm:
    """Deep-merge a per-env ``override`` into a term and rebuild it.

    Reconstructing the ``LossTerm`` validates the merged payload (unknown fields, bad shapes,
    unsupported combos) — at config time from the validator, and at resolve time from the orchestrator.
    """
    return LossTerm(**deep_merge(term.model_dump(), override))


def default_losses() -> list[LossTerm]:
    """Default loss list: RL only (reproduces the pre-``losses`` default)."""
    return [LossTerm(name="rl", loss=DPPOKLCoreConfig(), advantage=GRPOAdvantageConfig())]


# --------------------------------------------------------------------------------------------------
# Resolved internal types + conversions (Phase 1: map the schema onto today's execution engine).
# These are NOT part of the ``losses`` surface; they are produced from ``LossTerm``s at consume time.
# --------------------------------------------------------------------------------------------------


class RLLossConfig(BaseConfig):
    """Resolved DPPO+KL params consumed by ``default_loss_fn`` (built from a dppo_kl primary term)."""

    type: Literal["rl"] = "rl"
    name: str = "rl"

    dppo_mask_low: float = Field(0.2, ge=0)
    dppo_mask_high: float = Field(0.2, ge=0)
    kl_tau: float = Field(1e-3, ge=0)
    # No adv_tau: the advantage weight (× tau) is resolved orchestrator-side, so the core just
    # consumes the already-scaled advantage in LossInputs.advantages.


def to_rl_loss_config(term: LossTerm) -> RLLossConfig:
    """Resolve a ``dppo_kl`` primary term into ``RLLossConfig`` (core knobs only; the advantage
    weight's ``tau`` is applied orchestrator-side, not here)."""
    assert isinstance(term.loss, DPPOKLCoreConfig)
    return RLLossConfig(
        name=term.name,
        dppo_mask_low=term.loss.dppo_mask_low,
        dppo_mask_high=term.loss.dppo_mask_high,
        kl_tau=term.loss.kl_tau,
    )


class SystemRoleEchoConfig(BaseConfig):
    alpha: float = Field(1.0, allow_inf_nan=False)


class UserRoleEchoConfig(BaseConfig):
    alpha: float = Field(1.0, allow_inf_nan=False)


class AssistantRoleEchoConfig(BaseConfig):
    alpha: float = Field(1.0, allow_inf_nan=False)


class ToolRoleEchoConfig(BaseConfig):
    alpha: float = Field(1.0, allow_inf_nan=False)
    tool_names: set[str] | None = Field(None, min_length=1)


class EchoFilterConfig(BaseConfig):
    """A resolved echo token filter (from a ``custom`` filter on a ce term)."""

    import_path: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class EchoLossConfig(BaseConfig):
    """Resolved per-role echo overlay consumed by the orchestrator's ``build_echo_annotations``.

    Echo CE runs on the rollout's temperature-scaled logprobs (the same ones RL uses), not true
    ``T=1`` NLL — scale ``alpha`` to compensate. Negative ``alpha`` suppresses tokens (anti-echo)."""

    name: str = "echo"

    system: SystemRoleEchoConfig | None = None
    user: UserRoleEchoConfig | None = None
    assistant: AssistantRoleEchoConfig | None = None
    tool: ToolRoleEchoConfig | None = None
    filters: list[EchoFilterConfig] = Field(default_factory=list)
    """Custom token filters, intersected (AND) on top of the role baseline."""

    @model_validator(mode="after")
    def validate_roles(self) -> "EchoLossConfig":
        if self.system is self.user is self.assistant is self.tool is None:
            raise ValueError("EchoLossConfig requires at least one of system, user, assistant, or tool.")
        return self


def overlay_terms(losses: list[LossTerm]) -> list[LossTerm]:
    """The additive (non-primary, constant-weighted) overlay terms, in list order."""
    return [term for term in losses if not is_primary(term)]


def to_echo_config(term: LossTerm) -> EchoLossConfig:
    """Resolve one overlay term (role + optional custom filters) into an ``EchoLossConfig`` for the
    orchestrator's per-token alpha builder. The term's core (ce/custom) is applied trainer-side, not
    here. Constant weight bakes its alpha in directly; advantage weight uses 1.0 as an eligibility
    marker that the orchestrator scales by the rollout's advantage (x tau) once advantages exist."""
    alpha = term.weight.alpha if term.weight.type == "constant" else 1.0
    # Filters chain by AND: role filters intersect their role sets (and tool_names), and every custom
    # filter is kept (the orchestrator intersects their masks on top of the role baseline).
    roles: set[str] | None = None
    tool_names: set[str] | None = None
    custom_filters: list[EchoFilterConfig] = []
    for f in term.filters:
        if f.type == "role":
            roles = set(f.roles) if roles is None else (roles & set(f.roles))
            if "tool" in f.roles and f.tool_names is not None:
                tool_names = set(f.tool_names) if tool_names is None else (tool_names & f.tool_names)
        elif f.type == "custom":
            custom_filters.append(EchoFilterConfig(import_path=f.import_path, kwargs=f.kwargs))
    roles = roles or set()
    return EchoLossConfig(
        name=term.name,
        system=SystemRoleEchoConfig(alpha=alpha) if "system" in roles else None,
        user=UserRoleEchoConfig(alpha=alpha) if "user" in roles else None,
        assistant=AssistantRoleEchoConfig(alpha=alpha) if "assistant" in roles else None,
        tool=ToolRoleEchoConfig(alpha=alpha, tool_names=tool_names) if "tool" in roles else None,
        filters=custom_filters,
    )
