"""Composable loss-term config DSL (the unified ``losses`` surface).

A training run defines a list of named loss *terms*; per-env ``enabled_losses`` selects which apply.
Each term is a free composition of three independently-chosen axes, every one a discriminated-union
preset (``type`` + kwargs) with a ``custom`` import-path escape hatch:

- ``loss``    — the core (trainer-side): ``dppo_kl`` (RL) · ``ce`` (echo / NLL) · ``custom``.
- ``filters`` — token eligibility → mask (orchestrator-side): ``completion`` · ``role`` · ``custom``.
- ``weight``  — per-token weight (orchestrator-side): ``constant`` · ``advantage`` · ``custom``.

The terms are summed over one shared forward → one backward. Default ``losses`` reproduces today's
DPPO+KL training. sft/opd remain separate ``training_mode`` paths with fixed cores.

**Phase 1**: the 3-axis schema is the surface, but the execution engine is unchanged (one rl/custom
primary + one merged echo stream). The supported combos are mapped onto that engine via the resolved
internal types (``RLLossConfig`` / ``EchoLossConfig``) below; combos the engine can't run yet are
rejected at validation. Phase 2 generalizes the wire + trainer and removes the conversion.
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
    """Context tokens attributed to the given roles (needs renderer prompt_attribution for prompt
    roles; assistant/completion tokens are always available)."""

    type: Literal["role"] = "role"

    roles: list[Role] = Field(min_length=1)
    """Roles whose content tokens are eligible."""

    tool_names: set[str] | None = Field(None, min_length=1)
    """When ``"tool"`` is among the roles, restrict to these tool function names; None = all tools."""


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
    """A fixed per-token weight (echo's alpha). ``0`` = supervise with zero gradient; negative
    suppresses the tokens (anti-echo)."""

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
# The term: a free composition of the three axes.
# --------------------------------------------------------------------------------------------------


def _default_filters() -> list[TokenFilterConfig]:
    return [CompletionFilterConfig()]


class LossTerm(BaseConfig):
    """A single loss term — one core, one filter chain, one weight — summed into the total loss."""

    name: str
    """Unique term name (referenced by per-env ``enabled_losses`` / ``loss_overrides``)."""

    loss: LossCoreConfig
    """The core (required)."""

    filters: list[TokenFilterConfig] = Field(default_factory=_default_filters)
    """Token eligibility; the chain intersects (each filter narrows). Default: the completion."""

    weight: WeightConfig = Field(default_factory=ConstantWeightConfig)
    """Per-token weight. Default: constant ``1.0``."""

    @model_validator(mode="before")
    @classmethod
    def _expand_preset(cls, data: Any) -> Any:
        """Expand the one-line preset sugar — ``{type = "rl", ...}`` / ``{type = "echo", ...}`` — into
        the canonical core/filters/weight form. The full three-axis form (no top-level ``type``) and
        already-built terms pass through unchanged; ``custom`` lives only on the individual axes."""
        if not isinstance(data, dict) or "type" not in data:
            return data
        data = dict(data)
        preset = data.pop("type")
        if preset == "rl":
            core: dict[str, Any] = {"type": "dppo_kl"}
            for k in ("kl_tau", "dppo_mask_low", "dppo_mask_high"):
                if k in data:
                    core[k] = data.pop(k)
            weight: dict[str, Any] = {"type": "advantage"}
            if "tau" in data:
                weight["tau"] = data.pop("tau")
            expanded = {
                "name": data.pop("name", "rl"),
                "loss": core,
                "filters": [{"type": "completion"}],
                "weight": weight,
            }
        elif preset == "echo":
            role_filter: dict[str, Any] = {"type": "role", "roles": data.pop("roles", ["assistant"])}
            if "tool_names" in data:
                role_filter["tool_names"] = data.pop("tool_names")
            constant: dict[str, Any] = {"type": "constant"}
            if "alpha" in data:
                constant["alpha"] = data.pop("alpha")
            expanded = {
                "name": data.pop("name", "echo"),
                "loss": {"type": "ce"},
                "filters": [role_filter],
                "weight": constant,
            }
        else:
            raise ValueError(
                f"unknown loss preset type {preset!r}; use 'rl', 'echo', or the full core/filters/weight form."
            )
        if data:
            raise ValueError(f"unexpected kwargs for the {preset!r} loss preset: {sorted(data)}.")
        return expanded

    @model_validator(mode="after")
    def validate_supported(self) -> "LossTerm":
        # The primary (rl objective) is the completion-filtered dppo_kl/custom term weighted by the
        # advantage; everything else is an additive overlay over role/custom-filtered context tokens
        # (ce/custom core, constant- or advantage-weighted). Custom *weight* needs the weight resolver
        # (later phase) and is rejected here.
        core = self.loss.type
        weight = self.weight.type
        filter_types = [f.type for f in self.filters]
        if "completion" in filter_types:  # the rl objective (primary)
            if filter_types != ["completion"] or core not in ("dppo_kl", "custom") or weight != "advantage":
                raise ValueError(
                    f"loss term {self.name!r}: a completion-filtered (primary) term needs a dppo_kl/custom core, "
                    f"weight=advantage, and no other filters."
                )
        else:  # additive overlay over context tokens
            if core not in ("ce", "custom"):
                raise ValueError(f"loss term {self.name!r}: an overlay needs a ce or custom core.")
            if "role" not in filter_types or any(t not in ("role", "custom") for t in filter_types):
                raise ValueError(
                    f"loss term {self.name!r}: an overlay needs at least one role filter (plus optional custom filters)."
                )
            # Overlay weight may be constant, advantage, or a custom per-rollout resolver (all OK).
        return self


def is_primary(term: LossTerm) -> bool:
    """The rl objective trains the sampled completion (completion filter); overlays target context tokens."""
    return any(f.type == "completion" for f in term.filters)


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


def check_loss_overrides(loss_names: set[str], overlay_names: set[str], overrides: dict[str, dict], where: str) -> None:
    """Raise if ``overrides`` references unknown terms or any non-overlay (primary) term."""
    for name in overrides:
        if name not in loss_names:
            raise ValueError(f"{where}: loss_overrides key {name!r} not found in losses {sorted(loss_names)}.")
        if name not in overlay_names:
            raise ValueError(f"{where}: loss_overrides currently applies only to overlay terms, got {name!r}.")


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
    return [
        LossTerm(
            name="rl",
            loss=DPPOKLCoreConfig(),
            filters=[CompletionFilterConfig()],
            weight=AdvantageWeightConfig(),
        )
    ]


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
    filter: EchoFilterConfig | None = None

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
    roles: set[str] = set()
    tool_names: set[str] | None = None
    echo_filter: EchoFilterConfig | None = None
    for f in term.filters:
        if f.type == "role":
            roles.update(f.roles)
            if "tool" in f.roles:
                tool_names = f.tool_names
        elif f.type == "custom":
            echo_filter = EchoFilterConfig(import_path=f.import_path, kwargs=f.kwargs)
    return EchoLossConfig(
        name=term.name,
        system=SystemRoleEchoConfig(alpha=alpha) if "system" in roles else None,
        user=UserRoleEchoConfig(alpha=alpha) if "user" in roles else None,
        assistant=AssistantRoleEchoConfig(alpha=alpha) if "assistant" in roles else None,
        tool=ToolRoleEchoConfig(alpha=alpha, tool_names=tool_names) if "tool" in roles else None,
        filter=echo_filter,
    )
