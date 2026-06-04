"""Composable loss-term config DSL (the unified ``losses`` surface).

A training run defines a list of named loss *terms*; per-env ``enabled_losses``
selects which apply. Each term is a preset discriminated by ``type``:

- ``rl``     — DPPO+KL policy-gradient core (trains the sampled completion).
- ``echo``   — per-role CE overlay on *context* tokens (system/user/tool/assistant).
- ``custom`` — a user import path resolved to a loss core.

``rl`` carries trainer-side core params; ``echo`` carries orchestrator-side params
(per-role alphas + an optional filter). sft/opd are separate ``training_mode`` paths
with fixed cores — they are *not* loss-list terms (so a run uses at most one rl/custom
term + optional echo). The list is shared (defined once, propagated to both the
trainer and the orchestrator); each process reads the slots it executes.
"""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AfterValidator, Field, model_validator

from prime_rl.utils.config import BaseConfig


class SystemRoleEchoConfig(BaseConfig):
    """Echo supervision for system-message content tokens."""

    alpha: float = Field(1.0, allow_inf_nan=False)
    """Per-token echo weight."""


class UserRoleEchoConfig(BaseConfig):
    """Echo supervision for user-message content tokens."""

    alpha: float = Field(1.0, allow_inf_nan=False)
    """Per-token echo weight."""


class AssistantRoleEchoConfig(BaseConfig):
    """Echo supervision for assistant-message content and completion tokens."""

    alpha: float = Field(1.0, allow_inf_nan=False)
    """Per-token echo weight. ``alpha=0`` keeps the token supervised but gives it zero gradient."""


class ToolRoleEchoConfig(BaseConfig):
    """Echo supervision for tool-message content tokens."""

    alpha: float = Field(1.0, allow_inf_nan=False)
    """Per-token echo weight."""

    tool_names: set[str] | None = Field(None, min_length=1)
    """Restrict echo to these tool function names; None = all tools."""


class EchoFilterConfig(BaseConfig):
    """Optional callable that narrows role-selected echo tokens per rollout."""

    import_path: str
    """Dotted import path to the filter callable, e.g. ``"my_module.filter_warnings"``."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments forwarded to the filter as ``**kwargs``."""


class RLLossConfig(BaseConfig):
    """DPPO+KL policy-gradient term (trainer-side core; trains the completion)."""

    type: Literal["rl"] = "rl"
    name: str = "rl"

    dppo_mask_low: float = Field(0.2, ge=0)
    """Lower DPPO masking threshold."""

    dppo_mask_high: float = Field(0.2, ge=0)
    """Upper DPPO masking threshold."""

    adv_tau: float = Field(1.0, ge=0)
    """Temperature for the advantage term."""

    kl_tau: float = Field(1e-3, ge=0)
    """Temperature for the KL term."""


class EchoLossConfig(BaseConfig):
    """Per-role CE echo overlay (orchestrator-side weights; trainer-side CE core).

    Echo CE is computed on the rollout's temperature-scaled logprobs (the same ones RL
    uses), not true ``T=1`` NLL — scale ``alpha`` to compensate if needed. Negative
    ``alpha`` is allowed and *suppresses* the selected tokens (anti-echo)."""

    type: Literal["echo"] = "echo"
    name: str = "echo"

    system: SystemRoleEchoConfig | None = None
    """System-message echo (default: disabled)."""

    user: UserRoleEchoConfig | None = None
    """User-message echo (default: disabled)."""

    assistant: AssistantRoleEchoConfig | None = None
    """Assistant-message echo (default: disabled)."""

    tool: ToolRoleEchoConfig | None = None
    """Tool-message echo (default: disabled)."""

    filter: EchoFilterConfig | None = None
    """Optional per-token filter on top of the role baseline."""

    @model_validator(mode="after")
    def validate_roles(self) -> "EchoLossConfig":
        if self.system is self.user is self.assistant is self.tool is None:
            raise ValueError("EchoLossConfig requires at least one of system, user, assistant, or tool.")
        return self


class CustomLossTermConfig(BaseConfig):
    """A custom loss core resolved from a dotted import path."""

    type: Literal["custom"] = "custom"
    name: str
    """Unique term name (referenced by per-env ``enabled_losses``)."""

    import_path: str
    """Dotted import path to the loss core (``def core(inputs, **kwargs) -> LossOutputs``)."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments forwarded to the core as ``**kwargs``."""


LossTermConfig: TypeAlias = Annotated[
    RLLossConfig | EchoLossConfig | CustomLossTermConfig,
    Field(discriminator="type"),
]
"""A single loss term. The list of these is the unified ``losses`` surface."""


def validate_loss_list(losses: list[LossTermConfig]) -> list[LossTermConfig]:
    """Reject duplicate term names and more than one primary (rl/custom) term.

    The rl-path core comes from the single rl/custom term, so >1 is ambiguous.
    """
    names = [term.name for term in losses]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate loss term names: {duplicates}. Each term in `losses` needs a unique name.")
    primaries = [term.name for term in losses if term.type in ("rl", "custom")]
    if len(primaries) > 1:
        raise ValueError(f"At most one primary (rl/custom) loss term is allowed, got {primaries}.")
    return losses


LossList: TypeAlias = Annotated[list[LossTermConfig], AfterValidator(validate_loss_list)]
"""``list[LossTermConfig]`` with a unique-name check; the field type for ``losses``."""


def check_enabled_losses(loss_names: set[str], echo_names: set[str], enabled: list[str], where: str) -> None:
    """Raise if ``enabled`` references unknown terms or selects more than one echo term."""
    unknown = sorted(set(enabled) - loss_names)
    if unknown:
        raise ValueError(f"{where}: enabled_losses {unknown} not found in losses {sorted(loss_names)}.")
    enabled_echo = sorted(set(enabled) & echo_names)
    if len(enabled_echo) > 1:
        raise ValueError(f"{where}: at most one echo term may be enabled per env, got {enabled_echo}.")


def check_loss_overrides(loss_names: set[str], echo_names: set[str], overrides: dict[str, dict], where: str) -> None:
    """Raise if ``overrides`` references unknown terms or any non-echo term."""
    for name in overrides:
        if name not in loss_names:
            raise ValueError(f"{where}: loss_overrides key {name!r} not found in losses {sorted(loss_names)}.")
        if name not in echo_names:
            raise ValueError(f"{where}: loss_overrides currently applies only to echo terms, got {name!r}.")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins on leaves)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_echo_override(echo_term: EchoLossConfig, override: dict) -> EchoLossConfig:
    """Deep-merge a per-env ``override`` into an echo term and rebuild it.

    Constructing the ``EchoLossConfig`` here validates the merged payload (unknown
    fields, bad shapes, non-float alphas) — at config time when called from the
    validator, and at resolve time when called from the orchestrator.
    """
    return EchoLossConfig(**deep_merge(echo_term.model_dump(), override))


def default_losses() -> list[LossTermConfig]:
    """Default loss list: RL only (reproduces the pre-``losses`` default)."""
    return [RLLossConfig()]
