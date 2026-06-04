"""Composable loss-term config DSL (the unified ``losses`` surface).

A training run defines a list of named loss *terms*; per-env ``enabled_losses``
selects which apply. Each term is a preset discriminated by ``type``:

- ``rl``     — DPPO+KL policy-gradient core (trains the sampled completion).
- ``sft``    — masked NLL core (trains the completion as supervised tokens).
- ``opd``    — on-policy-distillation core (teacher-KL signal; needs a teacher).
- ``echo``   — per-role CE overlay on *context* tokens (system/user/tool/assistant).
- ``custom`` — a user import path resolved to a loss core.

``rl``/``sft``/``opd`` carry trainer-side core params; ``echo`` carries
orchestrator-side params (per-role alphas + an optional filter). The list is
shared (defined once, propagated to both the trainer and the orchestrator); each
process reads the slots it executes.
"""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

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


class SFTLossConfig(BaseConfig):
    """Masked-NLL term (trainer-side core)."""

    type: Literal["sft"] = "sft"
    name: str = "sft"


class OPDLossConfig(BaseConfig):
    """On-policy-distillation term (trainer-side core; requires a teacher)."""

    type: Literal["opd"] = "opd"
    name: str = "opd"


class EchoLossConfig(BaseConfig):
    """Per-role CE echo overlay (orchestrator-side weights; trainer-side CE core)."""

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
    RLLossConfig | SFTLossConfig | OPDLossConfig | EchoLossConfig | CustomLossTermConfig,
    Field(discriminator="type"),
]
"""A single loss term. The list of these is the unified ``losses`` surface."""


def default_losses() -> list[LossTermConfig]:
    """Default loss list: RL only (reproduces the pre-``losses`` default)."""
    return [RLLossConfig()]
