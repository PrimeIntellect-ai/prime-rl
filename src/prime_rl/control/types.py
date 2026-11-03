from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, Field

from prime_rl.configs.diff import ConfigChange

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import OrchestratorConfig

ActionState = Literal["pending", "accepted", "committed", "applied"]
"""A pause goes pending -> accepted -> committed, a config update pending -> applied."""
ActionType = Literal["pause", "update_config"]


class Action(BaseModel):
    id: str
    type: ActionType
    state: ActionState
    step: int | None = None
    """The step the action applies to: the requested one, else set once the controller applies it.
    For a config update, the first step whose rollouts use the new config."""
    changes: list[ConfigChange] | None = None
    """What a config update changes."""


class PauseRequest(BaseModel):
    type: Literal["pause"]
    step: int | None = None
    """The step to apply the action at, which the orchestrator requires and the trainer picks itself."""


class UpdateConfigRequest(BaseModel):
    type: Literal["update_config"]
    config: dict[str, Any]
    """The full new config, as returned by ``GET /v1/config``."""


ActionRequest = Annotated[PauseRequest | UpdateConfigRequest, Field(discriminator="type")]


class ConfigSnapshot(BaseModel):
    revision: int
    step: int | None
    """The last step the run shipped, None before the first."""
    config: dict[str, Any]


@dataclass(frozen=True)
class ConfigUpdate:
    """A validated live config update, as the orchestrator's components receive it."""

    revision: int
    step: int
    """The first step whose rollouts use the new config."""
    config: OrchestratorConfig
    """The full new config, never mutated in place."""
    changes: tuple[ConfigChange, ...]


class Reconfigurable:
    """An orchestrator component that the orchestrator reconfigures on every applied config
    update. Components that read live config values override ``reconfigure``."""

    def reconfigure(self, update: ConfigUpdate) -> None:
        """Apply a validated update at a step boundary. Cannot fail: the update was
        validated when it was submitted."""


class ActionNotFoundError(Exception):
    pass


class ControlConflictError(Exception):
    pass


class InvalidActionError(Exception):
    pass


class InvalidConfigError(InvalidActionError):
    def __init__(
        self,
        message: str,
        *,
        errors: list[Any] | None = None,
        frozen: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.detail = {"message": message, "errors": errors or [], "frozen": frozen or []}
