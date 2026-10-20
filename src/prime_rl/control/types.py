from typing import Literal

from pydantic import BaseModel

ActionState = Literal["pending", "accepted", "committed"]
ActionType = Literal["pause"]


class Action(BaseModel):
    id: str
    type: ActionType
    state: ActionState
    step: int | None = None
    """The step the action applies to: the requested one, else set once the controller applies it."""


class ActionNotFoundError(Exception):
    pass


class ActionRequest(BaseModel):
    type: ActionType
    step: int | None = None
    """The step to apply the action at, which the orchestrator requires and the trainer picks itself."""


class ControlConflictError(Exception):
    pass


class InvalidActionError(Exception):
    pass
