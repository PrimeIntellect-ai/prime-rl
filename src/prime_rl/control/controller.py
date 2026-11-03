import json
import threading
import uuid
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from prime_rl.configs.diff import ConfigChange, diff_config, frozen_raw_changes
from prime_rl.control.types import (
    Action,
    ActionNotFoundError,
    ConfigSnapshot,
    ControlConflictError,
    InvalidActionError,
    InvalidConfigError,
    UpdateConfigRequest,
)
from prime_rl.utils.config import dump_resolved_config
from prime_rl.utils.logger import get_logger

CheckConfig = Callable[[dict[str, Any]], tuple[BaseModel, list[ConfigChange]]]
"""Validate a submitted config against the running one; returns it and its changes, or raises
``InvalidConfigError``."""
ApplyConfig = Callable[[BaseModel, list[ConfigChange]], None]
"""Apply a validated config as the next revision."""


class Controller:
    """Run control. A pause only queues when submitted; ``update`` applies it at a step, so each
    step runs with one consistent decision. A config update likewise waits for ``apply_update``."""

    def __init__(
        self,
        *,
        requires_step: bool,
        on_submit: Callable[[], None] = lambda: None,
        read_config: Callable[[], tuple[int, BaseModel]] | None = None,
        check_config: CheckConfig | None = None,
    ) -> None:
        """``requires_step``: whether a pause names its step (orchestrator) or applies at the next
        one (trainer). ``on_submit`` runs after a pause queues, from the API thread. ``read_config`` and
        ``check_config``: read the running config with its revision and validate a new one, for
        components that update live."""
        self.requires_step = requires_step
        self.on_submit = on_submit
        self.id_to_action: dict[str, Action] = {}
        self.pending: Action | None = None
        self.saved_steps: set[int] = set()
        self.step: int | None = None
        self.stop_step: int | None = None
        self.read_config = read_config
        self.check_config = check_config
        self.pending_update: tuple[Action, BaseModel] | None = None
        # The API serves from its own thread while the step loop applies actions.
        self.lock = threading.Lock()

    def get(self, action_id: str) -> Action:
        action = self.id_to_action.get(action_id)
        if action is None:
            raise ActionNotFoundError(
                f"Unknown action {action_id!r}: the run may have restarted since it was submitted; submit it again."
            )
        with self.lock:
            if action.state == "accepted" and action.step in self.saved_steps:
                action.state = "committed"
        return action

    def mark_saved(self, step: int) -> None:
        """Record that this process saved ``step``, the step its pause applied to."""
        with self.lock:
            self.saved_steps.add(step)

    def submit(self, action_type: Literal["pause"], step: int | None = None) -> Action:
        self._validate_step(step)
        with self.lock:
            self._validate_accepting(step)
            action = Action(id=uuid.uuid4().hex, type=action_type, state="pending", step=step)
            self.id_to_action[action.id] = action
            self.pending = action
        self.on_submit()
        return action

    def update(self, step: int) -> bool:
        """Apply a pause due at ``step``. Returns whether the run pauses there."""
        with self.lock:
            self.step = step
            action = self.pending
            if action is None or action.step not in (None, step):
                return False
            self.pending = None
            action.state, action.step = "accepted", step
            self.stop_step = step
        get_logger().info(f"Control action {action.id}: {action.type} at step {step}")
        return True

    def get_config(self) -> ConfigSnapshot:
        if self.read_config is None:
            raise InvalidActionError("This component takes no config updates.")
        with self.lock:
            step, (revision, config) = self.step, self.read_config()
        return ConfigSnapshot(revision=revision, step=step, config=dump_resolved_config(config))

    def submit_config(self, request: UpdateConfigRequest) -> Action:
        """Validate a full new config and queue it for the next step boundary."""
        if self.check_config is None:
            raise InvalidActionError("This component takes no config updates; only the orchestrator does.")
        with self.lock:
            self._validate_accepting_update()
            revision, _ = self.read_config()
        config, changes = self.check_config(request.config)
        with self.lock:
            self._validate_accepting_update()
            if self.read_config()[0] != revision:
                raise ControlConflictError("The config changed while this update was validated; submit it again.")
            action = Action(id=uuid.uuid4().hex, type="update_config", state="pending", changes=changes)
            self.id_to_action[action.id] = action
            self.pending_update = (action, config)
        return action

    def apply_update(self, step: int, apply: ApplyConfig) -> None:
        """Apply the queued config update, if any, from ``step`` on."""
        with self.lock:
            if self.pending_update is None:
                return
            action, config = self.pending_update
            self.pending_update = None
            apply(config, action.changes)
            action.state, action.step = "applied", step

    def _validate_accepting(self, step: int | None) -> None:
        if self.stop_step is not None:
            raise ControlConflictError(
                f"The run paused at step {self.stop_step} and takes no more actions; resume it to submit new ones."
            )
        if self.pending is not None:
            raise ControlConflictError("A pause is already queued and the run takes no more actions.")
        if step is not None and self.step is not None and step < self.step:
            raise ControlConflictError(f"Step {step} has passed, this component is at step {self.step}.")

    def _validate_accepting_update(self) -> None:
        self._validate_accepting(None)
        if self.pending_update is not None:
            raise ControlConflictError(
                "A config update is already queued; wait until it is applied, then submit again."
            )

    def _validate_step(self, step: int | None) -> None:
        if self.requires_step and step is None:
            raise InvalidActionError("The orchestrator pauses at a step you name: pass the trainer's accepted 'step'.")
        if not self.requires_step and step is not None:
            raise InvalidActionError("The trainer pauses at its next step and takes no 'step'.")


def validate_update(current: BaseModel, raw: dict[str, Any]) -> tuple[BaseModel, list[ConfigChange]]:
    """Validate a submitted full config and diff it against ``current``. Rejects changes to
    frozen fields; every other field is applied live."""
    # Checked on the raw dict first: a frozen change can itself make the config invalid, and
    # validation may rewrite the dict in place.
    frozen = frozen_raw_changes(current, dump_resolved_config(current), raw)
    try:
        config = type(current).model_validate(raw)
    except ValidationError as error:
        errors = json.loads(error.json(include_url=False, include_input=False))
        raise InvalidConfigError("The config is invalid.", errors=errors, frozen=frozen) from error
    diff = diff_config(current, config)
    if diff.frozen:
        raise InvalidConfigError(
            "The config changes fields that cannot change while the run is going.", frozen=diff.frozen
        )
    return config, diff.changes
