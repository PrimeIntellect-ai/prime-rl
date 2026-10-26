import threading
import uuid
from collections.abc import Callable

from prime_rl.control.types import Action, ActionNotFoundError, ActionType, ControlConflictError, InvalidActionError
from prime_rl.utils.logger import get_logger


class Controller:
    """Run control. A pause only queues when submitted; ``update`` applies it at a step, so each
    step runs with one consistent decision."""

    def __init__(self, *, requires_step: bool, on_submit: Callable[[], None] = lambda: None) -> None:
        """``requires_step``: whether a pause names its step (orchestrator) or applies at the next
        one (trainer). ``on_submit`` runs after a pause queues, from the API thread."""
        self.requires_step = requires_step
        self.on_submit = on_submit
        self.id_to_action: dict[str, Action] = {}
        self.pending: Action | None = None
        self.saved_steps: set[int] = set()
        self.step: int | None = None
        self.stop_step: int | None = None
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

    def submit(self, action_type: ActionType, step: int | None = None) -> Action:
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

    def _validate_accepting(self, step: int | None) -> None:
        if self.stop_step is not None:
            raise ControlConflictError(
                f"The run paused at step {self.stop_step} and takes no more actions; resume it to submit new ones."
            )
        if self.pending is not None:
            raise ControlConflictError("A pause is already queued and the run takes no more actions.")
        if step is not None and self.step is not None and step < self.step:
            raise ControlConflictError(f"Step {step} has passed, this component is at step {self.step}.")

    def _validate_step(self, step: int | None) -> None:
        if self.requires_step and step is None:
            raise InvalidActionError("The orchestrator pauses at a step you name: pass the trainer's accepted 'step'.")
        if not self.requires_step and step is not None:
            raise InvalidActionError("The trainer pauses at its next step and takes no 'step'.")
