import pytest

from prime_rl.control.controller import Controller
from prime_rl.control.types import Action, ActionNotFoundError, ControlConflictError, InvalidActionError


@pytest.mark.parametrize(
    ("action_type", "requires_step", "requested_step", "steps_before", "steps_after", "expected"),
    [
        ("pause", False, None, [], [16], ["pause"]),
        ("pause", True, 17, [16], [16, 17], [None, "pause"]),
        ("pause", True, 16, [16], [16], ["pause"]),
        ("checkpoint", False, None, [], [16, 17], ["checkpoint", None]),
        ("checkpoint", True, 17, [16], [16, 17], [None, "checkpoint"]),
    ],
    ids=[
        "pause_trainer_next_step",
        "pause_orchestrator_later_step",
        "pause_orchestrator_last_shipped_step",
        "checkpoint_trainer_next_step",
        "checkpoint_orchestrator_later_step",
    ],
)
def test_update_applies_at_due_step(action_type, requires_step, requested_step, steps_before, steps_after, expected):
    controller = Controller(requires_step=requires_step)
    for step in steps_before:
        controller.update(step)
    controller.submit(action_type, requested_step)

    applied = [controller.update(step) for step in steps_after]

    assert applied == expected


def test_update_without_pending_action():
    controller = Controller(requires_step=False)

    applied = controller.update(step=16)

    assert applied is None


@pytest.mark.parametrize(
    ("action_type", "applied"),
    [("pause", False), ("pause", True), ("checkpoint", False)],
    ids=["pause_queued", "pause_applied", "checkpoint_queued"],
)
def test_submit_while_action_pending_or_paused(action_type, applied):
    controller = Controller(requires_step=False)
    controller.submit(action_type)
    if applied:
        controller.update(step=16)

    with pytest.raises(ControlConflictError):
        controller.submit("checkpoint")


def test_submit_after_checkpoint_applied():
    controller = Controller(requires_step=False)
    controller.submit("checkpoint")
    controller.update(step=16)

    action = controller.submit("pause")

    assert action.state == "pending"


def test_submit_passed_step():
    controller = Controller(requires_step=True)
    controller.update(step=16)

    with pytest.raises(ControlConflictError):
        controller.submit("pause", 15)


@pytest.mark.parametrize(("requires_step", "step"), [(True, None), (False, 16)], ids=["missing", "unexpected"])
def test_submit_invalid_step(requires_step, step):
    controller = Controller(requires_step=requires_step)

    with pytest.raises(InvalidActionError):
        controller.submit("pause", step)


@pytest.mark.parametrize(
    ("action_type", "saved_steps", "expected_state"),
    [("pause", [6], "committed"), ("pause", [], "accepted"), ("checkpoint", [6], "committed")],
    ids=["pause_saved", "pause_not_saved_yet", "checkpoint_saved"],
)
def test_get_commits_once_saved(action_type, saved_steps, expected_state):
    controller = Controller(requires_step=False)
    action = controller.submit(action_type)
    controller.update(step=6)
    for step in saved_steps:
        controller.mark_saved(step)

    observed = controller.get(action.id)

    assert observed == Action(id=action.id, type=action_type, state=expected_state, step=6)


def test_get_unknown_action():
    controller = Controller(requires_step=False)

    with pytest.raises(ActionNotFoundError):
        controller.get("missing")
