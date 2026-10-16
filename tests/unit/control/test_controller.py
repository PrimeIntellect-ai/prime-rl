import pytest

from prime_rl.control.controller import Controller
from prime_rl.control.types import Action, ActionNotFoundError, ControlConflictError, InvalidActionError


@pytest.mark.parametrize(
    ("requires_step", "requested_step", "steps_before", "steps_after", "expected"),
    [
        (False, None, [], [16], [True]),
        (True, 17, [16], [16, 17], [False, True]),
        (True, 16, [16], [16], [True]),
    ],
    ids=["trainer_next_step", "orchestrator_later_step", "orchestrator_last_shipped_step"],
)
def test_update_pauses_at_due_step(requires_step, requested_step, steps_before, steps_after, expected):
    controller = Controller(requires_step=requires_step)
    for step in steps_before:
        controller.update(step)
    controller.submit("pause", requested_step)

    paused = [controller.update(step) for step in steps_after]

    assert paused == expected


def test_update_without_pending_pause():
    controller = Controller(requires_step=False)

    paused = controller.update(step=16)

    assert paused is False


@pytest.mark.parametrize("applied", [False, True], ids=["pause_queued", "pause_applied"])
def test_submit_while_pausing(applied):
    controller = Controller(requires_step=False)
    controller.submit("pause")
    if applied:
        controller.update(step=16)

    with pytest.raises(ControlConflictError):
        controller.submit("pause")


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
    ("saved_steps", "expected_state"),
    [([6], "committed"), ([], "accepted")],
    ids=["saved", "not_saved_yet"],
)
def test_get_commits_once_saved(saved_steps, expected_state):
    controller = Controller(requires_step=False)
    action = controller.submit("pause")
    controller.update(step=6)
    for step in saved_steps:
        controller.mark_saved(step)

    observed = controller.get(action.id)

    assert observed == Action(id=action.id, type="pause", state=expected_state, step=6)


def test_get_unknown_action():
    controller = Controller(requires_step=False)

    with pytest.raises(ActionNotFoundError):
        controller.get("missing")
