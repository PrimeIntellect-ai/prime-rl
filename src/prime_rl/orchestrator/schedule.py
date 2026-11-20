"""The step schedule of a training run: at which steps the trainer validates and broadcasts its weights for online
evals. The orchestrator plans every step with it and tells the trainer."""

from prime_rl.configs.orchestrator import ScheduledEvalConfig
from prime_rl.configs.supervised import ValConfig
from prime_rl.transports.steps import StepPlan, StepRequest


def plan_step(
    request: StepRequest, start_step: int, val: ValConfig | None, eval: ScheduledEvalConfig | None
) -> StepPlan:
    """The plan of a trainer whose first step is ``start_step``: it validates at the steps ``val`` gives and broadcasts
    at every step an eval source is due, and at its last step, unless a pause ended the run there."""
    validate = val is not None and val.due(request.step, start_step)
    if eval is None:
        return StepPlan(validate=validate)
    if request.stop is not None:
        return StepPlan(validate=validate, broadcast=request.stop == "finish")
    return StepPlan(
        validate=validate, broadcast=any(request.step % interval == 0 for interval in eval.intervals.values())
    )
