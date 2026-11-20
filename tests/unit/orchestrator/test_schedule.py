import pytest

from prime_rl.configs.orchestrator import ScheduledEvalConfig
from prime_rl.configs.val import ValConfig
from prime_rl.orchestrator.schedule import plan_step
from prime_rl.transports.steps import StepPlan, StepRequest

VAL = ValConfig(interval=10, eval_on_start=True, source=[{"type": "documents", "name": "docs", "path": "docs"}])
EVAL = ScheduledEvalConfig(
    source=[
        {"name": name, "interval": interval, "env": {"taskset": {"id": "reverse-text"}}}
        for name, interval in (("every-3", 3), ("every-7", 7))
    ]
)


@pytest.mark.parametrize(
    "start_step,step_request,plan",
    [
        (1, StepRequest(1), StepPlan(validate=True)),
        (11, StepRequest(11), StepPlan(validate=True)),
        (1, StepRequest(10), StepPlan(validate=True)),
        (1, StepRequest(3), StepPlan(broadcast=True)),
        (1, StepRequest(7), StepPlan(broadcast=True)),
        (1, StepRequest(8), StepPlan()),
        (1, StepRequest(8, stop="finish"), StepPlan(broadcast=True)),
        (1, StepRequest(9, stop="pause"), StepPlan()),
    ],
)
def test_plan_step(start_step, step_request, plan):
    """Validation every val.interval steps and on the trainer's first step; a broadcast at every step an eval source
    is due and for the final eval, none for a paused run."""
    assert plan_step(step_request, start_step, VAL, EVAL) == plan
