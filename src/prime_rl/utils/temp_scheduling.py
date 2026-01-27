import math

from prime_rl.orchestrator.config import SamplingConfig


def compute_temperature(step: int, sampling_config: SamplingConfig, max_steps: int | None) -> float:
    """Compute temperature for the given step based on sampling config.

    Either sampling_config.temperature or sampling_config.temp_scheduler must be set (not both).
    """
    if sampling_config.temperature is not None:
        return sampling_config.temperature

    schedule = sampling_config.temp_scheduler
    assert schedule is not None, "Either temperature or temp_scheduler must be set"

    total_steps = schedule.total_steps if schedule.total_steps is not None else max_steps
    assert total_steps is not None, "total_steps must be set when max_steps is None"

    if total_steps <= 1:
        progress = 1.0
    else:
        capped_step = min(max(step, 0), total_steps - 1)
        progress = capped_step / float(total_steps - 1)

    if schedule.type == "linear":
        factor = progress
    elif schedule.type == "cosine":
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported temperature schedule: {schedule.type}")

    return schedule.start_temperature + (schedule.end_temperature - schedule.start_temperature) * factor
