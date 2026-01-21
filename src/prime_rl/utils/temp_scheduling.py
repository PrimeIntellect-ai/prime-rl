import math

from prime_rl.orchestrator.config import SamplingConfig


def compute_temperature(step: int, sampling_config: SamplingConfig, max_steps: int | None) -> float:
    schedule = sampling_config.temp_scheduler
    if schedule is None:
        return sampling_config.temperature

    start_temp = sampling_config.temperature if schedule.start_temperature is None else schedule.start_temperature
    if schedule.type == "constant":
        if schedule.start_temperature is None and schedule.end_temperature is not None:
            return schedule.end_temperature
        return start_temp

    if schedule.end_temperature is None:
        raise ValueError("temp_scheduler.end_temperature must be set for linear/cosine schedules")

    total_steps = schedule.total_steps if schedule.total_steps is not None else max_steps
    if total_steps is None:
        raise ValueError("temp_scheduler.total_steps must be set when max_steps is None")

    end_temp = schedule.end_temperature
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

    return start_temp + (end_temp - start_temp) * factor
