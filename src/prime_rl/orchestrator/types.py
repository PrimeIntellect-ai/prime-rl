from typing import TypedDict


class RolloutStep(TypedDict):
    """Trajectory step used for training."""

    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    is_truncated: bool
    reward: float


class RolloutState(TypedDict):
    """Rollout data stored in the buffer and consumed by the trainer."""

    example_id: int
    task: str
    reward: float
    metrics: dict[str, float]
    steps: list[RolloutStep]
    is_truncated: bool
    advantage: float | None
