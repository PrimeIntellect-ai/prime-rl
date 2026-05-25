import numpy as np


def compute_eval_step(
    step: int,
    prev_step: int,
    last_eval_step: int,
    interval: int,
    eval_base_model: bool = True,
) -> int | None:
    """Determine which orchestrator step (if any) should trigger an eval.

    Handles the case where step jumps over interval boundaries by finding
    the highest interval-aligned step in (prev_step, step].

    Returns the interval step to eval at, or None if no eval should run.
    """
    if step <= prev_step:
        return None
    highest_interval_step = (step // interval) * interval
    if highest_interval_step > prev_step and highest_interval_step > last_eval_step:
        if highest_interval_step == 0:
            if step == 0 and eval_base_model:
                return 0
        else:
            return highest_interval_step
    return None


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    Computes 1 - C(n-c, k) / C(n, k) in a numerically stable way.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k(rewards: list[float]) -> dict[str, float]:
    n = len(rewards)
    c = sum(r == 1.0 for r in rewards)
    ks = [2**i for i in range(n.bit_length())]
    return {f"pass@{k}": _pass_at_k(n, c, k) for k in ks}
