from collections.abc import Iterable

import numpy as np


def compute_eval_ckpt_step(
    ckpt_step: int,
    prev_ckpt_step: int,
    last_eval_step: int,
    interval: int,
    eval_base_model: bool = True,
) -> int | None:
    """Determine which checkpoint step (if any) should trigger an eval.

    Handles the case where ckpt_step jumps over interval boundaries by finding
    the highest interval-aligned step in (prev_ckpt_step, ckpt_step].

    Returns the interval step to eval at, or None if no eval should run.
    """
    if ckpt_step <= prev_ckpt_step:
        return None
    highest_interval_step = (ckpt_step // interval) * interval
    if highest_interval_step > prev_ckpt_step and highest_interval_step > last_eval_step:
        if highest_interval_step == 0:
            if ckpt_step == 0 and eval_base_model:
                return 0
        else:
            return highest_interval_step
    return None


def estimate_pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased estimator of pass@k (Chen et al., 2021).

    Computes 1 - C(n-c, k) / C(n, k) in a numerically stable way.
    """
    if not 0 <= c <= n:
        raise ValueError(f"Expected 0 <= c <= n, got n={n}, c={c}")
    if k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _pass_at_k(n: int, c: int, k: int) -> float:
    value = estimate_pass_at_k(n, c, k)
    if value is None:
        raise ValueError(f"Cannot compute pass@{k} from only {n} samples")
    return value


def compute_pass_at_k(rewards: list[float], ks: Iterable[int] | None = None) -> dict[str, float]:
    n = len(rewards)
    c = sum(r == 1.0 for r in rewards)
    if ks is None:
        ks = [2**i for i in range(n.bit_length())]
    pass_at_k = {}
    for k in ks:
        value = estimate_pass_at_k(n, c, int(k))
        if value is not None:
            pass_at_k[f"pass@{int(k)}"] = value
    return pass_at_k
