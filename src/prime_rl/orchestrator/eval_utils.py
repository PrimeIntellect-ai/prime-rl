import numpy as np


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
