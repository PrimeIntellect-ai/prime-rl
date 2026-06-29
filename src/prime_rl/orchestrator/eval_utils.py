import math


def compute_pass_metrics(rewards: list[float]) -> dict[str, float]:
    """Unbiased pass@k and pass^k for one example's binary (0/1) rewards.

    pass@k = 1 - C(n-c, k) / C(n, k)  (at least one of k samples correct)
    pass^k = C(c, k) / C(n, k)        (all k samples correct)

    ``n`` = number of rewards, ``c`` = number correct, ``k`` = powers of 2 in [1, n].
    ``math.comb`` returns 0 when ``k`` exceeds its first argument, so the edge cases
    (``n - c < k`` → pass@k = 1; ``c < k`` → pass^k = 0) fall out without branching.
    """
    n = len(rewards)
    c = sum(1 for r in rewards if r == 1.0)
    out: dict[str, float] = {}
    k = 1
    while k <= n:
        n_choose_k = math.comb(n, k)
        out[f"pass@{k}"] = 1.0 - math.comb(n - c, k) / n_choose_k
        out[f"pass^{k}"] = math.comb(c, k) / n_choose_k
        k *= 2
    return out
