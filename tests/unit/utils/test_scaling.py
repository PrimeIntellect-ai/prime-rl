import math

import numpy as np

from prime_rl.utils.scaling import Heuristic, fit_power_law


def test_heuristic_formulas():
    h = Heuristic()
    assert h.batch_size(tokens=1e10, seq_len=4096) == 2 ** round(math.log2(6.6 * 1e5 / 4096))
    assert h.lr(tokens=1e10, hidden=1024, batch_tokens=2**20) == 0.087571 * 1e10**-0.3461 * 1024**-0.3448 * 2**10
    assert h.lr(tokens=1.0, hidden=1, batch_tokens=2**20) == h.max_lr
    assert h.beta2(131_072) == 0.999
    assert h.beta2(2**30) == h.min_beta2
    assert h.eps(tokens=4e10, batch_tokens=4e6) == 9.676e-18 * 100


def test_fit_power_law_recovers_parameters():
    compute = np.logspace(18, 22, 6)
    floor, coeff, alpha = fit_power_law(compute, 1.7 + 300 * compute**-0.12)
    assert abs(floor - 1.7) < 1e-2
    assert abs(alpha - 0.12) / 0.12 < 1e-2
    assert abs(coeff - 300) / 300 < 0.05
