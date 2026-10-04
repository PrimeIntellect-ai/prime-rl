"""Compute-scaled hyperparameters and power-law fits for pretraining scaling ladders.

Used by ``tools/scaling_ladder.py`` and ``tools/fit_scaling_law.py``; see ``docs/scaling-ladder.md``.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Heuristic:
    """Hyperparameters as fitted power laws of the token budget, the width and the batch.

    batch_tokens = batch_coeff * tokens^batch_tokens_exp
    lr           = min(max_lr, lr_coeff * tokens^lr_tokens_exp * hidden^lr_hidden_exp * sqrt(batch_tokens))
    beta2        = clip(beta2_base^(batch_tokens / beta2_reference_batch_tokens), min_beta2, max_beta2)
    eps          = eps_coeff * sqrt(tokens / batch_tokens)

    ``lr`` is the peak LR of every parameter group: our Muon runs with ``adjust_lr="rms_norm"``,
    so its Muon and AdamW groups share one LR. The schedule is WSD: linear warmup over
    ``warmup_fraction`` of the steps, constant, then linear decay to ``min_lr_ratio * lr`` over the
    last ``decay_fraction``.

    The LR, beta2 and epsilon defaults are Marin's moe_hero_ep AdamW fit (seq_len 8192) and the
    batch default is a square-root rule. All of them are placeholders until refitted on LR/batch
    sweeps of our own recipe.
    """

    batch_coeff: float = 6.6
    batch_tokens_exp: float = 0.5
    lr_coeff: float = 0.087571
    lr_tokens_exp: float = -0.3461
    lr_hidden_exp: float = -0.3448
    max_lr: float = 0.05
    beta2_base: float = 0.999
    beta2_reference_batch_tokens: int = 131_072
    min_beta2: float = 0.95
    max_beta2: float = 0.9999
    eps_coeff: float = 9.676e-18
    warmup_fraction: float = 0.01
    decay_fraction: float = 0.2
    min_lr_ratio: float = 0.05

    def batch_size(self, tokens: float, seq_len: int) -> int:
        """Sequences per step, rounded to a power of two."""
        sequences = self.batch_coeff * tokens**self.batch_tokens_exp / seq_len
        return 2 ** max(0, round(math.log2(sequences)))

    def lr(self, tokens: float, hidden: int, batch_tokens: int) -> float:
        lr = self.lr_coeff * tokens**self.lr_tokens_exp * hidden**self.lr_hidden_exp * math.sqrt(batch_tokens)
        return min(self.max_lr, lr)

    def beta2(self, batch_tokens: int) -> float:
        beta2 = self.beta2_base ** (batch_tokens / self.beta2_reference_batch_tokens)
        return min(self.max_beta2, max(self.min_beta2, beta2))

    def eps(self, tokens: float, batch_tokens: int) -> float:
        return self.eps_coeff * math.sqrt(tokens / batch_tokens)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of ``y = E + A * x^-alpha``; returns ``(E, A, alpha)``.

    For each candidate floor ``E`` in ``[0, min(y))`` the rest is a line in log-log space;
    the floor with the smallest squared error in ``y`` wins.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    best = None
    for floor in np.linspace(0, y.min(), 2000, endpoint=False):
        slope, intercept = np.polyfit(np.log(x), np.log(y - floor), 1)
        error = np.sum((floor + np.exp(intercept) * x**slope - y) ** 2)
        if best is None or error < best[0]:
            best = (error, float(floor), float(np.exp(intercept)), float(-slope))
    return best[1:]
