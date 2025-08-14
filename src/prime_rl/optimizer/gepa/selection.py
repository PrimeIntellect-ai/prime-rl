from __future__ import annotations

import random
from typing import Iterable

from prime_rl.optimizer.gepa.config import SelectionConfig


def top_k(scores: list[tuple[int, float]], k: int) -> list[int]:
    return [i for i, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]


def tournament(scores: list[tuple[int, float]], k: int, t_size: int, rng: random.Random) -> list[int]:
    winners: list[int] = []
    indices = [i for i, _ in scores]
    for _ in range(k):
        pool = rng.sample(indices, min(t_size, len(indices)))
        pool_scores = [(i, dict(scores)[i]) for i in pool]
        winners.append(max(pool_scores, key=lambda x: x[1])[0])
    return winners


def select_indices(cfg: SelectionConfig, fitness: Iterable[float], rng: random.Random) -> list[int]:
    scores = list(enumerate(fitness))
    if cfg.strategy == "top-k":
        idxs = top_k(scores, cfg.k)
    else:
        idxs = tournament(scores, cfg.k, cfg.tournament_size, rng)
    # Add elites (ensure uniqueness, preserve order)
    elites = top_k(scores, cfg.keep_elite)
    seen = set()
    ordered = []
    for i in elites + idxs:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


