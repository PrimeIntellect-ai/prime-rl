"""Pure-pandas metric helpers for the RL orchestrator.

Kept in a separate module with no heavy imports so they can be unit-tested
without pulling in torch / transformers / verifiers.
"""

from __future__ import annotations

import pandas as pd


def compute_solve_rates(
    df: pd.DataFrame,
    rollouts_per_example: int,
) -> tuple[float, float, float]:
    """Compute solve_none, solve_all, and effective_batch_size for a set of rollouts.

    Groups by **(env_name, example_id)** so that problems from different
    environments with the same integer example_id are never merged.  Grouping
    by example_id alone causes cross-environment collisions when every
    environment auto-assigns IDs from ``range(len(dataset))``, which is the
    default behaviour for all built-in environments.

    Args:
        df: DataFrame with at minimum columns ``env_name``, ``example_id``,
            and ``reward``.  Must contain at least one row.
        rollouts_per_example: Number of rollouts generated per problem.
            A problem is "fully solved" when the sum of its binary rewards
            equals this value (i.e. every rollout scored 1).

    Returns:
        (solve_none, solve_all, effective_batch_size) where each value is a
        float in [0, 1] and the three values sum to 1.0.

        - ``solve_none``: fraction of problems where *no* rollout was correct.
        - ``solve_all``: fraction of problems where *every* rollout was correct.
        - ``effective_batch_size``: fraction of problems that are neither fully
          solved nor fully failed (the "interesting" portion of the batch for
          GRPO learning).
    """
    reward_per_problem = df.groupby(["env_name", "example_id"]).reward.sum()
    solve_none = float((reward_per_problem == 0).mean())
    solve_all = float((reward_per_problem == rollouts_per_example).mean())
    return solve_none, solve_all, 1.0 - solve_none - solve_all
