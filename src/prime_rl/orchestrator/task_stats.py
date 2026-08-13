"""Per-task outcome statistics — the task sampler's memory.

Every finalized train group is a free experiment: ``group_size`` episodes of
the current policy against one task. ``TaskStats`` keeps that evidence as
discounted success/failure pseudo-counts (a Beta posterior with forgetting)
plus reward EMAs, keyed by ``(env_name, task_key, agent role)``. The discount
is the staleness answer — the policy moves, so old outcomes must fade — and
nothing is ever evicted: estimates drift back toward the prior when a task
goes unobserved, so no task is written off permanently.

Keys are content hashes of the task's data, recomputable from the trace echo
(``rollout.task.data``), so stats survive dataset reordering and resumes and
can be joined against saved trace records offline. Roles are kept separate
because multi-agent groups mix reward scales (a proposer's reward says nothing
about solver difficulty).

Nothing here decides anything: this module only remembers and reports.
Sampling weights that *read* the posterior arrive with the weighted-sampling
config surface.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout

# Evidence discount per observed group: old outcomes fade as the policy moves.
DECAY = 0.9
# Beta prior pseudo-counts (alpha, beta): an unseen task sits at p_hat = 0.5.
PRIOR = (1.0, 1.0)
# A trace counts as a success when its reward reaches this threshold.
SUCCESS_THRESHOLD = 0.5
# p_hat bands for the pool occupancy metrics.
HOPELESS_BELOW = 0.05
SATURATED_ABOVE = 0.95


def task_key(task_data: dict) -> str:
    """Stable content key for one task: blake2b over the canonical JSON of its
    dumped data. Identical content hashes identically wherever it round-trips
    (dispatch request, trace echo, saved records)."""
    canonical = json.dumps(task_data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(canonical.encode(), digest_size=8).hexdigest()


@dataclass
class TaskStat:
    """Discounted evidence for one ``(env, task, role)``."""

    s: float = 0.0
    """Discounted success count."""
    f: float = 0.0
    """Discounted failure count."""
    reward_mean: float = 0.0
    reward_std: float = 0.0
    """EMA of the per-group reward std — the signal proxy for non-binary rewards."""
    draws_per_group: float = 0.0
    """EMA of role traces observed per group (1 for single-agent envs)."""
    visits: int = 0
    last_seen_version: int = 0

    @property
    def p_hat(self) -> float:
        """Posterior mean success rate under the Beta prior."""
        alpha, beta = PRIOR
        return (alpha + self.s) / (alpha + beta + self.s + self.f)

    def update(
        self, *, successes: int, failures: int, reward_mean: float, reward_std: float, draws: int, version: int
    ) -> None:
        self.s = DECAY * self.s + successes
        self.f = DECAY * self.f + failures
        if self.visits == 0:
            self.reward_mean = reward_mean
            self.reward_std = reward_std
            self.draws_per_group = float(draws)
        else:
            w = 1.0 - DECAY
            self.reward_mean += w * (reward_mean - self.reward_mean)
            self.reward_std += w * (reward_std - self.reward_std)
            self.draws_per_group += w * (draws - self.draws_per_group)
        self.visits += 1
        self.last_seen_version = max(self.last_seen_version, version)


class TaskStats:
    """The store plus per-tick counters for the sampler metric family.

    ``observe`` consumes one finalized train group; ``metrics`` drains the
    tick counters and snapshots the pool occupancy. Only clean, trainable
    traces update evidence — errored episodes and off-policy cancellations
    (which arrive as error markers) say nothing about task difficulty — but a
    degenerate zero-signal group is itself the strongest difficulty datum and
    updates counts like any other outcome.
    """

    def __init__(self) -> None:
        # env -> task_key -> role -> TaskStat
        self.stats: dict[str, dict[str, dict[str, TaskStat]]] = {}
        # Per-tick counters, drained by ``metrics()``.
        self._groups: dict[str, int] = defaultdict(int)
        self._signal_groups: dict[str, int] = defaultdict(int)
        self._tokens: dict[str, int] = defaultdict(int)
        self._wasted_tokens: dict[str, int] = defaultdict(int)

    def observe(self, group: list[Rollout]) -> None:
        env_name = group[0].env_name
        tokens = sum(r.num_total_tokens for r in group)
        # A group bought gradient iff any rollout carries a nonzero advantage.
        signal = any(r.is_trainable for r in group)
        self._groups[env_name] += 1
        self._signal_groups[env_name] += int(signal)
        self._tokens[env_name] += tokens
        if not signal:
            self._wasted_tokens[env_name] += tokens

        clean = [r for r in group if not r.has_error and r.agent.trainable]
        if not clean:
            return
        key = task_key(clean[0].task.data.model_dump(mode="json"))
        by_role: dict[str, list[Rollout]] = defaultdict(list)
        for rollout in clean:
            by_role[rollout.agent.name].append(rollout)
        for role, rollouts in by_role.items():
            rewards = [r.reward for r in rollouts]
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
            successes = sum(1 for x in rewards if x >= SUCCESS_THRESHOLD)
            stat = self.stats.setdefault(env_name, {}).setdefault(key, {}).setdefault(role, TaskStat())
            stat.update(
                successes=successes,
                failures=len(rewards) - successes,
                reward_mean=mean,
                reward_std=std,
                draws=len(rewards),
                version=max(r.policy_version for r in rollouts),
            )

    def metrics(self, num_tasks: dict[str, int | None]) -> dict[str, float]:
        """Sampler metric family, per env. Pool occupancy counts one unit per
        tracked ``(task, role)`` stat; ``unseen``/``coverage`` need the finite
        table size and are skipped for infinite tasksets. Tick counters
        (realized signal rate, wasted tokens) are drained on read."""
        out: dict[str, float] = {}
        for env_name, total in num_tasks.items():
            tracked = self.stats.get(env_name, {})
            units = [stat for roles in tracked.values() for stat in roles.values()]
            if units:
                hopeless = sum(1 for u in units if u.p_hat < HOPELESS_BELOW)
                saturated = sum(1 for u in units if u.p_hat > SATURATED_ABOVE)
                out[f"sampler/{env_name}/pool/hopeless"] = float(hopeless)
                out[f"sampler/{env_name}/pool/saturated"] = float(saturated)
                out[f"sampler/{env_name}/pool/learnable"] = float(len(units) - hopeless - saturated)
                out[f"sampler/{env_name}/p_hat/mean"] = sum(u.p_hat for u in units) / len(units)
            if total is not None:
                seen = len(tracked)
                out[f"sampler/{env_name}/pool/unseen"] = float(max(0, total - seen))
                out[f"sampler/{env_name}/coverage"] = seen / total if total else 0.0
            groups = self._groups.pop(env_name, 0)
            if groups:
                out[f"sampler/{env_name}/groups_observed"] = float(groups)
                out[f"sampler/{env_name}/realized_signal_rate"] = self._signal_groups.pop(env_name, 0) / groups
            tokens = self._tokens.pop(env_name, 0)
            if tokens:
                out[f"sampler/{env_name}/wasted_token_frac"] = self._wasted_tokens.pop(env_name, 0) / tokens
        return out

    def state_dict(self) -> dict:
        return {
            env: {key: {role: asdict(stat) for role, stat in roles.items()} for key, roles in tasks.items()}
            for env, tasks in self.stats.items()
        }

    def load_state_dict(self, state: dict) -> None:
        self.stats = {
            env: {key: {role: TaskStat(**stat) for role, stat in roles.items()} for key, roles in tasks.items()}
            for env, tasks in state.items()
        }
