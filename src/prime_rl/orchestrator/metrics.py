"""Train and eval rollout metrics.

A rollout container (``TrainRollouts`` / ``EvalRollouts``) owns the rollout list and exposes
``.effective`` (the clean subset, as the same container type), ``.by_env()`` / ``.by_agent()``
(slices of the same type) and ``.metrics`` (``TrainMetrics`` / ``EvalMetrics``). The metrics object
exposes each distributional / rate metric as a ``Stat`` — so
``rollouts.metrics.num_input_tokens.mean()`` works — and assembles the full
``{prefix}/{subset}/<metric>/<stat>`` wandb dict via ``.to_wandb(...)``.

No I/O, no pandas — plain Python over the ``vf.Trace`` properties each rollout exposes. Aggregation
is flat over the rollout list except the solve rates and pass@k, which group by
``(group_id, agent_name)``.

Reward-derived metrics (the reward distribution, its per-component breakdown, the solve rates,
pass@k, ``avg@k``) read the *policy view* of their pool — see :func:`_scored`. Cost, shape and
error metrics cover every rollout: an untrainable agent's tokens and failures are real.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

from prime_rl.orchestrator.utils import compute_pass_metrics

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout

Subset = Literal["all", "effective"]

DEFAULT_AGENT_NAME = "agent"
"""Slice key for a rollout with no agent info (the legacy v0 bridge); matches
``vf.AgentInfo.name``'s own default."""


def _scored(rollouts: list[Rollout]) -> list[Rollout]:
    """The policy view of a pool: its trainable rollouts, or all of them when none is trainable.

    A reward metric is about the policy, and an untrainable agent (a frozen judge, a pinned user
    sim) isn't it — usually carrying no rewards at all, so its traces score a structural zero. The
    fallback keeps an all-frozen pool reporting its own rewards rather than nothing. Same rule
    verifiers applies in ``v1/push.py`` and the eval dashboard."""
    return [r for r in rollouts if r.trainable] or rollouts


def _by_group_and_agent(rollouts: list[Rollout]) -> dict[tuple, list[Rollout]]:
    """Group rollouts into the unit a solve rate / pass@k is defined over: one example as attempted
    by one agent. Agents' rewards aren't comparable — a zero-sum pair sums to 0 whatever the policy
    does — so each gets its own group. One group per ``group_id`` for a single-agent env."""
    groups: dict[tuple, list[Rollout]] = {}
    for r in rollouts:
        groups.setdefault((r.group_id, r.agent_name), []).append(r)
    return groups


class Stat:
    """A distribution of per-rollout values with mean/max/min and p10/p90 accessors."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def max(self) -> float:
        return float(max(self.values)) if self.values else 0.0

    def min(self) -> float:
        return float(min(self.values)) if self.values else 0.0

    def percentile(self, q: float) -> float:
        """Linear-interpolated ``q``-th percentile (numpy's default method); 0.0 if empty."""
        if not self.values:
            return 0.0
        s = sorted(self.values)
        rank = q / 100 * (len(s) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(s) - 1)
        return float(s[lo] + (s[hi] - s[lo]) * (rank - lo))

    def p10(self) -> float:
        return self.percentile(10)

    def p90(self) -> float:
        return self.percentile(90)

    def to_dict(self, prefix: str) -> dict[str, float]:
        """``{prefix}/mean,max,min,p10,p90``; ``{}`` for an empty distribution."""
        if not self.values:
            return {}
        return {
            f"{prefix}/mean": self.mean(),
            f"{prefix}/max": self.max(),
            f"{prefix}/min": self.min(),
            f"{prefix}/p10": self.p10(),
            f"{prefix}/p90": self.p90(),
        }


class StatGroup:
    """A nested group of named ``Stat``s. ``to_dict`` emits ``{prefix}/<name>/<stat>`` for each
    distribution and ``group[name]`` returns one; subclasses supply the names via ``stats()``."""

    def __init__(self, rollouts: list[Rollout]) -> None:
        self.rollouts = rollouts

    def stats(self) -> dict[str, Stat]:
        raise NotImplementedError

    def __getitem__(self, name: str) -> Stat:
        return self.stats()[name]

    def to_dict(self, prefix: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, stat in self.stats().items():
            out |= stat.to_dict(f"{prefix}/{name}")
        return out


class TimingMetrics(StatGroup):
    """Per-phase rollout durations, nested so ``metrics.timing.setup.mean()`` reads naturally.
    ``total`` is the per-rollout sum across all phases."""

    PHASES = ("setup", "generation", "finalize", "scoring")

    @property
    def setup(self) -> Stat:
        return Stat([r.timing.setup.duration for r in self.rollouts])

    @property
    def generation(self) -> Stat:
        return Stat([r.timing.generation.duration for r in self.rollouts])

    @property
    def generation_model(self) -> Stat:
        """The share of the generation phase spent inside model calls (inference)."""
        return Stat([r.timing.generation.model.duration for r in self.rollouts])

    @property
    def generation_harness(self) -> Stat:
        """The share of the generation phase spent outside model calls (harness, tools,
        user simulation)."""
        return Stat([r.timing.generation.harness.duration for r in self.rollouts])

    @property
    def finalize(self) -> Stat:
        return Stat([r.timing.finalize.duration for r in self.rollouts])

    @property
    def scoring(self) -> Stat:
        return Stat([r.timing.scoring.duration for r in self.rollouts])

    @property
    def total(self) -> Stat:
        return Stat([sum(getattr(r.timing, p).duration for p in self.PHASES) for r in self.rollouts])

    def stats(self) -> dict[str, Stat]:
        return {
            **{phase: getattr(self, phase) for phase in self.PHASES},
            "generation/model": self.generation_model,
            "generation/harness": self.generation_harness,
            "total": self.total,
        }


class CustomMetrics(StatGroup):
    """Per-key ``Stat``s over a dynamic per-rollout dict attribute (env ``@metric``s or reward
    components), each averaged over the rollouts that report the key. ``value`` extracts the
    float from each entry (rewards are ``vf.Reward`` records; metrics are plain floats)."""

    def __init__(self, rollouts: list[Rollout], attr: str, value: Callable[[Any], float] = float) -> None:
        super().__init__(rollouts)
        self.attr = attr
        self.value = value

    def stats(self) -> dict[str, Stat]:
        names = sorted({name for r in self.rollouts for name in getattr(r, self.attr)})
        return {
            name: Stat(
                [self.value(getattr(r, self.attr)[name]) for r in self.rollouts if name in getattr(r, self.attr)]
            )
            for name in names
        }


class RolloutMetrics:
    """Metrics shared by train and eval over a rollout list. Distributional metrics are ``Stat``s
    (mean/max/min); boolean metrics are ``Stat``s of 0/1 (use ``.mean()`` for the rate). ``to_wandb``
    assembles the full ``{prefix}/{subset}/...`` dict; ``TrainMetrics`` / ``EvalMetrics`` extend it."""

    def __init__(self, rollouts: list[Rollout]) -> None:
        self.rollouts = rollouts

    # Distributional metrics
    @property
    def num_total_tokens(self) -> Stat:
        return Stat([float(r.num_total_tokens) for r in self.rollouts])

    @property
    def num_input_tokens(self) -> Stat:
        return Stat([float(r.num_input_tokens) for r in self.rollouts])

    @property
    def num_output_tokens(self) -> Stat:
        return Stat([float(r.num_output_tokens) for r in self.rollouts])

    @property
    def num_turns(self) -> Stat:
        return Stat([float(r.num_turns) for r in self.rollouts])

    @property
    def num_branches(self) -> Stat:
        return Stat([float(r.num_branches) for r in self.rollouts])

    @property
    def timing(self) -> TimingMetrics:
        return TimingMetrics(self.rollouts)

    @property
    def metrics(self) -> CustomMetrics:
        """Env custom ``@metric`` outputs, keyed by name (``metrics.metrics["acc"].mean()``)."""
        return CustomMetrics(self.rollouts, "metrics")

    @property
    def rewards(self) -> CustomMetrics:
        """Per-component reward breakdown over the policy view, keyed by name (each entry's
        weighted ``value``, summed into the scalar ``reward``)."""
        return CustomMetrics(_scored(self.rollouts), "rewards", value=lambda reward: reward.value)

    # Boolean rate metrics (0/1 distributions — ``.mean()`` is the rate)
    @property
    def is_truncated(self) -> Stat:
        return Stat([float(r.is_truncated) for r in self.rollouts])

    @property
    def is_completed(self) -> Stat:
        return Stat([float(r.is_completed) for r in self.rollouts])

    @property
    def has_error(self) -> Stat:
        return Stat([float(r.has_error) for r in self.rollouts])

    def stop_conditions(self) -> dict[str, float]:
        """``generation_truncated`` over all rollouts, then each recorded ``stop_condition``'s rate
        over the rollouts that recorded one."""
        out = {
            "generation_truncated": sum(
                1 for r in self.rollouts if r.is_truncated and r.stop_condition != "prompt_too_long"
            )
            / len(self.rollouts)
        }
        conditions = [r.stop_condition for r in self.rollouts if r.stop_condition is not None]
        for condition in sorted(set(conditions)):
            out[condition] = conditions.count(condition) / len(conditions)
        return out

    def error_types(self) -> dict[str, int]:
        """Count of errored rollouts by error type (the rollout's last error — e.g. ``Cancelled``,
        ``ProviderError``)."""
        types = [r.error.type for r in self.rollouts if r.has_error]
        return {t: types.count(t) for t in sorted(set(types))}

    def solve_rates(self) -> dict[str, float]:
        """Per-(example, agent) solve rates over the policy view, assuming binary 0/1 rewards
        (unspecified for other reward ranges): ``solved_none`` (the group earned no reward),
        ``solved_all`` (every rollout scored 1.0), and ``solved_some`` (the mixed remainder — the
        GRPO-signal groups)."""
        groups = _by_group_and_agent(_scored(self.rollouts))
        n_groups = len(groups)
        solved_none = sum(1 for g in groups.values() if sum(r.reward for r in g) == 0)
        solved_all = sum(1 for g in groups.values() if all(r.reward == 1.0 for r in g))
        return {
            "solved_none": solved_none / n_groups,
            "solved_all": solved_all / n_groups,
            "solved_some": 1 - (solved_none + solved_all) / n_groups,
        }

    def to_wandb(self, *, prefix: str, subset: Subset) -> dict[str, float]:
        """The common metric dict for one ``{prefix}/{subset}`` slice. Empty input → ``{}``."""
        if not self.rollouts:
            return {}
        p = f"{prefix}/{subset}"
        out: dict[str, float] = {}
        out |= self.num_total_tokens.to_dict(f"{p}/num_total_tokens")
        out |= self.num_input_tokens.to_dict(f"{p}/num_input_tokens")
        out |= self.num_output_tokens.to_dict(f"{p}/num_output_tokens")
        out |= self.num_turns.to_dict(f"{p}/num_turns")
        out |= self.num_branches.to_dict(f"{p}/num_branches")
        out |= self.timing.to_dict(f"{p}/timing")
        out |= self.metrics.to_dict(f"{p}/metrics")
        out |= self.rewards.to_dict(f"{p}/rewards")
        out[f"{p}/is_truncated/mean"] = self.is_truncated.mean()
        out[f"{p}/is_completed/mean"] = self.is_completed.mean()
        # errors live only on the `all` subset (effective drops them), so emit the rate + the
        # per-type counts there only
        if subset == "all":
            out[f"{p}/has_error/mean"] = self.has_error.mean()
            out |= {f"{p}/error/{t}": float(count) for t, count in self.error_types().items()}
        out |= {f"{p}/stop_condition/{k}": v for k, v in self.stop_conditions().items()}
        out |= {f"{p}/{k}": v for k, v in self.solve_rates().items()}
        return out


class TrainMetrics(RolloutMetrics):
    """Common metrics plus the reward distribution and filter-pipeline rates."""

    @property
    def reward(self) -> Stat:
        """The policy's reward distribution (untrainable agents excluded — see :func:`_scored`)."""
        return Stat([float(r.reward) for r in _scored(self.rollouts)])

    @property
    def is_trainable(self) -> Stat:
        return Stat([float(r.is_trainable) for r in self.rollouts])

    @property
    def is_filtered(self) -> Stat:
        return Stat([float(r.is_filtered) for r in self.rollouts])

    def filter_rates(self) -> dict[str, float]:
        """Per-filter detection rate over all rollouts."""
        names = sorted({name for r in self.rollouts for name in r.filter_results})
        return {
            name: sum(1 for r in self.rollouts if r.filter_results.get(name)) / len(self.rollouts) for name in names
        }

    def to_wandb(self, *, prefix: str, subset: Subset) -> dict[str, float]:
        out = super().to_wandb(prefix=prefix, subset=subset)
        if not self.rollouts:
            return out
        p = f"{prefix}/{subset}"
        out |= self.reward.to_dict(f"{p}/reward")
        out[f"{p}/is_trainable/mean"] = self.is_trainable.mean()
        out[f"{p}/is_filtered/mean"] = self.is_filtered.mean()
        out |= {f"{p}/filters/{k}/mean": v for k, v in self.filter_rates().items()}
        return out


class EvalMetrics(RolloutMetrics):
    """Common metrics plus the ``avg@<group_size>`` score and (on the effective subset, for
    binary-reward tasks) pass@k / pass^k. ``group_size`` (the ``avg@k`` k) is supplied by the
    container so the ``all`` and ``effective`` subsets share one stable key."""

    def __init__(self, rollouts: list[Rollout], group_size: int) -> None:
        super().__init__(rollouts)
        self.group_size = group_size

    @property
    def reward(self) -> Stat:
        """The policy's reward distribution (untrainable agents excluded — see :func:`_scored`)."""
        return Stat([float(r.reward) for r in _scored(self.rollouts)])

    def pass_at_k(self) -> dict[str, float]:
        """pass@k / pass^k averaged over (example, agent) over the policy view; ``{}`` for
        non-binary rewards."""
        scored = _scored(self.rollouts)
        if not {r.reward for r in scored}.issubset({0.0, 1.0}):
            return {}
        per_example = [compute_pass_metrics([r.reward for r in g]) for g in _by_group_and_agent(scored).values()]
        keys = sorted({k for d in per_example for k in d})
        return {k: sum(d[k] for d in per_example if k in d) / sum(1 for d in per_example if k in d) for k in keys}

    def to_wandb(self, *, prefix: str, subset: Subset) -> dict[str, float]:
        out = super().to_wandb(prefix=prefix, subset=subset)
        if not self.rollouts:
            return out
        p = f"{prefix}/{subset}"
        out[f"{p}/avg@{self.group_size}"] = self.reward.mean()
        if subset == "effective":
            out |= {f"{p}/{k}": v for k, v in self.pass_at_k().items()}
        return out


class TrainRollouts:
    """A list of train rollouts (everything that came back, errored + filtered + untrainable
    included). ``effective`` is the clean trainable subset (a view of the same traces);
    ``metrics`` builds ``TrainMetrics`` over them."""

    def __init__(self, rollouts: list[Rollout] | None = None) -> None:
        self.rollouts = rollouts if rollouts is not None else []

    def append(self, rollout: Rollout) -> None:
        self.rollouts.append(rollout)

    def __len__(self) -> int:
        return len(self.rollouts)

    def __iter__(self) -> Iterator[Rollout]:
        return iter(self.rollouts)

    @property
    def effective(self) -> TrainRollouts:
        return TrainRollouts([r for r in self.rollouts if not r.has_error and not r.is_filtered and r.trainable])

    def by_env(self) -> dict[str, TrainRollouts]:
        grouped: dict[str, list[Rollout]] = {}
        for r in self.rollouts:
            grouped.setdefault(r.env_name, []).append(r)
        return {env: TrainRollouts(rs) for env, rs in grouped.items()}

    def by_agent(self) -> dict[str, TrainRollouts]:
        """Per-agent slices, keyed by ``agent_name``. One key for a single-agent env; callers
        emit the slices only when there are several (see ``Orchestrator``)."""
        grouped: dict[str, list[Rollout]] = {}
        for r in self.rollouts:
            grouped.setdefault(r.agent_name or DEFAULT_AGENT_NAME, []).append(r)
        return {name: TrainRollouts(rs) for name, rs in grouped.items()}

    @property
    def metrics(self) -> TrainMetrics:
        return TrainMetrics(self.rollouts)


class EvalRollouts:
    """A list of eval rollouts (errored + untrainable included). ``effective`` is the
    non-errored trainable subset (a view).
    ``group_size`` (rollouts per example, the ``avg@k`` k) is derived from the full epoch and carried
    onto ``effective`` so both subsets share one stable key; ``metrics`` builds ``EvalMetrics``."""

    def __init__(self, rollouts: list[Rollout] | None = None, group_size: int | None = None) -> None:
        self.rollouts = rollouts if rollouts is not None else []
        self._group_size = group_size

    def __len__(self) -> int:
        return len(self.rollouts)

    def __iter__(self) -> Iterator[Rollout]:
        return iter(self.rollouts)

    @property
    def group_size(self) -> int:
        """The ``avg@k`` k. ``EvalSink`` supplies the env's configured group size and a subview
        carries its parent's, so ``k`` holds across subsets and across epochs that lost rollouts.

        Derived when nothing was supplied: the most *episodes* one example was rolled out for.
        Episodes, not traces — one attempt at an example is one episode however many agents it
        takes, so a two-agent env at ``group_size=4`` stays ``avg@4``."""
        if self._group_size is not None:
            return self._group_size
        episodes: dict = {}
        for r in self.rollouts:
            episodes.setdefault(r.group_id, set()).add(r.episode_id)
        return max((len(e) for e in episodes.values()), default=0)

    @property
    def effective(self) -> EvalRollouts:
        return EvalRollouts([r for r in self.rollouts if not r.has_error and r.trainable], group_size=self.group_size)

    def by_agent(self) -> dict[str, EvalRollouts]:
        """Per-agent slices, keyed by ``agent_name``, each carrying this pool's ``group_size`` so
        ``avg@k`` stays comparable across slices."""
        grouped: dict[str, list[Rollout]] = {}
        for r in self.rollouts:
            grouped.setdefault(r.agent_name or DEFAULT_AGENT_NAME, []).append(r)
        return {name: EvalRollouts(rs, group_size=self.group_size) for name, rs in grouped.items()}

    @property
    def metrics(self) -> EvalMetrics:
        return EvalMetrics(self.rollouts, self.group_size)


def agent_metrics(pool: TrainRollouts | EvalRollouts, *, prefix: str, subset: Subset) -> dict[str, float]:
    """The ``{prefix}/agent/{name}/{subset}/...`` slices for one pool: the per-agent breakdown the
    ``{agg,<env>}`` axes average over. A two-agent zero-sum env reports reward 0.0 at env level
    whatever each agent is doing, so for those envs this is the signal.

    ``{}`` unless the pool holds more than one agent — a single-agent slice would only restate the
    env's own keys. So a frozen-judge env emits slices on ``all`` but not on ``effective``, where
    dropping the untrainable agent leaves one."""
    pools = pool.by_agent()
    if len(pools) < 2:
        return {}
    out: dict[str, float] = {}
    for name, agent_pool in sorted(pools.items()):
        out |= agent_pool.metrics.to_wandb(prefix=f"{prefix}/agent/{name}", subset=subset)
    return out
