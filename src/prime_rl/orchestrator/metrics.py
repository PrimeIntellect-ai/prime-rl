"""Train and eval rollout metrics.

A rollout container (``TrainRollouts`` / ``EvalRollouts``) owns the rollout list and exposes
``.effective`` (the clean subset, as the same container type) and ``.metrics`` (``TrainMetrics`` /
``EvalMetrics``). The metrics object exposes each distributional / rate metric as a ``Stat`` — so
``rollouts.metrics.num_input_tokens.mean()`` works — and assembles the full
``{prefix}/{subset}/<metric>/<stat>`` wandb dict via ``.to_wandb(...)``.

The wandb layout mirrors the episode/trace hierarchy, one aggregation per level:

- ``{prefix}/{subset}/<metric>/<stat>`` — episode level, one value per episode summing its traces
  (matching ``vf.Episode``'s aggregates). Only the count metrics live here.
- ``{prefix}/{subset}/<agent>/<metric>/<stat>`` — agent level, one value per trace, grouped by agent
  name (``vf.Episode.by_agent``) so agents never mix into one distribution. Everything trace-scoped
  lives here: reward, rates, timing, custom metrics, the pipeline verdicts, the eval scores.

A single-agent env has one trace per episode, so the count metrics coincide across levels.

No I/O, no pandas — plain Python over the ``vf.Trace`` properties each rollout exposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

from prime_rl.orchestrator.utils import compute_pass_metrics

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import EvalEpisode, Rollout, TrainEpisode, TrainRollout

Subset = Literal["all", "effective"]


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

    PHASES = ("setup", "agent", "finalize", "scoring")

    @property
    def setup(self) -> Stat:
        return Stat([r.timing.setup.duration for r in self.rollouts])

    @property
    def agent(self) -> Stat:
        return Stat([r.timing.agent.duration for r in self.rollouts])

    @property
    def agent_model(self) -> Stat:
        """The share of the agent phase spent inside model calls (inference)."""
        return Stat([r.timing.agent.model.duration for r in self.rollouts])

    @property
    def agent_harness(self) -> Stat:
        """The share of the agent phase spent outside model calls (harness, tools,
        user simulation)."""
        return Stat([r.timing.agent.harness.duration for r in self.rollouts])

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
            "agent/model": self.agent_model,
            "agent/harness": self.agent_harness,
            "total": self.total,
        }


class CustomMetrics(StatGroup):
    """Per-key ``Stat``s over a dynamic per-rollout dict attribute (env ``@metric``s or reward
    components), each over the rollouts that carry the key. Scoring seeds every expected key
    with ``None`` before invoking it, so a ``None`` value means the signal never produced a
    score and counts as 0.0 — the ``effective`` subset excludes errored rollouts and gives the
    clean means. ``value`` extracts the float from each scored entry (rewards are ``vf.Reward``
    records; metrics are plain floats)."""

    def __init__(self, rollouts: list[Rollout], attr: str, value: Callable[[Any], float] = float) -> None:
        super().__init__(rollouts)
        self.attr = attr
        self.value = value

    def stats(self) -> dict[str, Stat]:
        names = sorted({name for r in self.rollouts for name in getattr(r, self.attr)})
        return {
            name: Stat(
                [
                    self.value(scores[name]) if scores[name] is not None else 0.0
                    for r in self.rollouts
                    if name in (scores := getattr(r, self.attr))
                ]
            )
            for name in names
        }


class TraceMetrics(StatGroup):
    """Trace-level metrics for one agent, every one of them flat over that agent's traces: one
    sample is one trace, so a fan-out (n same-agent traces in one episode, e.g. n solvers) simply
    contributes n samples. Weighing whole episodes against each other is the episode level's job;
    inside a seat the trace is the unit, which is also what the advantage computation samples."""

    DISTRIBUTIONS = ("reward", "num_total_tokens", "num_input_tokens", "num_output_tokens", "num_turns", "num_branches")
    RATES = ("is_truncated", "is_completed")

    def stats(self) -> dict[str, Stat]:
        return {
            name: Stat([float(getattr(r, name)) for r in self.rollouts]) for name in (*self.DISTRIBUTIONS, *self.RATES)
        }

    @property
    def timing(self) -> TimingMetrics:
        return TimingMetrics(self.rollouts)

    @property
    def metrics(self) -> CustomMetrics:
        """Env custom ``@metric`` outputs, keyed by name."""
        return CustomMetrics(self.rollouts, "metrics")

    @property
    def rewards(self) -> CustomMetrics:
        """Per-component reward breakdown, keyed by name (each entry's weighted ``value``,
        summed into the scalar ``reward``)."""
        return CustomMetrics(self.rollouts, "rewards", value=lambda reward: reward.value)

    @property
    def has_error(self) -> Stat:
        return Stat([float(r.has_error) for r in self.rollouts])

    def stop_conditions(self) -> dict[str, float]:
        """``generation_truncated`` over the agent's traces, then each recorded
        ``stop_condition``'s rate over the traces that recorded one."""
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
        """Count of errored traces by error type (the trace's last error — e.g. ``Cancelled``,
        ``ProviderError``)."""
        types = [r.last_error.type for r in self.rollouts if r.has_error and r.last_error is not None]
        return {t: types.count(t) for t in sorted(set(types))}

    def solve_rates(self) -> dict[str, float]:
        """Per-group solve rates over the agent's traces, assuming binary 0/1 rewards (unspecified
        for other reward ranges): ``solved_none`` (the group earned no reward), ``solved_all``
        (every trace scored 1.0), and ``solved_some`` (the mixed remainder — the GRPO-signal
        groups)."""
        groups: dict = {}
        for r in self.rollouts:
            groups.setdefault(r.group_id, []).append(r)
        n_groups = len(groups)
        solved_none = sum(1 for g in groups.values() if sum(r.reward for r in g) == 0)
        solved_all = sum(1 for g in groups.values() if all(r.reward == 1.0 for r in g))
        return {
            "solved_none": solved_none / n_groups,
            "solved_all": solved_all / n_groups,
            "solved_some": 1 - (solved_none + solved_all) / n_groups,
        }

    def to_dict(self, prefix: str, *, subset: Subset) -> dict[str, float]:
        """Full ``<stat>`` fan-out for the distributions; ``/mean`` only for the 0/1 rates.
        Errors live only on the ``all`` subset (``effective`` drops them by construction)."""
        stats = self.stats()
        out: dict[str, float] = {}
        for name in self.DISTRIBUTIONS:
            out |= stats[name].to_dict(f"{prefix}/{name}")
        for name in self.RATES:
            out[f"{prefix}/{name}/mean"] = stats[name].mean()
        out |= self.timing.to_dict(f"{prefix}/timing")
        out |= self.metrics.to_dict(f"{prefix}/metrics")
        out |= self.rewards.to_dict(f"{prefix}/rewards")
        if subset == "all":
            out[f"{prefix}/has_error/mean"] = self.has_error.mean()
            out |= {f"{prefix}/error/{t}": float(count) for t, count in self.error_types().items()}
        out |= {f"{prefix}/stop_condition/{k}": v for k, v in self.stop_conditions().items()}
        out |= {f"{prefix}/{k}": v for k, v in self.solve_rates().items()}
        return out


class EpisodeMetrics:
    """Metrics shared by train and eval over a list of episodes. The count metrics (tokens/turns/
    branches) are read straight off ``vf.Episode``, which already sums an episode's traces, and are
    the only per-metric keys ``to_wandb`` emits at the env level; every trace-level metric is
    emitted per agent via ``by_agent()``. The boolean ``Stat`` properties (0/1 distributions,
    ``.mean()`` is the rate) serve the console log lines. ``TrainMetrics`` / ``EvalMetrics`` extend
    ``to_wandb`` with the train pipeline rates and the eval scores."""

    def __init__(self, episodes: list[Episode]) -> None:
        self.episodes = episodes
        self.rollouts: list[Rollout] = [t for e in episodes for t in e.traces]

    def by_agent(self) -> dict[str, TraceMetrics]:
        """Per-agent metric views, merging each episode's own ``vf.Episode.by_agent`` grouping."""
        per_agent: dict[str, list[Rollout]] = {}
        for episode in self.episodes:
            for name, traces in episode.by_agent.items():
                per_agent.setdefault(name, []).extend(traces)
        return {name: TraceMetrics(rollouts) for name, rollouts in sorted(per_agent.items())}

    # Episode-level count metrics, one value per episode — ``vf.Episode``'s own aggregates.
    @property
    def num_total_tokens(self) -> Stat:
        return Stat([float(e.num_total_tokens) for e in self.episodes])

    @property
    def num_input_tokens(self) -> Stat:
        return Stat([float(e.num_input_tokens) for e in self.episodes])

    @property
    def num_output_tokens(self) -> Stat:
        return Stat([float(e.num_output_tokens) for e in self.episodes])

    @property
    def num_turns(self) -> Stat:
        return Stat([float(e.num_turns) for e in self.episodes])

    @property
    def num_branches(self) -> Stat:
        return Stat([float(sum(t.num_branches for t in e.traces)) for e in self.episodes])

    # Boolean rate metrics for the console log lines (0/1 distributions — ``.mean()`` is the
    # rate); to_wandb emits their per-agent counterparts instead.
    @property
    def is_truncated(self) -> Stat:
        return Stat([float(r.is_truncated) for r in self.rollouts])

    @property
    def has_error(self) -> Stat:
        return Stat([float(r.has_error) for r in self.rollouts])

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
        for agent, agent_metrics in self.by_agent().items():
            out |= agent_metrics.to_dict(f"{p}/{agent}", subset=subset)
        return out


class TrainMetrics(EpisodeMetrics):
    """Common metrics plus the per-agent filter-pipeline rates. ``reward`` (flat over all traces)
    serves the console log lines; the wandb reward stats are per-agent."""

    @property
    def reward(self) -> Stat:
        return Stat([float(r.reward) for r in self.rollouts])

    def to_wandb(self, *, prefix: str, subset: Subset) -> dict[str, float]:
        out = super().to_wandb(prefix=prefix, subset=subset)
        # The pipeline verdicts are per-trace (an untrainable seat is 0.0 throughout, and filters
        # only ever run on trainable survivors), so they read per agent like the rest.
        for agent, traces in self.by_agent().items():
            p = f"{prefix}/{subset}/{agent}"
            rollouts = traces.rollouts
            out[f"{p}/is_trainable/mean"] = sum(float(r.is_trainable) for r in rollouts) / len(rollouts)
            out[f"{p}/is_filtered/mean"] = sum(float(r.is_filtered) for r in rollouts) / len(rollouts)
            names = sorted({name for r in rollouts for name in r.filter_results})
            out |= {
                f"{p}/filters/{name}/mean": sum(1 for r in rollouts if r.filter_results.get(name)) / len(rollouts)
                for name in names
            }
        return out


def pass_at_k(rollouts: list[Rollout]) -> dict[str, float]:
    """pass@k / pass^k averaged over examples; ``{}`` for non-binary rewards."""
    rewards = [r.reward for r in rollouts]
    if not set(rewards).issubset({0.0, 1.0}):
        return {}
    by_example: dict = {}
    for r in rollouts:
        by_example.setdefault(r.group_id, []).append(r.reward)
    per_example = [compute_pass_metrics(rs) for rs in by_example.values()]
    keys = sorted({k for d in per_example for k in d})
    return {k: sum(d[k] for d in per_example if k in d) / sum(1 for d in per_example if k in d) for k in keys}


class EvalMetrics(EpisodeMetrics):
    """Common metrics plus the per-agent ``avg@<group_size>`` score and (on the effective subset,
    for binary-reward tasks) pass@k / pass^k. Both score an agent's own traces, so they live in its
    subtree like every other trace-level metric. ``group_size`` (the ``avg@k`` k, the run's rollouts
    per example) is supplied by the container so the ``all`` and ``effective`` subsets — and every
    agent — share one stable key."""

    def __init__(self, episodes: list[Episode], group_size: int) -> None:
        super().__init__(episodes)
        self.group_size = group_size

    @property
    def reward(self) -> Stat:
        return Stat([float(r.reward) for r in self.rollouts])

    def to_wandb(self, *, prefix: str, subset: Subset) -> dict[str, float]:
        out = super().to_wandb(prefix=prefix, subset=subset)
        for agent, traces in self.by_agent().items():
            p = f"{prefix}/{subset}/{agent}"
            out[f"{p}/avg@{self.group_size}"] = traces.stats()["reward"].mean()
            if subset == "effective":
                out |= {f"{p}/{k}": v for k, v in pass_at_k(traces.rollouts).items()}
        return out


def keep_traces(episode: Episode, keep: Callable[[Rollout], bool]) -> Episode | None:
    """The episode narrowed to the traces that pass ``keep``, or ``None`` if none do. A subset view
    stays a list of episodes so the episode-level aggregates keep describing what survived."""
    traces = [t for t in episode.traces if keep(t)]
    return episode.model_copy(update={"traces": traces}) if traces else None


class TrainRollouts:
    """The train episodes of one window (everything that came back, errored + filtered +
    untrainable included). ``effective`` is the clean trainable subset — the same episodes, each
    narrowed to its surviving traces; ``metrics`` builds ``TrainMetrics`` over them. Sized and
    iterated by *rollout*, since that is the unit almost every consumer wants."""

    def __init__(self, episodes: list[TrainEpisode] | None = None) -> None:
        self.episodes = episodes if episodes is not None else []

    def append(self, episode: TrainEpisode) -> None:
        self.episodes.append(episode)

    @property
    def rollouts(self) -> list[TrainRollout]:
        return [t for e in self.episodes for t in e.rollouts]

    def __len__(self) -> int:
        return sum(len(e.traces) for e in self.episodes)

    def __iter__(self) -> Iterator[TrainRollout]:
        return iter(self.rollouts)

    @property
    def effective(self) -> TrainRollouts:
        kept = (
            keep_traces(e, lambda r: not r.has_error and not r.is_filtered and r.agent.trainable) for e in self.episodes
        )
        return TrainRollouts([e for e in kept if e is not None])

    def by_env(self) -> dict[str, TrainRollouts]:
        grouped: dict[str, list[TrainEpisode]] = {}
        for episode in self.episodes:
            grouped.setdefault(episode.env_name, []).append(episode)
        return {env: TrainRollouts(episodes) for env, episodes in grouped.items()}

    @property
    def metrics(self) -> TrainMetrics:
        return TrainMetrics(self.episodes)


class EvalRollouts:
    """The eval episodes of one epoch (errored + untrainable included). ``effective`` is the
    non-errored trainable subset — the same episodes, each narrowed to its surviving traces.
    ``group_size`` (rollouts per example, the ``avg@k`` k) is derived from the full epoch and carried
    onto ``effective`` so both subsets share one stable key; ``metrics`` builds ``EvalMetrics``."""

    def __init__(self, episodes: list[EvalEpisode] | None = None, group_size: int | None = None) -> None:
        self.episodes = episodes if episodes is not None else []
        self._group_size = group_size

    @property
    def rollouts(self) -> list[Rollout]:
        return [t for e in self.episodes for t in e.traces]

    def __len__(self) -> int:
        return sum(len(e.traces) for e in self.episodes)

    def __iter__(self) -> Iterator[Rollout]:
        return iter(self.rollouts)

    @property
    def group_size(self) -> int:
        """The largest group in trainable (policy) traces — equals the configured group size
        whenever one example kept all its rollouts; untrainable traces don't count, so a
        multi-agent episode doesn't inflate ``k``. A subview carries its parent's value so
        ``avg@k`` doesn't drift across subsets."""
        if self._group_size is not None:
            return self._group_size
        counts: dict = {}
        for r in self.rollouts:
            if r.agent.trainable:
                counts[r.group_id] = counts.get(r.group_id, 0) + 1
        return max(counts.values(), default=0)

    @property
    def effective(self) -> EvalRollouts:
        kept = (keep_traces(e, lambda r: not r.has_error and r.agent.trainable) for e in self.episodes)
        return EvalRollouts([e for e in kept if e is not None], group_size=self.group_size)

    @property
    def metrics(self) -> EvalMetrics:
        return EvalMetrics(self.episodes, self.group_size)
