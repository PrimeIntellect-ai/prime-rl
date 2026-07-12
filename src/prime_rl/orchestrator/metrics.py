"""Train and eval graph metrics.

A graph container (``TrainGraphs`` / ``EvalGraphs``) owns the graph list and exposes
``.effective`` (the clean subset, as the same container type) and ``.metrics`` (``TrainMetrics`` /
``EvalMetrics``). The metrics object exposes each distributional / rate metric as a ``Stat`` — so
``graphs.metrics.num_input_tokens.mean()`` works — and assembles the full
``{prefix}/{subset}/<metric>/<stat>`` wandb dict via ``.to_wandb(...)``.

No I/O, no pandas — plain Python over the sole trainable trace projected by each graph. Aggregation
is flat over the graph list except the solve rates, which group by ``group_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Literal

from prime_rl.orchestrator.utils import compute_pass_metrics

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import AgentGraph

Subset = Literal["all", "effective"]


class Stat:
    """A distribution of per-graph values with mean/max/min and p10/p90 accessors."""

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

    def __init__(self, graphs: list[AgentGraph]) -> None:
        self.graphs = graphs

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
    """Per-phase graph durations, nested so ``metrics.timing.setup.mean()`` reads naturally.
    ``total`` is the per-graph sum across all phases."""

    PHASES = ("setup", "generation", "finalize", "scoring")

    @property
    def setup(self) -> Stat:
        return Stat([r.timing.setup.duration for r in self.graphs])

    @property
    def generation(self) -> Stat:
        return Stat([r.timing.generation.duration for r in self.graphs])

    @property
    def finalize(self) -> Stat:
        return Stat([r.timing.finalize.duration for r in self.graphs])

    @property
    def scoring(self) -> Stat:
        return Stat([r.timing.scoring.duration for r in self.graphs])

    @property
    def total(self) -> Stat:
        return Stat([sum(getattr(r.timing, p).duration for p in self.PHASES) for r in self.graphs])

    def stats(self) -> dict[str, Stat]:
        return {**{phase: getattr(self, phase) for phase in self.PHASES}, "total": self.total}


class CustomMetrics(StatGroup):
    """Per-key ``Stat``s over a dynamic per-graph dict attribute (env ``@metric``s or reward
    components), each averaged over the graphs that report the key."""

    def __init__(self, graphs: list[AgentGraph], attr: str) -> None:
        super().__init__(graphs)
        self.attr = attr

    def stats(self) -> dict[str, Stat]:
        names = sorted({name for r in self.graphs for name in getattr(r, self.attr)})
        return {
            name: Stat([getattr(r, self.attr)[name] for r in self.graphs if name in getattr(r, self.attr)])
            for name in names
        }


class GraphMetrics:
    """Metrics shared by train and eval over a graph list. Distributional metrics are ``Stat``s
    (mean/max/min); boolean metrics are ``Stat``s of 0/1 (use ``.mean()`` for the rate). ``to_wandb``
    assembles the full ``{prefix}/{subset}/...`` dict; ``TrainMetrics`` / ``EvalMetrics`` extend it."""

    def __init__(self, graphs: list[AgentGraph]) -> None:
        self.graphs = graphs

    # Distributional metrics
    @property
    def num_total_tokens(self) -> Stat:
        return Stat([float(r.num_total_tokens) for r in self.graphs])

    @property
    def num_input_tokens(self) -> Stat:
        return Stat([float(r.num_input_tokens) for r in self.graphs])

    @property
    def num_output_tokens(self) -> Stat:
        return Stat([float(r.num_output_tokens) for r in self.graphs])

    @property
    def num_turns(self) -> Stat:
        return Stat([float(r.num_turns) for r in self.graphs])

    @property
    def num_branches(self) -> Stat:
        return Stat([float(r.num_branches) for r in self.graphs])

    @property
    def timing(self) -> TimingMetrics:
        return TimingMetrics(self.graphs)

    @property
    def metrics(self) -> CustomMetrics:
        """Env custom ``@metric`` outputs, keyed by name (``metrics.metrics["acc"].mean()``)."""
        return CustomMetrics(self.graphs, "metrics")

    @property
    def rewards(self) -> CustomMetrics:
        """Per-component reward breakdown, keyed by name (summed into the scalar ``reward``)."""
        return CustomMetrics(self.graphs, "rewards")

    # Boolean rate metrics (0/1 distributions — ``.mean()`` is the rate)
    @property
    def is_truncated(self) -> Stat:
        return Stat([float(r.is_truncated) for r in self.graphs])

    @property
    def is_completed(self) -> Stat:
        return Stat([float(r.is_completed) for r in self.graphs])

    @property
    def has_error(self) -> Stat:
        return Stat([float(r.has_error) for r in self.graphs])

    def stop_conditions(self) -> dict[str, float]:
        """``generation_truncated`` over all graphs, then each recorded ``stop_condition``'s rate
        over the graphs that recorded one."""
        out = {
            "generation_truncated": sum(
                1 for r in self.graphs if r.is_truncated and r.stop_condition != "prompt_too_long"
            )
            / len(self.graphs)
        }
        conditions = [r.stop_condition for r in self.graphs if r.stop_condition is not None]
        for condition in sorted(set(conditions)):
            out[condition] = conditions.count(condition) / len(conditions)
        return out

    def error_types(self) -> dict[str, int]:
        """Count of errored graphs by error type (the graph's failure — e.g. ``Cancelled``,
        ``ProviderError``)."""
        types = [r.failure.type for r in self.graphs if r.failure is not None]
        return {t: types.count(t) for t in sorted(set(types))}

    def solve_rates(self) -> dict[str, float]:
        """Per-group solve rates, assuming binary 0/1 rewards (unspecified for other reward ranges):
        ``solved_none`` (the group earned no reward), ``solved_all`` (every graph scored 1.0), and
        ``solved_some`` (the mixed remainder — the GRPO-signal groups)."""
        groups: dict = {}
        for r in self.graphs:
            groups.setdefault(r.group_id, []).append(r)
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
        if not self.graphs:
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


class TrainMetrics(GraphMetrics):
    """Common metrics plus the reward distribution and filter-pipeline rates."""

    @property
    def reward(self) -> Stat:
        return Stat([float(r.reward) for r in self.graphs])

    @property
    def is_trainable(self) -> Stat:
        return Stat([float(r.is_trainable) for r in self.graphs])

    @property
    def is_filtered(self) -> Stat:
        return Stat([float(r.is_filtered) for r in self.graphs])

    def filter_rates(self) -> dict[str, float]:
        """Per-filter detection rate over all graphs."""
        names = sorted({name for r in self.graphs for name in r.filter_results})
        return {name: sum(1 for r in self.graphs if r.filter_results.get(name)) / len(self.graphs) for name in names}

    def to_wandb(self, *, prefix: str, subset: Subset) -> dict[str, float]:
        out = super().to_wandb(prefix=prefix, subset=subset)
        if not self.graphs:
            return out
        p = f"{prefix}/{subset}"
        out |= self.reward.to_dict(f"{p}/reward")
        out[f"{p}/is_trainable/mean"] = self.is_trainable.mean()
        out[f"{p}/is_filtered/mean"] = self.is_filtered.mean()
        out |= {f"{p}/filters/{k}/mean": v for k, v in self.filter_rates().items()}
        return out


class EvalMetrics(GraphMetrics):
    """Common metrics plus the ``avg@<group_size>`` score and (on the effective subset, for
    binary-reward tasks) pass@k / pass^k. ``group_size`` (the ``avg@k`` k) is supplied by the
    container so the ``all`` and ``effective`` subsets share one stable key."""

    def __init__(self, graphs: list[AgentGraph], group_size: int) -> None:
        super().__init__(graphs)
        self.group_size = group_size

    @property
    def reward(self) -> Stat:
        return Stat([float(r.reward) for r in self.graphs])

    def pass_at_k(self) -> dict[str, float]:
        """pass@k / pass^k averaged over examples; ``{}`` for non-binary rewards."""
        rewards = [r.reward for r in self.graphs]
        if not set(rewards).issubset({0.0, 1.0}):
            return {}
        by_example: dict = {}
        for r in self.graphs:
            by_example.setdefault(r.group_id, []).append(r.reward)
        per_example = [compute_pass_metrics(rs) for rs in by_example.values()]
        keys = sorted({k for d in per_example for k in d})
        return {k: sum(d[k] for d in per_example if k in d) / sum(1 for d in per_example if k in d) for k in keys}

    def to_wandb(self, *, prefix: str, subset: Subset) -> dict[str, float]:
        out = super().to_wandb(prefix=prefix, subset=subset)
        if not self.graphs:
            return out
        p = f"{prefix}/{subset}"
        out[f"{p}/avg@{self.group_size}"] = self.reward.mean()
        if subset == "effective":
            out |= {f"{p}/{k}": v for k, v in self.pass_at_k().items()}
        return out


class TrainGraphs:
    """Training graphs returned during one batch window."""

    def __init__(self, graphs: list[AgentGraph] | None = None) -> None:
        self.graphs = graphs if graphs is not None else []

    def append(self, graph: AgentGraph) -> None:
        self.graphs.append(graph)

    def __len__(self) -> int:
        return len(self.graphs)

    def __iter__(self) -> Iterator[AgentGraph]:
        return iter(self.graphs)

    @property
    def effective(self) -> TrainGraphs:
        return TrainGraphs([graph for graph in self.graphs if not graph.has_error and not graph.is_filtered])

    def by_env(self) -> dict[str, TrainGraphs]:
        grouped: dict[str, list[AgentGraph]] = {}
        for graph in self.graphs:
            grouped.setdefault(graph.env_name, []).append(graph)
        return {env: TrainGraphs(graphs) for env, graphs in grouped.items()}

    @property
    def metrics(self) -> TrainMetrics:
        return TrainMetrics(self.graphs)


class EvalGraphs:
    """Evaluation graphs returned during one eval epoch."""

    def __init__(self, graphs: list[AgentGraph] | None = None, group_size: int | None = None) -> None:
        self.graphs = graphs if graphs is not None else []
        self._group_size = group_size

    def __len__(self) -> int:
        return len(self.graphs)

    def __iter__(self) -> Iterator[AgentGraph]:
        return iter(self.graphs)

    @property
    def group_size(self) -> int:
        """The largest group (equals the configured group size whenever one example kept all its
        graphs); a subview carries its parent's value so ``avg@k`` doesn't drift across subsets."""
        if self._group_size is not None:
            return self._group_size
        counts: dict = {}
        for graph in self.graphs:
            counts[graph.group_id] = counts.get(graph.group_id, 0) + 1
        return max(counts.values(), default=0)

    @property
    def effective(self) -> EvalGraphs:
        return EvalGraphs([graph for graph in self.graphs if not graph.has_error], group_size=self.group_size)

    @property
    def metrics(self) -> EvalMetrics:
        return EvalMetrics(self.graphs, self.group_size)
