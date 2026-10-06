"""Episode-native train and evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import verifiers.v1 as vf

from prime_rl.orchestrator.algo.routing import is_trainable, scalar_advantage
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation
from prime_rl.orchestrator.utils import compute_pass_metrics, episode_env_name, episode_group_id

Subset = Literal["all", "effective"]


def dispatch_failure_metrics(failures: list[DispatchFailure], *, prefix: str, total_attempts: int) -> dict[str, float]:
    """Metrics for requests that failed outside the verifier episode boundary."""
    if total_attempts == 0:
        return {}
    out = {f"{prefix}/dispatch_failure/mean": len(failures) / total_attempts}
    error_types = [failure.error.type for failure in failures]
    out |= {
        f"{prefix}/dispatch_failure/{error_type}": float(error_types.count(error_type))
        for error_type in sorted(set(error_types))
    }
    return out


@dataclass(frozen=True)
class TraceRecord:
    """A trace joined with its episode and training selection state.
    ``cancelled`` marks episodes the pipeline voided (stale drop) — a
    selection verdict disjoint from trace errors."""

    episode: vf.Episode
    trace: vf.Trace
    sampled: bool
    admitted: bool
    cancelled: bool


def _records(
    episodes: list[vf.Episode],
    sampled_trace_ids: set[str],
    admitted: set[str],
    cancelled: set[str],
) -> list[TraceRecord]:
    return [
        TraceRecord(
            episode=episode,
            trace=trace,
            sampled=trace.id in sampled_trace_ids,
            admitted=episode.id in admitted,
            cancelled=episode.id in cancelled,
        )
        for episode in episodes
        for trace in episode.traces
    ]


class Stat:
    """A distribution with mean, extrema, and percentile accessors."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def max(self) -> float:
        return float(max(self.values)) if self.values else 0.0

    def min(self) -> float:
        return float(min(self.values)) if self.values else 0.0

    def percentile(self, q: float) -> float:
        if not self.values:
            return 0.0
        values = sorted(self.values)
        rank = q / 100 * (len(values) - 1)
        low = int(rank)
        high = min(low + 1, len(values) - 1)
        return float(values[low] + (values[high] - values[low]) * (rank - low))

    def p10(self) -> float:
        return self.percentile(10)

    def p90(self) -> float:
        return self.percentile(90)

    def to_dict(self, prefix: str) -> dict[str, float]:
        if not self.values:
            return {}
        return {
            f"{prefix}/mean": self.mean(),
            f"{prefix}/max": self.max(),
            f"{prefix}/min": self.min(),
            f"{prefix}/p10": self.p10(),
            f"{prefix}/p90": self.p90(),
        }


@dataclass
class MetricWindow:
    """Mergeable numeric observations, owned by a finite reporting window.

    Distributions retain scalars for exact percentiles. Rate families share a
    denominator so categories absent from a group still count toward its total.
    No episode, trace, or error objects survive extraction.
    """

    distributions: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)
    rates: dict[str, tuple[dict[str, float], int]] = field(default_factory=dict)

    def distribution(self, key: str, stat: Stat) -> None:
        self.distributions.setdefault(key, []).extend(stat.values)

    def rate(self, prefix: str, counts: dict[str, float], total: int) -> None:
        current, denominator = self.rates.get(prefix, ({}, 0))
        for name, count in counts.items():
            current[name] = current.get(name, 0.0) + count
        self.rates[prefix] = (current, denominator + total)

    def count(self, key: str, value: float) -> None:
        self.counts[key] = self.counts.get(key, 0.0) + value

    def extend(self, other: MetricWindow) -> None:
        for key, values in other.distributions.items():
            self.distributions.setdefault(key, []).extend(values)
        for key, value in other.counts.items():
            self.count(key, value)
        for prefix, (counts, total) in other.rates.items():
            self.rate(prefix, counts, total)

    def to_dict(self) -> dict[str, float]:
        out = dict(self.counts)
        for key, values in self.distributions.items():
            out |= Stat(values).to_dict(key)
        for prefix, (counts, total) in self.rates.items():
            out |= {
                f"{prefix}/{name}" if name else prefix: count / total if total else 0.0
                for name, count in counts.items()
            }
        return out


class StatGroup:
    def __init__(self, records: list[TraceRecord]) -> None:
        self.records = records

    @property
    def traces(self) -> list[vf.Trace]:
        return [record.trace for record in self.records]

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
    PHASES = ("setup", "agent", "finalize", "scoring")

    @property
    def setup(self) -> Stat:
        return Stat([trace.timing.setup.duration for trace in self.traces])

    @property
    def agent(self) -> Stat:
        return Stat([trace.timing.agent.duration for trace in self.traces])

    @property
    def agent_model(self) -> Stat:
        return Stat([trace.timing.agent.model.duration for trace in self.traces])

    @property
    def agent_harness(self) -> Stat:
        return Stat([trace.timing.agent.harness.duration for trace in self.traces])

    @property
    def finalize(self) -> Stat:
        return Stat([trace.timing.finalize.duration for trace in self.traces])

    @property
    def scoring(self) -> Stat:
        return Stat([trace.timing.scoring.duration for trace in self.traces])

    @property
    def total(self) -> Stat:
        return Stat([sum(getattr(trace.timing, phase).duration for phase in self.PHASES) for trace in self.traces])

    def stats(self) -> dict[str, Stat]:
        return {
            **{phase: getattr(self, phase) for phase in self.PHASES},
            "agent/model": self.agent_model,
            "agent/harness": self.agent_harness,
            "total": self.total,
        }


class CustomMetrics(StatGroup):
    def __init__(
        self,
        records: list[TraceRecord],
        attr: str,
        value: Callable[[Any], float] = float,
    ) -> None:
        super().__init__(records)
        self.attr = attr
        self.value = value

    def stats(self) -> dict[str, Stat]:
        names = sorted({name for trace in self.traces for name in getattr(trace, self.attr)})
        return {
            name: Stat(
                [
                    self.value(scores[name]) if scores[name] is not None else 0.0
                    for trace in self.traces
                    if name in (scores := getattr(trace, self.attr))
                ]
            )
            for name in names
        }


class TraceMetrics(StatGroup):
    DISTRIBUTIONS = (
        "reward",
        "num_total_tokens",
        "num_input_tokens",
        "num_output_tokens",
        "num_turns",
        "num_branches",
    )
    RATES = ("is_truncated", "is_timeout", "is_completed")

    def stats(self) -> dict[str, Stat]:
        return {
            name: Stat([float(getattr(trace, name)) for trace in self.traces])
            for name in (*self.DISTRIBUTIONS, *self.RATES)
        }

    @property
    def timing(self) -> TimingMetrics:
        return TimingMetrics(self.records)

    @property
    def metrics(self) -> CustomMetrics:
        return CustomMetrics(self.records, "metrics")

    @property
    def rewards(self) -> CustomMetrics:
        return CustomMetrics(self.records, "rewards", value=lambda reward: reward.value)

    @property
    def has_error(self) -> Stat:
        return Stat([float(trace.has_error) for trace in self.traces])

    @property
    def cancelled(self) -> Stat:
        """Pipeline cancellations (stale drop) — disjoint from errors."""
        return Stat([float(record.cancelled) for record in self.records])

    def error_types(self) -> dict[str, int]:
        types = [trace.last_error.type for trace in self.traces if trace.has_error and trace.last_error is not None]
        return {error_type: types.count(error_type) for error_type in sorted(set(types))}

    def solve_rates(self) -> dict[str, float]:
        groups: dict = {}
        for record in self.records:
            groups.setdefault(episode_group_id(record.episode), []).append(record.trace)
        num_groups = len(groups)
        solved_none = sum(sum(trace.reward for trace in group) == 0 for group in groups.values())
        solved_all = sum(all(trace.reward == 1.0 for trace in group) for group in groups.values())
        return {
            "solved_none": solved_none / num_groups,
            "solved_all": solved_all / num_groups,
            "solved_some": 1 - (solved_none + solved_all) / num_groups,
        }

    def snapshot(self, prefix: str, *, subset: Subset) -> MetricWindow:
        stats = self.stats()
        out = MetricWindow()
        for name in self.DISTRIBUTIONS:
            out.distribution(f"{prefix}/{name}", stats[name])
        for name in self.RATES:
            out.rate(f"{prefix}/{name}", {"mean": sum(stats[name].values)}, len(stats[name].values))
        for name, group in (("timing", self.timing), ("metrics", self.metrics), ("rewards", self.rewards)):
            for key, stat in group.stats().items():
                out.distribution(f"{prefix}/{name}/{key}", stat)
        if subset == "all":
            out.rate(f"{prefix}/has_error", {"mean": sum(self.has_error.values)}, len(self.records))
            out.rate(f"{prefix}/cancelled", {"mean": sum(self.cancelled.values)}, len(self.records))
            for key, value in self.error_types().items():
                out.count(f"{prefix}/error/{key}", float(value))
            num_groups = len({episode_group_id(record.episode) for record in self.records})
            out.rate(prefix, {key: value * num_groups for key, value in self.solve_rates().items()}, num_groups)
        conditions = [trace.stop_condition for trace in self.traces if trace.stop_condition is not None]
        out.rate(
            f"{prefix}/stop_condition",
            {condition: float(conditions.count(condition)) for condition in set(conditions)},
            len(conditions),
        )
        # Truncation uses all traces; named stop conditions use only recorded conditions.
        out.rate(
            f"{prefix}/stop_condition/generation_truncated",
            {"": sum(trace.is_truncated and trace.stop_condition != "prompt_too_long" for trace in self.traces)},
            len(self.records),
        )
        return out

    def to_dict(self, prefix: str, *, subset: Subset) -> dict[str, float]:
        return self.snapshot(prefix, subset=subset).to_dict()


class EpisodeMetrics:
    def __init__(
        self,
        episodes: list[vf.Episode],
        records: list[TraceRecord],
        cancelled: set[str] | None = None,
    ) -> None:
        self.episodes = episodes
        self.records = records
        self.cancelled_ids = cancelled if cancelled is not None else set()

    def by_agent(self) -> dict[str, TraceMetrics]:
        per_agent: dict[str, list[TraceRecord]] = {}
        for record in self.records:
            per_agent.setdefault(record.trace.agent.name, []).append(record)
        return {name: TraceMetrics(records) for name, records in sorted(per_agent.items())}

    def _episode_stat(self, attr: str) -> Stat:
        by_episode = {id(episode): 0.0 for episode in self.episodes}
        for record in self.records:
            by_episode[id(record.episode)] += float(getattr(record.trace, attr))
        return Stat(list(by_episode.values()))

    @property
    def num_total_tokens(self) -> Stat:
        return self._episode_stat("num_total_tokens")

    @property
    def num_input_tokens(self) -> Stat:
        return self._episode_stat("num_input_tokens")

    @property
    def num_output_tokens(self) -> Stat:
        return self._episode_stat("num_output_tokens")

    @property
    def num_turns(self) -> Stat:
        return self._episode_stat("num_turns")

    @property
    def num_branches(self) -> Stat:
        return self._episode_stat("num_branches")

    @property
    def is_truncated(self) -> Stat:
        return Stat([float(record.trace.is_truncated) for record in self.records])

    @property
    def is_timeout(self) -> Stat:
        return Stat([float(record.trace.is_timeout) for record in self.records])

    @property
    def has_error(self) -> Stat:
        return Stat(
            [float(not episode.ok or any(trace.has_error for trace in episode.traces)) for episode in self.episodes]
        )

    @property
    def cancelled(self) -> Stat:
        """Per-episode pipeline-cancellation rate — same denominator as
        ``has_error``. Read from the id-set, not the trace records, so
        trace-less episodes still count."""
        return Stat([float(episode.id in self.cancelled_ids) for episode in self.episodes])

    def snapshot(self, *, prefix: str, subset: Subset) -> MetricWindow:
        out = MetricWindow()
        if not self.episodes:
            return out
        metric_prefix = f"{prefix}/{subset}"
        for name in ("num_total_tokens", "num_input_tokens", "num_output_tokens", "num_turns", "num_branches"):
            out.distribution(f"{metric_prefix}/{name}", getattr(self, name))
        if subset == "all":
            out.rate(f"{metric_prefix}/has_error", {"mean": sum(self.has_error.values)}, len(self.episodes))
            out.rate(f"{metric_prefix}/cancelled", {"mean": sum(self.cancelled.values)}, len(self.episodes))
        for agent, metrics in self.by_agent().items():
            out.extend(metrics.snapshot(f"{metric_prefix}/{agent}", subset=subset))
        return out

    def to_wandb(self, *, prefix: str, subset: Subset) -> dict[str, float]:
        return self.snapshot(prefix=prefix, subset=subset).to_dict()


class TrainMetrics(EpisodeMetrics):
    @property
    def reward(self) -> Stat:
        return Stat([float(record.trace.reward) for record in self.records])

    def snapshot(self, *, prefix: str, subset: Subset) -> MetricWindow:
        out = super().snapshot(prefix=prefix, subset=subset)
        for agent, traces in self.by_agent().items():
            metric_prefix = f"{prefix}/{subset}/{agent}"
            out.rate(
                f"{metric_prefix}/is_trainable",
                {"mean": sum(float(is_trainable(record.trace)) for record in traces.records)},
                len(traces.records),
            )
            out.rate(
                f"{metric_prefix}/is_admitted",
                {"mean": sum(float(record.admitted) for record in traces.records)},
                len(traces.records),
            )
        return out


def pass_at_k(records: list[TraceRecord]) -> dict[str, float]:
    rewards = [record.trace.reward for record in records]
    if not set(rewards).issubset({0.0, 1.0}):
        return {}
    by_example: dict = {}
    for record in records:
        by_example.setdefault(episode_group_id(record.episode), []).append(record.trace.reward)
    per_example = [compute_pass_metrics(group) for group in by_example.values()]
    keys = sorted({key for result in per_example for key in result})
    return {
        key: sum(result[key] for result in per_example if key in result) / sum(key in result for result in per_example)
        for key in keys
    }


class EvalMetrics(EpisodeMetrics):
    def __init__(self, episodes: list[vf.Episode], records: list[TraceRecord], group_size: int) -> None:
        super().__init__(episodes, records)
        self.group_size = group_size

    @property
    def reward(self) -> Stat:
        return Stat([float(record.trace.reward) for record in self.records])

    def to_wandb(self, *, prefix: str, subset: Subset) -> dict[str, float]:
        out = super().to_wandb(prefix=prefix, subset=subset)
        for agent, traces in self.by_agent().items():
            metric_prefix = f"{prefix}/{subset}/{agent}"
            out[f"{metric_prefix}/avg@{self.group_size}"] = traces.stats()["reward"].mean()
            if subset == "effective":
                out |= {f"{metric_prefix}/{key}": value for key, value in pass_at_k(traces.records).items()}
        return out


class EpisodeCollection:
    def __init__(
        self,
        episodes: list[vf.Episode] | None = None,
        sampled_trace_ids: set[str] | None = None,
        admitted: set[str] | None = None,
        cancelled: set[str] | None = None,
        predicate: Callable[[TraceRecord], bool] | None = None,
    ) -> None:
        self.episodes = episodes if episodes is not None else []
        self.sampled_trace_ids = sampled_trace_ids if sampled_trace_ids is not None else set()
        self.admitted = admitted if admitted is not None else {episode.id for episode in self.episodes}
        self.cancelled = cancelled if cancelled is not None else set()
        self._predicate = predicate

    @property
    def records(self) -> list[TraceRecord]:
        records = _records(self.episodes, self.sampled_trace_ids, self.admitted, self.cancelled)
        if self._predicate is None:
            return records
        return [record for record in records if self._predicate(record)]

    @property
    def selected_episodes(self) -> list[vf.Episode]:
        if self._predicate is None:
            return self.episodes
        selected = {id(record.episode) for record in self.records}
        return [episode for episode in self.episodes if id(episode) in selected]

    @property
    def num_traces(self) -> int:
        return len(self.records)

    @property
    def num_total_tokens(self) -> int:
        return sum(record.trace.num_total_tokens for record in self.records)

    @property
    def vf_episodes(self) -> list[vf.Episode]:
        if self._predicate is None:
            return self.selected_episodes
        by_episode: dict[int, list[vf.Trace]] = {}
        for record in self.records:
            trace = record.trace
            advantage = scalar_advantage(trace)
            if advantage is not None:
                trace = trace.model_copy(update={"info": {**trace.info, "advantage": advantage}})
            by_episode.setdefault(id(record.episode), []).append(trace)
        return [
            episode.model_copy(update={"traces": by_episode[id(episode)]})
            for episode in self.selected_episodes
            if id(episode) in by_episode
        ]

    def append(
        self,
        episode: vf.Episode,
        *,
        sampled_trace_ids: set[str] | None = None,
        admitted: bool = True,
        cancelled: bool = False,
    ) -> None:
        self.episodes.append(episode)
        self.sampled_trace_ids.update(sampled_trace_ids or set())
        if admitted:
            self.admitted.add(episode.id)
        if cancelled:
            self.cancelled.add(episode.id)

    def extend(
        self,
        episodes: list[vf.Episode],
        *,
        sampled_trace_ids: set[str] | None = None,
        admitted: bool = True,
        cancelled: bool = False,
    ) -> None:
        self.episodes.extend(episodes)
        self.sampled_trace_ids.update(sampled_trace_ids or set())
        if admitted:
            self.admitted.update(episode.id for episode in episodes)
        if cancelled:
            self.cancelled.update(episode.id for episode in episodes)

    def __len__(self) -> int:
        return len(self.selected_episodes)

    def __iter__(self) -> Iterator[vf.Episode]:
        return iter(self.selected_episodes)


class TrainEpisodes(EpisodeCollection):
    @property
    def effective(self) -> TrainEpisodes:
        return TrainEpisodes(
            self.episodes,
            self.sampled_trace_ids,
            self.admitted,
            self.cancelled,
            predicate=lambda record: (
                record.admitted
                and record.sampled
                and not record.cancelled
                and not record.trace.has_error
                and record.trace.agent.trainable
            ),
        )

    def by_env(self) -> dict[str, TrainEpisodes]:
        grouped: dict[str, list[vf.Episode]] = {}
        for episode in self.selected_episodes:
            grouped.setdefault(episode_env_name(episode), []).append(episode)
        return {
            env_name: TrainEpisodes(episodes, self.sampled_trace_ids, self.admitted, self.cancelled, self._predicate)
            for env_name, episodes in grouped.items()
        }

    @property
    def metrics(self) -> TrainMetrics:
        return TrainMetrics(self.selected_episodes, self.records, self.cancelled)


@dataclass
class RolloutWindow:
    """Completed group accounting independent of whether a training batch fills."""

    metrics: MetricWindow = field(default_factory=MetricWindow)
    units: int = 0
    tokens: int = 0
    traces: int = 0
    groups: int = 0
    attempts: int = 0
    discarded: int = 0
    stale: int = 0
    errored: int = 0

    def observe(
        self,
        episodes: TrainEpisodes,
        failures: list[DispatchFailure],
        cancellation: GroupCancellation | None,
        *,
        env_name: str,
        queued_trace_ids: set[str],
    ) -> None:
        cancelled = cancellation.count if cancellation is not None else 0
        attempts = len(episodes) + len(failures) + cancelled
        self.units += max(attempts, episodes.num_traces)
        self.tokens += episodes.num_total_tokens
        self.traces += episodes.num_traces
        self.groups += 1
        self.attempts += attempts
        discarded = [episode for episode in episodes if not any(t.id in queued_trace_ids for t in episode.traces)]
        self.discarded += len(discarded) + len(failures) + cancelled
        self.stale += sum(episode.id in episodes.cancelled for episode in discarded)
        if cancellation is not None and cancellation.reason == "stale":
            self.stale += cancelled
        self.errored += len(failures) + sum(
            episode.id not in episodes.cancelled
            and (not episode.ok or any(trace.has_error for trace in episode.traces))
            for episode in discarded
        )
        for prefix in ("train/agg", f"train/{env_name}"):
            self.metrics.extend(episodes.metrics.snapshot(prefix=prefix, subset="all"))
            self.metrics.rate(
                f"{prefix}/all/dispatch_failure", {"mean": float(len(failures))}, len(episodes) + len(failures)
            )
            for failure in failures:
                self.metrics.count(f"{prefix}/all/dispatch_failure/{failure.error.type}", 1.0)
        self.metrics.rate("rollout/env_share", {env_name: float(episodes.num_traces)}, episodes.num_traces)
        self.metrics.count("rollout/traces/queued", float(len(queued_trace_ids)))
        self.metrics.count("rollout/traces/discarded", float(episodes.num_traces - len(queued_trace_ids)))


class EvalEpisodes(EpisodeCollection):
    def __init__(
        self,
        episodes: list[vf.Episode],
        group_size: int,
        predicate: Callable[[TraceRecord], bool] | None = None,
    ) -> None:
        super().__init__(episodes, predicate=predicate)
        self.group_size = group_size

    @property
    def effective(self) -> EvalEpisodes:
        return EvalEpisodes(
            self.episodes,
            self.group_size,
            predicate=lambda record: not record.trace.has_error and record.trace.agent.trainable,
        )

    @property
    def metrics(self) -> EvalMetrics:
        return EvalMetrics(self.selected_episodes, self.records, self.group_size)
