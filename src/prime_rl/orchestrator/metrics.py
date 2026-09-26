"""Episode metrics: one view over groups that filters itself.

``Episodes(groups)`` is every episode and trace of the groups. Each view
(``clean``, ``sampled``, ``admitted``, ``by_env()``, ``by_agent()``) is another
``Episodes`` narrowed by a predicate, so callers compose the subset they mean —
``batch.clean.sampled`` is what trained — and read stats or ``metrics()`` off it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Literal

import verifiers.v1 as vf

from prime_rl.orchestrator.algo.routing import is_trainable, scalar_advantage
from prime_rl.orchestrator.types import Group, has_error, is_cancelled
from prime_rl.orchestrator.utils import compute_pass_metrics

Subset = Literal["all", "effective"]

Keep = Callable[[Group, vf.Episode, vf.Trace], bool]

TIMING_PHASES = ("setup", "agent", "finalize", "scoring")


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

    def to_dict(self, prefix: str) -> dict[str, float]:
        if not self.values:
            return {}
        return {
            f"{prefix}/mean": self.mean(),
            f"{prefix}/max": self.max(),
            f"{prefix}/min": self.min(),
            f"{prefix}/p10": self.percentile(10),
            f"{prefix}/p90": self.percentile(90),
        }


class Episodes:
    def __init__(self, groups: list[Group], keep: Keep | None = None) -> None:
        self.groups = groups
        self._keep = keep

    # ── views ──────────────────────────────────────────────────────────────

    def filter(self, keep: Keep) -> Episodes:
        outer = self._keep
        if outer is None:
            return Episodes(self.groups, keep)
        return Episodes(
            self.groups, lambda group, episode, trace: outer(group, episode, trace) and keep(group, episode, trace)
        )

    @property
    def clean(self) -> Episodes:
        """Trainable-agent traces that neither errored nor were voided."""
        return self.filter(
            lambda _, episode, trace: not is_cancelled(episode) and not trace.has_error and trace.agent.trainable
        )

    @property
    def sampled(self) -> Episodes:
        """Traces that made it into a trainer payload."""
        return self.filter(lambda group, _, trace: trace.id in group.samples)

    @property
    def admitted(self) -> Episodes:
        return self.filter(lambda group, _, __: group.admitted)

    def by_env(self) -> dict[str, Episodes]:
        names = sorted({group.env for group in self.groups})
        return {name: Episodes([group for group in self.groups if group.env == name], self._keep) for name in names}

    def by_agent(self) -> dict[str, Episodes]:
        names = sorted({trace.agent.name for trace in self.traces})
        return {name: self.filter(lambda _, __, trace, name=name: trace.agent.name == name) for name in names}

    # ── contents ───────────────────────────────────────────────────────────

    @property
    def records(self) -> list[tuple[Group, vf.Episode, vf.Trace]]:
        return [
            (group, episode, trace)
            for group in self.groups
            for episode in group.episodes
            for trace in episode.traces
            if self._keep is None or self._keep(group, episode, trace)
        ]

    @property
    def traces(self) -> list[vf.Trace]:
        return [trace for _, _, trace in self.records]

    @property
    def episodes(self) -> list[vf.Episode]:
        """Every episode of the groups, trace-less ones included, until a view narrows
        it to the owners of the traces it keeps."""
        if self._keep is None:
            return [episode for group in self.groups for episode in group.episodes]
        kept = {id(episode) for _, episode, _ in self.records}
        return [episode for group in self.groups for episode in group.episodes if id(episode) in kept]

    @property
    def vf_episodes(self) -> list[vf.Episode]:
        """The kept episodes narrowed to their kept traces, for the trace stream; a
        narrowed trace carries its scalar advantage in ``info``."""
        if self._keep is None:
            return self.episodes
        traces: dict[int, list[vf.Trace]] = {}
        for _, episode, trace in self.records:
            advantage = scalar_advantage(trace)
            if advantage is not None:
                trace = trace.model_copy(update={"info": {**trace.info, "advantage": advantage}})
            traces.setdefault(id(episode), []).append(trace)
        return [episode.model_copy(update={"traces": traces[id(episode)]}) for episode in self.episodes]

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self) -> Iterator[vf.Episode]:
        return iter(self.episodes)

    @property
    def num_traces(self) -> int:
        return len(self.records)

    @property
    def num_total_tokens(self) -> int:
        return sum(trace.num_total_tokens for trace in self.traces)

    # ── stats ──────────────────────────────────────────────────────────────

    def trace_stat(self, value: Callable[[vf.Trace], float]) -> Stat:
        return Stat([float(value(trace)) for trace in self.traces])

    def episode_stat(self, value: Callable[[vf.Trace], float]) -> Stat:
        """One value per kept episode: ``value`` summed over its kept traces."""
        by_episode = {id(episode): 0.0 for episode in self.episodes}
        for _, episode, trace in self.records:
            by_episode[id(episode)] += float(value(trace))
        return Stat(list(by_episode.values()))

    @property
    def reward(self) -> Stat:
        return self.trace_stat(lambda trace: trace.reward)

    @property
    def is_truncated(self) -> Stat:
        return self.trace_stat(lambda trace: trace.is_truncated)

    @property
    def num_turns(self) -> Stat:
        return self.episode_stat(lambda trace: trace.num_turns)

    @property
    def num_branches(self) -> Stat:
        return self.episode_stat(lambda trace: trace.num_branches)

    @property
    def has_error(self) -> Stat:
        return Stat([float(has_error(episode)) for episode in self.episodes])

    @property
    def cancelled(self) -> Stat:
        return Stat([float(is_cancelled(episode)) for episode in self.episodes])

    def error_types(self) -> dict[str, int]:
        types = []
        for episode in self.episodes:
            if not has_error(episode):
                continue
            last = episode.last_error or next((trace.last_error for trace in episode.traces if trace.last_error), None)
            types.append(last.type if last is not None else "unknown")
        return {error_type: types.count(error_type) for error_type in sorted(set(types))}

    def solve_rates(self) -> dict[str, float]:
        rewards: dict[str, list[float]] = {}
        for group, _, trace in self.records:
            rewards.setdefault(group.id, []).append(trace.reward)
        if not rewards:
            return {}
        n = len(rewards)
        none = sum(sum(values) == 0 for values in rewards.values())
        every = sum(all(value == 1.0 for value in values) for values in rewards.values())
        return {"solved_none": none / n, "solved_all": every / n, "solved_some": 1 - (none + every) / n}

    def stop_conditions(self) -> dict[str, float]:
        traces = self.traces
        if not traces:
            return {}
        out = {
            "generation_truncated": sum(t.is_truncated and t.stop_condition != "prompt_too_long" for t in traces)
            / len(traces)
        }
        conditions = [trace.stop_condition for trace in traces if trace.stop_condition is not None]
        for condition in sorted(set(conditions)):
            out[condition] = conditions.count(condition) / len(conditions)
        return out

    def pass_at_k(self) -> dict[str, float]:
        rewards_by_group: dict[str, list[float]] = {}
        for group, _, trace in self.records:
            rewards_by_group.setdefault(group.id, []).append(trace.reward)
        if not all(set(values) <= {0.0, 1.0} for values in rewards_by_group.values()):
            return {}
        per_group = [compute_pass_metrics(values) for values in rewards_by_group.values()]
        keys = sorted({key for result in per_group for key in result})
        return {
            key: sum(result[key] for result in per_group if key in result) / sum(key in result for result in per_group)
            for key in keys
        }

    # ── reporting ──────────────────────────────────────────────────────────

    def metrics(self, prefix: str, *, subset: Subset) -> dict[str, float]:
        """Episode-level stats, then per-agent trace-level stats, under ``{prefix}/{subset}``."""
        if not self.episodes:
            return {}
        prefix = f"{prefix}/{subset}"
        out: dict[str, float] = {}
        for name in ("num_total_tokens", "num_input_tokens", "num_output_tokens", "num_turns", "num_branches"):
            out |= self.episode_stat(lambda trace, name=name: getattr(trace, name)).to_dict(f"{prefix}/{name}")
        if subset == "all":
            out[f"{prefix}/has_error/mean"] = self.has_error.mean()
            out[f"{prefix}/cancelled/mean"] = self.cancelled.mean()
            out |= {f"{prefix}/error/{key}": float(value) for key, value in self.error_types().items()}
        for agent, traces in self.by_agent().items():
            out |= traces.agent_metrics(f"{prefix}/{agent}", subset=subset)
        return out

    def agent_metrics(self, prefix: str, *, subset: Subset) -> dict[str, float]:
        traces = self.traces
        out: dict[str, float] = {}
        for name in (
            "reward",
            "num_total_tokens",
            "num_input_tokens",
            "num_output_tokens",
            "num_turns",
            "num_branches",
        ):
            out |= self.trace_stat(lambda trace, name=name: getattr(trace, name)).to_dict(f"{prefix}/{name}")
        for name in ("is_truncated", "is_completed"):
            out[f"{prefix}/{name}/mean"] = self.trace_stat(lambda trace, name=name: getattr(trace, name)).mean()
        for phase in TIMING_PHASES:
            out |= self.trace_stat(lambda t, p=phase: getattr(t.timing, p).duration).to_dict(f"{prefix}/timing/{phase}")
        out |= self.trace_stat(lambda t: t.timing.agent.model.duration).to_dict(f"{prefix}/timing/agent/model")
        out |= self.trace_stat(lambda t: t.timing.agent.harness.duration).to_dict(f"{prefix}/timing/agent/harness")
        out |= self.trace_stat(lambda t: sum(getattr(t.timing, p).duration for p in TIMING_PHASES)).to_dict(
            f"{prefix}/timing/total"
        )
        for name in sorted({name for trace in traces for name in trace.metrics}):
            values = [trace.metrics[name] for trace in traces if name in trace.metrics]
            out |= Stat([float(v) if v is not None else 0.0 for v in values]).to_dict(f"{prefix}/metrics/{name}")
        for name in sorted({name for trace in traces for name in trace.rewards}):
            values = [trace.rewards[name] for trace in traces if name in trace.rewards]
            out |= Stat([v.value if v is not None else 0.0 for v in values]).to_dict(f"{prefix}/rewards/{name}")
        if subset == "all":
            out[f"{prefix}/has_error/mean"] = self.trace_stat(lambda trace: trace.has_error).mean()
            out |= {f"{prefix}/{key}": value for key, value in self.solve_rates().items()}
        out |= {f"{prefix}/stop_condition/{key}": value for key, value in self.stop_conditions().items()}
        return out

    def train_metrics(self, prefix: str, *, subset: Subset) -> dict[str, float]:
        out = self.metrics(prefix, subset=subset)
        for agent, traces in self.by_agent().items():
            out[f"{prefix}/{subset}/{agent}/is_trainable/mean"] = traces.trace_stat(is_trainable).mean()
            out[f"{prefix}/{subset}/{agent}/is_admitted/mean"] = Stat(
                [float(group.admitted) for group, _, _ in traces.records]
            ).mean()
        return out

    def eval_metrics(self, prefix: str, *, subset: Subset, k: int) -> dict[str, float]:
        out = self.metrics(prefix, subset=subset)
        for agent, traces in self.by_agent().items():
            out[f"{prefix}/{subset}/{agent}/avg@{k}"] = traces.reward.mean()
            if subset == "effective":
                out |= {f"{prefix}/{subset}/{agent}/{key}": value for key, value in traces.pass_at_k().items()}
        return out
