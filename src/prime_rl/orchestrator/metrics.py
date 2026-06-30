"""Train and eval rollout metrics.

A rollout container (``TrainRollouts`` / ``EvalRollouts``) owns the rollout list and exposes
``.effective`` (the clean subset, as the same container type) and ``.metrics`` (``TrainMetrics`` /
``EvalMetrics``). The metrics object exposes each distributional / rate metric as a ``Stat`` — so
``rollouts.metrics.num_input_tokens.mean()`` works — and assembles the full
``{prefix}/{subset}/<metric>/<stat>`` wandb dict via ``.to_wandb(...)``.

No I/O, no pandas — plain Python over the ``vf.Trace`` properties each rollout exposes. Aggregation
is flat over the rollout list except the solve rates, which group by ``group_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Literal

from prime_rl.orchestrator.eval_utils import compute_pass_metrics

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout

Subset = Literal["all", "effective"]


class Stat:
    """A distribution of per-rollout values with mean/max/min accessors."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def max(self) -> float:
        return float(max(self.values)) if self.values else 0.0

    def min(self) -> float:
        return float(min(self.values)) if self.values else 0.0

    def to_dict(self, prefix: str) -> dict[str, float]:
        """``{prefix}/mean,max,min``; ``{}`` for an empty distribution."""
        if not self.values:
            return {}
        return {f"{prefix}/mean": self.mean(), f"{prefix}/max": self.max(), f"{prefix}/min": self.min()}


class RolloutMetrics:
    """Metrics shared by train and eval over a rollout list. Distributional metrics are ``Stat``s
    (mean/max/min); boolean metrics are ``Stat``s of 0/1 (use ``.mean()`` for the rate). ``to_wandb``
    assembles the full ``{prefix}/{subset}/...`` dict; ``TrainMetrics`` / ``EvalMetrics`` extend it."""

    def __init__(self, rollouts: list["Rollout"]) -> None:
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
    def timing_setup(self) -> Stat:
        return Stat([r.timing.setup.duration for r in self.rollouts])

    @property
    def timing_generation(self) -> Stat:
        return Stat([r.timing.generation.duration for r in self.rollouts])

    @property
    def timing_finalize(self) -> Stat:
        return Stat([r.timing.finalize.duration for r in self.rollouts])

    @property
    def timing_scoring(self) -> Stat:
        return Stat([r.timing.scoring.duration for r in self.rollouts])

    @property
    def timing_total(self) -> Stat:
        return Stat(
            [
                r.timing.setup.duration
                + r.timing.generation.duration
                + r.timing.finalize.duration
                + r.timing.scoring.duration
                for r in self.rollouts
            ]
        )

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

    def custom(self, attr: str) -> dict[str, Stat]:
        """Per-key ``Stat``s for a dynamic per-rollout dict attribute (``metrics`` or ``rewards``),
        each averaged over the rollouts that report the key."""
        names = sorted({name for r in self.rollouts for name in getattr(r, attr)})
        return {
            name: Stat([getattr(r, attr)[name] for r in self.rollouts if name in getattr(r, attr)]) for name in names
        }

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

    def solve_rates(self) -> dict[str, float]:
        """Per-group: ``solved_none`` (no rollout earned reward), ``solved_all`` (every rollout in
        the group did), and ``solved_some`` (the mixed-reward remainder — the GRPO-signal groups)."""
        groups: dict = {}
        for r in self.rollouts:
            groups.setdefault(r.group_id, []).append(r)
        n_groups = len(groups)
        solved_none = sum(1 for g in groups.values() if sum(r.reward for r in g) == 0)
        solved_all = sum(1 for g in groups.values() if all(r.reward for r in g))
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
        out |= self.timing_setup.to_dict(f"{p}/timing/setup")
        out |= self.timing_generation.to_dict(f"{p}/timing/generation")
        out |= self.timing_finalize.to_dict(f"{p}/timing/finalize")
        out |= self.timing_scoring.to_dict(f"{p}/timing/scoring")
        out |= self.timing_total.to_dict(f"{p}/timing/total")
        for name, stat in self.custom("metrics").items():
            out |= stat.to_dict(f"{p}/metrics/{name}")
        for name, stat in self.custom("rewards").items():
            out |= stat.to_dict(f"{p}/rewards/{name}")
        out[f"{p}/is_truncated/mean"] = self.is_truncated.mean()
        out[f"{p}/is_completed/mean"] = self.is_completed.mean()
        # has_error is structurally 0 on the effective subset, so log it on `all` only
        if subset == "all":
            out[f"{p}/has_error/mean"] = self.has_error.mean()
        out |= {f"{p}/stop_condition/{k}": v for k, v in self.stop_conditions().items()}
        out |= {f"{p}/{k}": v for k, v in self.solve_rates().items()}
        return out


class TrainMetrics(RolloutMetrics):
    """Common metrics plus the reward distribution and filter-pipeline rates."""

    @property
    def reward(self) -> Stat:
        return Stat([float(r.reward) for r in self.rollouts])

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
        out[f"{p}/is_filtered/mean"] = self.is_filtered.mean()
        out |= {f"{p}/filters/{k}/mean": v for k, v in self.filter_rates().items()}
        return out


class EvalMetrics(RolloutMetrics):
    """Common metrics plus the ``avg@<group_size>`` score and (on the effective subset, for
    binary-reward tasks) pass@k / pass^k."""

    def __init__(self, rollouts: list["Rollout"], group_size: int) -> None:
        super().__init__(rollouts)
        self.group_size = group_size

    @property
    def reward(self) -> Stat:
        return Stat([float(r.reward) for r in self.rollouts])

    def pass_at_k(self) -> dict[str, float]:
        """pass@k / pass^k averaged over examples; ``{}`` for non-binary rewards."""
        rewards = [r.reward for r in self.rollouts]
        if not set(rewards).issubset({0.0, 1.0}):
            return {}
        by_example: dict = {}
        for r in self.rollouts:
            by_example.setdefault(r.group_id, []).append(r.reward)
        per_example = [compute_pass_metrics(rs) for rs in by_example.values()]
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
    """A list of train rollouts (everything that came back, errored + filtered included). ``effective``
    is the clean subset (a view of the same traces); ``metrics`` builds ``TrainMetrics`` over them."""

    def __init__(self, rollouts: list["Rollout"] | None = None) -> None:
        self.rollouts = rollouts if rollouts is not None else []

    def append(self, rollout: "Rollout") -> None:
        self.rollouts.append(rollout)

    def __len__(self) -> int:
        return len(self.rollouts)

    def __iter__(self) -> Iterator["Rollout"]:
        return iter(self.rollouts)

    @property
    def effective(self) -> "TrainRollouts":
        return TrainRollouts([r for r in self.rollouts if not r.has_error and not r.is_filtered])

    def by_env(self) -> dict[str, "TrainRollouts"]:
        grouped: dict[str, list["Rollout"]] = {}
        for r in self.rollouts:
            grouped.setdefault(r.env_name, []).append(r)
        return {env: TrainRollouts(rs) for env, rs in grouped.items()}

    @property
    def metrics(self) -> TrainMetrics:
        return TrainMetrics(self.rollouts)


class EvalRollouts:
    """A list of eval rollouts (errored included). ``effective`` is the non-errored subset (a view);
    ``metrics`` builds ``EvalMetrics`` over them. ``group_size`` names the ``avg@<k>`` key."""

    def __init__(self, rollouts: list["Rollout"] | None = None, group_size: int = 1) -> None:
        self.rollouts = rollouts if rollouts is not None else []
        self.group_size = group_size

    def append(self, rollout: "Rollout") -> None:
        self.rollouts.append(rollout)

    def __len__(self) -> int:
        return len(self.rollouts)

    def __iter__(self) -> Iterator["Rollout"]:
        return iter(self.rollouts)

    @property
    def effective(self) -> "EvalRollouts":
        return EvalRollouts([r for r in self.rollouts if not r.has_error and not r.is_filtered], self.group_size)

    @property
    def metrics(self) -> EvalMetrics:
        return EvalMetrics(self.rollouts, self.group_size)
