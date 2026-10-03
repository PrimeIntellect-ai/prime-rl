"""GAR — groupwise advantage redistribution (MiMo-V2.6 §4.3.2).

A mixed group (rewards not all equal) is graded by an agent that sees the whole group
in one sandbox: the ``group-grade`` verifiers taskset, served on the source's ``grade``
env server and run on a frozen grader model. The grader ranks the candidates above
the group mean reward (P) and audits every candidate for hacks. A confirmed hack's
reward drops to the group's minimum before the group statistics;
:func:`redistribute` turns the ranking into advantages (Eq. 3).
Any grader failure keeps the group's plain GRPO advantages.
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf
from pydantic import ValidationError
from verifiers.v1.serve import EnvClient
from verifiers.v1.tasksets.group_grade import GroupGradeData, GroupVerdict, quote_found

from prime_rl import monitors
from prime_rl.configs.algorithm import GARAlgoConfig
from prime_rl.monitors.file.traces.update import make_update
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.utils import episode_group_id, train_work
from prime_rl.utils.async_utils import safe_cancel_all
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient
    from prime_rl.orchestrator.types import Progress

FALLBACK_REASONS = ("stale", "timeout", "error", "invalid")


def win_rates(ranking: list[list[str]]) -> dict[str, float]:
    """Each ranked label's win rate against the other ranked labels: a label in a
    lower tier counts 1, a tie 1/2. A lone label wins outright."""
    labels = [label for tier in ranking for label in tier]
    if len(labels) < 2:
        return {label: 1.0 for label in labels}
    rates = {}
    below = len(labels)
    for tier in ranking:
        below -= len(tier)
        for label in tier:
            rates[label] = (below + 0.5 * (len(tier) - 1)) / (len(labels) - 1)
    return rates


def redistribute(rewards: list[float], quality: list[float], lambda_max: float) -> tuple[list[float], float]:
    """Eq. 3: ``A = R - mean(R)``; each member of P, the rollouts with ``A > 0``, gets
    ``lambda * f * A`` with ``lambda = sum_P A / sum_P f A`` capped at ``lambda_max``;
    then the group is re-centered to zero mean (a no-op unless the cap binds)."""
    mean = sum(rewards) / len(rewards)
    advantages = [reward - mean for reward in rewards]
    above = [i for i, advantage in enumerate(advantages) if advantage > 0]
    if not above:
        return advantages, 1.0
    lam = min(lambda_max, sum(advantages[i] for i in above) / sum(quality[i] * advantages[i] for i in above))
    for i in above:
        advantages[i] *= lam * quality[i]
    shift = sum(advantages) / len(advantages)
    return [advantage - shift for advantage in advantages], lam


class GARAlgorithm(GRPOAlgorithm):
    """GRPO whose mixed groups are graded by an agentic group grader. Owns the
    grader: its frozen model pool and the client onto its env server. Grading takes
    minutes, so groups finalize in the background."""

    finalize_in_background = True

    def __init__(self, config: GARAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.config = config
        self.grader: EnvClient | None = None
        self.grader_clients: InferenceClient | None = None
        self.grader_address_file: Path | None = None
        self.progress: Progress | None = None
        self.max_off_policy_steps: int | None = None
        self.slots = asyncio.Semaphore(config.grader.max_concurrent)

    def bind(self, progress: Progress, max_off_policy_steps: int, grader_address_file: Path) -> None:
        """Wire in the orchestrator's step clock and staleness bound, and where the
        launcher-managed grader server publishes its address."""
        self.progress = progress
        self.max_off_policy_steps = max_off_policy_steps
        self.grader_address_file = grader_address_file

    async def setup(self) -> None:
        from prime_rl.orchestrator.envs import ENV_SERVER_STARTUP_TIMEOUT, wait_for_address

        self.grader_clients = await self.connect(self.config.grader.model)
        address = self.config.grader.serve.address
        if address is None:
            assert self.grader_address_file is not None
            address = await wait_for_address(self.grader_address_file, timeout=ENV_SERVER_STARTUP_TIMEOUT)
        self.grader = EnvClient(address=address)
        await self.grader.wait_for_server_startup(timeout=ENV_SERVER_STARTUP_TIMEOUT)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        # One candidate per episode: the trace carrying its patch (else its first
        # trainable trace); its credit applies to every trainable trace of the episode.
        members: list[tuple[vf.Trace, list[vf.Trace]]] = []
        for episode in episodes:
            traces = [trace for _, trace in iter_trainable_traces([episode])]
            if traces:
                members.append((next((trace for trace in traces if "patch" in trace.info), traces[0]), traces))
        rewards = [candidate.reward for candidate, _ in members]
        if len(set(rewards)) < 2:
            await super().score_group(episodes)
            return
        mean = sum(rewards) / len(rewards)
        passed = [reward > mean for reward in rewards]

        data, labels = GroupGradeData.from_traces(
            [candidate for candidate, _ in members], passed, random.Random(episode_group_id(episodes[0]))
        )
        traces_by_id = {candidate.id: traces for candidate, traces in members}
        traces_by_label = {label: traces_by_id[trace_id] for label, trace_id in labels.items()}

        started = time.monotonic()
        grade, verdict, reason = await self.grade(data, self.fallback_step(episodes))
        metrics = {
            "gar/graded": float(verdict is not None),
            "gar/latency": time.monotonic() - started,
            **{f"gar/fallback/{name}": float(reason == name) for name in FALLBACK_REASONS},
        }
        common: dict[str, Any] = {"fallback_reason": reason, "grader_trace_id": None}
        if grade is not None:
            common["grader_trace_id"] = grade.traces[0].id if grade.traces else None
            if (usage := vf.Usage.aggregate(t.usage for t in grade.traces if t.usage is not None)) is not None:
                metrics["gar/grader_tokens"] = float(usage.prompt_tokens + usage.completion_tokens)
                if usage.cost is not None:
                    metrics["gar/grader_cost"] = usage.cost
            await self.log_grade(grade, episodes, labels)

        if verdict is None:
            await super().score_group(episodes)
            info = {label: {"label": label, **common} for label in labels}
        else:
            info = self.apply(verdict, data, traces_by_label, common)

        for label, traces in traces_by_label.items():
            for trace in traces:
                trace.info["gar"] = info[label]
                trace.record_metrics(metrics)
        await monitors.log_annotations(
            [
                make_update(trace.id, info={"gar": info[label]})
                for label, traces in traces_by_label.items()
                for trace in traces
            ]
        )

    async def grade(
        self, data: GroupGradeData, fallback_step: int | None
    ) -> tuple[vf.Episode | None, GroupVerdict | None, str | None]:
        """Run one grader episode: the episode (if any), the validated verdict, or the
        reason the group falls back. ``grader.timeout`` covers the wait for a grader
        slot too, and the group falls back once ``progress.step`` reaches
        ``fallback_step`` — the last step whose batch can still take it."""
        grading = asyncio.create_task(self.run_grader(data))
        stale = asyncio.create_task(self.until_step(fallback_step))
        try:
            done, _ = await asyncio.wait(
                {grading, stale}, timeout=self.config.grader.timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            await safe_cancel_all([grading, stale])
        if grading not in done:
            return None, None, "stale" if stale in done else "timeout"
        try:
            grade = grading.result()
        except Exception as e:  # noqa: BLE001 - a grader outage falls back per group
            get_logger().warning(f"GAR grader request failed: {e!r}")
            return None, None, "error"
        if not grade.ok or not grade.traces:
            return grade, None, "error"
        try:
            verdict = GroupVerdict.model_validate(grade.traces[0].info.get("group_verdict"))
            verdict.check(data)
        except (ValidationError, ValueError) as e:
            get_logger().warning(f"GAR grader verdict rejected: {e}")
            return grade, None, "invalid"
        return grade, verdict, None

    async def run_grader(self, data: GroupGradeData) -> vf.Episode:
        assert self.grader is not None and self.grader_clients is not None, (
            "grader not connected — setup() must run first"
        )
        async with self.slots:
            return await self.grader.run(
                client=self.grader_clients.eval_client,
                model=self.config.grader.model.name,
                sampling=vf.SamplingConfig(),
                task_data=data.model_dump(mode="json"),
            )

    async def until_step(self, step: int | None) -> None:
        assert self.progress is not None, "bind() must run first"
        if step is None:  # frozen-sourced episodes never go stale
            await asyncio.Event().wait()
        while self.progress.step < step:
            await asyncio.sleep(1.0)

    def fallback_step(self, episodes: list[vf.Episode]) -> int | None:
        """The last step whose batch can still admit this group: at the next one the
        insertion sweep (``max_off_policy_steps``) would drop it."""
        assert self.max_off_policy_steps is not None, "bind() must run first"
        starts = [policy.start for episode in episodes if (policy := train_work(episode).policy) is not None]
        return min(starts) + self.max_off_policy_steps + 1 if starts else None

    def apply(
        self,
        verdict: GroupVerdict,
        data: GroupGradeData,
        traces_by_label: dict[str, list[vf.Trace]],
        common: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Drop confirmed hacks to the group's minimum reward, rescale P by the ranking, and assign the
        advantages. Returns each label's ``info.gar`` record."""
        candidates = {candidate.label: candidate for candidate in data.candidates}
        verdicts = {candidate.label: candidate for candidate in verdict.candidates}
        hacks = {
            # A confirmed hack must quote its cited turns verbatim; otherwise it is only suspected.
            label: "suspected"
            if v.hack == "confirmed" and not all(quote_found(candidates[label].turns, e) for e in v.evidence)
            else v.hack
            for label, v in verdicts.items()
        }
        ranked = [label for label, candidate in candidates.items() if candidate.passed and hacks[label] != "confirmed"]
        rates = win_rates([[label for label in tier if label in ranked] for tier in verdict.ranking])
        f_min = self.config.f_min
        # A candidate the ranking leaves out sits mid-table: a pass whose confirmed hack
        # failed its citation, or one lifted above the mean by a hack's dropped reward.
        quality = {label: f_min + (1 - f_min) * rates.get(label, 0.5) for label in candidates}

        traces = [(label, trace) for label, label_traces in traces_by_label.items() for trace in label_traces]
        # The group's worst observed outcome: 0 in a mixed binary group, and no reward
        # bounds assumed otherwise; a hack never ends above an honest rollout.
        floor = min(trace.reward for _, trace in traces)
        rewards = [floor if hacks[label] == "confirmed" else trace.reward for label, trace in traces]
        advantages, lam = redistribute(rewards, [quality[label] for label, _ in traces], self.config.lambda_max)
        for (label, trace), advantage in zip(traces, advantages, strict=True):
            assign_advantages(trace, advantage)
            trace.record_metric("gar/lambda", lam)
            trace.record_metric("gar/all_fail", float(not ranked))
            if candidates[label].passed:
                trace.record_metric("gar/hack_confirmed", float(hacks[label] == "confirmed"))
                trace.record_metric("gar/hack_suspected", float(hacks[label] == "suspected"))
            if label in rates:
                trace.record_metric("gar/f", quality[label])

        tiers = {label: index for index, tier in enumerate(verdict.ranking) for label in tier}
        return {
            label: {
                "label": label,
                **common,
                "hack": hacks[label],
                "evidence": [evidence.model_dump() for evidence in verdicts[label].evidence],
                "tier": tiers.get(label),
                "w": rates.get(label),
                "f": quality[label] if label in rates else None,
                "lambda": lam,
            }
            for label in candidates
        }

    async def log_grade(self, grade: vf.Episode, episodes: list[vf.Episode], labels: dict[str, str]) -> None:
        """Log the grader's episode to the trace stream as kind ``grade``, linked to
        its group and, by label, to the candidate traces. The candidates' transcripts
        are left out: the candidate traces are already in the stream."""
        data = vf.WireTaskData.model_validate(grade.task.data.model_dump(mode="json", exclude={"candidates"}))
        task = grade.task.model_copy(update={"data": data})
        traces = []
        for trace in grade.traces:
            info = {**trace.info, "kind": "grade", "group_id": episode_group_id(episodes[0]), "candidates": labels}
            traces.append(trace.model_copy(update={"task": task, "info": info}))
        logged = grade.model_copy(update={"task": task, "traces": traces})
        await monitors.log([logged], train_work(episodes[0]).step, "grade", "all")
