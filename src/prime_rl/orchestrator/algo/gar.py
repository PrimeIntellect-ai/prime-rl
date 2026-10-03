"""GAR — groupwise advantage redistribution (MiMo-V2.6 §4.3.2).

A mixed-outcome group (at least one pass and one fail) is graded by an agent that
sees the whole group in one sandbox: the ``group-grade`` verifiers taskset, served
on the source's ``grade`` env server and run on a frozen grader model. Its verdict
zeroes the reward of confirmed hacks before the group statistics and ranks the
remaining passes; :func:`redistribute` turns the ranking into advantages (Eq. 3).
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
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient

FALLBACK_REASONS = ("timeout", "error", "invalid")

# How far a ranking is trusted, by the grader's own confidence.
CONFIDENCE_SHRINK = {"high": 1.0, "medium": 0.5, "low": 0.0}


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
    """Eq. 3: ``A = R - mean(R)``; each pass (``R == 1``) gets ``lambda * f * A`` with
    ``lambda = sum_P A / sum_P f A`` capped at ``lambda_max``; then the group is
    re-centered to zero mean (a no-op unless the cap binds)."""
    mean = sum(rewards) / len(rewards)
    advantages = [reward - mean for reward in rewards]
    passes = [i for i, reward in enumerate(rewards) if reward == 1.0]
    if not passes or len(passes) == len(rewards):
        return advantages, 1.0
    lam = min(lambda_max, sum(advantages[i] for i in passes) / sum(quality[i] * advantages[i] for i in passes))
    for i in passes:
        advantages[i] *= lam * quality[i]
    shift = sum(advantages) / len(advantages)
    return [advantage - shift for advantage in advantages], lam


class GARAlgorithm(GRPOAlgorithm):
    """GRPO whose mixed groups are graded by an agentic group grader. Owns the
    grader: its frozen model pool and the client onto its env server."""

    def __init__(self, config: GARAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.config = config
        self.grader: EnvClient | None = None
        self.grader_clients: InferenceClient | None = None
        self.grader_address_file: Path | None = None
        """Where the launcher-managed grader server publishes its address; set by ``TrainEnvs``."""
        self.slots = asyncio.Semaphore(config.grader.max_concurrent)

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
        passed = [candidate.reward == 1.0 for candidate, _ in members]
        if all(passed) or not any(passed):
            await super().score_group(episodes)
            return

        data, labels = GroupGradeData.from_traces(
            [candidate for candidate, _ in members], passed, random.Random(episode_group_id(episodes[0]))
        )
        traces_by_id = {candidate.id: traces for candidate, traces in members}
        traces_by_label = {label: traces_by_id[trace_id] for label, trace_id in labels.items()}

        started = time.monotonic()
        grade, verdict, reason = await self.grade(data)
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

    async def grade(self, data: GroupGradeData) -> tuple[vf.Episode | None, GroupVerdict | None, str | None]:
        """Run one grader episode: the episode (if any), the validated verdict, or the
        reason the group falls back."""
        assert self.grader is not None and self.grader_clients is not None, (
            "grader not connected — setup() must run first"
        )
        async with self.slots:
            try:
                async with asyncio.timeout(self.config.grader.timeout):
                    grade = await self.grader.run(
                        client=self.grader_clients.eval_client,
                        model=self.config.grader.model.name,
                        sampling=vf.SamplingConfig(),
                        task_data=data.model_dump(mode="json"),
                    )
            except TimeoutError:
                return None, None, "timeout"
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

    def apply(
        self,
        verdict: GroupVerdict,
        data: GroupGradeData,
        traces_by_label: dict[str, list[vf.Trace]],
        common: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Zero confirmed hacks, rescale the passes by the ranking, and assign the
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
        passes = [label for label, candidate in candidates.items() if candidate.passed and hacks[label] != "confirmed"]
        shrink = CONFIDENCE_SHRINK[verdict.confidence]
        rates = win_rates([[label for label in tier if label in passes] for tier in verdict.ranking])
        rates = {label: 0.5 + shrink * (rate - 0.5) for label, rate in rates.items()}
        f_min = self.config.f_min
        # A pass the ranking leaves out (a confirmed hack that failed its citation) sits mid-table.
        quality = {label: 1.0 if len(passes) == 1 else f_min + (1 - f_min) * rates.get(label, 0.5) for label in passes}

        traces = [(label, trace) for label, label_traces in traces_by_label.items() for trace in label_traces]
        rewards = [0.0 if hacks[label] == "confirmed" else trace.reward for label, trace in traces]
        advantages, lam = redistribute(
            rewards, [quality.get(label, 1.0) for label, _ in traces], self.config.lambda_max
        )
        for (label, trace), advantage in zip(traces, advantages, strict=True):
            assign_advantages(trace, advantage)
            trace.record_metric("gar/lambda", lam)
            trace.record_metric("gar/all_fail", float(not passes))
            if candidates[label].passed:
                trace.record_metric("gar/hack_confirmed", float(hacks[label] == "confirmed"))
                trace.record_metric("gar/hack_suspected", float(hacks[label] == "suspected"))
            if label in quality:
                trace.record_metric("gar/f", quality[label])

        tiers = {label: index for index, tier in enumerate(verdict.ranking) for label in tier}
        return {
            label: {
                "label": label,
                **common,
                "hack": hacks[label],
                "hack_kind": verdicts[label].hack_kind,
                "evidence": [evidence.model_dump() for evidence in verdicts[label].evidence],
                "axes": verdicts[label].axes,
                "tier": tiers.get(label),
                "w": rates.get(label),
                "f": quality.get(label),
                "lambda": lam,
            }
            for label in candidates
        }

    async def log_grade(self, grade: vf.Episode, episodes: list[vf.Episode], labels: dict[str, str]) -> None:
        """Log the grader's episode to the trace stream as kind ``grade``, linked to
        its group and, by label, to the candidate traces."""
        for trace in grade.traces:
            trace.info["kind"] = "grade"
            trace.info["group_id"] = episode_group_id(episodes[0])
            trace.info["candidates"] = labels
        await monitors.log([grade], train_work(episodes[0]).step, "grade", "all")
