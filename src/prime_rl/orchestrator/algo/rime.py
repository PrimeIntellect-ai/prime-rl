"""Task-position-relative ACT/WRITE credit for complete Alchemy stream cohorts."""

from __future__ import annotations

import asyncio

import verifiers.v1 as vf

from prime_rl.configs.algorithm import RIMEAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.rime_credit import assign_credit
from prime_rl.orchestrator.algo.routing import assign_advantages, assign_reference_logprobs
from prime_rl.orchestrator.trajectories import iter_trainable_branches


class RIMEAlgorithm(Algorithm):
    preserve_zero_advantages = True

    def __init__(self, config: RIMEAlgoConfig, clients):
        super().__init__(config, clients)
        self.horizon = config.horizon
        self.beta = config.beta
        self.reference = config.reference
        self.reference_clients = None
        self.expected_group_size = config.expected_group_size
        self.max_sequence_tokens = config.max_sequence_tokens

    async def setup(self) -> None:
        if self.reference is not None:
            self.reference_clients = await self.connect(self.reference)

    async def score_episode(self, episode: vf.Episode) -> None:
        if self.reference is not None and self.reference_clients is None:
            raise RuntimeError("Frozen initial-policy reference must be connected")
        branches = []
        for _, trace in iter_trainable_traces([episode]):
            sampling = trace.agent.config.sampling
            if sampling is None or sampling.temperature != 1.0 or sampling.top_p != 1.0:
                raise ValueError("RIME reference scoring requires temperature=1 and top_p=1")
            if getattr(sampling, "top_k", 0) not in (0, -1, None):
                raise ValueError("RIME reference scoring requires full-vocabulary sampling")
            for branch, _ in iter_trainable_branches(trace):
                if len(branch.token_ids) > self.max_sequence_tokens:
                    raise ValueError("RIME sequence exceeds the configured training context")
                branches.append(branch)
        if self.reference is None:
            return
        scores = await asyncio.gather(*(self.reference_clients.score(branch.token_ids.tolist()) for branch in branches))
        for branch, logprobs in zip(branches, scores, strict=True):
            assign_reference_logprobs(branch, logprobs)

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        if (
            len(episodes) != self.expected_group_size
            or len({episode.id for episode in episodes}) != len(episodes)
            or any(not episode.ok for episode in episodes)
        ):
            raise ValueError("RIME requires a complete, clean sibling cohort")
        rows = []
        indexed = []
        identity = None
        for episode in episodes:
            members = []
            row = None
            for trace in episode.traces:
                if not trace.agent.trainable or trace.is_dropped:
                    continue
                if trace.has_error or trace.num_turns == 0:
                    raise ValueError("Incomplete RIME training trace")
                meta = trace.info.get("rime")
                if not isinstance(meta, dict):
                    raise ValueError("Trainable trace is missing RIME stream metadata")
                candidate = (
                    meta["episode_index"],
                    meta["task_rewards"],
                    meta["exploration_rewards"],
                    meta["format_gates"],
                )
                if row is None:
                    row = candidate
                elif row != candidate:
                    raise ValueError("Inconsistent rewards within one RIME sibling")
                role, position = meta["role"], meta["position"]
                if role not in {"actor", "writer"} or type(position) is not int:
                    raise ValueError("Invalid RIME trace role or task position")
                members.append((trace, role, position))
            if row is None:
                raise ValueError("RIME sibling has no trainable ACT/WRITE traces")
            if identity is None:
                identity = row[0]
            if row[0] != identity:
                raise ValueError("RIME siblings must face the same task stream")
            tasks = len(row[1])
            if any(not 0 <= position < tasks for _, _, position in members):
                raise ValueError("RIME task position outside the stream")
            for t in range(tasks):
                if sum(role == "actor" and position == t for _, role, position in members) != 1:
                    raise ValueError("Need one complete actor trajectory per trial")
                writers = sum(role == "writer" and position == t for _, role, position in members)
                if writers != int(t < tasks - 1):
                    raise ValueError("Need one writer per nonterminal trial and no terminal writer")
            rows.append(row)
            indexed.append(members)
        credit = assign_credit(
            [row[1] for row in rows],
            [row[2] for row in rows],
            [row[3] for row in rows],
            horizon=self.horizon,
            beta=self.beta,
        )
        for i, members in enumerate(indexed):
            for trace, role, t in members:
                advantage = float((credit.actor if role == "actor" else credit.writer)[i, t])
                assign_advantages(trace, advantage)
                trace.info["advantage"] = advantage
                if role == "writer":
                    trace.record_metric("rime/writer_improvement_reward", float(credit.writer_reward[i, t]))
