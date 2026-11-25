from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.configs.algorithm import OPDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.routing import assign_reference_logprobs
from prime_rl.orchestrator.clients import ModelScorer, setup_client
from prime_rl.orchestrator.trajectories import iter_trainable_branches

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


class OPDAlgorithm(Algorithm):
    """On-policy distillation. Needs a teacher: the frozen reference model the
    per-token reverse KL is computed against.

    The policy samples its own rollouts; as each finishes, the env worker that
    ran it prefill-scores each sample's full context under the teacher
    (``ref_logprobs``), and the trainer evaluates the KL against the live policy. No
    credit is assigned — rollouts keep ``advantages=None`` and samples ship no
    advantage stream; ``group_size`` only fans out sampling."""

    def __init__(self, config: OPDAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.teacher = config.teacher
        self.teacher_clients: ModelScorer | None = None  # frozen teacher endpoint, bound in the env worker

    async def setup(self) -> None:
        # The orchestrator never scores; connecting checks the teacher is up before the run starts.
        await self.connect(self.teacher)

    def bind(self, client: vf.ClientConfig, model: str) -> None:
        self.teacher_clients = ModelScorer(setup_client(self.teacher), self.teacher.name)

    async def score_episode(self, episode: vf.Episode) -> None:
        teacher = self.teacher_clients
        assert teacher is not None, "teacher not bound — Algorithm.bind() must run first"
        branches = [
            branch for _, trace in iter_trainable_traces([episode]) for branch, _ in iter_trainable_branches(trace)
        ]
        scores = await asyncio.gather(*(teacher.score(branch.token_ids.tolist()) for branch in branches))
        for branch, logprobs in zip(branches, scores, strict=True):
            assign_reference_logprobs(branch, logprobs)
