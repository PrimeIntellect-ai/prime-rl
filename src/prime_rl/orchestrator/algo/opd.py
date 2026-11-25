from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf

from prime_rl.configs.algorithm import OPDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.clients import ModelScorer, setup_client

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


class OPDAlgorithm(Algorithm):
    """On-policy distillation. Needs a teacher: the frozen reference model the
    per-token reverse KL is computed against.

    The policy samples its own rollouts; as each finishes, its replica's
    sample server prefill-scores each sample's full context under the teacher
    (``ref_logprobs``), and the trainer evaluates the KL against the live policy. No
    credit is assigned — rollouts keep ``advantages=None`` and samples ship no
    advantage stream; ``group_size`` only fans out sampling."""

    def __init__(self, config: OPDAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.teacher = config.teacher
        self.teacher_clients: ModelScorer | None = None  # frozen teacher endpoint, bound in the sample server

    async def setup(self) -> None:
        # The orchestrator never scores; connecting checks the teacher is up before the run starts.
        await self.connect(self.teacher)

    def bind(self, client: vf.ClientConfig, model: str) -> None:
        self.teacher_clients = ModelScorer(setup_client(self.teacher), self.teacher.name)

    async def score_sample(self, token_ids: np.ndarray, sampled: np.ndarray, context: dict) -> np.ndarray:
        teacher = self.teacher_clients
        assert teacher is not None, "teacher not bound — Algorithm.bind() must run first"
        logprobs = await teacher.score(token_ids.tolist())
        return np.where(sampled, np.asarray(logprobs, dtype=np.float32), np.float32(0.0))
