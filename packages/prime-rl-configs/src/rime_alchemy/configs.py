"""Lightweight schema and task identity; no Alchemy runtime imports."""

from typing import Literal

import verifiers.v1 as vf
from pydantic import Field, model_validator
from verifiers.v1.judges.decision import DecisionJudgeConfig

MEMORY_LEVELS = [
    "No useful change: redundant, cosmetic, unsupported, or harmful changes",
    "Minor useful addition with little actionable new information",
    "Meaningful discovery, supported correction, or concrete useful experimental lead",
    "Substantial actionable knowledge, with uncertainty represented and little redundancy",
    "Several substantial discoveries or corrections, concise and useful for future trials",
]


def memory_judge_config():
    return DecisionJudgeConfig(model="cloudflare/clef-flash", levels=list(MEMORY_LEVELS))


class StreamData(vf.TaskData):
    episode_index: int


class StreamTask(vf.Task[StreamData]):
    pass


class RIMEAlchemyTasksetConfig(vf.TasksetConfig):
    split: str = "train"


class RIMEAlchemyEnvConfig(vf.EnvConfig):
    actor: vf.AgentConfig = vf.AgentConfig()
    writer: vf.AgentConfig = vf.AgentConfig()
    exploration_judge: DecisionJudgeConfig = Field(default_factory=memory_judge_config)
    memory_chars: int = Field(4096, ge=1)
    carry_memory: bool = True
    writer_frozen: bool = False
    actor_frozen: bool = False
    score_exploration: bool = True
    action_decoding: Literal["original", "visible_legal_schema"] = "original"
    eval_untrainable: bool = False
    memory_mode: Literal["written", "history", "shuffled", "fork"] = "written"
    fork_trial: int = Field(4, ge=1, le=8)
    fork_horizon: int = Field(2, ge=1, le=5)
    fork_repetitions: int = Field(2, ge=1, le=8)
    judge_max_concurrent: int = Field(1, ge=1)
    judge_max_retries: int = Field(5, ge=0)
    judge_retry_timeout: float = Field(180.0, gt=0)
    judge_local: bool = False

    @model_validator(mode="after")
    def require_shared_policy(self):
        if self.memory_mode == "fork" and (
            self.fork_horizon > self.fork_trial or self.fork_trial + self.fork_horizon > 10
        ):
            raise ValueError("Memory forks need a full past window and must fit inside ten trials")
        if self.action_decoding == "visible_legal_schema" and (
            not self.actor_frozen or self.score_exploration or self.judge_local or self.memory_mode != "written"
        ):
            raise ValueError("Legal ACT requires frozen actor, written memory and no judge")
        if self.actor_frozen and self.writer_frozen and not self.eval_untrainable:
            raise ValueError("At least one RIME role must remain trainable")
        if any(agent.model is not None or agent.client is not None for agent in (self.actor, self.writer)):
            raise ValueError("ACT and WRITE must inherit the same trainable run policy")
        return self
