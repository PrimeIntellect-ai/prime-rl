"""Lightweight schema and task identity; no Alchemy runtime imports."""

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

    @model_validator(mode="after")
    def require_shared_policy(self):
        if any(agent.model is not None or agent.client is not None for agent in (self.actor, self.writer)):
            raise ValueError("ACT and WRITE must inherit the same trainable run policy")
        return self
