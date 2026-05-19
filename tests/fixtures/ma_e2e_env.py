from __future__ import annotations

from typing import Any

from verifiers.envs.multi_agent_env import MultiAgentEnv
from verifiers.envs.multi_agent_kernel import StaticSchedule, TurnSlot
from verifiers.rubrics.multi_agent_rubric import MultiAgentRubric
from verifiers.types import MARScore, MemberScore, Messages, State


class SmokeRubric(MultiAgentRubric):
    async def build_marscore(self, state: State) -> MARScore:
        return MARScore(
            members=[
                MemberScore(member_id="debater_a", reward=1.0),
                MemberScore(member_id="debater_b", reward=-1.0),
                MemberScore(member_id="judge", reward=0.0),
            ],
            episode_scalar=1.0,
        )


class SmokeMultiAgentEnv(MultiAgentEnv):
    async def build_prompt(self, state: State, member_id: str, slot: TurnSlot) -> Messages:
        messages: Messages = [
            {"role": "system", "content": f"You are {member_id}."},
            *state["prompt"],
        ]
        for utt in state["_kernel"].transcript:
            messages.append(
                {
                    "role": "user",
                    "content": f"[{utt.member_id}/{utt.phase}] {utt.public_channel}",
                }
            )
        messages.append({"role": "user", "content": f"phase={slot.phase}"})
        return messages

    async def render_completion(self, state: State) -> None:
        state["completion"] = [message for step in state["trajectory"] for message in step["completion"]]


def load_environment(**_: Any) -> SmokeMultiAgentEnv:
    return SmokeMultiAgentEnv(
        schedule=StaticSchedule(
            (
                TurnSlot(slot_id=0, agents=("debater_a",), phase="propose"),
                TurnSlot(slot_id=1, agents=("debater_b",), phase="propose"),
                TurnSlot(slot_id=2, agents=("debater_a",), phase="critique"),
                TurnSlot(slot_id=3, agents=("debater_b",), phase="critique"),
                TurnSlot(slot_id=4, agents=("judge",), phase="final"),
            )
        ),
        members=["debater_a", "debater_b", "judge"],
        dataset=lambda: None,
        rubric=SmokeRubric(members=["debater_a", "debater_b", "judge"]),
    )
