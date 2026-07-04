"""replay-recheck-v1 — re-serve finished rollouts with a check-your-work turn.

A thin derivation over the `verifiers.v1.tasksets.replay` base: seeds a saved rollout's
final conversation plus an appended instruction to re-verify and fix the answer, and
the new rollout is scored by the original (`inner`) taskset.
"""

import verifiers.v1 as vf
from verifiers.v1.tasksets.replay import (
    ReplayTask,
    ReplayTaskset,
    ReplayTasksetConfig,
    build_children,
    main_tree,
    recheck_seed,
)

RECHECK_INSTRUCTION = (
    "Carefully check your work above. Re-verify the reasoning and the final answer, and "
    "fix any mistakes you find. Then state your final answer."
)


class ReplayRecheckConfig(ReplayTasksetConfig):
    instruction: str = RECHECK_INSTRUCTION
    """The user turn appended to the finished conversation."""


class ReplayRecheckTaskset(ReplayTaskset, vf.Taskset[ReplayTask, ReplayRecheckConfig]):
    def build_prompt(self, record: dict, anchor: int | None) -> list[dict]:
        nodes = record["nodes"]
        children, _ = build_children(nodes)
        return recheck_seed(nodes, children, main_tree(children), self.config.instruction)
