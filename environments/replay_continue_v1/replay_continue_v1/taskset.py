"""replay-continue-v1 — resume saved rollouts mid-way and let the model finish the task.

A thin derivation over the `verifiers.v1.tasksets.replay` base: anchors a saved rollout
at a context-compaction point (the seed is the post-compaction prompt the original
harness restarted from) or right after a tool result, and the new rollout is scored by
the original (`inner`) taskset.
"""

import random
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.replay import (
    ReplayTask,
    ReplayTaskset,
    ReplayTasksetConfig,
    compaction_forks,
    continue_seed,
    tool_call_anchors,
)


class ReplayContinueConfig(ReplayTasksetConfig):
    anchor: Literal["compaction", "tool-call"] = "compaction"
    """Where to resume: `"compaction"` seeds the post-compaction prompt of each
    compaction point (one task per compaction; only compacted rollouts are sources).
    `"tool-call"` seeds the conversation up to a tool result and resumes right after it
    (one deterministically sampled resume point per source rollout)."""


class ReplayContinueTaskset(ReplayTaskset, vf.Taskset[ReplayTask, ReplayContinueConfig]):
    def record_anchors(self, record, children, roots, tree) -> list[int | None]:
        nodes = record["nodes"]
        if self.config.anchor == "compaction":
            # One task per compaction point.
            return compaction_forks(nodes, children, tree)
        # Tool calls are numerous (~dozens per rollout); one deterministically sampled
        # resume point per source keeps the index one-candidate-per-rollout and
        # identical across pool workers.
        anchors = tool_call_anchors(nodes, children, tree)
        return [random.Random(record.get("id", "")).choice(anchors)] if anchors else []

    def build_prompt(self, record: dict, anchor: int | None) -> list[dict]:
        return continue_seed(record["nodes"], anchor)
