"""Temporary manifest-backed SWE-rebench tasksets for the NGU experiment."""

import json
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from swerebench_v2.taskset import SWERebenchV2Data, SWERebenchV2Task, repo_workdir

__all__ = ["NGUSWETaskset", "NGUSWEEnv"]


class NGUSWEData(SWERebenchV2Data):
    difficulty: Literal["easy", "medium", "hard", "extra-hard", "unclassified"] | None = None


class NGUSWETask(SWERebenchV2Task, vf.Task[NGUSWEData]):
    pass


class NGUSWEConfig(vf.TasksetConfig):
    manifest: Path


class NGUSWETaskset(vf.Taskset[NGUSWETask, NGUSWEConfig]):
    def load(self):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        manifest = json.loads(self.config.manifest.read_text())
        ids = manifest["task_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate task IDs in NGU manifest")
        path = hf_hub_download(
            manifest["dataset"], manifest["filename"], repo_type="dataset", revision=manifest["revision"]
        )
        selected = set(ids)
        rows = pq.read_table(path).to_pylist()
        by_id = {row["instance_id"]: (index, row) for index, row in enumerate(rows) if row["instance_id"] in selected}
        if set(by_id) != selected:
            raise ValueError(f"Missing manifest tasks: {sorted(selected - set(by_id))}")
        for task_id in ids:
            index, row = by_id[task_id]
            yield NGUSWETask(
                NGUSWEData(
                    idx=index,
                    name=task_id,
                    difficulty=manifest.get("difficulty_by_task", {}).get(task_id, manifest.get("bucket")),
                    prompt=row["problem_statement"],
                    image=row["image_name"],
                    workdir=repo_workdir(row["repo"]),
                    resources=vf.TaskResources(cpu=4, memory=4, disk=10),
                    install_config=row["install_config"],
                    base_commit=row.get("base_commit") or "",
                    test_patch=row.get("test_patch") or "",
                    gold_patch=row.get("patch") or "",
                    fail_to_pass=list(row.get("FAIL_TO_PASS") or []),
                    pass_to_pass=list(row.get("PASS_TO_PASS") or []),
                ),
                self.config.task,
            )


class NGUSWEEnv(vf.SingleAgentEnv):
    """Count exhaustion of the solve budget as a failed attempt, including on resume."""

    async def finalize(self, task, episode):
        for trace in episode.traces:
            timed_out = (
                not trace.ok
                and len(trace.errors) == 1
                and trace.errors[0].type == "HarnessError"
                and trace.errors[0].message == "agent timeout: rollout exceeded its 3600s budget"
            )
            trace.record_metric("solve_timeout", float(timed_out))
            if timed_out:
                trace.record_reward("solved", 0.0)
                # Keep the original error for diagnosis; timeout is a valid zero in this measurement.
                trace.ok = True
