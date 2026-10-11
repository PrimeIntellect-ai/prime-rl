from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import Progress


def saved_episode(policy_start: int) -> str:
    return vf.WireEpisode.model_validate(
        {
            "task": {"type": "test", "data": {"idx": 0}},
            "run": {
                "type": "train",
                "id": "run",
                "work": {"type": "train", "step": policy_start, "policy": {"start": policy_start, "end": policy_start}},
            },
        }
    ).model_dump_json()


def test_resume_drops_restored_group_members_past_max_off_policy_steps() -> None:
    group_size = SimpleNamespace(config=SimpleNamespace(group_size=4))
    sink = TrainSink(
        SimpleNamespace(max_off_policy_steps=2),
        tokenizer=None,
        train_envs=SimpleNamespace(get=lambda name: group_size),
        progress=Progress(step=10),
        batch_size=8,
    )
    task = vf.Task(vf.TaskData(idx=0, prompt="task"))
    # Batch 10 trains v9, so v7 is the oldest dispatch version still within 2 steps.
    requests = sink.load_state_dict(
        {
            "pending_batch": {},
            "queued": {},
            "open_groups": {
                "mixed": ("env", task, [saved_episode(9), saved_episode(7), saved_episode(6)]),
                "stale": ("env", task, [saved_episode(6)]),
            },
        }
    )

    assert [episode.run.work.policy.start for episode in sink.pending_groups["mixed"]] == [9, 7]
    assert "stale" not in sink.pending_groups
    assert [(request.group_id, request.rollouts) for request in requests] == [("mixed", 2), ("stale", 4)]
