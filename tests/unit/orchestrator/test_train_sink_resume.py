from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import Progress


def saved_episode(policy_start: int) -> dict:
    return vf.WireEpisode.model_validate(
        {
            "task": {"type": "test", "data": {"idx": 0}},
            "run": {
                "type": "train",
                "id": "run",
                "work": {"type": "train", "step": policy_start, "policy": {"start": policy_start, "end": policy_start}},
            },
        }
    ).model_dump()


def test_resume_drops_queued_traces_with_missing_payload_files(tmp_path) -> None:
    group_size = SimpleNamespace(config=SimpleNamespace(group_size=4))
    sink = TrainSink(
        SimpleNamespace(max_staleness=2),
        tokenizer=None,
        train_envs=SimpleNamespace(get=lambda name: group_size),
        progress=Progress(step=10),
        batch_size=8,
    )
    present = tmp_path / "present.bin"
    present.touch()

    def sample(file):
        return SimpleNamespace(payload=[SimpleNamespace(file=str(file))])

    sink.load_state_dict(
        {
            "pending_batch": {"kept": [sample(present)], "read": [sample(tmp_path / "deleted.bin")]},
            "queued_episodes": {"kept": saved_episode(9), "read": saved_episode(9)},
            "open_groups": {},
        }
    )

    assert list(sink.pending_batch) == list(sink.episode_by_trace) == ["kept"]
