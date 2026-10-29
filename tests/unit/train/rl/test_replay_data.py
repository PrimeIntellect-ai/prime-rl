import numpy as np
import pytest
import torch

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.trainer.rl.data import ReplayDataLoader
from prime_rl.trainer.world import reset_world
from prime_rl.transports.batch.replay import ReplayMeta, write_replay_meta, write_replay_step
from prime_rl.transports.batch.types import TrainingSample

SEQ_LEN = 32
NUM_RANKS = 2


def sample(length: int, first_id: int) -> TrainingSample:
    mask = np.arange(length) >= length // 2
    return TrainingSample(
        token_ids=np.arange(first_id, first_id + length, dtype=np.int32),
        mask=mask,
        logprobs=np.where(mask, -0.25, 0.0).astype(np.float32),
        temperatures=np.full(length, 0.5, dtype=np.float32),
        env_name="env",
        advantages=np.where(mask, 1.5, 0.0).astype(np.float32),
        trace_id=f"trace-{first_id}",
        branch_index=0,
    )


@pytest.fixture
def replay_dir(tmp_path):
    steps = [
        prepare_batch([sample(n, 100 * i) for i, n in enumerate(lengths)], SEQ_LEN, NUM_RANKS, sum)
        for lengths in ([20, 12, 9, 30], [17, 5, 26])
    ]
    for step, grid in enumerate(steps):
        write_replay_step(tmp_path, step, grid)
    write_replay_meta(tmp_path, ReplayMeta(num_steps=len(steps), seq_len=SEQ_LEN, num_train_workers=NUM_RANKS))
    return tmp_path, steps


@pytest.fixture
def world(monkeypatch):
    """Rank 3 of 4: data rank 1 of 2 with two ranks (e.g. CP) per data rank."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    reset_world()
    yield
    reset_world()


def test_replay_loader_returns_its_data_rank_of_each_step_and_cycles(replay_dir, world):
    directory, steps = replay_dir
    loader = ReplayDataLoader(directory, SEQ_LEN, NUM_RANKS, 8)
    assert loader.dp_rank == 1

    for step in [0, 1, 0]:
        loader.wait_for_batch()
        batch = loader.get_batch()
        expected = steps[step][1]
        assert len(batch) == len(expected)
        for tensors, micro_batch in zip(batch, expected):
            num_tokens = len(micro_batch.input_ids)
            padding = -num_tokens % 8
            assert tensors["input_ids"].shape == (1, num_tokens + padding) and tensors["input_ids"].dtype == torch.int64
            assert not tensors["loss_mask"][0, num_tokens:].any()
            for name in ("input_ids", "position_ids", "loss_mask", "advantages", "inference_logprobs", "temperatures"):
                written = torch.from_numpy(getattr(micro_batch, name)).to(tensors[name].dtype)
                torch.testing.assert_close(tensors[name][0, :num_tokens], written)
            padded_lengths = micro_batch.sequence_lengths[:-1] + [micro_batch.sequence_lengths[-1] + padding]
            assert tensors["seq_lens"].tolist() == padded_lengths
            assert tensors["sequence_lengths"] == padded_lengths
            assert tensors["padding"] == padding
            assert tensors["trace_ids"] == micro_batch.trace_ids
            assert tensors["routed_experts"] is None and tensors["top_logprobs_ids"] is None


@pytest.mark.parametrize(
    "seq_len,dp_world_size,message",
    [
        (SEQ_LEN, 1, "packed for 2 data ranks, but the trainer has 1"),
        (64, NUM_RANKS, "packed for seq_len 32"),
    ],
)
def test_replay_loader_rejects_a_mismatched_topology(replay_dir, world, seq_len, dp_world_size, message):
    with pytest.raises(ValueError, match=message):
        ReplayDataLoader(replay_dir[0], seq_len, dp_world_size, 8)


def test_replay_config_excludes_fake_data_score_centering_and_router_replay(tmp_path):
    with pytest.raises(ValueError, match="at most one of data.fake and data.replay"):
        TrainerConfig(data={"fake": {}, "replay": tmp_path})
    with pytest.raises(ValueError, match="requires loss.score_centering = false"):
        TrainerConfig(data={"replay": tmp_path}, loss={"type": "icepop", "score_centering": True})
    with pytest.raises(ValueError, match="requires enable_router_replay = false"):
        TrainerConfig(data={"replay": tmp_path}, loss={"type": "icepop"}, enable_router_replay=True)
    assert TrainerConfig(data={"replay": tmp_path}, loss={"type": "icepop"}).data.replay == tmp_path
